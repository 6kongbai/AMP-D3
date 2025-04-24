import pytorch_lightning as pl
import torch
from tqdm import tqdm

from src.modules.sequence.encode import *
from src.modules.sequence.transformer import SeqTransformer, AmpTransformer
from src.utils.constant import seq_length_freq, get_seq_constant_init
from src.utils.diffusion_util import get_para_schedule
from src.utils.embed.embedding import sequence_embedding
from src.utils.embed.sequence import index_to_fasta


class MyModel(pl.LightningModule):
    def __init__(
            self,
            seq_n_class=20,
            seq_n_seq_emb=80,
            seq_n_hidden=128,
            seq_clamp=-50,
            seq_n_blocks=12,
            slice_size=5,
            n_timestep=1000,
            n_self_atte_head=4,
            beta_schedule="linear",
            beta_start=1.e-7,
            beta_end=2.e-2,
            temperature=0.1,
            learning_rate_struct=5e-3,
            learning_rate_seq=5e-3,
            learning_rate_cont=5e-3,
            loss_weight=0.9,
    ):
        super().__init__()
        self.learning_rate_struct = learning_rate_struct
        self.learning_rate_seq = learning_rate_seq
        self.learning_rate_cont = learning_rate_cont
        self.loss_weight = loss_weight
        self.temperature = temperature

        self.time_sampler = torch.distributions.Categorical(torch.ones(n_timestep))
        self.seq_constant_data = get_seq_constant_init(self.device)

        betas, alphas, alphas_bar = get_para_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timestep=n_timestep,
            device=self.device
        )

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)

        self.num_timestep = n_timestep
        self.n_class = seq_n_class
        self.clamp = seq_clamp

        self.transformer = AmpTransformer(
            input_dim=seq_n_seq_emb,
            output_dim=seq_n_hidden,
            n_block=seq_n_blocks,
            slice_size=slice_size,
        )

        self.seq_ffn = SeqFFN(seq_n_hidden, seq_n_class)
        self.seq_attention = SelfAttention(
            n_emb=seq_n_hidden,
            n_head=n_self_atte_head
        )
        self.seq_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)
        self.metric_loss = MetricLoss(temperature=temperature)

    def get_loss(self, batch):
        batch_size = batch['logit'].shape[0]
        time_step = torch.ones(batch_size, device=self.device, dtype=torch.int64) * self.time_sampler.sample(
            [batch_size]).to(self.device)
        amp_x0_real, amp_x0_pred, amp_sentence_cont_emb, attention_masks = self.seq_forward(time_step, batch, "AMP")
        _, _, nonamp_sentence_cont_emb, _ = self.seq_forward(time_step, batch, "nonAMP")
        seq_kl_loss = multinomial_kl(amp_x0_pred, amp_x0_real, mask=attention_masks)
        diff_loss = seq_kl_loss
        self.log("diff_loss", diff_loss, on_step=True, on_epoch=False, prog_bar=True)
        seq_metric_loss = self.metric_loss(amp_sentence_cont_emb, nonamp_sentence_cont_emb)
        intra_loss = seq_metric_loss
        contrast_loss = intra_loss
        self.log("contrast_loss", contrast_loss, on_step=True, on_epoch=False, prog_bar=True)
        total_loss = self.loss_weight * diff_loss + (1 - self.loss_weight) * contrast_loss
        self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss, amp_x0_pred, amp_x0_real, attention_masks

    def seq_pred(self, seq_data, time_step, mask):
        seq_emb = self.transformer(seq_data, time_step, mask)
        output = self.seq_ffn(seq_emb)
        seq_pred = F.softmax(output, dim=-1).float()
        return seq_pred, seq_emb

    def seq_forward(self, seq_time_steps, batch, seq_type, diff_statue: bool = True):
        x0_real = batch['logit'] if seq_type == "AMP" else batch['nonamp_logit']
        mask = batch['amp_mask'] if seq_type == "AMP" else batch['nonamp_mask']
        x0_real, mask = x0_real.to(self.device), mask.to(self.device)
        alphas_bar = self.alphas_bar[seq_time_steps]
        noise = get_seq_noise(device=self.device)
        Qt_weight = get_Qt_weight(alphas_bar, noise, device=self.device, n_class=self.n_class)
        x_t = torch.matmul(x0_real, Qt_weight)
        x_t_emb, attention_masks = batch_sequence_embedding(x_t, mask, self.device)
        x0_pred, token_emb = self.seq_pred(x_t_emb, seq_time_steps, attention_masks)
        if diff_statue:
            token_emb, attn = self.seq_attention(token_emb, attention_masks)
            sentence_emb = get_attn_emb(token_emb, attn)
            sentence_cont_emb = self.seq_predictor(sentence_emb)
        else:
            sentence_cont_emb = None

        return x0_real, x0_pred, sentence_cont_emb, attention_masks

    def training_step(self, batch, batch_idx):
        total_loss, amp_x0_pred, amp_x0_real, attention_masks = self.get_loss(batch)
        score = token_aa_acc(amp_x0_pred, amp_x0_real, self.device, mask=attention_masks)

        self.log("seq_score", score, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def q_posterior(self, x0, time_step):
        time_step = time_step.clamp(0, self.num_timestep - 1)

        alphas = self.alphas[time_step]
        alphas_bar_t = self.alphas_bar[time_step]
        alphas_bar_t_1 = self.alphas_bar[time_step - 1]

        noise = get_seq_noise(device=self.device)
        # with marginal distribution

        # q(xt|x0) : x0 -> xt
        # xt_from_x0 = token_alphas_bar_t.sqrt() * x0 + noise * (1.0 - token_alphas_bar_t).sqrt()
        Qt_weight = get_Qt_weight(alphas_bar_t, noise, self.device, self.n_class)
        xt_from_x0 = torch.matmul(x0, Qt_weight)

        # q(xt|xt_1,x0) -> q(xt|xt_1)
        # xt_from_xt_1 = token_alphas.sqrt() * x0 + (1 - token_alphas).sqrt() * noise
        Qt_weight = get_Qt_weight(alphas, noise, self.device, self.n_class)
        xt_from_xt_1 = torch.matmul(x0, Qt_weight)

        # q(xt-1|x0)
        # xt_1_from_x0 = token_alphas_bar_t_1.sqrt() * x0 + noise * (1.0 - token_alphas_bar_t_1).sqrt()
        Qt_weight = get_Qt_weight(alphas_bar_t_1, noise, self.device, self.n_class)
        xt_1_from_x0 = torch.matmul(x0, Qt_weight)

        # p(x0|xt)
        # x_part = torch.log(x0) - torch.log(xt_from_x0)
        # log(p(x0|xt))-log(q(xt|x0))=log(p(x0|xt)/q(xt|x0))
        # x_log_sum_exp = torch.logsumexp(x_part, dim=-1, keepdim=True)
        # x_part = x_part - x_log_sum_exp

        xt_1_from_xt = torch.log(x0) - torch.log(xt_from_x0) + torch.log(xt_from_xt_1) + torch.log(xt_1_from_x0)
        xt_1_from_xt = torch.clamp(xt_1_from_xt, self.clamp, 0)
        # log(p_theta(xt_1|xt))=log(p(x0|xt)) - log(q(xt|x0)) + log(q(xt|xt-1,x0)) + log(q(xt-1|x0))
        xt_1_from_xt = torch.exp(xt_1_from_xt)
        # p_theta(xt_1|xt)
        # xt_1_from_xt = F.softmax(xt_1_from_xt, dim=-1)
        return xt_1_from_xt

    @torch.no_grad()
    def denoise_seq_sample(self, n_seq=1, seq_length=None, fasta_out_statue: bool = False):
        seq_freq = torch.tensor(seq_length_freq, device=self.device)
        D = torch.distributions.Categorical(seq_freq)

        out_seq_list = []
        out_seq_traj = []
        for i in range(n_seq):
            if seq_length is None:
                seq_len = D.sample()
            else:
                seq_len = seq_length[i]
                # seq_len = seq_length

            seq_init = get_seq_noise(seq_len, self.device)
            seq_index_t = logit_to_index(seq_init, random_state=True).unsqueeze(dim=0)

            mask = torch.zeros([1, 50], device=self.device, dtype=torch.float32)
            mask[:, :seq_len] = 1.0

            t_list = torch.arange(self.num_timestep - 1, 0, -1).to(self.device)
            print("denoise {}-th sequence".format(i + 1))

            for time_steps in tqdm(t_list):
                seq_emb, attn_mask = sequence_embedding(indexes=seq_index_t, mask=mask, device=self.device)
                time_steps = time_steps.unsqueeze(dim=0)
                seq0_pred, _ = self.seq_pred(seq_emb, time_steps, mask)
                seq_t = self.q_posterior(x0=seq0_pred, time_step=time_steps)
                seq_index_t = logit_to_index(seq_t, random_state=True)
                out_seq_traj.append(index_to_fasta(seq_index_t[0], mask[0]))

            seq_index_final = seq_index_t[0]
            seq_fasta = index_to_fasta(seq_index_final, mask[0])
            out_seq_list.append(seq_fasta)

        if fasta_out_statue:
            record_path = save_output_seq(out_seq_list)
        else:
            record_path = None

        return out_seq_list, record_path, out_seq_traj

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.transformer.parameters(), 'lr': self.learning_rate_seq},
            {'params': self.seq_ffn.parameters(), 'lr': self.learning_rate_seq},
            {'params': self.seq_attention.parameters(), 'lr': self.learning_rate_cont},
            {'params': self.seq_predictor.parameters(), 'lr': self.learning_rate_cont},
        ], weight_decay=1e-2)
        return optimizer
