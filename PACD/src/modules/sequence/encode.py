import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pytorch_metric_learning import miners, distances, losses, reducers
from torch import nn

from src.utils.constant import aa_count_freq
from src.utils.embed.embedding import sequence_embedding


class SelfAttention(nn.Module):
    def __init__(self, n_emb, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert n_emb % n_head == 0
        self.attn = nn.MultiheadAttention(embed_dim=n_emb, num_heads=n_head, dropout=attn_drop, batch_first=True)
        self.proj = nn.Linear(n_emb, n_emb)
        self.resid_drop = nn.Dropout(resid_drop)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = ~mask.bool()

        attn_output, attn_weights = self.attn(query=x, key=x, value=x, key_padding_mask=mask)
        y = self.resid_drop(self.proj(attn_output))
        return y, attn_weights


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=2000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class SeqFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="silu", dropout=0.1):
        super().__init__()

        self.dim_list = [input_dim, input_dim * 4, input_dim * 2, input_dim, input_dim // 2, input_dim // 4, output_dim]

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 1):
            self.layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class MetricLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        self.miner = miners.MultiSimilarityMiner()
        self.distances = distances.CosineSimilarity()
        self.reducer = reducers.MeanReducer()

        self.metric_trip_loss = losses.TripletMarginLoss(
            distance=self.distances,
            reducer=self.reducer,
        )

        self.metric_cont_loss = losses.ContrastiveLoss(
            distance=self.distances,
            pos_margin=1,
            neg_margin=0
        )

        self.cont_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, AMP_emb, nonAMP_emb, loss_type="cont"):
        assert loss_type in {"cont", "metric_cont", "metric_trip"}, print("metric_loss_type error")

        emb = torch.concat((AMP_emb, nonAMP_emb), dim=0)

        AMP_label = torch.ones(len(AMP_emb), device=AMP_emb.device)
        nonAMP_label = torch.zeros(len(nonAMP_emb), device=nonAMP_emb.device)
        label = torch.concat((AMP_label, nonAMP_label))

        if loss_type == "cont":
            loss = self.cont_loss(emb, label)
        elif loss_type == "metric_cont":
            loss = self.metric_cont_loss(emb, label)
        else:
            hard_pairs = self.miner(emb, label)
            loss = self.trip_loss(emb, label, hard_pairs)

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        n = labels.shape[0]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

        mask_pos = torch.ones_like(similarity_matrix, device=features.device) * (
            labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_neg = torch.ones_like(mask_pos, device=features.device) - mask_pos

        similarity_matrix = torch.exp(similarity_matrix / self.T)

        mask_diag = (torch.ones(n, n) - torch.eye(n, n)).to(features.device)
        similarity_matrix = similarity_matrix * mask_diag

        sim_pos = mask_pos * similarity_matrix
        sim_neg = similarity_matrix - sim_pos
        sim_neg = torch.sum(sim_neg, dim=1).repeat(n, 1).T
        sim_total = sim_pos + sim_neg

        loss = torch.div(sim_pos, sim_total)
        loss = mask_neg + loss + torch.eye(n, n, device=features.device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

        return loss


class MatchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right, match_type="graph"):
        assert match_type in {"node", "graph"}, print("match_type error")
        device = feature_left.device

        if match_type == "node":
            similarity = F.cosine_similarity(feature_left, feature_right, dim=1).to(device)
            similarity = torch.exp(similarity / self.T)
            loss = torch.mean(-torch.log(similarity))
        else:
            n = len(feature_left)
            similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2).to(device)
            similarity = torch.exp(similarity / self.T)

            mask_pos = torch.eye(n, n, device=device, dtype=bool)
            sim_pos = torch.masked_select(similarity, mask_pos)

            sim_total_row = torch.sum(similarity, dim=0)
            loss_row = torch.div(sim_pos, sim_total_row)
            loss_row = -torch.log(loss_row)

            sim_total_col = torch.sum(similarity, dim=1)
            loss_col = torch.div(sim_pos, sim_total_col)
            loss_col = -torch.log(loss_col)

            loss = loss_row + loss_col
            loss = torch.sum(loss) / (2 * n)

        return loss


class MetricPredictorLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.activate = nn.ReLU()

        self.project_emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            self.activate,
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            self.activate,
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, feature):
        out = self.project_emb(feature)
        return out


def index_to_onehot(x, num_classes=20):
    x = torch.tensor(x)

    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'

    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)

    return x_onehot.float()


# sum P(x)log(P(x)/Q(x))
def multinomial_kl(prob1, prob2, mask=None):
    prob1 = torch.softmax(prob1, dim=-1)  # [B, T, C]
    prob2 = torch.softmax(prob2, dim=-1)  # [B, T, C]
    kl_div = prob1 * (torch.log(prob1 + 1e-10) - torch.log(prob2 + 1e-10))  # [B, T, C]
    kl_div = kl_div.sum(dim=-1)

    if mask is not None:
        kl_div = kl_div * mask
        total_valid_tokens = mask.sum()
        kl_div = kl_div.sum() / (total_valid_tokens + 1e-8)
    else:
        kl_div = kl_div.mean()

    return kl_div


def get_time_steps(n_seq, n_timestep, device=None):
    # time_step = torch.randint(
    #     0, n_timestep, size=(n_seq // 2 + 1,), device=device)
    # time_step = torch.cat(
    #     [time_step, n_timestep - time_step - 1], dim=0)[:n_seq]

    time_step = torch.randint(1, n_timestep, size=(n_seq,), device=device)

    return time_step


def get_seq_noise(seq_len=1, device=None, noise_state="dmd", n_class=20):
    # dud: discrete uniform distribution
    # dmd: discrete marginal distribution

    if noise_state == "dud":
        noise = torch.ones([seq_len, n_class], device=device) / n_class
    else:
        noise = torch.tensor(aa_count_freq, device=device).unsqueeze(dim=0).repeat(seq_len, 1)

    return noise


# Qt = alphas_bar * I + (1 - alphas_bar) * K
def get_Qt_weight(alphas_bar, noise, device, n_class=20):
    """
    :param alphas_bar: [batch_size]
    :param noise: [1, n_class]
    :param device:
    :param n_class:
    :return:
    """
    Qt_weight = [bar_t * torch.eye(n_class, device=device) + (1 - bar_t) * noise for bar_t in alphas_bar]
    Qt_weight = torch.stack(Qt_weight).float()
    # [batch_size, n_class, n_class]
    return Qt_weight


def sample_seq_init(batch_size, sample_logit, device):
    D = torch.distributions.Categorical(sample_logit)
    token_index = D.sample([batch_size])
    return token_index.to(device)


def logit_to_index(logit_p, random_state=False):
    if random_state:
        D = torch.distributions.Categorical(logit_p)
        token_index = D.sample()
    else:
        token_index = logit_p.argmax(dim=-1)

    return token_index


def batch_sequence_embedding(seq_logit, mask, device):
    indexes = logit_to_index(seq_logit)
    embeddings, attention_masks = sequence_embedding(indexes=indexes, mask=mask, max_length=50, device=device)
    return embeddings, attention_masks


def token_aa_acc(pred, real, device, mask=None):
    y_pred = torch.argmax(pred, dim=-1)
    y_real = torch.argmax(real, dim=-1)

    correct = (y_pred == y_real).float()

    if mask is not None:
        assert mask.shape == correct.shape, f"Mask shape {mask.shape} does not match prediction shape {correct.shape}"
        correct = correct * mask
        valid_count = mask.sum()
        if valid_count > 0:
            score = correct.sum() / valid_count
        else:
            score = torch.tensor(0.0, device=device)
    else:
        score = correct.sum() / correct.numel()

    return score


def get_attn_emb(seq_emb, seq_attn):
    B, T, C = seq_emb.size()
    attn = seq_attn.mean(dim=1)
    attn_emb = torch.bmm(attn.unsqueeze(1), seq_emb).squeeze(1)
    return attn_emb


def save_output_seq(out_seq_list):
    record_list = []

    for i, seq_str in enumerate(out_seq_list):
        seq_id = "AMP_{}".format(i)
        seq_desc = ""

        record = SeqRecord(Seq(seq_str), id=seq_id, description=seq_desc)
        record_list.append(record)

    time_str = str(pd.Timestamp.now())[:16]
    record_path = "data/output/fasta/AMP_{}.fasta".format(time_str)
    print("generate " + record_path)
    SeqIO.write(record_list, record_path, "fasta")

    return record_path
