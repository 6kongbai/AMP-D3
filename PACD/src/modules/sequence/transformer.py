import math

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from src.modules.sequence.encode import SinusoidalPosEmb, SelfAttention


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SeqBlock(nn.Module):
    def __init__(self, n_emb, n_head, attn_drop, resid_drop, n_diff_step, n_seq_max, emb_type):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_emb, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_emb)
        self.dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        self.attn = SelfAttention(n_emb=n_emb, n_head=n_head, attn_drop=attn_drop, resid_drop=attn_drop)

        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, 2 * n_emb),
            nn.GELU(),
            nn.Linear(2 * n_emb, n_emb),
            nn.Dropout(resid_drop),
        )

        if emb_type == "pos_emb":
            self.emb_t = TimestepEmbedder(n_emb)
            self.emb_pos = SinusoidalPosEmb(n_seq_max, n_emb)
        else:
            self.emb_t = nn.Embedding(n_diff_step, n_emb)
            self.emb_pos = nn.Embedding(n_seq_max, n_emb)

        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(n_emb, n_emb)
        self.linear_pos = nn.Linear(n_emb, n_emb)

    def forward(self, x, time_step, mask=None):
        """
        Args:
            x: Input tensor of shape (B, L, D)
            time_step: Time step tensor of shape (B,)
            mask: Optional mask tensor of shape (B, L), 1 for valid positions, 0 for masked
        Returns:
            x: Output tensor of shape (B, L, D)
            att: Attention weights of shape (B, H, L, L)
        """
        B, L, D = x.shape

        # Time embedding
        time_emb = self.silu(self.linear_t(self.emb_t(time_step)))  # [B, D]
        time_emb = time_emb.unsqueeze(1)  # [B, 1, D]

        # Positional embedding
        pos = torch.arange(L, device=x.device)  # [L]
        pos_emb = self.silu(self.linear_pos(self.emb_pos(pos)))  # [L, D]
        pos_emb = pos_emb.unsqueeze(0)  # [1, L, D]

        # Add embeddings
        x = x + time_emb + pos_emb  # Broadcasting: [B, L, D]

        # Attention with mask
        a, att = self.attn(x, mask=mask)
        x = self.dropout(x + a)
        x = self.ln1(x)

        # MLP
        x = self.dropout(x + self.mlp(x))
        x = self.ln2(x)

        return x, att


class Block(nn.Module):
    """
    A block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Combines global self-attention, local attention, and MLP with conditioning.
    """

    def __init__(self, n_emb, n_head, window_size, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        self.slice_size = window_size
        self.enable_block_overlap = False

        # Layer Normalization (no learnable parameters)
        self.norm_input = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.norm_local = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.norm_global = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)
        self.norm_mlp = nn.LayerNorm(n_emb, elementwise_affine=False, eps=1e-6)

        # Attention Mechanisms
        self.global_attn = SelfAttention(n_emb=n_emb, n_head=n_head, attn_drop=attn_drop, resid_drop=resid_drop)
        self.local_attn = SelfAttention(n_emb, n_head=1)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, 2 * n_emb),
            nn.GELU(),
            nn.Linear(2 * n_emb, n_emb),
            nn.Dropout(resid_drop),
        )

        # Adaptive Layer Normalization (adaLN-Zero) modulation networks
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_emb, 9 * n_emb, bias=True)
        )

        # self.adaLN_modulation2 = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(n_emb, 4 * n_emb, bias=True)  # Generates 6 parameters
        # )

    def forward(self, x, c, mask=None):
        """
        Args:
            x: Input tensor of shape (B, L, D)
            c: Conditioning tensor of shape (B, D)
            mask: Padding mask of shape (B, L), True for padded positions, False for valid positions
        """
        # Generate modulation parameters from conditioning tensor
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_nat, scale_nat, gate_nat = self.adaLN_modulation(c).chunk(9, dim=1)

        # shift_mlp, scale_mlp = self.adaLN_modulation2(c).chunk(2, dim=1)
        # Normalize input
        x_normalized = self.norm_input(x)

        # Process local attention with mask
        x_local = self._apply_local_attention(x_normalized, shift_nat, scale_nat, gate_nat, mask)

        # Process global attention with mask
        x_global = self._apply_global_attention(x_normalized, shift_msa, scale_msa, gate_msa, mask)

        # Concatenate local and global features
        x_concat = x_global + x_local
        # x_concat = torch.concat([x_global, x_local], dim=-1)

        # Process MLP
        x_output = self._apply_mlp(x_concat, x_normalized, shift_mlp, scale_mlp, gate_mlp)

        return x_output

    def _apply_local_attention(self, x, shift, scale, gate, mask=None):
        b, l, d = x.shape
        if l % self.slice_size != 0:
            raise ValueError(f"Sequence length {l} must be divisible by slice_size {self.slice_size}")

        x_slices = rearrange(x, 'b (i j) d -> b i j d', i=self.slice_size, j=l // self.slice_size)
        x_slices = rearrange(x_slices, 'b i j d -> (b i) j d')

        if mask is not None:
            mask_slices = rearrange(mask, 'b (i j) -> b i j', i=self.slice_size, j=l // self.slice_size)
            mask_slices = rearrange(mask_slices, 'b i j -> (b i) j')
        else:
            mask_slices = None

        if mask_slices is not None:
            valid_slices = mask_slices.sum(dim=-1) > 0  # 每个切片是否有有效位置
        else:
            valid_slices = torch.ones(x_slices.size(0), dtype=torch.bool, device=x.device)

        x_local = torch.zeros_like(x_slices)
        if valid_slices.any():
            valid_x_slices = x_slices[valid_slices]
            valid_mask_slices = mask_slices[valid_slices] if mask_slices is not None else None
            attn_output, _ = self.local_attn(valid_x_slices, mask=valid_mask_slices)
            x_local[valid_slices] = attn_output.to(x_local.dtype)

        x_local = rearrange(x_local, '(b i) j d -> b i j d', b=b, i=self.slice_size)
        x_local = rearrange(x_local, 'b i j d -> b (i j) d')

        x_local = x + gate.unsqueeze(1) * self.modulate(self.norm_local(x_local), shift, scale)
        return x_local

    def _apply_global_attention(self, x, shift, scale, gate, mask=None):
        """
        Applies global attention with modulation and residual connection.
        """
        x_modulated = self.modulate(self.norm_global(x), shift, scale)
        x_global, _ = self.global_attn(x_modulated, mask=mask)
        x_global = x + gate.unsqueeze(1) * x_global
        return x_global

    def _apply_mlp(self, x_concat, x_original, shift, scale, gate):
        """
        Applies MLP with modulation and residual connection.
        """
        x_modulated = self.modulate(self.norm_mlp(x_concat), shift, scale)
        x_mlp = x_original + gate.unsqueeze(1) * self.mlp(x_modulated)
        return x_mlp

    @staticmethod
    def modulate(x, shift, scale):
        """
        Modulates the input tensor using shift and scale parameters.
        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AmpTransformer(nn.Module):
    def __init__(self,
                 input_dim=None,
                 output_dim=128,
                 slice_size=5,
                 n_emb=128,
                 n_head=16,
                 attn_drop=0.1,
                 resid_drop=0.1,
                 n_block=8,
                 ):
        super().__init__()
        self.cont_emb = nn.Linear(input_dim, n_emb)

        self.register_buffer('pos_embed', get_1d_sincos_pos_embed(n_emb, 50).unsqueeze(0))

        self.t_embedder = TimestepEmbedder(n_emb)
        self.output_emb = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, output_dim),
        )

        self.blocks = nn.ModuleList([
            Block(
                n_emb=n_emb,
                n_head=n_head,
                window_size=slice_size,
                attn_drop=attn_drop,
                resid_drop=resid_drop,
            ) for _ in range(n_block)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights for the DiT model.
        """

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)  # Zero-out adaLN modulation layers
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(self, x, time_step, mask=None):
        """
        Args:
            x: Input tensor of shape (B, L, input_dim)
            time_step: Time step tensor of shape (B,)
            mask: Optional mask tensor of shape (B, L), 1 for valid positions, 0 for masked
        Returns:
            output: Output tensor of shape (B, L, output_dim)
        """

        x_emb = self.cont_emb(x) + self.pos_embed  # [B, L, n_emb]
        c = self.t_embedder(time_step)
        x_input = x_emb
        for block in self.blocks:
            x_emb = block(x_emb, c, mask)
        x_emb = x_emb + x_input
        output = self.output_emb(x_emb)
        return output


def get_1d_sincos_pos_embed(embed_dim, seq_len):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / embed_dim))

    pos_embed = torch.zeros(seq_len, embed_dim, dtype=torch.float32)

    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)

    return pos_embed


class SeqTransformer(nn.Module):
    def __init__(
            self,
            input_dim=None,
            output_dim=128,
            n_emb=128,
            n_head=16,
            attn_drop=0.1,
            resid_drop=0.1,
            n_diff_step=500,
            n_block=8,
            emb_type="pos_emb",
            n_seq_max=50
    ):
        super().__init__()

        self.cont_emb = nn.Linear(input_dim, n_emb)
        self.n_block = n_block

        self.output_emb = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, output_dim),
        )

        self.blocks = nn.ModuleList([
            SeqBlock(
                n_emb=n_emb,
                n_head=n_head,
                attn_drop=attn_drop,
                resid_drop=resid_drop,
                n_diff_step=n_diff_step,
                emb_type=emb_type,
                n_seq_max=n_seq_max
            ) for _ in range(n_block)
        ])

    def forward(self, x, time_step, mask=None):
        """
        Args:
            x: Input tensor of shape (B, L, input_dim)
            time_step: Time step tensor of shape (B,)
            mask: Optional mask tensor of shape (B, L), 1 for valid positions, 0 for masked
        Returns:
            output: Output tensor of shape (B, L, output_dim)
        """
        x_emb = self.cont_emb(x)  # [B, L, n_emb]

        for block in self.blocks:
            x_emb, _ = block(x_emb, time_step, mask)

        output = self.output_emb(x_emb)  # [B, L, output_dim]
        return output
