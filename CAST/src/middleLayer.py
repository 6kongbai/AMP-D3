import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8):
        super(MultiHeadCrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.query_fc = nn.Linear(input_dim, hidden_dim)
        self.key_fc = nn.Linear(input_dim, hidden_dim)
        self.value_fc = nn.Linear(input_dim, hidden_dim)
        self.attn_fc = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

        # Learnable FiLM parameters (scale and shift)
        self.film_scale = nn.Linear(input_dim, hidden_dim)
        self.film_shift = nn.Linear(input_dim, hidden_dim)

    def forward(self, mod1_feat, mod2_feat):
        batch_size = mod1_feat.size(0)

        # Compute Q, K, V for mod1 and mod2 features
        Q = self.query_fc(mod1_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_fc(mod2_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_fc(mod2_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # FiLM adjustment for mod2 (scale and shift)
        scale = self.film_scale(mod2_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        shift = self.film_shift(mod2_feat).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply FiLM adjustments (scale and shift) to mod2's key and value
        K = K * scale + shift
        V = V * scale + shift

        # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.head_dim ** 0.5
        attn_scores = self.softmax(attn_scores)

        # Apply attention to values: [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_scores, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Final linear layer
        output = self.attn_fc(attn_output)
        return output





