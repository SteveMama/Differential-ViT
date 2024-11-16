import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diff_attn import DifferentialAttention, PatchEmbedding


class TransformerEncoder(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4. ,
                 qkv_bias = False,
                 drop=0. ,
                 attn_drop=0.,
                 act_layer = nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.diff_attn = DifferentialAttention(dim,
                                               num_heads=num_heads,
                                               qkv_bias=qkv_bias,
                                               attn_drop=attn_drop,
                                               proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.diff_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


