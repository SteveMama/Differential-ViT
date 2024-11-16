import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=28, patch_size=7, in_channels=1, embed_dim = 64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Sequential(
            # Breaking the images into patches
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)

        # Adding a Classification Token
        cls_tokens = self.cls_token.expand(b, -1 , -1)
        x = torch.cat([cls_tokens, x], dim =1)

        # position embeddings
        x = x + self.pos_embedding

        return x
