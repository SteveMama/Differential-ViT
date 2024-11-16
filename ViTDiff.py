from diff_attn import *
from Encoder import *

class VisionTransformerDiff(nn.Module):
    def __init__(self,
                 image_size = 28,
                 patch_size=7,
                 in_channels=1,
                 num_classes=10,
                 embed_dim = 64,
                 depth=6,
                 num_heads =8,
                 mlp_ratio=4. ,
                 qkv_bias = False,
                 drop_rate =0.,
                 attn_drop_rate= 0.):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.blocks = nn.Sequential(*[
            TransformerEncoder(
                dim = embed_dim,
                num_heads= num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop = drop_rate,
                attn_drop= attn_drop_rate
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[: , 0]
        x = self.head(x)
        return x
