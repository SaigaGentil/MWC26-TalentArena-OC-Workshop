from torch import nn
import torch
from .attention import MultiHeadAttention
from .norm import LayerNorm
from .ff import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, gpt_config: "model.GPTConfig"):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=gpt_config.emb_dim,
            d_out=gpt_config.emb_dim,
            context_length=gpt_config.context_length,
            dropout=gpt_config.drop_rate,
            num_heads=gpt_config.n_heads,
            qkv_bias=gpt_config.qkv_bias,
        )
        self.ff = FeedForward(gpt_config=gpt_config)
        self.norm1 = LayerNorm(emb_dim=gpt_config.emb_dim)
        self.norm2 = LayerNorm(emb_dim=gpt_config.emb_dim)
        self.drop_shortcut = nn.Dropout(p=gpt_config.drop_rate)

    def forward(self, x):
        shortcut = x  # residual connection for attention
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x  # residual connection for ff bloc
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut

        return x
