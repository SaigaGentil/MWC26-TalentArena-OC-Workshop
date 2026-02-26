from dataclasses import dataclass
import torch
from torch import nn
from .norm import LayerNorm
from .transformer import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int = 50257  # Vocabulary size
    context_length: int = 1024  # Context length
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of layers
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # Query-Key-Value bias


class GPTModel(nn.Module):
    def __init__(self, gpt_config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(
            num_embeddings=gpt_config.vocab_size, embedding_dim=gpt_config.emb_dim
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=gpt_config.context_length, embedding_dim=gpt_config.emb_dim
        )
        self.drop_emb = nn.Dropout(gpt_config.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[
                TransformerBlock(gpt_config=gpt_config)
                for _ in range(gpt_config.n_layers)
            ]
        )

        self.final_norm = LayerNorm(emb_dim=gpt_config.emb_dim)
        self.out_head = nn.Linear(
            in_features=gpt_config.emb_dim,
            out_features=gpt_config.vocab_size,
            bias=False,
        )

        self.out_head.weight = (
            self.tok_emb.weight
        )  # tying the output head weights with the token embedding weights

    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        in_idx = in_idx.type(dtype=torch.long)
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
