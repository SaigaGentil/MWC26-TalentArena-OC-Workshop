from torch import nn
import torch


class FeedForward(nn.Module):
    def __init__(self, gpt_config: "model.GPTConfig"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                in_features=gpt_config.emb_dim, out_features=4 * gpt_config.emb_dim
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(
                in_features=4 * gpt_config.emb_dim, out_features=gpt_config.emb_dim
            ),
        )

    def forward(self, x):
        return self.layers(x)
