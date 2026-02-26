from torch import nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert not (d_out % num_heads), "d_out must be divisible by num_heads because we need to split the output into num_heads different heads for multi head attention"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads

        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )  # masking upper triangular part of the attention scores matrix to prevent attending to future tokens

    def forward(self, x):
        b, num_tokens, d_in = x.shape # x shape: (batch_size, seq_length, emb_dim)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # adding num_heads dimension for efficient multi head mm
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # keys shape: (batch_size, seq_length, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # queries shape: (batch_size, seq_length, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) # values shape: (batch_size, seq_length, num_heads, head_dim)

        # transposing from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim) in order to perform multi head mm efficiently
        keys = keys.transpose(1, 2) # keys shape: (batch_size, num_heads, seq_length, head_dim)
        queries = queries.transpose(1, 2) # queries shape: (batch_size, num_heads, seq_length, head_dim)
        values = values.transpose(1, 2) # values shape: (batch_size, num_heads, seq_length, head_dim)

        attn_scores = queries @ keys.transpose(2, 3) # attn_scores shape: (batch_size, num_heads, seq_length, seq_length)
        mask_bool = self.mask[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf) # masking the upper triangular part of the attention scores matrix to prevent attending to future tokens

        attn_weights = torch.softmax(attn_scores / self.head_dim ** 0.5, dim=-1) # attn_weights shape: (batch_size, num_heads, seq_length, seq_length)

        attn_weights = self.dropout(attn_weights) # applying dropout to attention weights for regularization

        context_vec = (attn_weights @ values).transpose(1, 2) # context_vec shape: (batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, num_heads, head_dim)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # context_vec shape: (batch_size, seq_length, d_out)

        context_vec = self.out_proj(context_vec) # context_vec shape: (batch_size, seq_length, d_out)
        return context_vec