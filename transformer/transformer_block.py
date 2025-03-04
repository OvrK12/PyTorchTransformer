import torch
from torch import nn

from transformer.mha import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, forward_dim: int, emb_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm_2 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

        self.mha = MultiHeadAttention(emb_dim, num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )
    
    def forward(self, query: torch.tensor, key_value: torch.tensor = None, mask: torch.BoolTensor = None):
        if key_value == None:
            key_value = query
        # input has shape: batch_size x seq_length x emb_dim
        attention = self.mha(query, key_value, mask)
        residual_1 = self.norm_1(query + self.dropout(attention))

        out = self.ffnn(residual_1)
        return self.norm_2(residual_1 + self.dropout(out))