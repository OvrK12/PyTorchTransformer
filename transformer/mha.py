import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)

        self.w0 = nn.Linear(emb_dim, emb_dim)
    
    def forward(self, query: torch.tensor, key_value: torch.tensor = None, mask: torch.BoolTensor = None):
        if key_value is None:
            key_value = query
        batch_size = query.shape[0]

        # shape after this step: batch_size x num_heads x seq_length x emb_dim
        q = self.Q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.K(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.V(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # output has shape batch_size x num_heads x seq_length x seq_length
        out = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        if mask is not None:
            out = out.masked_fill(mask == 0, float("-inf"))

        out = nn.functional.softmax(out, dim=-1)
        # output has shape batch_size x num_heads x seq_length x emb_dim
        out = torch.matmul(out, v)
        # output has shape batch_size x seq_length x emb_dim
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.emb_dim)

        return self.w0(out)