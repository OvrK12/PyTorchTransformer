import math
import torch
from torch import nn
from transformer_block import TransformerBlock

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, forward_dim: int, emb_dim: int, num_heads: int, dropout_rate: int, num_layers: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(self.get_sinusoid_table(max_len, emb_dim), freeze=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(forward_dim, emb_dim, num_heads, dropout_rate) for _ in range(num_layers)
            ])
        
    def get_sinusoid_table(self, max_len, emb_dim):
        def get_angle(pos, i, emb_dim):
            return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

        sinusoid_table = torch.zeros(max_len, emb_dim)
        for pos in range(max_len):
            for i in range(emb_dim):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
                else:
                    sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
        return sinusoid_table
        
    def forward(self, x, mask):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        positions = torch.arange(0,seq_length).unsqueeze(0).repeat(batch_size,1).to(x.device)
        out = self.dropout(self.embedding(x) + self.pos_embedding(positions))
        for layer in self.transformer_blocks:
            out = layer(out,mask)

        return out