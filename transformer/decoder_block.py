from torch import nn
from transformer.mha import MultiHeadAttention
from transformer.transformer_block import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, forward_dim: int, emb_dim: int, num_heads: int, dropout_rate: int):
        super().__init__()
        self.attention_block = MultiHeadAttention(emb_dim, num_heads)
        self.transformer_block = TransformerBlock(forward_dim, emb_dim, num_heads, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, query, key_value, src_mask, tgt_mask):
        attention = self.attention_block(query, mask=tgt_mask)
        # residual connection
        out = self.norm(query + self.dropout(attention))

        # use key/value from encoder and query from self attention
        return self.transformer_block(out, key_value, mask=src_mask)
