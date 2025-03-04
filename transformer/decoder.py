import math
import torch
from torch import nn
from transformer.decoder_block import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, forward_dim: int, emb_dim: int, num_heads: int, num_layers: int, max_len: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(self.get_sinusoid_table(max_len, emb_dim), freeze=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(forward_dim,emb_dim, num_heads, dropout_rate) for _ in range(num_layers)
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

    def forward(self, input, encoder_out, src_mask, tgt_mask):
        batch_size = input.shape[0]
        seq_length = input.shape[1]
        positions = torch.arange(0,seq_length).unsqueeze(0).repeat(batch_size,1).to(input.device)
        out = self.dropout(self.embedding(input) + self.pos_embedding(positions))

        for layer in self.decoder_blocks:
            out = layer(out, encoder_out, src_mask, tgt_mask)
        
        # return the contextualized embeddings -> decoding happens in main transformer class
        return out