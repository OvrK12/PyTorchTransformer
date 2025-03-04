import torch
from torch import nn

from transformer.encoder import Encoder
from transformer.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size = 1000,
            padding_idx = 0,
            forward_dim=2048,
            emb_dim=512,
            num_heads=8,
            num_layers=6,
            max_len=128,
            dropout_rate=0.1,
            ):
        super().__init__()
        self.padding_idx = padding_idx
        self.encoder = Encoder(vocab_size, forward_dim, emb_dim, num_heads, num_layers, max_len, dropout_rate)
        self.decoder = Decoder(vocab_size, forward_dim, emb_dim, num_heads, num_layers, max_len, dropout_rate)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def create_src_mask(self, src):
        device = src.device
        src_mask = (src != self.padding_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.padding_idx).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask * torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        ).to(device)
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)

        # dimensions are batch_size x seq_length x vocab_size
        return self.linear(decoder_out)
