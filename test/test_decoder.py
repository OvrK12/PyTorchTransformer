import sys
sys.path.append('..')
import unittest
import torch
from transformer.decoder import Decoder

class TestDecoder(unittest.TestCase):
    def test_output_shape(self):
        """
        Test whether input/output shapes match
        """ 
        vocab_size = 1000
        num_layers = 8
        max_len = 64
        batch_size = 8
        seq_length = 10
        emb_dim = 512
        num_heads = 8
        forward_dim = 4 * emb_dim
        dropout_rate = 0.1
    
        decoder = Decoder(vocab_size, forward_dim, emb_dim, num_heads, num_layers, max_len, dropout_rate)
        x = torch.randint(0, vocab_size, (batch_size, seq_length))
        src_mask = None
        tgt_mask = None

        output = decoder(x, x, src_mask, tgt_mask)
        
        self.assertEqual(output.shape, (batch_size, seq_length, emb_dim))

if __name__ == "__main__":
    unittest.main()