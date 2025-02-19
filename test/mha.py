import sys
sys.path.append('..')
import torch
import unittest
from transformer.mha import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        batch_size = 8
        seq_length = 10
        emb_dim = 512
        num_heads = 8
        
        mha = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads)
        
        x = torch.randn(batch_size, seq_length, emb_dim)
        mask = None
        output = mha(x, mask)
        
        self.assertEqual(output.shape, (batch_size, seq_length, emb_dim))


if __name__ == "__main__":
    unittest.main()