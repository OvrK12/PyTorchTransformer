import sys
sys.path.append('..')
import torch
import unittest
from transformer.transformer_block import TransformerBlock

class TestTransformerBlock(unittest.TestCase):
    def test_output_shape(self):
        """
        Test whether input/output shapes match
        """        
        batch_size = 8
        seq_length = 10
        emb_dim = 512
        num_heads = 8
        forward_dim = 4 * emb_dim
        dropout_rate = 0.1

        transformer = TransformerBlock(forward_dim, emb_dim, num_heads, dropout_rate)
        x = torch.randn(batch_size, seq_length, emb_dim)
        mask = None

        output = transformer(x, mask)
        
        self.assertEqual(output.shape, (batch_size, seq_length, emb_dim))

    def test_deterministic_output(self):
        """
        Test whether same input produces same output across different forward passes
        """       
        torch.manual_seed(42)
        batch_size = 8
        seq_length = 10
        emb_dim = 512
        num_heads = 8
        forward_dim = 4 * emb_dim
        dropout_rate = 0.1

        transformer = TransformerBlock(forward_dim, emb_dim, num_heads, dropout_rate)
        # set to eval mode to disable dropout layers
        transformer.eval()
        x = torch.randn(batch_size, seq_length, emb_dim)
        mask = None

        output_1 = transformer(x, mask)
        output_2 = transformer(x, mask)
        
        self.assertTrue(torch.equal(output_1, output_2))

if __name__ == "__main__":
    unittest.main()