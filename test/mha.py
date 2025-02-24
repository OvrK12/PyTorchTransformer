import sys
sys.path.append('..')
import torch
import unittest
from transformer.mha import MultiHeadAttention
from torch.testing import assert_close

class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        """
        Test whether input/output shapes match
        """ 
        batch_size = 8
        seq_length = 10
        emb_dim = 512
        num_heads = 8
        
        mha = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads)
        
        x = torch.randn(batch_size, seq_length, emb_dim)
        mask = None
        output = mha(x, mask)
        
        self.assertEqual(output.shape, (batch_size, seq_length, emb_dim))
    
    def test_with_known_values(self):
        """
        Test correctness of output using known values
        """ 
        model = MultiHeadAttention(emb_dim=5, num_heads=1)

        test_embeddings = [torch.full((5,),0.5),
                           torch.full((5,),1.0),
                           torch.full((5,),2.0),
                           torch.full((5,),3.0),
                           torch.full((5,),4.0),
                           torch.full((5,),5.0),
                        ]
        embedding_list = torch.stack(test_embeddings).unsqueeze(0)
        # Set weights to known values
        with torch.no_grad():
            model.Q.weight.fill_(0.5)
            model.Q.bias.fill_(0)

            model.K.weight.fill_(1.0)
            model.K.bias.fill_(0)

            model.V.weight.fill_(2.0)
            model.V.bias.fill_(0)

            model.w0.weight.fill_(0.5)
            model.w0.bias.fill_(0)
        
        out = model(embedding_list)
        correct_result = torch.full((6,5),125.0).unsqueeze(0)
        assert_close(out, correct_result, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()