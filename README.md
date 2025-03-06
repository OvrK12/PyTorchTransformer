# PyTorch Transformer
PyTorch implementation of a transformer from scratch according to the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.

### Transformer Implementation
All src files for the transformer are located under ```/transformer```
### Test Cases
For each module of the transformer there are test cases implemented in ```/test```. You can run all test cases in one go by using ```python run_all.py```
### Train Loop
A PyTorch train loop is implemented in ```train.py```. This trains the custom transformer on the wmt14 machine translation dataset (de-en split). After the training is completed, the model is saved to an out_path. Training params can be adjusted by changing the consts at the beginning of the file.

TODO: add details about training (GPU requirements, training time, features)
### Eval
TODO
