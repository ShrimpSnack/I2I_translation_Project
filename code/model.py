import torch
import torch.nn as nn


'''
Generator
- Instance normalization in all layers (except the last output layer)
Discriminator
- Leaky ReLU with a negatice slope of 0.01

nd = the number of domain
nc = the dimension of domain labels(nd+2 when training with both CelebA and RaFD datasets, otherwise same as nd)
N = the number of output channels
K = kernel size
S = stride size
P = padding size
IN = instance Normalization
'''
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    
  def forward(self, x, c):
    
