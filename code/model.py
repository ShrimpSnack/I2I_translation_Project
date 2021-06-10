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

'''
# x와 c를 왜 concat해주는지 의문
'''
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    layers = []
    
    
    
    self.main = nn.Sequential(*layers)
    
  def forward(self, x, c): 
    # real image x 와 Target domain c
    # x는 128x128크기, batchsize 16개, dimension 3 -> [16,3, 128, 128]
    # c는 16x7의 사이즈 (16개의 이미지에 대한 label 7개)
    c = c.view(c.size(0), c.size(1), 1, 1) # [16, 7, 1, 1]
    c = c.repeat(1,1, x.size(2), x.size(3)) # [16, 7, 128, 128]
    x = torch.cat([x,c], dim =1) # [16, 10, 128, 128] 
    return self.main(x)
    
