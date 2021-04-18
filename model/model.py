import torch
import torch.nn as nn
import torch.nn.fuctional as F
import numpy as np

## Generator

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    layers = []
    layers.append(nn.Conv2d(3+c_dim, kernel_size = 7, stride = 1, padding= 3, bias = Flase))
    layers.append(nn.InstanceNorm2d(conv_dim, affine = True, track_running_stats= True))
    layers.append(nn.ReLU(inplace = True))
    
    
  def forward(self, image, domain):
    domain = domain.view(domain.size(0), domain.size(1), 1, 1)
    domain = domain.repeat(1, 1, domain.size(2), domain.size(3))
    image = torch.cat([image, target], dim = 1)
    return self.layer(image)
  
