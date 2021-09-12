import torch
import torch.nn as nn

# Generator(U-Net)
class UNetDown(nn.Module):
  def __init__(self, in_channels, out_channels, normalize = True, dropout = 0.0):
    super(UNetDown, self).__init__()
    
    layers = []
    layers.append(nn.Conv2d(in_size, out_size, kernel_size = 4, stride=2, padding = 1, bias= False))
    if normalize:
      layers.append(nn.InstanceNorm2d(out_size))
    layers.append(nn.LeakyReLU(0.2))
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)
  def forward(self, x):
    return self.model(x)
