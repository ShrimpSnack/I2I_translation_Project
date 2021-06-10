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
InstanceNorm2d(num_features, eps, momentum, affine, track_running_stats)
- affine: True로 설정하면 이 모듈에 학습가능한 아핀 매개변수가 있으며 배치 정규화와 동일한 방식으로 초기화되는 부울 값
- track_running_stats: True로 설정하면 이 모듈이 실행 평균 및 분산을 추적하고 False로 설정하면 이 모듈이 이러한 통계를 추정하지 않고 항상 학습및 평가 모드에서 배치 통계를 사용하는 부울 값
'''
'''
# x와 c를 왜 concat해주는지 의문
# instance Norm2d - > affine, track_running_stats
# bottleneck할때 Conv2d의 bias를 False
'''
class ResidualBlock(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(ResidualBlock, self).__init__()
    self.main = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
                              nn.InstanceNorm2d(dim_out, affine = True, track_running_stats = True),
                              nn.ReLU(inplace = True),
                              ## 논문에서 없던 layer 추가
                              nn.Conv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
                              nn.InstanceNorm2d(dim_out, affine = True, track_running_stats = True))
  def forward(self,x):
    return x + self.main(x)
                              


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    ## affine=True, learnable parameter에서 affine 변환을 해줌
    
    #downsampling
    self.downsampling = nn.Sequential(nn.Conv2d(3,64,kernel_size = 7, stride = 1, padding = 3),
                                      nn.InstanceNorm2d(64, affine = True, track_running_stats = True),
                                      nn.ReLU(inplace = True),
                                     nn.Conv2d(64, 128,kernel_size = 4, stride = 2, padding = 1),
                                      nn.InstanceNorm2d(64, affine = True, track_running_stats = True),
                                      nn.ReLU(inplace = True),
                                     nn.Conv2d(128, 256,kernel_size = 4, stride = 2, padding = 1),
                                      nn.InstanceNorm2d(128, affine = True, track_running_stats = True),
                                      nn.ReLU(inplace = True))
    # Bottleneck
    layers = []
    for i in range(6):
      layers.append(ResidualBlock(dim_in=256, dim_out=256))
 
    self.bottleneck = nn.Sequential(*layers)
  
    # upsampling layers
    self.upsampling = nn.Sequential(nn.ConvTranpose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
                                    nn.InstanceNorm2d(128, affine = True, track_running_stats = True),
                                    nn.ReLU(inplace = True),
                                    nn.ConvTranpose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
                                    nn.InstanceNorm2d(64, affine = True, track_running_stats = True),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 3, kernel_size = 7, stride = 1, padding = 3, bias = False),
                                    nn.Tanh())
                                    
    
                                   
  def forward(self, x, c): 
    # real image x 와 Target domain c
    # x는 128x128크기, batchsize 16개, dimension 3 -> [16,3, 128, 128]
    # c는 16x7의 사이즈 (16개의 이미지에 대한 label 7개)
    c = c.view(c.size(0), c.size(1), 1, 1) # [16, 7, 1, 1]
    c = c.repeat(1,1, x.size(2), x.size(3)) # [16, 7, 128, 128]
    result = torch.cat([x,c], dim =1) # [16, 10, 128, 128] 
    
    result = self.downsampling(result)
    result= self.Bottleneck(result)
    result = self.upsampling(result)
    return result
 '''
 kernel_size 계산하는거 확인
 output에서 src와 cls다르게 하는 거 확인
 '''

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    #input layer
    self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1)
    self.LRelu = nn.LeakyReLU(0.01)
    
    # Hidden Layer
    layers= []
    dim = 64
    for i in range(5):
      layers.append(nn.Conv2d(dim, dim*2, kernel_size = 4, stride =2, padding = 1))
      layers.append(nn.LeakyReLU(0.01))
      dim = dim*2
    
    self.hidden = nn.Sequential(*layers)
    
    kernel_size = int(image_size/ np.power(2, 6)) 
    self.src = nn.Conv2d(2048, 1, kernel_size = 3, stride = 1, padding = 1)
    self.cls = nn.Conv2d(2048, c_dim, kernel_size = kernel_size, bias = False)
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.LRelu(x)
    x = self.hidden(x)
    out_src = self.src(x)
    out_cls = self.cls(x)
    return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
    
    
    
