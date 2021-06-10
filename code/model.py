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
'''
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    ## affine=True, learnable parameter에서 affine 변환을 해줌
    
    self.downsampling = nn.Sequential(nn.Conv2d(3,64,kernel_size = 7, stride = 1, padding = 3),
                                      nn.InstanceNorm2d(64, affine = True, reack_running_stats = True),
                                      nn.ReLU(inplace = True),
                                     nn.Conv2d(64, 128,kernel_size = 4, stride = 2, padding = 1),
                                      nn.InstanceNorm2d(64, affine = True, reack_running_stats = True),
                                      nn.ReLU(inplace = True),
                                     nn.Conv2d(128, 256,kernel_size = 4, stride = 2, padding = 1),
                                      nn.InstanceNorm2d(128, affine = True, reack_running_stats = True),
                                      nn.ReLU(inplace = True))
    
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
    
