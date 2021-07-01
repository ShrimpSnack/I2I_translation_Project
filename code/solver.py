'''
Build Model

solver class를 이용해서 training과 testing을 위하여 model를 build
-> solver = Solver(celeba_loader, rafd_loader, config)

generator의 input = (image+domain)정보를 이용
discriminator
discriminator에서 domain을 예측하는 classifier를 추가하여 단일 generator로 다양한 도메인간의 변환 가능


'''
from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

class Solver(object):
  def __init__(self, celeba_loader, rafd_loader, config):
    # Data loader.
    self.celeba_loader = celeba_loader
    self.rafd_loader = rafd_loader
    
  def build_model(self):
    if self.dataset in ['CelebA', 'RaFD']:
      self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
      self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
    elif self.dataset in ['Both']:
      self.G = Generator(self.g_conv_dim, self.c_dim + self.c_dim + 2, self.g_repeat_num)
      self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c_dim, self.d_repeat_num)
    
    self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
    self.d_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
    #self.print_network(self.G, 'G')
    #self.print_network(self.D, 'D')
    
 def train(self):
    # set data loader
    if self.dataset == 'CelebA':
      data_laoder  = self.celeba_loader
    elif self.dataset == 'RaFD':
      data_loader = self.rafd_loader
    
    # Fetch fixed inputs for debugging
    data_iter = iter(data_loader)
    x_fixed, c_org = next(data_iter)
    x_fixed = x_fixed.to(self.device)
    c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
    
    g_lr = self.g_lr
    d_lr = self.d_lr
    
    # Start training
    start_iters = 0
    if self.resum_iters:
      start_iters = self.resum_iters
      self.restore_model(self.resume_iters)
    
