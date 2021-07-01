'''
Build Model

training 과 testing을 위한 solver class를 이용해 instance를 생성
model build 함
-> solver = Solver(celeba_loader, rafd_loader, config)

generator으 input으로 input image 와 함께 domain 정보를 이용
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
  def print_network(self, model, name):
  def restore_model(self, resum_iters):
  def build_tensorboard(self):
  def update_lr(self, g_lr, d_lr):
  def reset_grad(self):
  def denorm(self, x):
  def gradient_penalty(self, y, x):
  def label2onehot(self,
