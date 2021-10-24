import torch
import torch.optim as optim
from torchvision.utils import save_image
import os

from models import Generator, Discriminator, weights_init_normal

class Solver:
  def __init__(self, config, loaders):
    #Paramters
    self.config = config
    self.loaders = loaders
    self.save_images_path = os.path.join(self.config.output_path, 'images/')
    self.save_models_path = os.path.join(self.config.output_path, 'models/')
    
    #Set Devices
    if self.config
