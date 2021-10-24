'''
load train and test Dataset
'''

import glob
import os

import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Loaders:
  '''
  Initialize dataloaders
  '''
  def __init__(self, config):
    self.dataset_path = config.dataset_path
    self.image_size = config.image_size
    self.batch_size = config.batch_size
    
    self.transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size), Image.BiCUBIC),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
    train_set = ImageFolder(os.path.join(self.dataset_path, 'train/'), self.transforms)
    test_set = ImageFolder(os.path.join(self.dataset_path, 'test/'), self.transforms)
    
 

class ImageFolder(Dataset):
  '''
  Load images given the path
  '''
  def __init__(self, path, transform):
    self.transform = transform
    self.samples = sorted(glob.glob(os.path.join(path + '/*.*')))
    
                          
