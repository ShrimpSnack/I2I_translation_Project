'''
dataloader에서 get_loader함수 생성

get_loader 함수에는 transform 부분과 dataset생성하는 부분

- transform 함수
RandomHorizontalFlip  
CenterCrop  
Resize  

- 이미지 크기
CelebA: 178x218
RaFD: 256x256

-> input크기를 맞추기 위해서 center 부분을 crop하고 resize

- 데이터 생성하는 방법
CelebA: custom dataset사용 (preprocessing 동반), 40개의 label중 일부만 사용  

> 178 x 218 >> 178 x 178 >> 128 x 128 >> 40 labels

: 보통 __getitem__과 __len__존재, preprocess()함수 추가(in __init__())

RaFD: torchvision의 ImageFolder를 이용, 8개의 label

> 256 x 256 >> 178 x 178 >> 128 x 128 >> 8 labels



'''

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

class CelebA(data.Dataset):
  def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
    self.image_dir = image_dir
    self.attr_path = attr_path
    self.selected_attrs = selected_attrs
    self.transform  = transform
    self.mode = mode
    
    self.train_dataset = []
    self.test_dataset = []
    self.attr2idx = {}
    self.idx2attr = {}
    
    # 전처리는 필수
    self.preprocess()
        
    if mode == 'train':
      self.num_images = len(self.train_dataset)
    else: 
      self.num_images = len(self.test_dataset)
    
  def preprocess(self):
    ## attribute 추출
    lines = [line.rstrip() for line in open(self.attr_path, 'r')]
    all_attr_names = lines[1].split()
    
    # attr_name을 index로 변환
    for i, attr_name in enumerate(all_attr_names):
      self.attr2idx[attr_name] = i
      self.idx2attr[i] = attr_name
      
    lines = lines[2:]
    random.seed(1234)
    random.shuffle(lines)
    
    for i, line in enumerate(lines):
      split = line.split()
      filename = split[0]
      values = split[1:]
      
      label = []
      for attr_name in self.selected_attrs:
        idx = self.attr2idx[attr_name]
        label.append(values[idx] == '1')
      
      ## 랜덤으로 train_dataset과 test_dataset 만들기
      if (i+1) < 2000:
        self.test_dataset.append([filename, label])
      else:
        self.train_dataset.append([filename, label])
        
    
  def __getitem__(self, index):
    ## dataset 불러오기
    dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
    
    ## 인덱스 별 나누기
    filename, label = dataset[index]
    image = Image.open(os.path.join(self.image_dir, filename))
    
    return self.transform(image), torch.FloatTensor(label)
    
  
  def __len__(self):
    return self.num_images
    
    
    
    
def get_loader(image_dir, attr_path, selected_attrs, crop_size = 178, image_size = 128,
               batch_size = 16, dataset = 'CelebA', mode = 'train', num_workers = 1):
  transform = []
  if mode == 'train':
    transform.append(T.RandomHorizontalFlip())
  transform.append(T.CenterCrop(crop_size))
  transform.append(T.Resize(image_size))
  transform.append(T.ToTensor())
  transform.append(T.Normalize(mean =(0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)))
  transform = T.Compose(transform)
  
  # Dataset 만드는 과정, RaFD는 torchvision 내 ImageFolder 사용
  if dataset == 'CelebA':
    dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
  elif dataset == 'RaFD':
    dataset = ImageFolder(image_dir, transform)
    
  data_loader = data.DataLoader(dataset= dataset, batch_size= batch_size, shuffle=(mode =='train'), num_workers = num_workers)
  
  return data_loader


