#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:49:13 2019

@author: diaa
"""

from torch.utils.data import Dataset
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
      
        
      
class ImageDataSets(Dataset):

  def __init__(self, root, train_test_data_loader, image_loader=default_loader, transform=None):
    
    self.image_loader = image_loader
    self.transform = transform
    
    self.image_files, self.labels = train_test_data_loader(root)

  def __len__(self):
    # Here, we need to return the number of samples in this dataset.
    return len(self.image_files)
  
  def __getitem__(self, indx):
   
    image = self.image_loader(self.image_files[indx])
    label = self.labels[indx]
   
    if self.transform:
        image = self.transform(image)
   
    return image,label