#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:00:06 2019

@author: diaa
"""
from torch import nn
from torchvision import  datasets, transforms, models



'''
Required
'''
num_classes = None # number of classes of your model

def train_test_data_loader(path):
    """
    Function:
        load each image file with its label
    
    Arguments:
        path -- path of the data dirs
    
    Returns: two list image_files, labels --
                image_files list contains images files,
                labels list contains labels for each image
    """
    image_files = []
    labels = []
    
    ### Write your logic here ###

    return image_files, labels


'''
Optional
'''
def custom_image_loader(path):
    """
    Function:
        load each image file in your way
    
    Arguments:
        path -- path of the image file
    
    Returns: image
    """
    image = None
    
    ### Write your logic here ###

    
    return image
        
        
class CutomBlock(nn.Module):
    
    expansion = None

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(CutomBlock, self).__init__()
        
        
    
    def forward(self, X):
        
        out = None
        
        return out
    
    
train_transforms = None # Ex: transforms.Compose([transforms.RandomHorizontalFlip(),
#                                      transforms.RandomRotation(10),
#                                      transforms.RandomResizedCrop(224),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                           std=[0.229, 0.224, 0.225]),
#                                      ])

test_transforms = None # Ex: transforms.Compose([transforms.Resize(255),
#                                      transforms.CenterCrop(224),
#                                      transforms.ToTensor(),
#                                        ])


