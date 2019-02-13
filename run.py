#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:37:31 2019

@author: diaa
"""
import argparse
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import  datasets, transforms, models
import torchvision.models as models

from torch import nn, optim
from util.blocks import *
from util.dataset import ImageDataSets
from util.workers import *
from setup import *


def manage():
    
    parser = argparse.ArgumentParser(description='Residual Network')

    parser.add_argument('--train', type=bool, default=False)

    parser.add_argument('--gpu', type=bool, default=False)  
    
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    
    parser.add_argument('--learning_rate', type=float)

    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--saving_path', type=str)
    parser.add_argument('--valid_ratio', type=float)
    
    
    parser.add_argument('--resnet_version', type=int)
    parser.add_argument('--pretrained', type=bool, default=False)
    
    parser.add_argument('--custom_image_loader', type=bool, default=False)
    parser.add_argument('--block', type=str)
    parser.add_argument('--custom_block', type=bool, default=False)

    parser.add_argument('--layers', type=int, nargs='+')

    parser.add_argument('--predict', type=bool, default=False)
    
    parser.add_argument('--predictions_data_path', type=str)
    parser.add_argument('--loading_path', type=str)

    args = parser.parse_args()
    
    if args.train:
        train_data_path = args.train_data_path
        test_data_path = args.test_data_path
        valid_ratio = args.valid_ratio
        train_sampler = None
        valid_sampler = None
        batch_size = args.batch_size
        custom_block = args.custom_block
        layers = args.layers
        block = args.block
        resnet_version = args.resnet_version
        pretrained = args.pretrained
        saving_path = args.saving_path
        n_epochs = args.n_epochs
        gpu = args.gpu
       
        checkpoint = {}
        
        checkpoint['custom_block'] = custom_block
        checkpoint['layers'] = layers
        checkpoint['block'] = block
        checkpoint['resnet_version'] = resnet_version


        
        if args.custom_image_loader:
            train_dataset = ImageDataSets(train_data_path, train_test_data_loader, custom_image_loader, transform=train_transforms)
            test_dataset = ImageDataSets(test_data_path, train_test_data_loader, custom_image_loader, transform=test_transforms)
      
        else:
            train_dataset = ImageDataSets(train_data_path, train_test_data_loader)
            test_dataset = ImageDataSets(test_data_path, train_test_data_loader)
                        
        if valid_ratio:
                        
            train_size = len(train_dataset)
            indices = list(range(train_size))
            np.random.shuffle(indices)
            split_size = int(valid_ratio * train_size)
            
            train_indices, valid_indices = indices[split_size:], indices[:split_size]
            
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)
        
        train_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=train_sampler)
        valid_loader = DataLoader(train_dataset,batch_size=batch_size,sampler=valid_sampler)
        test_loader = DataLoader(test_dataset,batch_size=batch_size)
        
        if custom_block:
            assert layers, 'layers is missed'
            model = ResNet(CutomBlock, layers, num_classes=num_classes)
        elif layers:
            assert block, 'block name is missed'
            if block.upper() =='BasicBlock'.upper():
                model = ResNet(BasicBlock, layers, num_classes=num_classes)
            elif block.upper() =='Bottleneck'.upper():
                model = ResNet(Bottleneck, layers, num_classes=num_classes)
            else:
                raise 'block name isn\'t recognize'
        elif resnet_version:
            if resnet_version == 18: 
                model =models.resnet18(pretrained=pretrained)
            elif resnet_version == 34: 
                model = models.resnet34(pretrained=pretrained)
            elif resnet_version == 50: 
                model = models.resnet50(pretrained=pretrained)
            elif resnet_version == 101: 
                model = models.resnet101(pretrained=pretrained)
            elif resnet_version == 152: 
                model = models.resnet152(pretrained=pretrained)
        else:
            raise "Residual Networks version isn't  recognize"
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        train(model, train_loader, valid_loader, optimizer, loss_function, checkpoint, saving_path,n_epochs,gpu)           
        test(model, test_loader, loss_function)
    elif args.predict:
        loading_path = args.loading_path
        predictions_data_path = args.predictions_data_path
        
        checkpoint = torch.load(loading_path)
        custom_block = checkpoint['custom_block']
        layers = checkpoint['layers']
        resnet_version = checkpoint['resnet_version']
        block = checkpoint['block']
        
        if custom_block:
            
            assert layers, 'layers is missed'
            model = ResNet(CutomBlock, layers, num_classes=num_classes)
        elif layers:
            assert block, 'block name is missed'
            if block.upper() =='BasicBlock'.upper():
                model = ResNet(BasicBlock, layers, num_classes=num_classes)
            elif block.upper() =='Bottleneck'.upper():
                model = ResNet(Bottleneck, layers, num_classes=num_classes)
            else:
                raise 'block name isn\'t recognize'
        elif resnet_version:
            if resnet_version == 18: 
                model = models.resnet18()
            elif resnet_version == 34: 
                model = models.resnet34(pretrained=pretrained)
            elif resnet_version == 50: 
                model = models.resnet50(pretrained=pretrained)
            elif resnet_version == 101: 
                model = models.resnet101(pretrained=pretrained)
            elif resnet_version == 152: 
                model = models.resnet152(pretrained=pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        predict(predictions_data_path,model,test_transforms)
            
        
manage()
#!python run.py --train True --train_data_path "/home/diaa/Desktop/NEW GITHUB/Residual Networks/flower_data/train" --test_data_path "/home/diaa/Desktop/NEW GITHUB/Residual Networks/flower_data/valid" --saving_path "check.pt" --n_epochs 1 --batch_size 64 --resnet_version 18
#!python run.py --predict True --predictions_data_path "/home/diaa/Desktop/NEW GITHUB/Residual Networks/flower_data/valid/1/image_06734.jpg" --loading_path "check.pt" 
