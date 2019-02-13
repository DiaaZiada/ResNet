#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:11:11 2019

@author: diaa
"""
import numpy as np
import torch
from time import time
from PIL import Image


def train(model, train_loader, val_loader, optimizer, loss_function, checkpoint, saving_path, n_epochs, train_on_gpu):
    """
    Function:
        train the model
    
    Arguments:
        model -- the resnet that want to train
        train_loader -- trainning data set
        val_loader -- validation data set
        optimizer -- the optimization method of trainning
        loss_fuction -- loss function to calculate the cost
        checkpoint -- python dict contains all necessary information about model to load it later
        saving_path -- file name with path that you want to save the model in
        n_epochs -- number of iterations 
        train_on_gpu -- bool value to check if you have gpu and you want to train on it
            
    """
    train_losses = []
    train_accs = []
  
    val_losses = []
    val_accs = []
    
    min_val_loss = np.Inf
    model.train()      
    for e in range(1,n_epochs+1):
        epoch_start = time()
        batch_number = 0        
        train_loss = 0
        train_acc = 0
        batch_start = time()
        for x,y in train_loader:
            batch_number += 1
      
            if train_on_gpu and torch.cuda.is_available() :
                x, y = x.cuda(), y.cuda()
        
            optimizer.zero_grad()
      
            y_ = model.forward(x)
      
            loss = loss_function(y_,y)
      
            loss.backward()
      
            optimizer.step()
      
            train_loss += loss.item()
      
            ps = torch.exp(y_)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y.view(*top_class.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor))
            delay = time()-batch_start
            print({"train batch finished" : batch_number/len(train_loader) *100. ,
                  "time left" : delay * (len(train_loader)-batch_number),
                   "delay":delay})
            batch_start = time()

        else:
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                model.eval()
        
                batch_number = 0
                batch_start = time()
                for x,y in val_loader:
                    batch_number += 1
                    if train_on_gpu and torch.cuda.is_available() :
                        x, y = x.cuda(), y.cuda()
                    x = x.squeeze()
                    y = y.squeeze()
                    y_ = model.forward(x)
                     
                    loss = loss_function(y_,y)
                    
                    val_loss += loss.item()
                    
                    ps = torch.exp(y_)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == y.view(*top_class.shape)
                     
                    val_acc += torch.mean(equals.type(torch.FloatTensor))
                    delay = time()-batch_start
                    print({"vaildation batch finished" : batch_number/len(val_loader) *100. ,
                           "time left" : (delay)*(len(val_loader)-batch_number),
                           "delay":delay})
                    batch_start = time()
               
        train_loss /= len(train_loader)     
        train_acc /= len(train_loader)  
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if min_val_loss > val_loss:
            print ('Validation loss decreased ({:.6f} --> {:.6f}). ')
            min_val_loss = val_loss
            if saving_path:         
                print(' Saving model ...'.format(min_val_loss, val_loss))
                checkpoint['state_dict'] = model.state_dict()
                torch.save(checkpoint,saving_path)
    
                       
        
        delay = time() - epoch_start
                       
        print({"Epoch" : e,
                   "Train Finished": e / n_epochs * 100. , 
                   "Time Left": delay * (n_epochs * e),
                   "Training Loss" : train_loss,
                   "Validation Loss" : val_loss,
                   "train Accuracy" : train_acc,
                   "Validation Accuracy" : val_acc,
                   "Delay" : delay
               })
          
      
#    return train_losses, train_accs, val_losses, val_accs
    


def test(model, test_loader, loss_function):
    """
    Function:
        test the model accuracy
    
    Arguments:
        model -- the resnet that want to train
        train_loader -- test data set
        loss_fuction -- loss function to calculate the cost
    """
    test_acc = 0
    test_loss = 0
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    for x, y in test_loader:
        if torch.cuda.is_available():
            x,y = x.cuda(), y.cuda()
            
        y_ = model.forward(x)
        
        loss = loss_function(y_,y)
        test_loss += loss.item()
           
        ps = torch.exp(y_)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == y.view(*top_class.shape) 
        test_acc += torch.mean(equals.type(torch.FloatTensor))
        
    test_acc /= len(test_loader)
    test_loss /= len(test_loader)

    print('Test Loss',test_loss,'Test Accuracy',test_acc)



def predict(image_path, model,transform, topk=1):
    """
    Function:
        Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image_path -- path of image file
        model -- trained model that will make the predictoin
        transform -- transformation of the image 
        topk -- range of class to display
    """
    
    image = Image.open(image_path)
    image = transform(image)
    y_ = model.forward(image[None])

    ps = torch.exp(y_)
    top_p, top_class = ps.topk(topk, dim=1)
    print('class ',int(top_class),' probability ', float(top_p))
    return top_p, top_class

			