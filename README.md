

# Residual Networks
ResNet is just a simple toolkit that allow you to make Residual Networks in simple way using command line

**Requirements**
- [Python](https://www.python.org/) 3.*
- [Numpy](http://www.numpy.org/)
- [Pytorch](https://pytorch.org/)


**Options that project support :**
1. modes 
	- train 
	- prediction
2. use GPU while training
3. number of epochs (iterations)
4. batch size
5. learning rate 
6. paths 
	* train data path
	* test data path   
	*  saving path  
	* predictions_data_path
	* loading path
7. valid ratio
8. resnet version
9. pretrained
10. layers

## SetUp

there are some functionality let for you to write your on logic on it all collected in file `setup.py`. 
there are two type of those functionality 
	* 	Required	
	* Optional



### **Required**
- number of classes that the model will classify between the 
	Ex: `num_classes = 10`
- data loader 
	```py
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
	```
### **Optional**
- custom image loader
	```py
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
	```
- custom residual block
	```py
	class CutomBlock(nn.Module):
    
	    expansion = None

	    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
	        
	        super(CutomBlock, self).__init__()
	        
	        
	    
	    def forward(self, X):
	        
	        out = None
	        
	        return out
    
	```
- training data transformation
	Ex:
	```py
	train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225]),
                                      ])

	```
- test data transformation
	Ex: 
	```py
	test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                        ])

	```
		
## Run
To Run ResNet toolkit and make your Residual model you have to know to tell the ResNet toolkit what you want to do 

**Modes**
	your should chose one mode of two mode
	- train
		`python run.py --train True`
	- predict
		`python run.py --predict True`

### Train Mode
in train mode you have to define some arguments subset is required and the other is optional 

**Required Args**
- train_data_path -- training data set directory path 
	Ex: `--train_data_path "/data/train"` 
- test_data_path -- testing data set directory path 
	Ex: `--test_data_path "/data/test"` 
- n_epochs -- number of iteration of training process 
	Ex:`--n_epochs 30`
- learning_rate -- learning rate of training 
	Ex: `--learning_rate 0.001`
-there are some args that are optional but you should select at least one of them such as 
	*	resnet_version
	*	block
	*	layers
will dict ripe them into  Optional Args section 

**Optional Args**

-  gpu -- if you have cuda gpu and you want to use it while training you just set the argument to True value
	Ex: `--gpu True`
- batch_size -- batch size for each updating step
	Ex: `--batch_size 64`
- saving_path -- the of file to save the model after training 
  Ex: `--saving_path "models/model.pt"`
- valid_ratio -- if you want to split training data set into train and validation sets just top the ratio between train and validation data set 
	Ex: `--valid_ratio 0.2`
- resnet_version -- the version of residual network that you want to use such as 
	* 18
	* 34
	* 50
	* 101
	* 152
	and there is an extra argument if you chose resnet_version argument that is 
	- pretrained -- the load the trained weights of the network not only the architecture 
Ex: 
 `--resnet_version 18`
 `--resnet_version 18 --pretrained True` 
- custom_image_loader -- if you write your own logic for loading images and you want to use that logic 
 Ex: `--custom_image_loader True`

- layers -- if you want to make residual network with specific layers, but there an extra arguments you should select one of them 
	*   block -- choose  one of two implemented
			*  BasicBlock
			* Bottleneck
	* custom_block -- if you have implemented your own block and you want to use it 
	Ex:
	`--layers 2 2 3 5 --block BasicBlock`
	`--layers 2 2 3 5 --custom_block True`

### Prediction Mode

**Required parameters**
- loading_path -- path of trained model file
	`Ex: --loading_path "dir/model.pt"`
- predictions_data_path -- path of the data set
	`Ex: --predictions_data_path "/dir1/dir2/file.jpg"`


## Date
**mon, jan 28, 2019**
