## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        

        
        # first convolutional layer
        self.conv1 = torch.nn.Conv2d(1,32,5) 
        
        # first Max-pooling layer
        self.pool1 = torch.nn.MaxPool2d(2,2) 
        
        # second convolutional layer
        self.conv2 = torch.nn.Conv2d(32,64,5) 
        
        # second Max-pooling layer
        self.pool2 = torch.nn.MaxPool2d(2,2)
        
        # Fully connected layer
        self.fc1 = torch.nn.Linear(64*21*21, 1024)   
        self.fc2 = torch.nn.Linear(1024, 512)       
        self.fc3 = torch.nn.Linear(512, 136)        
        self.drop1 = nn.Dropout(p=0.1)
        
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
          
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop1(x)
      
        # Flatten before passing to the fully-connected layers.
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

