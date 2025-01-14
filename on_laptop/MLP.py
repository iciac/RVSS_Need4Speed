import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
FLAG_GPU = True


class MLPNet(nn.Module):
    def __init__(self, width, hidden, ):
        super(MLPNet, self).__init__()

        # First fully connected layers input image is 28x28 = 784 dim.
        self.fc0 = nn.Linear(320*240, 256)  # nparam = 784*256 = 38400
        
        self.conv1 = nn.Conv2d(256,32, kernel_size=(10,10))
        
        
        
        self.fc1 = nn.Linear(width, width//3)
        self.fc2 = nn.Linear(width//3, 11)

    def forward(self, x):
        # Flattens the image like structure into vectors
        x = torch.flatten(x, start_dim=1)

        # fully connected layers with activations
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # Outputs are log(p) so softmax followed by log.
        # return(x)
        output = F.log_softmax(x, dim=1)
        return output
