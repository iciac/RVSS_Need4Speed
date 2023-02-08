## Imports
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib as plt
import numpy as np
import os


## Parameters
batch_size = 4

img_width = 320
img_height = 240

min_th = -0.5
max_th = 0.5
th_step = 0.1

## Data Wrangling
datapath = '../on_robot/archive/'

class PerfectDrivingDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# trainset = torch.utils.data.datasets.ImageLoader(datapath+'trial_06', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, 
    batch_size = batch_size, shuffle=True, num_workers=2)

# testset = torch.utils.data.datasets.ImageLoader(datapath+'trial_07', transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
    batch_size = batch_size, shuffle=True, num_workers=2)

classes = np.arange(min_th, max_th+th_step, th_step) 

## CNN Wrangling
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

## Loss Function and Optimiser
criterion = nn.CrossEntropyLoss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)