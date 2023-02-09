import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PenguinNet(nn.Module):
    def __init__(self, embedding='vgg', hidden_layers=2, dim_k=64):
        super().__init__()
        if embedding == 'pretrain_vgg':
            models.VGG11_BN_Weights(pretrained=True)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
