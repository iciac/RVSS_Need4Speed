
from tqdm import tqdm
from MLP import MLPNet
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


class Trainer():
    def __init__(self, width, hidden):
        self.width = width
        self.hidden = hidden

        self.MLP = MLPNet(width=self.width, hidden=self.hidden)
        self.optim = optim.Adam(params=self.MLP.parameters())
        self.criterion = nn.NLLLoss()
        
        
        transforms = nn.Sequential(
                transforms.CenterCrop(10),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.greyscale(),
                #transforms.functionaladjustbrightness()
                )
        
        
        
        

    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0

            # Simply for time keeping
            start_time = time.time()
            # Loop over all training data
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward
                if FLAG_GPU:
                    outputs = self.MLP(inputs.cuda())
                    loss = self.criterion(outputs, labels.cuda())
                else:
                    outputs = self.MLP(inputs)
                    loss = self.criterion(outputs, labels)

                # Compute Gradients
                loss.backward()
                # BackProp
                self.optim.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                # endif
            # end for over minibatches epoch finishes
            end_time = time.time()

            # test the network every epoch on test example
            correct = 0
            total = 0

            with torch.no_grad():
                #VIS = True
                for data in testloader:
                    # load images and labels
                    images, labels = data

                    if FLAG_GPU:
                        outputs = self.MLP(images.cuda())
                        # note here we take the max of all probability
                        _, predicted = torch.max(outputs.cpu(), 1)
                    else:
                        outputs = self.MLP(images)
                        # note here we take the max of all probability
                        _, predicted = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # end for
            # end with
            print('Epoch', epoch+1, 'took', end_time-start_time, 'seconds')
            print('Accuracy of the network after',
                  epoch+1, 'epochs is', 100*correct/total)

        print('Finished Training')
        torch.save(self.MLP.state_dict(), 'modelweights.pt')
