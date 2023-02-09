
from steerDS import SteerDataSet
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
FLAG_GPU = False


class Trainer():
    def __init__(self, width=256, hidden=0):
        self.width = width
        self.hidden = hidden

        self.MLP = MLPNet(width=self.width, hidden=self.hidden)
        self.optim = optim.Adam(params=self.MLP.parameters())
        self.criterion = nn.NLLLoss()
        #self.criterion = nn.CrossEntropyLoss()
        print("start")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(150),
                transforms.Grayscale(),
                # TODO: better crop
                #  transforms.Resize(size=(48, 64)),
                # transforms.ColorJitter(brightness=0.5),
                # AddGaussianNoise(0., 0.04),

                #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        # TODO (maybe (if there's time)): flip (and flip angle)

        #ds = SteerDataSet("../dev_data/training_data", ".jpg", transform)

        ds = SteerDataSet(
            "/Users/camerongordon/Documents/GitHub/RVSS_Need4Speed/on_robot/collect_data/archive/trial_01", ".jpg", transform)

        print("The dataset contains %d images " % len(ds))

        self.trainloader = DataLoader(ds, batch_size=128, shuffle=False)

        # self.data_transforms = transforms.Compose([
        #         transforms.Resize((48, 64)),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #         transforms.ToTensor(),
        #         ])

    def train_one_epoch(self):
        running_loss = 0
        start_time = time.time()
        # Loop over all training data
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # print(data)
            im = data["image"].type(torch.FloatTensor)
            y = data["steering"]
            print(y)
            inputs = im
            # y[torch.where(y,y < 0)] = 0
            # y[torch.where(y=0)] = 1
            # y[torch.where(y > 0)] = 2

            # print(y)
            # print(type(y))
            y = y.type(torch.LongTensor)
            print(y)

            # zero the parameter gradients
            self.optim.zero_grad()
            # print(im)

            # forward
            if FLAG_GPU:
                outputs = self.MLP(inputs.cuda())
                loss = self.criterion(outputs, y.cuda())
            else:
                outputs = self.MLP(inputs)
                #print(outputs, y)
                loss = self.criterion(outputs, y)
                #print(outputs, y, loss)

            # Compute Gradients
            loss.backward()
            # BackProp
            self.optim.step()

            # print statistics
            running_loss += loss.item()
            if i % 640 == 0:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            # endif
        # end for over minibatches epoch finishes
        end_time = time.time()

        # test the network every epoch on test example
        #correct = 0
        #total = 0

    def validate(self):

        with torch.no_grad():
            #VIS = True
            for data in self.testloader:
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


if __name__ == '__main__':
    # TODO: arguments here
    epochs = 1000

    # TODO: define the dataloader
    # trainloader =

    trainer = Trainer()

    # TODO: start training
    for epoch in tqdm(range(epochs)):
        print(epoch)
        trainer.train_one_epoch()
        # running_loss = 0.0

        # # TODO: run the train_one_epoch function
        # # Simply for time keeping
        # start_time = time.time()
        # # Loop over all training data
        # for i, data in enumerate(trainloader, 0):
        #     # get the inputs; data is a list of [inputs, labels]
        #     inputs, labels = data

        # TODO: do validation
        # if epoch % 5 == 0:
        #     validate()
