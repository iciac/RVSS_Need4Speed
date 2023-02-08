
import argparse
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
from model import PenguinNet
FLAG_GPU = True


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/2021_04_17_train_modelnet',
                        metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='../on_robot/collect_data/archive',
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('--workers', default=12, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for Embedding
    parser.add_argument('--embedding', default='vgg',
                        type=str, help='embedding functions to choose')
    parser.add_argument('--dim_k', default=64, type=int,
                        metavar='K', help='dim. of the feature vector')

    # settings for training.
    parser.add_argument('--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        metavar='METHOD', help='name of an optimizer')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--lr', type=float, default=1e-3,
                        metavar='D', help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, 
                        metavar='D', help='decay rate of learning rate')

    args = parser.parse_args(argv)
    return args


class Trainer():
    def __init__(self, width, hidden):
        self.width = width
        self.hidden = hidden

        self.MLP = MLPNet(width=self.width, hidden=self.hidden)
        self.optim = optim.Adam(params=self.MLP.parameters())
        self.criterion = nn.NLLLoss()
        
        
        self.data_transforms = transforms.Compose([
                transforms.Resize((48, 64)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.ToTensor(),
                ])

    def train_one_epoch(self, epochs):
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


if __name__ == '__main__':
    # TODO: arguments here
    args = options()
    
    trainset, evalset = get_datasets(args)
    trainer = Trainer()
    
    model = PenguinNet(args.embedding, args.dim_k, )
    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    print('resume epoch from {}'.format(args.start_epoch+1))

    evalloader = torch.utils.data.DataLoader(evalset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    min_loss = float('inf')
    min_info = float('inf')
    
    # TODO: define the dataloader
    trainloader = 
    
    
    # TODO: start training
    for epoch in tqdm(range(epochs)):
            running_loss = 0.0

            # TODO: run the train_one_epoch function
            # Simply for time keeping
            start_time = time.time()
            # Loop over all training data
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                
            # TODO: do validation
            if epoch % 5 == 0:
                validate()
                
                
            