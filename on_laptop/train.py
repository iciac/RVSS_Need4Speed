
import argparse
import os, glob
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
from steerDS import SteerDataSet, AddGaussianNoise, CropData


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/2021_04_17_train_modelnet', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='../on_robot/collect_data/archive', help='path to the input dataset')
    parser.add_argument('--workers', default=12, type=int, help='number of data loading workers')
    
    # data settings.
    parser.add_argument('--im_crop_xmin', default=40, type=int, help='xmin of image to crop')
    parser.add_argument('--im_crop_ymin', default=0, type=int, help='ymin of image to crop')
    parser.add_argument('--im_crop_height', default=200, type=int, help='height of image to crop')
    parser.add_argument('--im_crop_width', default=320, type=int, help='width of image to crop')
    parser.add_argument('--im_resize_height', default=40, type=int, help='height of resized image')
    parser.add_argument('--im_resize_width', default=64, type=int, help='width of resized image')

    # settings for Embedding
    parser.add_argument('--embedding', default='cnn', type=str, help='embedding functions to choose')
    parser.add_argument('--hidden_layers', default=2, type=int, help='number of hidden layers')
    parser.add_argument('--dim_k', default=64, type=int, help='dim. of the feature vector')

    # settings for training.
    parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--max_epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--optimizer', default='Adam', type=str, help='name of an optimizer')
    parser.add_argument('--device', default='cuda:0', type=str, help='use CUDA if available')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')

    args = parser.parse_args(argv)
    return args


class Trainer():
    def __init__(self, args):
        self.device = args.device
        self.width = args.dim_k
        self.hidden = args.hidden_layers

        if args.embedding == 'cnn':
            self.model = PenguinNet(args.embedding, self.hidden, self.width).to(args.device)
        self.optim = optim.Adam(params=self.model.parameters())
        self.criterion = nn.NLLLoss()

    # FIXME: need to see if the detail is correct
    def train_one_epoch(self, epoch, trainloader):
        running_loss = .0
        train_loss = .0
        
        # Loop over all training data
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs.to(self.device)
            labels.to(self.device)

            # zero the parameter gradients
            self.optim.zero_grad()

            # forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Compute Gradients
            loss.backward()
            # BackProp
            self.optim.step()

            # get statistics
            train_loss = train_loss + loss.item()
            running_loss = running_loss + loss.item()
            
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss / 100))
                running_loss = .0
        
        avg_train_loss = train_loss / i
            
        return avg_train_loss

    # TODO: evaluation, should be similar to training
    # FIXME: wrong details/code
    def eval_one_epoch(self, epoch, evalloader):
        total_num = 0
        correct = 0
        eval_loss = .0
        
        for i, data in enumerate(evalloader):
            inputs, labels = data
            inputs.to(self.device)
            labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # FIXME: here
                # note here we take the max of all probability
                _, predicted = torch.max(outputs, 1)
                
                eval_loss = eval_loss + loss.item()

                # compute statistics
                total_num = total_num + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
                # TODO: some printing results?
                
                                
        avg_eval_loss = eval_loss / i
        
        # FIXME: output of predictions? for visualize maybe?
        return avg_eval_loss, total_num, correct


if __name__ == '__main__':
    # ANCHOR: arguments here
    args = options()
    
    # ANCHOR: transform functions
    transform = transforms.Compose(
    [transforms.ToTensor(),
     CropData(args.im_crop_xmin, args.im_crop_ymin, args.im_crop_width, args.im_crop_height),
     transforms.Resize(size=(48, 64)),
     transforms.ColorJitter(brightness=0.5),
     AddGaussianNoise(0., 0.04),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
    ds_train = SteerDataSet('../on_robot/collect_data/archive', ".jpg", transform, mode='train', trial_file_name='./data/ds_train.txt')
    ds_eval = SteerDataSet('../on_robot/collect_data/archive', ".jpg", transform, mode='eval', trial_file_name='./data/ds_train.txt')

    trainer = Trainer()
    
    model = PenguinNet(args.embedding, args.hidden_layers, args.dim_k)
    model.to(args.device)

    ds_trainloader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    ds_evalloader = DataLoader(ds_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    min_loss = float('inf')
    
    # TODO: start training
    for epoch in tqdm(range(args.max_epochs)):
        running_loss = 0.0
        
        # Simply for time keeping
        start_time = time.time()
        
        trainer.train_one_epoch(epoch, ds_trainloader)
        
        # end for over minibatches epoch finishes
        end_time = time.time()
        
        # TODO: validate the network every epoch on eval examples
        if epoch % 5 == 0:
            avg_eval_loss, total_num, correct = trainer.eval_one_epoch(epoch, ds_evalloader)
            
        print('Epoch', epoch+1, 'took', end_time-start_time, 'seconds')
        print('Accuracy of the network after', epoch+1, 'epochs is', 100*correct/total_num)

    print('Finished Training')
    torch.save(self.model.state_dict(), 'modelweights.pt')
    
                
            