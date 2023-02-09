
import argparse
import os, glob
from tqdm import tqdm
from MLP import MLPNet
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model import PenguinNet, SimpleNet
from steerDS import SteerDataSet, AddGaussianNoise, CropData
from train import Trainer


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./checkpoints/2023_02_09_SimpleNet_01.pt', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='../on_robot/collect_data/archive', help='path to the input dataset')
    parser.add_argument('--workers', default=6, type=int, help='number of data loading workers')
    
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


if __name__ == '__main__':
    # ANCHOR: arguments here
    args = options()
    
    # ANCHOR: transform functions
    transform = transforms.Compose(
    [transforms.ToTensor(),
     CropData(args.im_crop_xmin, args.im_crop_ymin, args.im_crop_width, args.im_crop_height),
     transforms.Resize(size=(48, 64)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])
    ds_test = SteerDataSet('../on_robot/collect_data/archive', ".jpg", transform, mode='test', trial_file_name='./data/ds_test.txt')

    trainer = Trainer(args)
    
    if args.embedding == 'cnn':
        model = PenguinNet(args.embedding, args.hidden, args.width).to(args.device)
    elif args.embedding == 'simple':
        model = SimpleNet()
    
    model.to(args.device)

    ds_testloader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=False)
    
    # TODO: start training
    # Simply for time keeping
    start_time = time.time()
    
    info_tab = trainer.test_one_epoch(ds_testloader, model)
    
    # end for over minibatches epoch finishes
    end_time = time.time()
    
    # TODO: metrics
    print('Accuracy of the network is', 100*correct/total_num)

