#!/usr/bin/env python3
import time
import click
import math
import sys
import argparse
sys.path.append("..")
import cv2
import numpy as np
import penguinPi as ppi
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as transF
from model import SimpleNet


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/2021_04_17_train_modelnet', help='output filename (prefix)')
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


class CropData():
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height
        
    def __call__(self, tensor):
        return transF.crop(tensor, self.xmin, self.ymin, self.height, self.width)
    
    
# NOTE: arguments
args = options()    

# stop the robot 
ppi.set_velocity(0,0)
print("initialise camera")
camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')

#INITIALISE NETWORK HERE
model = SimpleNet()

#LOAD NETWORK WEIGHTS HERE
model.load_state_dict(torch.load('../on_laptop/checkpoints/2023_02_09_SimpleNet_01.pt', map_location='cpu'))

angle_tab = np.arange(-0.5, 0.6, 0.1)

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    while True:
        # get an image from the the robot
        image = camera.frame
        # image = Image.open('../collect_data/archive/trial_02/0000000.00.jpg')

        #TO DO: apply any image transformations
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    CropData(args.im_crop_xmin, args.im_crop_ymin, args.im_crop_width, args.im_crop_height),
                    transforms.Resize(size=(48, 64)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
        
        inp = transform(image).unsqueeze(0)
        # print(inp)
        # inp_show = inp.squeeze(0).permute(1,2,0).numpy()
        # plt.imshow(inp_show)
        # plt.show()
        # inp_show.show()
        
        # print(image.shape)   # 240*320*3
        # print(inp.shape)   # 3*48*64
        
        #TO DO: pass image through network to get a prediction for the steering angle
        angle_prob = model(inp)
        _, angle_idx = torch.max(angle_prob, 1)
        angle = angle_tab[angle_idx]
        angle = np.clip(angle, -0.5, 0.5)
        
        # NOTE: control
        Kd = 30 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 30 #how fast to turn when given an angle
        left  = int(Kd + (Ka+10)*angle)
        right = int(Kd - Ka*angle)
        
        print('angle is {}. left is {}, right is {}'.format(angle, left, right))
        
        ppi.set_velocity(left,right) 
        
        
except KeyboardInterrupt:    
    ppi.set_velocity(0,0)
