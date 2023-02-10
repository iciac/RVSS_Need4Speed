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
from torchvision import transforms, models
import torchvision.transforms.functional as transF
from model import SimpleNet, MiddleNet, PenguinNet


def options(argv=None):
    parser = argparse.ArgumentParser(description='train PenguinPi robot model')

    # io settings.
    parser.add_argument('--pretrained_model', type=str, default='../on_laptop/checkpoints/2023_02_09_SimpleNet_trial_01.pt', help='pretrained model dir')
    parser.add_argument('--dataset_path', type=str, default='../on_robot/collect_data/archive', help='path to the input dataset')    
    
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
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='use CUDA if available')

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
if args.embedding == 'vgg11_pretrained':
    # model = models.vgg11(pretrained=True)
    model = models.vgg11(pretrained=False)
    # torch.save(model.state_dict(), './pretrained_models/vgg11.pth')
    model.load_state_dict(torch.load('./pretrained_models/vgg11.pth'))
    # model = models.resnet50(pretrained=True)
    # print(model)
    # print(jjj)
    num_ftrs = model.classifier[0].in_features

    model.classifier = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 11),
                                    nn.LogSoftmax(dim=1))
elif args.embedding == 'resnet18_pretrained':
    # model = models.resnet18(pretrained=True)
    model = models.resnet18(pretrained=False)
    # torch.save(model.state_dict(), './pretrained_models/resnet18.pth')
    model.load_state_dict(torch.load('./pretrained_models/resnet18.pth'))
    # print(model)
    # print(jjj)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                nn.ReLU(),
                            #  nn.Dropout(0.2),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512, 11),
                                nn.LogSoftmax(dim=1))
elif args.embedding == 'cnn':
    model = PenguinNet(args.embedding, args.hidden, args.width).to(args.device)
elif args.embedding == 'simple':
    model = SimpleNet()
elif args.embedding == 'middle':
    model = MiddleNet()

#LOAD NETWORK WEIGHTS HERE
model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))

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
                    transforms.Resize(size=(args.im_resize_height, args.im_resize_width)),
                    # transforms.ColorJitter(brightness=(0.75, 2)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
        
        inp = transform(image).unsqueeze(0)
        # print(inp.shape)
        # inp_show = inp.squeeze(0).permute(1,2,0).numpy()
        # plt.imshow(inp_show)
        # plt.show()
        # inp_show.show()
        
        # print(image.shape)   # 240*320*3
        # print(inp.shape)   # 3*48*64
        
        #TO DO: pass image through network to get a prediction for the steering angle
        angle_prob = model(inp)
        print('angle probability (log): {}'.format(torch.max(angle_prob)))
        angle_prob_check = torch.exp(angle_prob)
        print('angle probability: {}'.format(torch.max(angle_prob_check)))
        _, angle_idx = torch.max(angle_prob, 1)
        angle = angle_tab[angle_idx]
        angle = np.clip(angle, -0.5, 0.5)
        
        # NOTE: control
        Kd = 30 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 30 #how fast to turn when given an angle
        # certainty code
        # if torch.max(angle_prob_check) < 0.65:
        #     Kd = 5
        #     Ka = 0
        
        # speed code
        # if np.abs(0-angle) < 0.1:
        #     Kd = 1.5 * Kd
        # else:
        #     Kd = (1-np.abs(angle)) * Kd
        left  = int(Kd + (Ka+10)*angle)
        # left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
        
        print('angle is {}. left is {}, right is {}'.format(angle, left, right))
        
        ppi.set_velocity(left,right) 
        
        
except KeyboardInterrupt:    
    ppi.set_velocity(0,0)
