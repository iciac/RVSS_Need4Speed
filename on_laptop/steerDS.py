import os, glob
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as transF

from torchvision import transforms
from torch.utils.data import Dataset


# thank you https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CropData():
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height
        
    def __call__(self, tensor):
        return transF.crop(tensor, self.xmin, self.ymin, self.height, self.width)
    

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None, mode='train', trial_file_name=None, seed=917):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        # NOTE: get data for specific mode
        if trial_file_name != None:
            trials_fi = sorted([self.root_folder + '/' + line.rstrip('\n') for line in open(trial_file_name)])
        else:
            trials_fi = sorted(glob.glob(os.path.join(self.root_folder, '*')))
        
        self.filenames = []
        for i in range(len(trials_fi)):
            self.filenames.append(sorted(glob.glob(os.path.join(trials_fi[i], '*.jpg'))))
        # NOTE: flatten list
        self.filenames = [item for sublist in self.filenames for item in sublist]
        
        # NOTE: randomly choose 80% for training, 20% for validation
        if mode != 'test':
            if mode == 'train':
                data_len = int(len(self.filenames) * 0.8)
            elif mode == 'eval':
                data_len = int(len(self.filenames) * 0.2)
            fi_idx = np.random.choice(len(self.filenames), data_len, replace=False)
            self.filenames = [self.filenames[index] for index in fi_idx]
            
        self.totensor = transforms.ToTensor()
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)
        
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(steering)        
    
        sample = {"image":img , "steering":steering}        
        
        return sample

    