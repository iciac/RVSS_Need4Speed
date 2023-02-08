import torch
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from os import path
import matplotlib.pyplot as plt

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
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


def test():

    # thank you https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
            
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

        

    transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.CenterCrop(150),
    # TODO: better crop
    #  transforms.Resize(size=(48, 64)),
     transforms.ColorJitter(brightness=0.5),
     AddGaussianNoise(0., 0.04),
     
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
     #TODO (maybe (if there's time)): flip (and flip angle)

    ds = SteerDataSet("../dev_data/training_data",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=False)
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["steering"]
        
        print(im.shape)
        print(im.max(),im.min())

        plt.imshow(im.detach().squeeze().permute(1,2,0).cpu().numpy().reshape(150,150,3))

        plt.show()
        print(y)
        break



if __name__ == "__main__":
    test()
