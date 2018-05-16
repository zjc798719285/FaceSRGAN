from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms as trf
import numpy as np
import torch

class ImageDataset(Dataset):

    def __init__(self, path, scales):
        self.path = path
        self.scales = scales
        self.imgs = []
        self.files = [os.path.join(path, i) for i in os.listdir(path)]
        print('Begin Initializing')
        for file_i in self.files:
           image_hr = cv2.imread(file_i); (m, n, c) = np.shape(image_hr)
           lr = (int(n / self.scales), int(m / self.scales))
           hr = (int(n / self.scales)*4, int(m / self.scales)*4)
           image_hr = cv2.resize(image_hr, hr)
           image_lr = cv2.resize(image_hr, lr)
           image_hr = trf.ToTensor()(image_hr); image_lr = trf.ToTensor()(image_lr)
           image_hr = trf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_hr)
           image_lr = trf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_lr)
           imgs = {'lr': image_lr, 'hr': image_hr, 'filename': file_i}
           self.imgs.append(imgs)
        print('Completed Initializing')



    def __getitem__(self, index):

        return self.imgs[index]

    def __len__(self):

        return len(self.files)