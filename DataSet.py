from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

class ImageDataset(Dataset):

    def __init__(self, path, scales):
        self.path = path
        self.files = os.listdir(path)
        self.scales = scales

    def __getitem__(self, index):
        image_hr = cv2.imread(os.path.join(self.path, self.files[index]))
        (m, n, c) = np.shape(image_hr)
        lr = (int(n/self.scales), int(m/self.scales))
        image_lr = cv2.resize(image_hr, lr)
        image_hr = image_hr.swapaxes(0, 2)
        image_hr = image_hr.swapaxes(1, 2)
        image_lr = image_lr.swapaxes(0, 2)
        image_lr = image_lr.swapaxes(1, 2)
        image_lr = image_lr.astype('float32')
        image_hr = image_hr.astype('float32')
        imgs = {'lr': image_lr, 'hr': image_hr}
        return imgs

    def __len__(self):

        return len(self.files)