from Models import *
import torch
from DataSet import ImageDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
data_path = 'E:\PROJECT\FaceSRGAN\dataset\\'
epochs = 10
# Initialize generator and discriminator
generator = GeneratorResNet().to('cuda')
discriminator = Discriminator().to('cuda')

# feature_extractor = FeatureExtractor()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

dataloader = DataLoader(ImageDataset(data_path, 4), batch_size=1, shuffle=True, num_workers=0)


if __name__ == '__main__':
    for epoch_i in range(epochs):
        t = 0
        t1 = time.time()
        for step, imgs in enumerate(dataloader):
            t3 = time.time()
            imgs_hr = (imgs['hr'])
            imgs_lr = (imgs['lr'])
            gen_hr = generator.forward(imgs_lr.cuda())
            t4 = time.time()
            print('inner loop:', t4-t3)
            t += (t4 - t3)
        t2 = time.time()
        print(epoch_i, t2 - t1, t)
#


