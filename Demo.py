from Models import *
import torch
from DataSet import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
import time, os
data_path = 'E:\PROJECT\FaceSRGAN\dataset\\'
save_path = 'E:\PROJECT\FaceSRGAN\checkpoint'
epochs = 10
# Initialize generator and discriminator
generator = GeneratorResNet().cuda()
discriminator = Discriminator().cuda()

# Losses
Loss_GAN = torch.nn.BCELoss().cuda()
Loss_content = torch.nn.L1Loss().cuda()

#label

real = torch.cuda.FloatTensor(np.ones((1, 1)).astype(np.float32))
fake = torch.cuda.FloatTensor(np.zeros((1, 1)).astype(np.float32))
dataloader = DataLoader(ImageDataset(data_path, 4), batch_size=1, shuffle=True, num_workers=0)
optimizer_G = torch.optim.Adadelta(generator.parameters(), lr=0.01, rho=0.7)
optimizer_D = torch.optim.Adadelta(discriminator.parameters(), lr=0.001, rho=0.7)

if __name__ == '__main__':
    for epoch_i in range(epochs):
        t1 = time.time()
        for step, imgs in enumerate(dataloader):
            try:
                imgs_hr = (imgs['hr']);imgs_lr = (imgs['lr'])
                gen_hr = generator(imgs_lr.cuda())
                loss_G = 1.5*Loss_GAN(discriminator(gen_hr), real)
                loss_G.backward()
                optimizer_G.step()

                loss_real = Loss_GAN(discriminator(imgs_hr.cuda()), real)
                gen_hr2 = generator(imgs_lr.cuda())
                loss_fake = Loss_GAN(discriminator(gen_hr2), fake)
                loss_D = (loss_real + 2*loss_fake) / 2
                loss_D.backward()
                optimizer_D.step()
                print('epoch', epoch_i, 'loss_G', float(loss_G), 'loss_D', float(loss_D),
                      'gen_hr', float(discriminator(gen_hr2)), 'hr', float(discriminator(imgs_hr.cuda())))
            except :
                continue
        model = 'generator' + str(epoch_i) + '.pth'
        torch.save(generator, os.path.join(save_path, model))
        t2 = time.time()
        print(epoch_i, t2 - t1)
#


