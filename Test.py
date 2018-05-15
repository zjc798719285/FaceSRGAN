from DataSet import ImageDataset
from torch.utils.data import DataLoader
import torch
from ImageTransform import TensorToImage
import cv2, os
data_path = 'E:\PROJECT\FaceSRGAN\dataset\\'
model_path = 'E:\PROJECT\FaceSRGAN\checkpoint\\generator2.pth'
results = 'E:\PROJECT\FaceSRGAN\Result\\'

dataloader = DataLoader(ImageDataset(data_path, 4), batch_size=1, shuffle=True, num_workers=0)
generator = torch.load(model_path)


if __name__ == '__main__':
    for step, imgs in enumerate(dataloader):
        try:
            print(step)
            fileName = imgs['filename'][0].split('\\', len(imgs['filename'][0]))[-1]
            imgs_hr = (imgs['hr']); imgs_lr = (imgs['lr'])
            gen_hr = generator(imgs_lr.cuda()); gen_cpu = gen_hr.cpu()
            image_hr = TensorToImage(imgs_hr.numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image_lr = TensorToImage(imgs_lr.numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image_gen = TensorToImage(gen_cpu.detach().numpy(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            path_hr = os.path.join(results, 'HR', fileName)
            path_lr = os.path.join(results, 'LR', fileName)
            path_gen = os.path.join(results, 'GEN', fileName)
            cv2.imwrite(path_hr, image_hr)
            cv2.imwrite(path_lr, image_lr)
            cv2.imwrite(path_gen, image_gen)
        except:
            continue

