import numpy as np
def TensorToImage(tensor, mean, std):
    mean = np.reshape(a=np.array(mean), newshape=(1, 1, 3))
    std = np.reshape(a=np.array(std), newshape=(1, 1, 3))
    img = np.transpose(tensor[0, :, :, :], (1, 2, 0))
    img = (img * std + mean) * 255
    image = img.astype('uint8')
    return image

