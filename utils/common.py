import torch
import numpy as np
from PIL import Image


def denormalize_image(img:torch.Tensor , mean = (0.5,0.5,0.5) , std = (0.5,0.5,0.5)):
    img = img.cpu().detach().transpose(0,2).transpose(0,1).numpy()
    mean = np.array(mean).reshape(1,1,3)
    std = np.array(std).reshape(1,1,3)

    img  = img * std + mean
    img = np.clip(img , 0,1)
    img = (img * 255).astype("uint8")
    return img
