"""
implementation of several  dataset classes for different models
presented by bardiya2254kariminia@github.com

now we cannot train them on large datasets 
so we are going to use the cifar 
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100 , STL10
from config.transform_config import Transform_class

# implementation of datasets
class Noisy_Cifar100(Dataset):
    def __init__(self , noise_std = 0.1):
        super(self,Noisy_Cifar100).__init__()
        self.cifar = CIFAR100(root="./data", train=True, download=True)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, index):
        image , label = self.cifar[index]
        clean_image = Transform_class.get_transform()["cifar100_transform"](image)
        noisy_image= clean_image + torch.rand_like(clean_image) * self.noise_std
        noisy_image = Transform_class.get_transform()["cifar100_transform"](noisy_image)
        return clean_image , noisy_image

class Noisy_STL10(Dataset):
    def __init__(self , noise_std = 0.1):
        super(self,Noisy_STL10).__init__()
        self.cifar = STL10(root="./data", train=True, download=True)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.cifar)
    
    def __getitem__(self, index):
        image , label = self.cifar[index]
        clean_image = Transform_class.get_transform()["cifar100_transform"](image)
        noisy_image= clean_image + torch.rand_like(clean_image) * self.noise_std
        noisy_image = Transform_class.get_transform()["cifar100_transform"](noisy_image)
        return clean_image , noisy_image