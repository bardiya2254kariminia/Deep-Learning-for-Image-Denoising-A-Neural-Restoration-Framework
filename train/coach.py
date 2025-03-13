
import os,sys,argparse
import tqdm as tqdm
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from  torch.utils.data import DataLoader
from Dataset.datasets import STL10 , CIFAR100
from Dataset.augmentation import Augmentation
from config.path_config import *
from models.networks import Unet,VAE
from utils.common import *
from torch.optim import Adam
import torch.nn.functional as F

class Coach(object):
    def __init__(self , opts):
        self.opts = opts
        self.net = self.get_model()
        self.load_net_weigths()
        # optimizers
        if self.opts.mode == "train":
            self.optimizer = Adam(params=self.net.parameters(),
                                betas=(self.opts.b1 , self.opts.b2),
                                lr=self.opts.learning_rate,
                                weight_decay=self.opts.weigth_decay)
            # criterions
            self.l1_loss = F.l1_loss()
        else:
            self.eval()
    
    def get_model(self):
        if self.opts.model == "U-net":
            return Unet(self.opts)
        elif self.opts.model == "VAE":
            return VAE(self.opts)

    def load_net_weigths(self):
        try:
            if self.opts.load_weigth_path:
                ckpt = torch.load(self.opts.load_weigth_path, map_location="cpu")
                self.net.load_state_dict(state_dict=ckpt , strict=False)
            print("weigths successfully loaded")
        except:
            print("something went wrong on loading")

    def teach(self, ):
        pass

    def forward_pass(self):
        pass

    def calc_loss(self):
        pass
    
    def test_single(self):
        pass
    
    def save_weigths(self):
        pass

    def load_weigths(self):
        pass

    def eval(self):
        self.net.eval()