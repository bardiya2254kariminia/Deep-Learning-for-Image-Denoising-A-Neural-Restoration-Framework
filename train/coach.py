
import os,sys,argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader ,random_split
from Dataset.datasets import Noisy_Cifar100 , Noisy_STL10
from Dataset.augmentation import Augmentation
from config.path_config import *
from models.networks import Unet,VAE
from utils.common import *
from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict

class Coach(object):
    def __init__(self , opts):
        self.opts = opts
        self.net = self.get_model().to(self.opts.device)
        self.load_net_weigths()
        if self.opts.mode == "train":
            # optimizers
            self.optimizer = Adam(params=self.net.parameters(),
                                betas=(self.opts.b1 , self.opts.b2),
                                lr=self.opts.learning_rate,
                                weight_decay=self.opts.weigth_decay)
            # criterions
            self.l1_loss = F.l1_loss()
            # Dataloader & Dattset
            self.dataset = self.configurate_dataset()
            self.train_dataset , self.val_dataset = random_split(dataset=self.dataset ,
                                                                  lengths=(self.opts.train_size , len(self.dataset) - self.opts.train_size))
            self.train_dataloader = DataLoader(dataset=self.train_dataset , 
                                         batch_size=self.opts.batch_size,
                                         shuffle=True,
                                         num_workers=self.opts.num_workers,
                                         drop_last=True)
            self.val_dataloader = DataLoader(dataset=self.val_dataset , 
                                         batch_size=self.opts.batch_size,
                                         shuffle=False,
                                         num_workers=self.opts.num_workers,
                                         drop_last=True)
        else:
            self.eval()
    
    def get_model(self):
        if self.opts.model == "U-net":
            return Unet(self.opts)
        elif self.opts.model == "VAE":
            return VAE(self.opts)

    def  configurate_dataset(self):
        if self.opts.dataset == "Cifar100":
            return Noisy_Cifar100()
        elif self.opts.dataset == "STL10":
            return Noisy_STL10()

    def load_net_weigths(self):
        try:
            if self.opts.load_weigth_path:
                ckpt = torch.load(self.opts.load_weigth_path, map_location="cpu")
                self.net.load_state_dict(state_dict=ckpt , strict=False)
            print("weigths successfully loaded")
        except:
            print("something went wrong on loading")

    def teach(self):
        for epoch in range(self.opts.epoches) :
            print(f"---------{epoch=}----------")
            # configure the asving methode later
            epoch_loss_dict = defaultdict(lambda : [])
            epoch_val_loss_dict = defaultdict(lambda : [])
            # train_phase
            for i, (clean_image , noisy_image) in tqdm(enumerate(self.train_dataloader , start=1),total= (self.train_dataloader)):
                clean_image = clean_image.to(self.opts.device)
                noisy_image = noisy_image.to(self.opts.device)
                self.net.train()
                self.optimizer.zero_grad()
                output_image = self.net(noisy_image)
                loss , loss_dict = self.calc_loss(output_image , clean_image)
                loss.backward()
                self.optimizer.zero_grad()
                for key in loss_dict.keys():
                    epoch_loss_dict[key] += loss_dict[key]
            print("train_losses")
            for k , v in epoch_loss_dict.items():
                print(f"{k} = {torch.mean(torch.tensor(v,dtype=torch.float32))}")
            
            # validation_phase
            for i, (clean_image , noisy_image) in tqdm(enumerate(self.train_dataloader , start=1),total= (self.train_dataloader)):
                clean_image = clean_image.to(self.opts.device)
                noisy_image = noisy_image.to(self.opts.device)
                self.net.eval()
                output_image = self.net(noisy_image)
                loss , loss_dict = self.calc_loss(output_image , clean_image)
                for key in loss_dict.keys():
                    epoch_val_loss_dict[key] += loss_dict[key]
            print("val_losses")
            for k , v in epoch_val_loss_dict.items():
                print(f"{k} = {torch.mean(torch.tensor(v,dtype=torch.float32))}")

            # saving weigths
            try:
                print(f"saving weigths in {self.opts.save_weigth_path}")
                self.save_weigths()
            except:
                print("didn't provide saving path")
                sys.exit()

    def calc_loss(self , output_image , target_image):
        if self.opts.model == "U-net"
        pass
    
    def test_single(self):
        pass
    
    def save_weigths(self):
        pass

    def load_weigths(self):
        pass

    def eval(self):
        self.net.eval()