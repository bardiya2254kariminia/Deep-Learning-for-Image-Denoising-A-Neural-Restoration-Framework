"""
implementation of several deep networks for calculating the outputs metrics.
presented by bardiya2254kariminia@github.com

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchsummary

# implementation of the models:


class Unet_Convblock(nn.Module):
    def __init__(self , in_channel ,out_channel):
        super(Unet_Convblock,self).__init__()
        self.body_list  = nn.ModuleList()
        self.body_list.add_module(
            "conv_1" , nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1
            )
        )
        self.body_list.add_module(
            "leakyrelu_1" , nn.LeakyReLU(0.2)
        )
        self.body_list.add_module(
            "conv_2" , nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1
            )
        )
        self.body_list.add_module(
            "leakyrelu_2" , nn.LeakyReLU(0.2)
        )
        self.body = nn.Sequential(*self.body_list)
    
    def forward(self , x):
        return self.body(x)

class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.down1 = nn.Sequential(
            Unet_Convblock(in_channel=3,out_channel=64),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Unet_Convblock(in_channel=64,out_channel=128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Unet_Convblock(in_channel=128,out_channel=256)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Unet_Convblock(in_channel=256,out_channel=512)
        )
        self.up4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            Unet_Convblock(in_channel=512,out_channel=1024),
            nn.ConvTranspose2d(in_channels=1024 ,out_channels=512 ,kernel_size=2,stride=2)
        )
        self.up3 = nn.Sequential(
            Unet_Convblock(in_channel=1024,out_channel=512),
            nn.ConvTranspose2d(in_channels=512 ,out_channels=256,kernel_size=2,stride=2)
        )
        self.up2 = nn.Sequential(
            Unet_Convblock(in_channel=512,out_channel=256),    
            nn.ConvTranspose2d(in_channels=256 ,out_channels=128,kernel_size=2,stride=2)
        )
        self.up1 = nn.Sequential(
            Unet_Convblock(in_channel=256,out_channel=128),    
            nn.ConvTranspose2d(in_channels=128 ,out_channels=64,kernel_size=2,stride=2)
        )
        self.final_conv = nn.Sequential(
            Unet_Convblock(in_channel=128,out_channel=64),
            nn.Conv2d(in_channels=64, out_channels=3,kernel_size=1)
        )
    
    def forward(self,x):
        out_down1 = self.down1(x)
        # print(f"{out_down1.shape=}")
        out_down2 = self.down2(out_down1)
        # print(f"{out_down2.shape=}")
        out_down3 = self.down3(out_down2)
        # print(f"{out_down3.shape=}")
        out_down4 = self.down4(out_down3)
        # print(f"{out_down4.shape=}")
        out_up4 = self.up4(out_down4)
        # print(f"{out_up4.shape=}")
        out_up3 = self.up3(torch.cat([out_down4 , out_up4],dim=1))
        out_up2 = self.up2(torch.cat([out_down3,out_up3],dim=1))
        out_up1 = self.up1(torch.cat([out_down2, out_up2],dim=1))
        out = self.final_conv(torch.cat([out_down1,out_up1],dim=1))
        return out



if __name__ == "__main__":
    model = Unet()
    out = model(torch.zeros(1,3,256,256))
    print(f"{out.shape=}")