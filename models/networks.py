"""
implementation of several deep networks for calculating the outputs metrics.
presented by bardiya2254kariminia@github.com

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as  F
import torchsummary

# implementation of the models:

# Unet
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

# VAE(variational auto encoder)[vectorwise format and cnn based]
class VAE(nn.Module):
    def __init__(self ,spatial = 32 , latent_dim=128):
        super(VAE, self).__init__()
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        encoder_channel =[3, spatial, spatial*2, latent_dim]
        decoder_channel =[latent_dim, spatial*2, spatial, 3]
        # Encoder architecture
        for i , (in_c,out_c) in enumerate(zip(encoder_channel[:-1], encoder_channel[1:]),start=1):
            self.encoder_list.add_module(
                f"conv_{i}" ,nn.Conv2d(in_c , out_c , kernel_size=4,stride=2,padding=1)
            )
            self.encoder_list.add_module(
                f"leakyRelu_{i}" ,nn.LeakyReLU(0.2)
            )
        self.encoder_list.add_module("fc_encoder" , nn.Flatten())
        self.encoder = nn.Sequential(*self.encoder_list)

        # Bottleneck space
        self.mean_fc = nn.Linear(in_features= 128*16*16 , out_features=latent_dim)
        self.log_variance_fc = nn.Linear(in_features= 128*16*16 , out_features=latent_dim)
        self.fc_upsample =  nn.Linear(in_features=latent_dim , out_features=128*16*16)

        # Decoder architecture
        for i , (in_c, out_c) in enumerate(zip(decoder_channel[:-1] , decoder_channel[1:]),start=1):
            self.decoder_list.add_module(
                f"conv_transpose_{i}", nn.ConvTranspose2d(in_c,out_c , kernel_size=4,stride=2,padding=1)
            )
            self.decoder_list.add_module(
                f"leakyRelu_{i}", nn.LeakyReLU(0.2)
            )
        self.decoder_list.add_module("tanh_decoder" ,nn.Tanh()) #for better mapping of  the outputs and numerical stability
        self.decoder = nn.Sequential(*self.decoder_list)
        
    def get_z(self, mean , log_variance):
        std = torch.exp((1/2) * log_variance) # -> exp(0.5  * log_variance) = log_variance ^ (1/2) = std
        z = torch.randn_like(std)
        return mean + z * std
    
    def forward(self,x:torch.Tensor):
        # encoding phase
        encoder_out = self.encoder(x)
        # bottleneck phase
        mean = self.mean_fc(encoder_out)
        log_variance = self.log_variance_fc(encoder_out)
        z = self.get_z(mean , log_variance)
        decoded_z = self.fc_upsample(z).view((-1 ,128,16,16))
        # decodding phase
        out = self.decoder(decoded_z)
        return out


if __name__ == "__main__":
    model = VAE()
    out = model(torch.zeros(1,3,128,128))