import os
import torch
import torchvision
from torch import nn

class EncodeBlock(nn.Module):
    def __init__(self, in_feature, out_feature, use_bn, act_type, use_pool):
        super(EncodeBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        layers = []
        layers.append(nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1))
        if use_bn: layers.append(nn.BatchNorm2d(out_feature))
        if (act_type=='LReLU'): layers.append(nn.LeakyReLU(0.2, inplace=True))
        if (act_type=='Tanh'): layers.append(nn.Tanh())
        if use_pool: layers.append(nn.AvgPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)      

class DecodeBlock(nn.Module):
    def __init__(self, in_feature, out_feature, use_bn, act_type, use_upsamp):
        super(DecodeBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        layers = []
        if use_upsamp: layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1))
        if use_bn: layers.append(nn.BatchNorm2d(out_feature))
        if (act_type=='LReLU'):layers.append(nn.LeakyReLU(0.2, inplace=True))
        if (act_type=='Tanh'):layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        nc = 3
        nf = 8
        # 入力： (nc) x 256 x 256
        self.enc1 = EncodeBlock(nc, nf, use_bn=False, act_type='LReLU', use_pool=True)
        # サイズ： (nf) x 128 x 128        
        self.enc2 = EncodeBlock(nf, nf * 2, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*2) x 64 x 64
        self.enc3 = EncodeBlock(nf * 2, nf * 4, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*4) x 32 x 32
        self.enc4 = EncodeBlock(nf * 4, nf * 8, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*8) x 16 x 16
        self.enc5 = EncodeBlock(nf * 8, nf * 16, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*16) x 8 x 8
        self.enc6 = EncodeBlock(nf * 16, nf * 32, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*32) x 4 x 4
        
        self.dec6 = DecodeBlock(nf * 32, nf * 16, use_bn=True, act_type='LReLU', use_upsamp=True)
        # サイズ： (nf*16) x 8 x 8
        self.dec5 = DecodeBlock(nf * 16, nf * 8, use_bn=True, act_type='LReLU', use_upsamp=True)
        # サイズ： (nf*8) x 16 x 16
        self.dec4 = DecodeBlock(nf * 8, nf * 4, use_bn=True, act_type='LReLU', use_upsamp=True)
        # サイズ： (nf*4) x 32 x 32
        self.dec3 = DecodeBlock(nf * 4, nf * 2, use_bn=True, act_type='LReLU', use_upsamp=True)      
        # サイズ： (nf*2) x 64 x 64        
        self.dec2 = DecodeBlock(nf * 2, nf, use_bn=True, act_type='LReLU', use_upsamp=True)
        # サイズ： (nf) x 128 x 128        
        self.dec1 = DecodeBlock(nf, nc, use_bn=False, act_type='Tanh', use_upsamp=True)
        # サイズ： (nc) x 256 x 256
        

        
    def forward(self, input):
        #Encode       
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        
        #Decode
        dec6 = self.dec6(enc6)
        dec5 = self.dec5(dec6)
        dec4 = self.dec4(dec5)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        
        x = dec1
        
        return x