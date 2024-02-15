import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

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
    def __init__(self, in_feature, out_feature, use_bn, act_type, decode_type):
        super(DecodeBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        layers = []
        if (decode_type=="upsamp"):
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if (decode_type=="Conv_T"):
            layers.append(nn.ConvTranspose2d( in_feature, out_feature, kernel_size=4, stride=2, padding=1, bias=False))
        elif (decode_type!="Conv_T"):
            layers.append(nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1))
        if use_bn: layers.append(nn.BatchNorm2d(out_feature))
        if (act_type=='LReLU'):layers.append(nn.LeakyReLU(0.2, inplace=True))
        if (act_type=='Tanh'):layers.append(nn.Tanh())
        if (act_type=='Sigmoid'):layers.append(nn.Sigmoid())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VAE(nn.Module):
    def __init__(self, linear_bottleneck, decode_type, dataset):
        super(VAE, self).__init__()
        # use datset 
        self.dataset = dataset# Mnist or egg
        self.modeltype ="conv"

        nf = 8

        # 入出力チャンネル
        if self.dataset=="egg":
            nc = 3
            bn_img_size = 4
            bottle_channel_num = nf * 8
        else:
            nc = 1
            bn_img_size = 7
            bottle_channel_num = 1

             

        self.nc=nc
        self.nf=nf
        self.bottle_channel_num = bottle_channel_num
        
        self.bn_img_size = bn_img_size
        
        self.linear_bottleneck = linear_bottleneck

        self.LReLU = nn.LeakyReLU(0.2, inplace=True)

        ### for MNIST ###
        self.MNIST_enc1 = EncodeBlock(nc, nf, use_bn=True, act_type='LReLU', use_pool=True)
        #  8 x 28 x 28
        self.MNIST_enc2 = EncodeBlock(nf, nf*2, use_bn=True, act_type='LReLU', use_pool=True)
        # 16 x 14 x 14
        self.MNIST_enc3 = EncodeBlock(nf*2, nf*4, use_bn=True, act_type='LReLU', use_pool=False)
        # 32 x 14 x 14
        self.MNIST_enc4 = EncodeBlock(nc*4, nf*8, use_bn=True, act_type='LReLU', use_pool=True)
        # 64 x 7 x 7

        self.for_Mnist_em = nn.Linear(nf*2*bn_img_size*bn_img_size, linear_bottleneck)
        self.for_Mnist_ev = nn.Linear(nf*2*bn_img_size*bn_img_size, linear_bottleneck)
        self.for_Mnist_dec_fc = nn.Linear(linear_bottleneck, nf*2*bn_img_size*bn_img_size)


        self.MNIST_dec4 = DecodeBlock(nf * 8, nf * 4, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： 32 x 14 x 14
        self.MNIST_dec3 = DecodeBlock(nf * 4, nf * 2, use_bn=True,  act_type='LReLU', decode_type=decode_type)      
        # サイズ： 16 x 14 x 14        
        self.MNIST_dec2 = DecodeBlock(nf * 2, nf, use_bn=True,  act_type='LReLU', decode_type=decode_type)
        # サイズ： 8 x 28 x 28        
        self.MNIST_dec1 = DecodeBlock(nf, nc, use_bn=False,  act_type='Sigmoid', decode_type=decode_type)
        # サイズ： 1 x 28 x 28

        #############  全結合のみ #############
        image_size=32*32*3
        self.image_size = image_size
        h1_dim=1024
        h_dim=512
        z_dim=32
        self.fc1 = nn.Linear(image_size, h1_dim)
        self.fc1_1 = nn.Linear(h1_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc4_1 = nn.Linear(h_dim, h1_dim)
        self.fc5 = nn.Linear(h1_dim, image_size)

        ### for egg #####################
        # 入力： (nc) x h x w
        self.enc1 = EncodeBlock(nc, nf, use_bn=False, act_type='LReLU', use_pool=True)
        # サイズ： (nf) x h/2 x w/2        
        self.enc2 = EncodeBlock(nf, nf * 2, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*2) x h/4 x w/4
        self.enc3 = EncodeBlock(nf * 2, nf * 4, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*4) x h/8 x w/8
        self.enc4 = EncodeBlock(nf * 4, nf * 8, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*8) x h/16 x w/16
        self.enc5 = EncodeBlock(nf * 8, nf * 16, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*16) x h/32 x w/32
        self.enc6 = EncodeBlock(nf * 16, nf * 32, use_bn=True, act_type='LReLU', use_pool=True)
        # サイズ： (nf*32) x h/64 x w/64

        
        self.em = nn.Linear(bottle_channel_num*bn_img_size*bn_img_size, linear_bottleneck)
        self.ev = nn.Linear(bottle_channel_num*bn_img_size*bn_img_size, linear_bottleneck)
        self.dec_fc = nn.Linear(linear_bottleneck, bottle_channel_num*bn_img_size*bn_img_size)

        self.dec6 = DecodeBlock(nf * 32, nf * 16, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf*16) x 8 x 8
        self.dec5 = DecodeBlock(nf * 16, nf * 8, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf*8) x 16 x 16
        self.dec4 = DecodeBlock(nf * 8, nf * 4, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf*4) x 32 x 32
        self.dec3 = DecodeBlock(nf * 4, nf * 2, use_bn=True,  act_type='LReLU', decode_type=decode_type)      
        # サイズ： (nf*2) x 64 x 64        
        #self.dec2 = DecodeBlock(nf * 2, nf, use_bn=True,  act_type='LReLU', decode_type=decode_type)
        self.dec2_1 = DecodeBlock(nf * 16, nf*8, use_bn=True,  act_type='LReLU', decode_type=False)
        self.dec2 = DecodeBlock(nf * 2, nf, use_bn=True,  act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf) x 128 x 128        
        self.dec1_2 = DecodeBlock(nf*4, nf*2, use_bn=False,  act_type='LReLU', decode_type=False)
        self.dec1_1 = DecodeBlock(nf*2, nf, use_bn=False,  act_type='LReLU', decode_type=False)
        self.dec1 = DecodeBlock(nf, nc, use_bn=False,  act_type='Tanh', decode_type=decode_type)
        # サイズ： (nc) x 256 x 256

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc1_1(h))
        return self.fc2(h), self.fc3(h)
    
    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc4_1(h))
        return  F.tanh(self.fc5(h))

    def mnist_decode(self, z):
        x = self.for_Mnist_dec_fc(z)
        x = x.view(-1, self.nf*2,self.bn_img_size,self.bn_img_size)
        x = self.LReLU(x)
        dec2 = self.MNIST_dec2(x)
        dec1 = self.MNIST_dec1(dec2)
        return dec1
    

    def forward(self, input):
        if self.dataset=="MNIST":
            #####   for MNIST  ###########
            ########### 全結合のみ ############
            """mu, log_var = self.encode(input.view(-1, self.image_size))
            z = self.reparameterize(mu, log_var)
            x_reconst = self.decode(z)
            x_reconst= x_reconst.view(-1, 1, 28, 28)"""

            ##########  conv #############
            #Encode       
            enc1 = self.MNIST_enc1(input) # サイズ：  8 x 14 x 14
            enc2 = self.MNIST_enc2(enc1)  # サイズ： 16 x  7 x  7

            x = enc2.view(-1, self.nf*2*self.bn_img_size*self.bn_img_size)
            mu = self.for_Mnist_em(x)
            log_var = self.for_Mnist_ev(x)
            z = self.reparameterize(mu, log_var)
            
            #Decode
            x = self.for_Mnist_dec_fc(z)
            x = x.view(-1, self.nf*2,self.bn_img_size,self.bn_img_size)
            x = self.LReLU(x)
            dec2 = self.MNIST_dec2(x)
            dec1 = self.MNIST_dec1(dec2)
            
            x_reconst = dec1
        
        ######  for egg  ###########
        elif self.dataset=="egg":

            # # 全結合のみ
            # mu, log_var = self.encode(input.view(-1, self.image_size))
            # z = self.reparameterize(mu, log_var)
            # x_reconst = self.decode(z)
            # x_reconst= x_reconst.view(-1, 3, 32, 32)
            
            ##################    Encode   ##################      
            enc1 = self.enc1(input)
            enc2 = self.enc2(enc1)
            enc3 = self.enc3(enc2)
            enc4 = self.enc4(enc3)
            #print(enc4[0])
            #enc5 = self.enc5(enc4)
            #enc6 = self.enc6(enc5)

            #     # pooling 2回 
            # enc1 = self.enc1(input)     # w/2 * h/2 * nf 8     (16*16*8=2048) 
            # enc1_1 = self.enc1_1(enc1)  # w/2 * h/2 * nf*2 16  (16*16*16=4096)
            # enc1_2 = self.enc1_2(enc1_1)# w/2 * h/2 * nf*4 32  (16*16*32=8192)
            # enc2 = self.enc2(enc1_2)    # w/4 * h/4 * nf*8 64  (8*8*64=4096)
            # enc2_1 = self.enc2_1(enc2)  # w/4 * h/4 * nf*16 128(8*8*128)
            
            x = enc4.view(-1, self.bottle_channel_num*self.bn_img_size*self.bn_img_size)
            mu = self.em(x)
            log_var = self.ev(x)
            z = self.reparameterize(mu, log_var)
            
            ####################   Decode  #########################
            x = self.dec_fc(z)
            x = x.view(-1, self.bottle_channel_num,self.bn_img_size,self.bn_img_size)
            x = self.LReLU(x)

            #     # pooling 2回
            # dec2_1 = self.dec2_1(enc2_1)
            # #dec2_1 = self.dec2_1(x)
            # dec2 = self.dec2(dec2_1)
            # dec1_2 = self.dec1_2(dec2)
            # dec1_1 = self.dec1_1(dec1_2)
            # dec1 = self.dec1(dec1_1)

            
            #dec6 = self.dec6(x)
            #dec5 = self.dec5(x)
            #dec5 = self.dec5(x)
            
            #dec5 = self.dec5(x)
            dec4 = self.dec4(x)
            dec3 = self.dec3(dec4)
            dec2 = self.dec2(dec3)
            dec1 = self.dec1(dec2)
            
            x_reconst = dec1
            #x_reconst = x_reconst.view(x.size())
            
        return x_reconst, mu, log_var, z

    def egg_decode(self, z):
                
        #Decode
        x = self.dec_fc(z)
        x = x.view(-1, self.bottle_channel_num,self.bn_img_size,self.bn_img_size)
        x = self.LReLU(x)
        
        #dec6 = self.dec6(x)
        #dec5 = self.dec5(dec6)
        #dec5 = self.dec5(x)
        
        #dec5 = self.dec5(x)
        dec4 = self.dec4(x)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        
        x_reconst = dec1
        
        return x_reconst




class AE(nn.Module):
    def __init__(self, decode_type):
        super(AE, self).__init__()
        # use datset 
        self.dataset = "egg"
        self.modeltype ="conv"

        # 入出力チャンネル
        if self.dataset=="egg":
            nc = 3
            bn_img_size = 8
        else:
            nc = 1
            bn_img_size = 7
        nf = 8      

        self.nc=nc
        self.nf=nf


        self.LReLU = nn.LeakyReLU(0.2, inplace=True)

        ### for MNIST ###
        self.MNIST_enc1 = EncodeBlock(nc, nf, use_bn=True, act_type='LReLU', use_pool=True)
        #  8 x 28 x 28
        self.MNIST_enc2 = EncodeBlock(nf, nf*2, use_bn=True, act_type='LReLU', use_pool=True)
        # 16 x 14 x 14
        self.MNIST_enc3 = EncodeBlock(nf*2, nf*4, use_bn=True, act_type='LReLU', use_pool=False)
        # 32 x 14 x 14
        self.MNIST_enc4 = EncodeBlock(nc*4, nf*8, use_bn=True, act_type='LReLU', use_pool=True)
        # 64 x 7 x 7

        self.MNIST_dec4 = DecodeBlock(nf * 8, nf * 4, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： 32 x 14 x 14
        self.MNIST_dec3 = DecodeBlock(nf * 4, nf * 2, use_bn=True,  act_type='LReLU', decode_type=decode_type)      
        # サイズ： 16 x 14 x 14        
        self.MNIST_dec2 = DecodeBlock(nf * 2, nf, use_bn=True,  act_type='LReLU', decode_type=decode_type)
        # サイズ： 8 x 28 x 28        
        self.MNIST_dec1 = DecodeBlock(nf, nc, use_bn=False,  act_type='Sigmoid', decode_type=decode_type)
        # サイズ： 1 x 28 x 28

        #############  全結合のみ #############
        image_size=28*28
        self.image_size = image_size
        h_dim=400
        z_dim=20
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

        ### for egg #####################
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

        self.dec6 = DecodeBlock(nf * 32, nf * 16, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf*16) x 8 x 8
        self.dec5 = DecodeBlock(nf * 16, nf * 8, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf*8) x 16 x 16
        self.dec4 = DecodeBlock(nf * 8, nf * 4, use_bn=True, act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf*4) x 32 x 32
        self.dec3 = DecodeBlock(nf * 4, nf * 2, use_bn=True,  act_type='LReLU', decode_type=decode_type)      
        # サイズ： (nf*2) x 64 x 64        
        self.dec2 = DecodeBlock(nf * 2, nf, use_bn=True,  act_type='LReLU', decode_type=decode_type)
        # サイズ： (nf) x 128 x 128        
        self.dec1 = DecodeBlock(nf, nc, use_bn=False,  act_type='Tanh', decode_type=decode_type)
        # サイズ： (nc) x 256 x 256

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std      
        
    def encode(self, x):
        h =  F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def decode(self, z):
        h =  F.relu(self.fc4(z))
        return  F.sigmoid(self.fc5(h))
    

    def forward(self, input):
        #####   for MNIST  ###########
        ########### 全結合のみ ############
        """mu, log_var = self.encode(input.view(-1, self.image_size))
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        x_reconst= x_reconst.view(-1, 1, 28, 28)"""

        ##########  conv #############
        #Encode       
        """enc1 = self.MNIST_enc1(input) # サイズ：  8 x 14 x 14
        enc2 = self.MNIST_enc2(enc1)  # サイズ： 16 x  7 x  7

        #Decode
        dec2 = self.MNIST_dec2(enc2)
        dec1 = self.MNIST_dec1(dec2)
        
        x_reconst = dec1"""

        ######  for egg  ###########

        #Encode       
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        #enc5 = self.enc5(enc4)
        #enc6 = self.enc6(enc5)

        
        #dec6 = self.dec6(enc6)
        #dec5 = self.dec5(enc5)
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        
        x_reconst = dec1
        
        return x_reconst

    def egg_decode(self, z):
                
        #Decode
        x = self.dec_fc(z)
        x = x.view(-1, self.nf*16,8,8)
        x = self.LReLU(x)
        
        #dec6 = self.dec6(x)
        #dec5 = self.dec5(dec6)
        #dec5 = self.dec5(x)
        
        dec5 = self.dec5(x)
        dec4 = self.dec4(dec5)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        
        x_reconst = dec1
        
        return x_reconst


class EncodeBlock1(nn.Module):
    def __init__(self, in_feature, out_feature, last_layer_flg):
        super(EncodeBlock1, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        layers = []
        if last_layer_flg:
            layers.append(nn.Conv2d(in_feature, out_feature, 4,1))
        else:
            layers.append(nn.Conv2d(in_feature, out_feature, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)    

class DecodeBlock1(nn.Module):
    def __init__(self, in_feature, out_feature, first_layer_flg):
        super(DecodeBlock1, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        layers = []
        if first_layer_flg:
            layers.append(nn.ConvTranspose2d( in_feature, out_feature, 4))
        else:
            layers.append(nn.ConvTranspose2d( in_feature, out_feature, kernel_size=4, stride=2, padding=0))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim, nc=1):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc

        self.ReLU = nn.ReLU(True)

        self.enc1 = nn.Conv2d(nc, 32, 4, 2, 1)          # B,  32, 32, 32
        self.enc2 = nn.Conv2d(32, 32, 4, 2, 1)          # B,  32, 16, 16
        self.enc3 = nn.Conv2d(32, 32, 4, 2, 1)          # B,  32,  8,  8
        self.enc4 = nn.Conv2d(32, 32, 4, 2, 1)          # B,  32,  4,  4
        self.enc_view = View((-1, 32*4*4))                  # B, 512
        self.enc_mnist_view = View((-1, 32*7*7))                  # B, 512
        self.enc_fc1 = nn.Linear(32*4*4, 256)              # B, 256
        self.enc_mnist_fc1 = nn.Linear(32*7*7, 256) 
        self.enc_fc2 = nn.Linear(256, 256)                 # B, 256
        self.enc_fc3 = nn.Linear(256, z_dim)             # B, z_dim*2
    

        self.dec_fc1 = nn.Linear(z_dim, 256)               # B, 256
        self.dec_fc2 = nn.Linear(256, 256)                 # B, 256
        self.dec_fc3 = nn.Linear(256, 32*4*4)              # B, 512
        self.dec_mnist_fc3 = nn.Linear(256, 32*7*7) 
        self.dec_view = View((-1, 32, 4, 4))                # B,  32,  4,  4
        self.dec_mnist_view = View((-1, 32, 7, 7))                # B,  32,  4,  4
        self.dec1 = nn.ConvTranspose2d(32, 32, 4, 2, 1) # B,  32,  8,  8
        self.dec2 = nn.ConvTranspose2d(32, 32, 4, 2, 1) # B,  32, 16, 16
        self.dec3 = nn.ConvTranspose2d(32, 32, 4, 2, 1) # B,  32, 32, 32
        self.dec4 = nn.ConvTranspose2d(32, nc, 4, 2, 1) # B,  nc, 64, 64


    def weight_init(self):
        for blk in self._modules:
            for m in self._modules[blk]:
                kaiming_init(m)

    def forward(self, x):

        #print(x.shape)
        distributions = self.enc1(x)
        #print(distributions.shape)
        distributions = self.ReLU(distributions)

        distributions = self.enc2(distributions)
        #print(distributions.shape)
        distributions = self.ReLU(distributions)

        # distributions = self.enc3(distributions)
        # print(distributions.shape)
        # distributions = self.ReLU(distributions)

        # distributions = self.enc4(distributions)
        # distributions = self.ReLU(distributions)

        #distributions = self.enc_view(distributions)
        #print(distributions.shape)
        distributions = self.enc_mnist_view(distributions)

        #distributions = self.enc_fc1(distributions)
        distributions = self.enc_mnist_fc1(distributions)
        
        distributions = self.ReLU(distributions)

        distributions = self.enc_fc2(distributions)
        distributions = self.ReLU(distributions)
        distributions = self.enc_fc3(distributions)

        #distributions = self._encode(x)
        mu = distributions
        logvar = distributions
        z = reparametrize(mu, logvar)
        #x_recon = self._decode(z)
        
        x_recon = self.dec_fc1(z)
        #x_recon = self.dec_fc1(distributions)

        x_recon = self.ReLU(x_recon)
        x_recon = self.dec_fc2(x_recon)
        x_recon = self.ReLU(x_recon)

        #x_recon = self.dec_fc3(x_recon)
        x_recon = self.dec_mnist_fc3(x_recon)
        x_recon = self.ReLU(x_recon)

        #x_recon = self.dec_view(x_recon)
        x_recon = self.dec_mnist_view(x_recon)

        # x_recon = self.dec1(x_recon)
        # x_recon = self.ReLU(x_recon)
        # x_recon = self.dec2(x_recon)
        # x_recon = self.ReLU(x_recon)
        x_recon = self.dec3(x_recon)
        x_recon = self.ReLU(x_recon)
        x_recon = self.dec4(x_recon)
        x_recon = x_recon.view(x.size())


        return x_recon, mu, logvar, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)