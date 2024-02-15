from __future__ import print_function
import argparse
import os
import random
from re import U, X
from cv2 import log
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from PIL import Image
from pathlib import Path
from tqdm import tqdm

from model.vae import VAE
import image_folder as ImageFolder
import log as log

def main():
    ############  parametar setting  ###############
    usedata = "egg" # egg or MNIST

    parser = argparse.ArgumentParser(description='vae train')
    parser.add_argument('--batch_size', help='batch size',default=32)
    parser.add_argument('--in_w', help='in_w',default=256)
    parser.add_argument('--lr', help='lr',default=0.001) 
    parser.add_argument('--b', help='beta',default=0)  #0.0001
    parser.add_argument('--linear_bn', help='linear_bottleneck',default= 1024)
    parser.add_argument('--seed', help='seed',default= 0)

    args = parser.parse_args()

    num_epochs = 100
    batch_size = int(args.batch_size)#32
    image_size = int(args.in_w)#256
    lr = float(args.lr)#0.001
    b = float(args.b)#0.0001  kl項の係数
    manualSeed = int(args.seed)
    linear_bottleneck = int(args.linear_bn)
    decode_type = 'upsamp'


    ################   folder setting  ##################
    data_dir = "/home/taki/egg/data/egg/nomal_egg"

    ##############  random seed setting  ############
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
   

    ###############   dataset setting  ###################
    if usedata=="egg":
        log_path = f'/home/taki/egg/log/ae/nomalegg_dim_16_16'
        # フォルダ作成
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        train_transform = torchvision.transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=360),
                                    transforms.ToTensor(),
                                    transforms.Resize((image_size, image_size)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = torchvision.transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((image_size, image_size)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
           
    
        test_dataset = ImageFolder.ImageFolder(data_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    elif usedata=="MNIST":
        log_path = f'/home/taki/egg/log/vae/mnist/0808_mse_bottleneck{linear_bottleneck}_b{b}'
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        img_transform = torchvision.transforms.Compose([
                                    transforms.ToTensor()
        ])
        # dataset
        dataset = MNIST('./data', train = True, download=False, transform=img_transform)
        train_size = int(len(dataset) * 0.8)
        validation_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        test_dataset = MNIST('./data', train = False, download=False, transform=img_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)


    ############### model setting ##################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # model = VAE(linear_bottleneck=linear_bottleneck, decode_type=decode_type, dataset=usedata).to(device)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    # model.apply(weights_init)

    # # loss
    # criterion = nn.MSELoss().to(device)
    # #criterion = nn.BCELoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  

    ################### test #####################
    #model_name = 'vae.pth'
    model_name = 'autoencoder.pth'
    model_path = os.path.join(log_path, model_name)
    original_imgs = []
    decoded_imgs = []
    diff_imgs = []

    scores = []

    #model = VAE(linear_bottleneck=linear_bottleneck, decode_type=decode_type, dataset=usedata)
    model = AutoEncoder()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:

            if usedata=="egg":
                original = batch
            else:
                original, _ = batch

            
            original = original.to(device, non_blocking=True)


            #decoded, _ , _ , _ = model(original)
            decoded = model(original)
            diff = torch.abs(original - decoded)
            score = torch.mean(diff.view(diff.shape[0], -1), dim=1)
            
            original_imgs.extend(original.detach().cpu().numpy())
            decoded_imgs.extend(decoded.detach().cpu().numpy())
            diff_imgs.extend(diff.detach().cpu().numpy())
            scores.extend(score.detach().cpu().numpy())

    original_imgs = np.array(original_imgs).transpose(0, 2, 3, 1)
    decoded_imgs = np.array(decoded_imgs).transpose(0, 2, 3, 1)
    diff_imgs = np.array(diff_imgs).transpose(0, 2, 3, 1)
    scores = np.array(scores)

    for i in range(23):
        plt.imshow(decoded_imgs[i])
        diff_name = f'egg_reconst{i}.png'
        diff_path = os.path.join(log_path, diff_name)
        plt.savefig(diff_path)
        plt.show()

    # 入力画像(左)、出力画像(中)、差分画像(右)の順に表示
    if usedata=="egg":
        plt.figure(figsize=(20, 20))
        for i in range(23):
            img = np.hstack((original_imgs[i], decoded_imgs[i], diff_imgs[i]))
            ax = plt.subplot(8, 3, i+1)
            img = np.clip((img+1)/2, 0., 1.)
            plt.imshow(img)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            #plt.title(str(scores[i]))

        diff_name = 'batch_egg_reconst.png'
        diff_path = os.path.join(log_path, diff_name)
        plt.savefig(diff_path)
        plt.show()
    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0.2, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0) 


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
        self.enc7 = EncodeBlock(nf * 32, nf * 64, use_bn=True, act_type='LReLU', use_pool=True)

        self.enc8 = EncodeBlock(nf * 64, nf * 128, use_bn=True, act_type='LReLU', use_pool=True)

        self.dec8 = DecodeBlock(nf * 128, nf * 64, use_bn=True, act_type='LReLU', use_upsamp=True)

        self.dec7 = DecodeBlock(nf * 64, nf * 32, use_bn=True, act_type='LReLU', use_upsamp=True)
        
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
        #enc5 = self.enc5(enc4)
        #enc6 = self.enc6(enc5)
        #enc7 = self.enc7(enc6)
        #enc8 = self.enc8(enc7)
        
        #Decode
        #dec8 = self.dec8(enc8)
        #dec7 = self.dec7(dec8)
        #dec6 = self.dec6(enc6)
        #dec5 = self.dec5(enc5)
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        
        x = dec1
        
        return x



if __name__ == '__main__':
    main()

