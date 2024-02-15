from __future__ import print_function
import argparse
from cProfile import label
import os
import random
from re import X
from cv2 import log
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from tqdm.notebook import trange, tqdm

from model.vae import VAE
import image_folder as ImageFolder
import log as log

def main():

    parser = argparse.ArgumentParser(description='vae train')

    parser.add_argument('--batch_size', help='batch size',default=32)
    parser.add_argument('--in_w', help='in_w',default=256)
    parser.add_argument('--lr', help='lr',default=0.001) 
    parser.add_argument('--b', help='beta',default=0)  #0.0001
    parser.add_argument('--linear_bn', help='linear_bottleneck',default= 1024)
    parser.add_argument('--seed', help='seed',default= 0)

    args = parser.parse_args()

    image_size = int(args.in_w)#256
    b = float(args.b)#0.0001  kl項の係数
    manualSeed = int(args.seed)
    linear_bottleneck = int(args.linear_bn)


    usedata = "MNIST"

    #linear_bottleneck = 1024
    b = 0.001
    manualSeed = 0
    decode_type = 'upsamp'

    data_dir = "/dataset/dataset/egg/nomal_egg"
    #data_dir = '/home/taki/egg/data/egg/nomal_egg'
    print(f"linear_bottleneck:{linear_bottleneck}")
    #log_path = f'/home/taki/egg/log/vae/for_egg/{decode_type}_linear{linear_bottleneck}_b{b}_nomalegg_dim128_8_8'
    log_path = "/home/taki/egg/log/vae/mnist/0808_mse_bottleneck2_b5e-05"

    # 再現性のためにrandom seedを設定する
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

   

    img_transform = torchvision.transforms.Compose([
                     transforms.ToTensor()
    ])
    # dataset
    dataset = MNIST('./data', train = True, download=False, transform=img_transform)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    test_dataset = MNIST('./data', train = False, download=False, transform=img_transform)
    test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=False, num_workers=4, drop_last=False)

    ############### setting ##################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ################### test #####################
    original_imgs = []
    decoded_imgs = []
    diff_imgs = []

    scores = []
    latent_dirs = []

    model_name = 'vae.pth'
    model_path = os.path.join(log_path, model_name)



    model = VAE(linear_bottleneck=linear_bottleneck, decode_type=decode_type)
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
                original, label = batch

            
            original = original.to(device, non_blocking=True)
            

            decoded, _ , _ , z = model(original)
            diff = torch.abs(original - decoded)
            score = torch.mean(diff.view(diff.shape[0], -1), dim=1)
            
            original_imgs.extend(original.detach().cpu().numpy())
            decoded_imgs.extend(decoded.detach().cpu().numpy())
            diff_imgs.extend(diff.detach().cpu().numpy())

            latent_dirs.extend(z)

    original_imgs = np.array(original_imgs).transpose(0, 2, 3, 1)
    decoded_imgs = np.array(decoded_imgs).transpose(0, 2, 3, 1)

    z = z.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    plt.figure(figsize=(10, 8)) 
    plt.scatter(z[:, 0], z[:, 1], c=label, cmap="rainbow", alpha=0.6)
    for i in range(10):
        m = get_mean(z, label, i)
        plt.text(m[0], m[1], "{}".format(i), fontsize=20)
    plt.colorbar()
    plt.grid()
    diff_path = os.path.join(log_path, "latent_space.png")
    plt.savefig(diff_path)

    # for i in range(10):
    #     decoded = model.mnist_decode(latent_dirs[i])
    #     decoded = decoded.detach().cpu().numpy()
    #     decoded_imgs = np.array(decoded).transpose(0, 2, 3, 1)
    #     #img = np.hstack(decoded_imgs[i])

    #     img = np.clip((decoded_imgs[0]+1)/2, 0., 1.)
    #     plt.imshow(img)
    #     plt.gray()
    #     diff_name = f"randomreconst{i}.png" 
    #     diff_path = os.path.join(log_path, diff_name)
    #     plt.savefig(diff_path)

def get_mean(z_mu, y_sample, num):
    for i in range(10):
        idx = y_sample==num
        return np.mean(z_mu[idx], axis=0)


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



if __name__ == '__main__':
    main()

