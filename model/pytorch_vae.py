import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from pathlib import Path

# VAE model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=128):
        return input.view(input.size(0), size, 8, 8)
        
class pytorch_VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=8*8*128, z_dim=10):
        super(pytorch_VAE, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            
            # 入力： 3 x 256 x 256
            nn.Conv2d(image_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
            # サイズ： 8 x 128 x 128
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),            
            # サイズ： 16 x 64 x 64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),  
            # サイズ： 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
            # サイズ： 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2),
            # サイズ： 128 x 8 x 8                         
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            

            # サイズ： 128 x 8 x 8
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # # サイズ： 64 x 16 x 16
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # # サイズ： 32 x 32 x 32
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # # サイズ： 16 x 64 x 64
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # サイズ： 8 x 128 x 128
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # サイズ： 3 x 256 x 256
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var, z


