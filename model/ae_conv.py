
import os

import torch
import torchvision
from torch import nn

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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()                             
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding = 1)  
        self.relu = nn.ReLU(True)                                   
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(32*7*7, 2) # 2 to 10
        
    def forward(self, x):   #in  [i, 1, 28, 28] 
        x = self.conv1(x)   #out [i, 16, 28, 28]
        x = self.relu(x)
        x = self.pool(x)    #out [i, 16, 14, 14]
        x = self.conv2(x)   #out [i, 32, 14, 14]
        x = self.relu(x)
        x = self.pool(x)    #out [i ,32, 7, 7]
        x = x.view(-1, 32*7*7)
        x = self.linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 32*7*7) # 10 to 2
        self.conv1_t = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv2_t = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()   # -1～1に変換

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 32, 7, 7)
        x = self.conv1_t(x)
        x = self.relu(x)
        x = self.conv2_t(x)
        #x = self.sigmoid(x)
        x = self.tanh(x)
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    # 特徴ベクトルを返すメソッド
    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


