from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from PIL import Image
from pathlib import Path
from tqdm import tqdm

def main():

    # 再現性のためにrandom seedを設定する
    manualSeed = 0
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    data_dir = "/dataset/dataset/egg/nomal_egg"
    log_path = '/home/taki/egg/log/ae/nomalegg_dim_16_16'
    # フォルダ作成
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    # 訓練中のバッチサイズ
    batch_size = 32

    # 訓練画像の高さと幅のサイズ
    image_size = 32
    num_epochs = 100
    lr = 0.001

    # Transform を作成する
    transform = torchvision.transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Dataset を作成する。
    dataset = ImageFolder(img_dir= data_dir, transform=transform)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    test_dataset = ImageFolder(data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)




    ############### setting ##################
    # 訓練画像のチャンネル数。カラー画像の場合は3。
    nc = 3

    # 特徴マップのサイズ
    nf = 8

    # 再現性のためにrandom seedを設定する
    manualSeed = 0

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = AutoEncoder()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    model.apply(weights_init)


    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)



    ############ treain #######
    history = {
    'train_loss': [],
    'valid_loss': [],}

    # Train Step
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))

        model.train()

        batch_loss = []

        for batch in loop:
            inputs = batch
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        
            batch_loss.append(loss)

        train_avg_loss = torch.tensor(batch_loss).mean()

        batch_loss = []

  

        # Validation Step
        with torch.no_grad():
            model.eval()
            loop_val = tqdm(val_loader, unit='batch', desc='| Test  | Epoch {:>3} |'.format(epoch + 1))

            for i, batch in enumerate(loop_val):
                inputs = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                batch_loss.append(loss)

            valid_avg_loss = torch.tensor(batch_loss).mean()

        history['train_loss'].append(train_avg_loss)
        history['valid_loss'].append(valid_avg_loss)
        
        print(f"epoch: {epoch+1}, train_loss: {train_avg_loss:.3f}, valid_loss: {valid_avg_loss:.3f}")

    print('Finished Training')

    plt.figure(figsize=(8, 4.5))
    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    #plt.ylim(0, 0.1)
    plt.legend(['Train', 'Valid'], loc='upper right')
    plt.grid(True)
    loss_name = 'loss.png'
    loss_path = os.path.join(log_path, loss_name)
    plt.savefig(loss_path)
    plt.show()

    model_name = 'autoencoder.pth'
    model_path = os.path.join(log_path, model_name)
    torch.save(model.state_dict(), model_path)


    original_imgs = []
    decoded_imgs = []
    diff_imgs = []

    scores = []

    model = AutoEncoder()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            original = batch
            original = original.to(device, non_blocking=True)
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

    # 入力画像(左)、出力画像(中)、差分画像(右)の順に表示
    plt.figure(figsize=(20, 20))
    for i in range(24):
        img = np.hstack((original_imgs[i], decoded_imgs[i], diff_imgs[i]))
        ax = plt.subplot(8, 3, i+1)
        img = np.clip((img+1)/2, 0., 1.)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #     if scores[i] > th[6]:
    #         plt.title(str(scores[i]), color='r')
    #     else:
    #         plt.title(str(scores[i]))

    diff_name = 'diff.png'
    diff_path = os.path.join(log_path, diff_name)
    plt.savefig(diff_path)
    plt.show()


#############  class and def ######################### 

# カスタムの重み初期化用関数。
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
        
        self.dec1_2 = DecodeBlock(nf*4, nf*2, use_bn=False,  act_type='Tanh', decode_type=False)
        self.dec1_1 = DecodeBlock(nf*2, nf, use_bn=False,  act_type='Tanh', decode_type=False)
        
        
    def forward(self, input):
        #Encode       
        # enc1 = self.enc1(input)
        # enc2 = self.enc2(enc1)
        # enc3 = self.enc3(enc2)
        # enc4 = self.enc4(enc3)
        # #enc5 = self.enc5(enc4)
        # #enc6 = self.enc6(enc5)
        # #enc7 = self.enc7(enc6)
        # #enc8 = self.enc8(enc7)
        
        # #Decode
        # #dec8 = self.dec8(enc8)
        # #dec7 = self.dec7(dec8)
        # #dec6 = self.dec6(enc6)
        # #dec5 = self.dec5(enc5)
        # dec4 = self.dec4(enc4)
        # dec3 = self.dec3(dec4)
        # dec2 = self.dec2(dec3)
        # dec1 = self.dec1(dec2)


        enc1 = self.enc1(input)     # w/2 * h/2 * nf
        enc1_1 = self.enc1_1(enc1)  # w/2 * h/2 * nf*2
        enc1_2 = self.enc1_2(enc1_1)# w/2 * h/2 * nf*4
        enc2 = self.enc2(enc1_2)    # w/4 * h/4 * nf*8
        enc2_1 = self.enc2_1(enc2)  # w/4 * h/4 * nf*216

        dec2_1 = self.dec2_1(enc2_1)
        dec2 = self.dec2(dec2_1)
        dec1_2 = self.dec1_2(dec2)
        dec1_1 = self.dec1_1(dec1_2)
        dec1 = self.dec1(dec1_1)

        x = dec1
        
        return x


class ImageFolder(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG"]

    def __init__(self, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_paths[index]

        # 画像を読み込む。
        img = Image.open(path)

        if self.transform is not None:
            # 前処理がある場合は行う。
            img = self.transform(img)

        return img

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。"""
        img_dir = Path(img_dir)
        img_paths = [
            p for p in img_dir.iterdir() if p.suffix in ImageFolder.IMG_EXTENSIONS
        ]
        img_paths.sort()   

        return img_paths

    def __len__(self):
        """ディレクトリ内の画像ファイルの数を返す。"""
        return len(self.img_paths)


if __name__ == '__main__':
    main()

