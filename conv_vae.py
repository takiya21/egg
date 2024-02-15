from __future__ import print_function
import argparse
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

from model.vae import VAE, BetaVAE_H
from model.pytorch_vae import pytorch_VAE
import image_folder as ImageFolder
import log as log


def main():
    ############  parametar setting  ###############

    parser = argparse.ArgumentParser(description='vae train')
    parser.add_argument('--batch_size', help='batch size',default=64)
    parser.add_argument('--in_w', help='in_w',default=256)
    parser.add_argument('--lr', help='lr',default=1e-4) 
    parser.add_argument('--b', help='beta',default=1)  #0.0001
    parser.add_argument('--linear_bn', help='linear_bottleneck',default= 30)
    parser.add_argument('--seed', help='seed',default= 0)
    parser.add_argument('--dataset', help='dataset',default= "egg")
    parser.add_argument('--scheduler', help='scheduler',default= "cos")
    parser.add_argument('--scheduler_gamma', help='scheduler_gamma',default= "1")


    args = parser.parse_args()

    num_epochs = 1000
    batch_size = int(args.batch_size)#32
    image_size = int(args.in_w)#256
    lr = float(args.lr)#0.001
    b = float(args.b)#0.0001  kl項の係数
    manualSeed = int(args.seed)
    linear_bottleneck = int(args.linear_bn)
    usedata = str(args.dataset)
    use_scheduler = str(args.scheduler)
    scheduler_gamma = float(args.scheduler_gamma)
    decode_type = 'upsamp'
    weight_decay = 0.0001
    h_dim = 8*8*128
    z_dim = 30


    ################   folder setting  ##################
    data_dir = "/dataset/dataset/egg/nomal_egg"
    print(f"linear_bottleneck:{linear_bottleneck}")



    ##############  random seed setting  ############
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
   

    ###############   dataset setting  ###################
    if usedata=="egg":
        #log_path = f'/home/taki/egg/log/vae/for_egg/0914_adamw{weight_decay}_lr{lr}_inW{image_size}_b{b}_btlsize16_16_64_linear{linear_bottleneck}_scheduler_{use_scheduler}{scheduler_gamma}_mse_epoch{num_epochs}'
        #log_path = f'/home/taki/egg/log/vae/for_egg/vae_test1118_lr{lr}_inW{image_size}_b{b}_btlsize1_1_256_linear{linear_bottleneck}_scheduler_{use_scheduler}{scheduler_gamma}_mse_epoch{num_epochs}'
        log_path = f'/home/taki/egg/log/vae/for_egg/1118_conv_vay.py_pytorch_vae_lr{lr}_inW{image_size}_b{b}_mse=sum_epoch{num_epochs}_h_dim{h_dim}_z_dim{z_dim}_batch{batch_size}_sheduler=cosin_dataaug=flip'

        # フォルダ作成
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        train_transform = torchvision.transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    #transforms.RandomRotation(degrees=360),
                                    transforms.ToTensor(),
                                    transforms.Resize((image_size, image_size)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = torchvision.transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((image_size, image_size)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
           
        dataset = ImageFolder.ImageFolder(img_dir= data_dir, transform=train_transform)
        train_size = int(len(dataset) * 0.8)
        validation_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
        test_dataset = ImageFolder.ImageFolder(data_dir, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    elif usedata=="MNIST":
        log_path = f'/home/taki/egg/log/vae/mnist/h_vae1116_lr{lr}_inW{image_size}_b{b}_btlsize1_1_256_linear{linear_bottleneck}_scheduler_{use_scheduler}{scheduler_gamma}_mse_epoch{num_epochs}'
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

    #model = VAE(linear_bottleneck=linear_bottleneck, decode_type=decode_type, dataset=usedata).to(device)
    #model = BetaVAE_H(z_dim=linear_bottleneck).to(device)
    model = pytorch_VAE(image_channels=3, h_dim=h_dim, z_dim=z_dim).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.apply(weights_init)

    # loss
    criterion = nn.MSELoss().to(device)
    #criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    # sheduler
    if use_scheduler=="step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif use_scheduler=="exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    elif use_scheduler=="cos":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-10)

    #################### train ####################
    original_imgs = []
    decoded_imgs = []
    diff_imgs = []
    scores = []
    scheduler_lr_list = []


    history = log.History(keys=('train_loss',
                                'val_loss',
                                'kl_loss',
                                'mse_loss',
                                'val_kl_loss',
                                'val_mse_loss',
                                'epoch_cnt',
                                'train_avg_loss',
                                'valid_avg_loss'),
                                 output_dir=log_path)

    # Train Step
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))

        # Train Step. {{{
        # =====
        batch_loss = []     
        model.train()
        G_meter = log.AverageMeter()
        kl_meter = log.AverageMeter()
        mse_meter = log.AverageMeter()
        
        for _, batch in enumerate(loop):
            

            if usedata=="egg":
                inputs = batch
            else:
                inputs, _ = batch
            inputs = inputs.to(device)
            
            
            #print(inputs.shape)
            x_reconst, mu, log_var, _ = model(inputs)
            #print(x_reconst.shape)
            #mse_loss = criterion(x_reconst, inputs)
            mse_loss = F.mse_loss(x_reconst, inputs, size_average=False)
            #mse_loss = F.binary_cross_entropy(x_reconst, inputs, size_average=False)
            
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            loss = mse_loss + b * kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get losses. {{{}}
            G_meter.update(loss.item(), inputs[0].size()[0])
            kl_meter.update((b * kl_div).item(), inputs[0].size()[0])
            mse_meter.update(mse_loss.item(), inputs[0].size()[0])
            history({'train_loss': loss.item()})


                # }}} get loss

            batch_loss.append(loss)

        # Print training log. {
        # =====
        msg = "[Train {}] Epoch {}/{}".format(
            'VAE', epoch + 1, num_epochs)
        msg += " - {}: {:.4f}".format('train_loss', G_meter.avg)
        msg += " - {}: {}".format('mse', mse_meter.avg)
        msg += " - {}: {}".format('b*kl_div', b * kl_div)
        msg += " - {}: {:.8f}".format('learning rate',
                                      scheduler.get_last_lr()[0])
        history({'epoch_cnt': epoch})
        print(msg)

        ## add histroy ##
        history({'kl_loss':kl_meter.avg})
        history({'mse_loss': mse_meter.avg})

        train_avg_loss = torch.tensor(batch_loss).mean()
        batch_loss = []
        model.eval()
        scheduler.step()
        G_meter.reset()
        kl_meter.reset()
        mse_meter.reset()
        # }}}Train Step.


        # # Validation Step. {
        # =====
        with torch.no_grad():
            model.eval()
            loop_val = tqdm(val_loader, unit='batch', desc='| Val  | Epoch {:>3} |'.format(epoch + 1))

            for i, batch in enumerate(loop_val):
                
                if usedata=="egg":
                    inputs = batch
                else:
                    inputs, _ = batch

                inputs = inputs.to(device)

                x_reconst, mu, log_var, _ = model(inputs)
                #mse_loss = criterion(x_reconst, inputs)
                mse_loss = F.mse_loss(x_reconst, inputs, size_average=False)
                #mse_loss = F.binary_cross_entropy(x_reconst, inputs, size_average=False)
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                

                diff = torch.abs(inputs - x_reconst)

                loss = mse_loss + b * kl_div
                batch_loss.append(loss)

                G_meter.update(loss.item(), inputs[0].size()[0])
                kl_meter.update((b * kl_div).item(), inputs[0].size()[0])
                mse_meter.update(mse_loss.item(), inputs[0].size()[0])
                history({'val_loss': loss.item()})

                inputs_img  = inputs.to('cpu').detach().numpy().copy()
                outputs_img = x_reconst.to('cpu').detach().numpy().copy()
                #diff_imgs.extend(diff.detach().cpu().numpy())
                diff_imgs = diff.detach().cpu().numpy()


            valid_avg_loss = torch.tensor(batch_loss).mean()
        # =====
        # Print validation log. {
        msg = "[Validation {}] Epoch {}/{}".format(
            'VAE', epoch + 1, num_epochs)
        msg += " - {}: {:.4f}".format('val_loss', G_meter.avg)
        msg += " - {}: {}".format('mse', mse_loss)
        msg += " - {}: {}".format('b*kl_div', b * kl_div)
        msg += " - {}: {:.8f}".format('learning rate',
                                      scheduler.get_last_lr()[0])
        history({'epoch_cnt': epoch})
        print(msg)
        # } val log

        ## add histroy ##
        history({'val_kl_loss' : kl_meter.avg})
        history({'val_mse_loss': mse_meter.avg})

        if epoch % 10 == 0 and epoch!=0:
            if usedata =="egg":
                inputs_img = np.array(inputs_img).transpose(0, 2, 3, 1)
                outputs_img = np.array(outputs_img).transpose(0, 2, 3, 1)
                diff_imgs = np.array(diff_imgs).transpose(0, 2, 3, 1)
                
                # 入力画像(左)、出力画像(中)、差分画像(右)の順に表示
                plt.figure(figsize=(20, 20))
                for i in range(24):
                    img = np.hstack((inputs_img[i], outputs_img[i], diff_imgs[i]))
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

                    diff_name = f'diff{epoch}.png'
                    diff_path = os.path.join(log_path, diff_name)
                    plt.savefig(diff_path)
            elif usedata=="MNIST":
                x_concat = torch.cat([inputs.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
                save_image(x_concat, os.path.join(log_path, 'val_reconst-{}.png'.format(epoch+1)))

        history({'train_avg_loss': train_avg_loss})
        history({'valid_avg_loss': valid_avg_loss})
        
    print('Finished Training')
    keylist=['kl_loss','mse_loss', 'val_kl_loss','val_mse_loss']
    history.plot_loss(keylist=keylist, filename='loss_.png')
    avg_keylist = ['train_avg_loss','valid_avg_loss']
    history.plot_loss(keylist=avg_keylist, filename='loss_avg.png')
    history.plot_loss(keylist=['kl_loss'], filename='kl_loss.png')
    history.plot_loss(keylist=['mse_loss'], filename='mse_loss.png')
    history.save()

    model_name = 'vae.pth'
    model_path = os.path.join(log_path, model_name)
    torch.save(model.state_dict(), model_path)

    ################### test #####################
    original_imgs = []
    decoded_imgs = []
    diff_imgs = []

    scores = []

    #model = VAE(linear_bottleneck=linear_bottleneck, decode_type=decode_type, dataset=usedata)
    #model = BetaVAE_H(z_dim=linear_bottleneck)
    model = pytorch_VAE(image_channels=3, h_dim=h_dim, z_dim=z_dim).to(device)
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

            decoded, _ , _ , _ = model(original)
            diff = torch.abs(original - decoded)
            score = torch.mean(diff.view(diff.shape[0], -1), dim=1)
            
            original_imgs.extend(original.detach().cpu().numpy())
            decoded_imgs.extend(decoded.detach().cpu().numpy())
            diff_imgs.extend(diff.detach().cpu().numpy())
            scores.extend(score.detach().cpu().numpy())

            # Save the sampled images
            z = torch.randn(30, linear_bottleneck).to(device)
            out = model.egg_decode(z).view(-1, 3, image_size, image_size)
            save_image(out, os.path.join(log_path, 'sampled.png'))

    original_imgs = np.array(original_imgs).transpose(0, 2, 3, 1)
    decoded_imgs = np.array(decoded_imgs).transpose(0, 2, 3, 1)
    diff_imgs = np.array(diff_imgs).transpose(0, 2, 3, 1)
    scores = np.array(scores)

    # 入力画像(左)、出力画像(中)、差分画像(右)の順に表示
    if usedata=="egg":
        plt.figure(figsize=(20, 20))
        for i in range(24):
            img = np.hstack((original_imgs[i], decoded_imgs[i], diff_imgs[i]))
            ax = plt.subplot(8, 3, i+1)
            img = np.clip((img+1)/2, 0., 1.)
            plt.imshow(img)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.title(str(scores[i]))

        diff_name = 'diff.png'
        diff_path = os.path.join(log_path, diff_name)
        plt.savefig(diff_path)
        plt.show()
    elif usedata=="MNIST":
        x_concat = torch.cat([inputs.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(log_path, 'test_reconst.png'))


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

