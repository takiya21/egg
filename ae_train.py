import datetime
import os
import copy
import sys
import csv
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from tqdm import tqdm

#from model.ae_conv import autoencoder
from model.ae_vgg_conv import AutoEncoder
import read_dataset
import image_folder as ImageFolder
import log as log

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

def main():
    parser = argparse.ArgumentParser(description='bento train')

    parser.add_argument('--batch_size', help='batch size',default=32)
    parser.add_argument('--in_w', help='in_w',default=256)
    parser.add_argument('--lr', help='lr',default=0.001)
    parser.add_argument('--weight_decay', help='weight decay',default=0.001)
    parser.add_argument('--optim', help='optim',default="SGD", type=str)
    parser.add_argument('--seed', help='seed',default= 1)

    args = parser.parse_args()

    print('~~~~~~~~~~ training start ~~~~~~~~~~~~~')
    # ~~~~~~~~~~~~~~~~ param ~~~~~~~~~~~~~~~~~~~
    batch_size = int(args.batch_size)#16
    in_w = int(args.in_w)#256
    in_h = in_w
    lr = float(args.lr)#0.001
    weight_decay = float(args.weight_decay)#0.001
    optim_flg = str(args.optim)
    seed = int(args.seed)

    num_epochs = 100

    data_dir = "/dataset/dataset/egg/cut_above"


    print('batch_size:',batch_size,
          ', in_w:', in_w, ', lr:', lr, 
          ', weight_decay:', weight_decay, 
          ' ,epoch:', num_epochs)

    # ~~~~~~~~~~~~~~~~ log folder ~~~~~~~~~~~~~~~~~~~~
    log_path = '/home/taki/egg/log'
    # フォルダ作成
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    param_folder = f'optim{optim_flg}_batch{batch_size}_w,h{in_w}_lr{lr}_wDecay{weight_decay}'
    path = os.path.join(log_path, param_folder)
    # フォルダ作成
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, f"seed{seed}")
    if not os.path.exists(path):
        os.mkdir(path)

    ##############################コメントアウト##################################
        
    # ~~~~~~~~~~~~ set data transforms ~~~~~~~~~~~~~~~
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomRotation(degrees=360),
        transforms.Resize((in_w, in_h)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5,), (0.5,)) #グレースケールのとき
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #これやるとrgb値変になる
    ])

    test_transform = transforms.Compose([
        transforms.Resize((in_w, in_h)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.5,), (0.5,))
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ##################  mydata #####################
    dataset = ImageFolder.ImageFolder(img_dir=data_dir, transform=train_transform)
    test_dataset  = ImageFolder.ImageFolder(img_dir=data_dir, transform=test_transform)

    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    test_dataset = ImageFolder.ImageFolder(data_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    """
    ################## MNSIT ##########################
    #datasetのtransform用
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # dataset(data数)
    dataset = MNIST('/home/data/MNIST', train = True, download=False, transform=img_transform)
    # dataloader(data数/bachsize)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    testset = MNIST('/home/data/MNIST', train = False, download=False, transform=img_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    """



#~~~~~~~~~~~~~~~~~~~  gpu setup~~~~~~~~~~~~~~~~~~~~~~~~
    # set seed
    random_seed = seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

    # gpuが使えるならgpuを使用、無理ならcpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  


    #~~~~~~~~~~~~~~~~~~~  net setup~~~~~~~~~~~~~~~~~~~~~~~~
    net = AutoEncoder().to(device)
    net.apply(weights_init)

    # optimizer の設定
    weight_decay = float(weight_decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # sheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # criterion
    criterion = nn.MSELoss()
    #l1_norm   = nn.L1Loss()


    # Observe that all parameters are being optimized
    if optim_flg == "SGD":
        optimizer = optim.SGD(  net.parameters(), 
                                lr=lr, momentum=0.9, 
                                weight_decay=weight_decay)
    elif optim_flg == "Adam" :
        optimizer = optim.AdamW(net.parameters(), 
                                lr=lr,
                                weight_decay=weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


    ############################  training ###################################
    # Training. {{{
    # =====
    history = log.History(keys=('train_loss',
                                'val_loss',
                                'epoch_cnt',
                                'train_L1_loss',
                                'val_L1_loss',
                                'test_L1_loss',),
                                 output_dir=path)

    """
    #best_model_wts = copy.deepcopy(net.state_dict())
    #best_acc = 0.0

    output_list = []
    input_list  = []
    score_list  = []
    index_list  = []
    """

    original_imgs = []
    decoded_imgs = []
    diff_imgs = []
    scores = []


    min_loss    = 10
    #min_L1_loss = 10
    best_model_wts = 0

    for epoch in range(num_epochs):# {{{epoch
        loop = tqdm(train_loader, unit='batch',desc='Epoch {:>3}'.format(epoch+1))

        # Train Step. {{{
        # =====

        # test,trainで使用されたりされないモードがあるので気を付ける
        net.train()
        for _, batch in enumerate(loop):
            # print(batch.shape)
            G_meter = log.AverageMeter()
            inputs= batch
           
            # gpuに送る
            inputs = inputs.to(device)

            # Update network. {{{
            # =====

            optimizer.zero_grad()  # 勾配を０
            
            # forward network
            outputs = net(inputs)

            # backward network{{{
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            # }}}backward network

            # Get losses. {{{}}
            G_meter.update(loss.item(), inputs[0].size()[0])
            history({'train_loss': loss.item()})
                # }}} get loss
            
            # }}}Update network

        # Print training log. {
        # =====
        msg = "[Train {}] Epoch {}/{}".format(
            'AutoEncoder', epoch + 1, num_epochs)
        msg += " - {}: {:.4f}".format('train_loss', G_meter.avg)
        msg += " - {}: {:.4f}".format('learning rate',
                                      scheduler.get_last_lr()[0])
        history({'epoch_cnt': epoch})

        print(msg)

        # }}}Train Step.

        # Validation Step. {
        # =====

        with torch.no_grad():  # 勾配
            net.eval()
            loop_val = tqdm(val_loader, unit='batch',desc='Epoch {:>3}'.format(epoch + 1))
            epoch_loss    = 0
            epoch_L1_loss = 0
            iter_cnt      = 0

            for _, batch in enumerate(loop_val):
                iter_cnt = iter_cnt + 1
                G_meter = log.AverageMeter()
                inputs = batch
                inputs = inputs.to(device)
                outputs = net(inputs)

                diff = torch.abs(inputs - outputs)
                score = torch.mean(diff.view(diff.shape[0], -1), dim=1)

                loss = criterion(outputs, inputs)

                epoch_loss    = epoch_loss + loss

                G_meter.update(loss.item(), inputs[0].size()[0])
                history({'val_loss': loss.item()})

                inputs_img  = inputs.to('cpu').detach().numpy().copy()
                output = outputs.to('cpu').detach().numpy().copy()
                diff_imgs.extend(diff.detach().cpu().numpy())
                #scores.extend(score.detach().cpu().numpy())


            #}}
        epoch_loss_mean    = epoch_loss / iter_cnt

        # deep copy the model{{{       
        if epoch_loss_mean < min_loss:
            min_loss = epoch_loss_mean
            best_model_wts = copy.deepcopy(net.state_dict())

        #}}}

        # } val step

        # Print validation log. {
        # =====
        msg = "[Validation {}] Epoch {}/{}".format(
            'CNN', epoch + 1, num_epochs)
        msg += " - {}: {:.4f}".format('val_loss', G_meter.avg)

        print(msg)
        # } val log

        # sheduler step
        scheduler.step()
        if epoch % 10 == 0:
            inputs_img = np.array(inputs_img).transpose(0, 2, 3, 1)
            outputs_img = np.array(output).transpose(0, 2, 3, 1)
            diff_img = np.array(diff_imgs).transpose(0, 2, 3, 1)
            scores = np.array(scores)

            # 入力画像(左)、出力画像(中)、差分画像(右)の順に表示
            plt.figure(figsize=(20, 20))
            for i in range(24):
                img = np.hstack((inputs_img[i], outputs_img[i], diff_img[i]))
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

                #save_image(inputs, log_path + '/inputimage_{}.png'.format(epoch))
                #save_image(outputs, log_path + '/outputimage_{}.png'.format(epoch))
    # }}}} epoch
    
    # 重み保存
    torch.save(best_model_wts, path+"/model_dict.pth")

    original_imgs = []
    decoded_imgs = []
    diff_imgs = []

    scores = []

    # ~~~~~~~~~~~~~~ testdataに対する推論 ~~~~~~~~~~~~~~~~~~~~~
    
    print("~~~~~~~~~~~~~~ eval test data ~~~~~~~~~~~~~~~~~~")
    with torch.no_grad():  # 勾配の消失
        for batch in test_loader:
            original = batch
            original = original.to(device, non_blocking=True)
            decoded  = net(original)
            diff = torch.abs(original - decoded)
            score = torch.mean(diff.view(diff.shape[0], -1), dim=1)
            
            original_imgs.extend(original.detach().cpu().numpy())
            decoded_imgs.extend(decoded.detach().cpu().numpy())
            diff_imgs.extend(diff.detach().cpu().numpy())
            scores.extend(score.detach().cpu().numpy())
    test_path = path + "/test_result"
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    original_imgs = np.array(original_imgs).transpose(0, 2, 3, 1)
    decoded_imgs = np.array(decoded_imgs).transpose(0, 2, 3, 1)
    diff_img = np.array(diff_imgs).transpose(0, 2, 3, 1)
    scores = np.array(scores)

    # 入力画像(左)、出力画像(中)、差分画像(右)の順に表示
    plt.figure(figsize=(20, 20))
    for i in range(24):
        img = np.hstack((original_imgs[i], decoded_imgs[i], diff_img[i]))
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
    diff_path = os.path.join(test_path, diff_name)
    plt.savefig(diff_path)
    plt.show()

    # ~~~~~~ plot graph ~~~~~~~~
    #print("# ~~~~~~ plotting graph ~~~~~~~~")
    plt.rcParams["figure.figsize"] = (6.4, 4.8)
    history.plot_loss("MSE") # 引数：MSE or MAE
    history.save()
    print("~~~~~~~~~~ completed ~~~~~~~~~~~~")

"""
    # ~~~~~~ save log ~~~~~~~~~ 
    #dt_now = datetime.datetime.now()
    savepath = os.path.join(log_path, 'testlog.csv')
    

    if not os.path.exists(savepath):
        with open(savepath, 'w', encoding='utf_8_sig') as f: # 'w' 上書き
            writer = csv.writer(f)
            writer.writerow(["seed",
                         "optim",
                         "batch_size",
                         "in_w",
                         "lr",
                         "weight_decay",
                         "min_L1_loss",
                         "corr_list",
                         "min_MSE_loss"])
        

    with open(savepath, 'a', encoding='utf_8_sig') as f: # 'a' 追記
        writer = csv.writer(f)
        writer.writerow([seed,
                         optim_flg,
                         batch_size,
                         in_w,
                         lr,
                         weight_decay,
                         min_L1_loss.to('cpu').detach().numpy().copy(),
                         0,
                         min_loss.to('cpu').detach().numpy().copy()])
"""

if __name__ == '__main__':
    main()

