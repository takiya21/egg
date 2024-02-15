import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from PIL import Image
from pathlib import Path

from model.pytorch_vae import pytorch_VAE
import image_folder as ImageFolder
import log as log



def main():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description='vae train')
    parser.add_argument('--b', help='beta',default=1)
    parser.add_argument('--z_dim', help='z_dim',default=30)
    parser.add_argument('--lr', help='lr',default=1e-3)
    parser.add_argument('--batch_size', help='batch size',default=64)
    args = parser.parse_args()


    # Hyper-parameters
    num_epochs = 1000
    image_size = 256
    h_dim = 8*8*128
    z_dim = int(args.z_dim)
    batch_size = int(args.batch_size)
    learning_rate = float(args.lr)
    b = int(args.b)


    # Create a directory if not exists
    log_path = f'/home/taki/egg/log/vae/for_egg/1121pytorchVAE_lr{learning_rate}_inW{image_size}_b{b}_mse=sum_epoch{num_epochs}_h_dim{h_dim}_z_dim{z_dim}_batch{batch_size}_sheduler=cosin_dataaug=flip'
    print(log_path)
    train_path = os.path.join(log_path, "train_log")
    val_path = os.path.join(log_path, "val_log")
    test_path = os.path.join(log_path, "test_log")
    make_dirs(log_path)
    make_dirs(train_path)
    make_dirs(val_path)
    make_dirs(test_path)


    ##############  random seed setting  ############
    manualSeed = 0
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    # MNIST dataset
    # dataset = torchvision.datasets.MNIST(root='../../data',
    #                                      train=True,
    #                                      transform=transforms.ToTensor(),
    #                                      download=True)

    # Data loader
    # data_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                           batch_size=batch_size, 
    #                                           shuffle=True)


    #########   set transform  #########
    train_transform = torchvision.transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                #transforms.RandomRotation(degrees=360),
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = torchvision.transforms.Compose([
                                #transforms.RandomHorizontalFlip(p=0.5),
                                #transforms.RandomVerticalFlip(p=0.5),
                                #transforms.RandomRotation(degrees=360),
                                transforms.ToTensor(),
                                transforms.Resize((image_size, image_size)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    ######  set dataset ##########
    dataset = ImageFolder.ImageFolder(img_dir="/dataset/dataset/egg/nomal_egg", transform=train_transform)

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)                                 
    test_dataset = ImageFolder.ImageFolder("/dataset/dataset/egg/nomal_egg", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        

    ######### set model  ###########
    model = pytorch_VAE(image_channels=3, h_dim=h_dim, z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-10)


    #######   log 管理用クラス #######
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


    # Start training
    for epoch in range(num_epochs):

        running_loss = 0
        kl_running_loss = 0
        mse_running_loss = 0

        val_running_loss = 0
        val_kl_running_loss = 0
        val_mse_running_loss = 0

        for i, x in enumerate(train_data_loader):


            # Forward pass
            x = x.to(device)# .view(-1, image_size) # 全結合
            x_reconst, mu, log_var, _ = model(x)
            
            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43

            reconst_loss = F.mse_loss(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Backprop and optimize
            loss = reconst_loss + b * kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            mse_running_loss += reconst_loss.item()
            kl_running_loss += kl_div.item()

        
            #####   iteration loss  ######
            # if (i+1) % 10 == 0:
                #print(f"Epoch[{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Reconst Loss: {reconst_loss.item()}, KL Div: {b * kl_div.item()},lr: {scheduler.get_last_lr()[0]}" )

        print(f"[Train loss ] Epoch[{epoch+1}/{num_epochs}], Reconst Loss: {mse_running_loss/(i+1):.6f}, KL Div: {kl_running_loss/(i+1):.6f},lr: {scheduler.get_last_lr()[0]:.10f}" )
        
        # Get losses.
        #print(f'train kl_loss={kl_running_loss}, train_i={i+1}, / = {kl_running_loss/(i+1)} ')
        history({'train_loss':running_loss/(i+1)})
        history({'kl_loss':kl_running_loss/(i+1)})
        history({'mse_loss':mse_running_loss/(i+1)})

        scheduler.step()  
        

        if epoch % 20 == 0 and epoch!=0:
            with torch.no_grad():
                # Save the reconstructed images
                out, _, _, _ = model(x)
                x_concat = torch.cat([x.view(-1, 3, 256, 256), out.view(-1, 3, 256, 256)], dim=3)
                save_image(x_concat, os.path.join(train_path, 'train_reconst-{}.png'.format(epoch+1)))

        # # Validation Step. {
        # =====
        with torch.no_grad():
            model.eval()
            for i, x in enumerate(val_data_loader):

                # Forward pass
                x = x.to(device)# .view(-1, image_size) # 全結合
                x_reconst, mu, log_var, _ = model(x)
                

                reconst_loss = F.mse_loss(x_reconst, x, size_average=False)
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Backprop and optimize
                loss = reconst_loss + b * kl_div

                val_running_loss += loss.item()
                val_mse_running_loss += reconst_loss.item()
                val_kl_running_loss += kl_div.item()

            if epoch % 20 == 0 and epoch!=0:
                # Save the reconstructed images
                x_concat = torch.cat([x.view(-1, 3, 256, 256), x_reconst.view(-1, 3, 256, 256)], dim=3)
                save_image(x_concat, os.path.join(val_path, 'val_reconst-{}.png'.format(epoch+1)))

                # Save the sampled images
                z = torch.randn(30, z_dim).to(device)
                out = model.decode(z).view(-1, 3, 256, 256)
                save_image(out, os.path.join(val_path, 'val_sampled-{}.png'.format(epoch+1)))
        
        print(f"[Val loss   ] Epoch[{epoch+1}/{num_epochs}], Reconst Loss: {val_mse_running_loss/(i+1):.6f}, KL Div: {val_kl_running_loss/(i+1):.6f},lr: {scheduler.get_last_lr()[0]:.10f}" )
        
        # Get losses.
        #print(f'val_kl_loss={val_kl_running_loss}, val_i={i+1}, / = {val_kl_running_loss/(i+1)} ')
        history({'val_loss':val_running_loss/(i+1)})
        history({'val_kl_loss':val_kl_running_loss/(i+1)})
        history({'val_mse_loss':val_mse_running_loss/(i+1)})

    ######  plot loss  #######
    keylist=['kl_loss','mse_loss']
    history.plot_loss(keylist=keylist, filename='loss_.png')
    history.plot_loss(keylist=['kl_loss','val_kl_loss'], filename='kl_loss.png')
    history.plot_loss(keylist=['mse_loss', 'val_mse_loss'], filename='mse_loss.png')
    history.save()




    ########   test  #############
    model_name = 'vae.pth'
    model_path = os.path.join(log_path, model_name)
    torch.save(model.state_dict(), model_path)

    model = pytorch_VAE(image_channels=3, h_dim=h_dim, z_dim=z_dim)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)


    model.eval()
    with torch.no_grad():
        for i, x in enumerate(test_loader):

            # Save the reconstructed images
            x = x.to(device)
            out, _, _, _ = model(x)

        x_concat = torch.cat([x.view(-1, 3, 256, 256), out.view(-1, 3, 256, 256)], dim=3)
        save_image(x_concat, os.path.join(test_path, 'test_reconst.png'))

        # Save the sampled images
        z = torch.randn(30, z_dim).to(device)
        out = model.decode(z).view(-1, 3, 256, 256)
        save_image(out, os.path.join(test_path, 'test_sampled.png'))





def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    main()
