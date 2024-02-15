import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.utils.data
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision.datasets import MNIST
from tqdm import tqdm
import log as log
import image_folder as ImageFolder

def main():

    # Create a directory if not exists
    log_path = f'/home/taki/egg/log/vae/mnist/pytorch-tutorial1'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Hyper-parameters
    image_size = 784
    h_dim = 400
    z_dim = 20
    num_epochs = 20
    batch_size = 128
    learning_rate = 1e-3


    img_transform = torchvision.transforms.Compose([transforms.ToTensor()])

    # MNIST dataset
    dataset = torchvision.datasets.MNIST(root='./data',train=True,transform=img_transform,download=False)
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    test_dataset = MNIST('./data', train = False, download=False, transform=img_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)


    # egg dataset
    # train_transform = torchvision.transforms.Compose([
    #                                 transforms.RandomHorizontalFlip(p=0.5),
    #                                 transforms.RandomVerticalFlip(p=0.5),
    #                                 transforms.RandomRotation(degrees=360),
    #                                 transforms.ToTensor(),
    #                                 transforms.Resize((image_size, image_size)),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])

    # test_transform = torchvision.transforms.Compose([
    #                             transforms.ToTensor(),
    #                             transforms.Resize((image_size, image_size)),
    #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
        
    # dataset = ImageFolder.ImageFolder(img_dir= data_dir, transform=train_transform)
    # train_size = int(len(dataset) * 0.8)
    # validation_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    # test_dataset = ImageFolder.ImageFolder(data_dir, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)



    # VAE model
    class VAE(nn.Module):
        def __init__(self, image_size=32*32*3,h1_dim=1024, h_dim=512, z_dim=32): #mnist:784(28*28*1), egg:32*32*3
            super(VAE, self).__init__()
            self.fc1 = nn.Linear(image_size, h1_dim)
            self.fc1_1 = nn.Linear(h1_dim, h_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(h_dim, z_dim)
            self.fc4 = nn.Linear(z_dim, h_dim)
            self.fc4_1 = nn.Linear(h_dim, h1_dim)
            self.fc5 = nn.Linear(h1_dim, image_size)
            
        def encode(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc1_1(h))
            return self.fc2(h), self.fc3(h)
        
        def reparameterize(self, mu, log_var):
            std = torch.exp(log_var/2)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc4_1(h))
            return F.sigmoid(self.fc5(h))
        
        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconst = self.decode(z)
            return x_reconst, mu, log_var


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)

    history = log.History(keys=('train_loss',
                                'val_loss',
                                'epoch_cnt',
                                'train_avg_loss',
                                'valid_avg_loss'),
                                output_dir=log_path)

    # Start training
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, unit='batch', desc='| Train | Epoch {:>3} |'.format(epoch+1))
        
        batch_loss = []  
        #model.train()
        for i, (x, _) in enumerate(loop):
            G_meter = log.AverageMeter()

            # Forward pass
            x = x.to(device).view(-1, image_size)
            x_reconst, mu, log_var = model(x)

            #reconst_loss = criterion(x_reconst, x)
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)

            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get losses. {{{}}
            G_meter.update(loss.item(), x[0].size()[0])
            history({'train_loss': loss.item()})
            # }}} get loss
            
            batch_loss.append(loss.item())


        msg = "[Train {}] Epoch {}/{}".format(
            'MNIST', epoch + 1, num_epochs)
        msg += " - {}: {:.4f}".format('train_loss', G_meter.avg)
        msg += " train loss:{}, Reconst Loss: {:.4f}, KL Div: {:.4f}".format(
                G_meter.avg, reconst_loss.item(), kl_div.item())

        print(msg)

        if epoch % 10 == 0:

            # Save the reconstructed images
            x_concat = torch.cat([x.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(log_path, 'train-{}.png'.format(epoch)))
        
        # print ("Epoch[{}/{}], Step [{}/{}],train loss:{}, Reconst Loss: {:.4f}, KL Div: {:.4f}" 
        #         .format(epoch+1, num_epochs, i+1, len(train_loader), G_meter.avg, reconst_loss.item(), kl_div.item()))

        # }}} train step

    # # Validation Step. {
        # =====
        with torch.no_grad():
            #odel.eval()
            loop_val = tqdm(val_loader, unit='batch', desc='| Test  | Epoch {:>3} |'.format(epoch + 1))

            for i, batch in enumerate(loop_val):
                inputs, _ = batch
                inputs = inputs.to(device).view(-1, image_size)

                x_reconst, mu, log_var = model(inputs)
                

                #mse_loss = criterion(x_reconst, inputs)
                mse_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
                
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                loss = mse_loss + kl_div
                batch_loss.append(loss)

                G_meter.update(loss.item(), inputs[0].size()[0])
                history({'val_loss': loss.item()})

                if epoch % 10 == 0:
                    # Save the reconstructed images
                    x_concat = torch.cat([inputs.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
                    save_image(x_concat, os.path.join(log_path, 'reconst-{}.png'.format(epoch)))


        # =====
        # Print validation log. {
        msg = "[Validation {}] Epoch {}/{}".format(
            'VAE', epoch + 1, num_epochs)
        msg += " - {}: {:.4f}".format('val_loss', G_meter.avg)
        print(msg)



    history.plot_loss()
    history.save()

    model_name = 'vae.pth'
    model_path = os.path.join(log_path, model_name)
    torch.save(model.state_dict(), model_path)

    ################### test #####################
    model = VAE()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    #model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs,_ = batch
            inputs = inputs.to(device).view(-1, image_size)

            x_reconst, mu, log_var = model(inputs)
            x_concat = torch.cat([inputs.view(-1, 1, 28, 28), x_reconst.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join(log_path, 'test-{}.png'.format(epoch)))



if __name__ == '__main__':
    main()

