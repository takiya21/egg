# https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import sys

import model.ae_conv
from model.ae_conv import autoencoder


# tanhによって出力の値が-1~1になっているので、それを0~1に変換することで画像表示できるようにしている
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

def main(arg1, arg2):
    # フォルダ作成
    if not os.path.exists('./log'):
        os.mkdir('./log')
    
    

    # パラメータ設定
    num_epochs = 1
    #batch_size = 32
    learning_rate = 0.001
    
    batch_size = int(arg1)
    #learning_rate = float(arg2)

    # フォルダ作成
    if not os.path.exists('./log/mlp_img/conv_{}_{}'.format(batch_size, learning_rate)):
        os.mkdir('./log/mlp_img/conv_{}_{}'.format(batch_size, learning_rate))

# フォルダ作成
    if not os.path.exists('./log/loss_img/conv_test_loss'):
        os.mkdir('./log/loss_img/conv_test_loss')

    # datasetのtransform用
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # dataset(data数)
    dataset = MNIST('./data', train = True, download=True, transform=img_transform)
    # dataloader(data数/bachsize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    testset = MNIST('./data', train = False, download=True, transform=img_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model
    model = autoencoder().to(device)

    # Build criterion
    criterion = nn.MSELoss() #MSE -> 平均二乗誤差

    # optimizer の設定
    weight_decay = float(arg2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # sheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # lossを保存するリスト
    train_loss = []

    # === training === {{{
    print("conv{}_{}".format(batch_size, learning_rate))
    for epoch in range(num_epochs):
        running_loss = 0.0
        for count, data in enumerate(dataloader, 1):
            img, _ = data
            
            img = img.to(device)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ===================loss++====================
            running_loss += loss.item()

        
        # ===================loss====================
        avg_loss = running_loss / count # 1epochの平均誤差
        train_loss.append(avg_loss)
        # ===================log========================
        print('epoch [{}/{}], train_loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data))
        scheduler.step()
        
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './log/mlp_img/image_{}.png'.format(epoch))

    
    #}}}
        

    # === test === {{{
    feat_vec = []   # 特徴ベクトルを保存するリスト
    num_label = []  # lavelを保存するリスト

    model.eval()    # modelを推論モードに切り替える
    with torch.no_grad():   # 重み更新をしない
        test_running_loss = 0.0
        cnt = 0
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            z = model.encode(inputs).detach().cpu() # 特徴ベクトルを取り出す
            feat_vec.append(z)

            num_label.append(labels.cpu())  # labelを取り出す

            loss = criterion(outputs, inputs)
            test_running_loss += loss.item()

            # テストデータに対する入力画像を保存
            if cnt % 10 == 0:
                in_pic = to_img(inputs.cpu().data)
                save_image(in_pic, './log/mlp_img/conv_{}_{}/input{}_{}_{}_image_+30sheduler-wdecay{}.png'.format(batch_size, learning_rate,cnt, batch_size, learning_rate,weight_decay))

                # テストデータに対する再構成画像を保存
                test_pic = to_img(outputs.cpu().data)
                save_image(test_pic, './log/mlp_img/conv_{}_{}/test{}_{}_{}_image_+30sheduler-wdecay{}.png'.format(batch_size, learning_rate, cnt, batch_size, learning_rate,weight_decay))

            cnt += 1

    test_avg_loss = test_running_loss / len(testloader)
    print("test_loss:", test_avg_loss)

    text = "{}".format(test_avg_loss)
    file = open('./log/loss_img/conv_test_loss/conv_test_loss_{}_{}_+30sheduler-wdecay{}.txt'.format(batch_size, learning_rate, weight_decay), 'w')
    file.write(text)
    file.close()
    #}}}

    
    # === plot losses === {{{
    plt.figure()
    plt.plot(train_loss, label="train_loss")
    #plt.plot(test_loss, color="blue", label="test_loss")
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("./log/loss_img/conv_loss{}_{}_30sheduler-wdecay{}.png".format(batch_size, learning_rate, weight_decay))
    # }}}

    # === 潜在空間をプロット === {
    vec = torch.cat([_vec for _vec in feat_vec]).numpy() # test時に得られたfear_vecを結合しnumpyに変換
    lab = torch.cat([_lab for _lab in num_label]).numpy()

    """
    # each_vec[0]には0の特徴ベクトルが格納
    vec_0 = []
    vec_1 = []
    vec_2 = []
    vec_3 = []
    vec_4 = []
    vec_5 = []
    vec_6 = []
    vec_7 = []
    vec_8 = []
    vec_9 = []

    vec_0 = vec[0 == lab]
    vec_1 = vec[1 == lab]
    vec_2 = vec[2 == lab]
    vec_3 = vec[3 == lab]
    vec_4 = vec[4 == lab]
    vec_5 = vec[5 == lab]
    vec_6 = vec[6 == lab]
    vec_7 = vec[7 == lab]
    vec_8 = vec[8 == lab]
    vec_9 = vec[9 == lab]


    av_vec = []
    av_vec.append(np.sum(vec_0, axis=0))
    av_vec.append(np.sum(vec_1, axis=0))
    av_vec.append(np.sum(vec_2, axis=0))
    av_vec.append(np.sum(vec_3, axis=0))
    av_vec.append(np.sum(vec_4, axis=0))
    av_vec.append(np.sum(vec_5, axis=0))
    av_vec.append(np.sum(vec_6, axis=0))
    av_vec.append(np.sum(vec_7, axis=0))
    av_vec.append(np.sum(vec_8, axis=0))
    av_vec.append(np.sum(vec_9, axis=0))
    
    
    av_vec = np.array(av_vec)

    one_ofvec0to1 = np.array(av_vec)

    vec0to1 = np.array(av_vec[0])

    diff = np.empty(10)
    for i in range(10):
        diff[i] = vec0to1[i] - av_vec[1][i]
    
    diff = diff / 10
  


    # 1の特徴
    av_vec_1 = av_vec[1]

    for i in range(10):
        one_ofvec0to1[i] = av_vec[0]

    av_vec = av_vec.astype(np.float32)
    av_vec = torch.from_numpy(av_vec).clone()
    av_vec = av_vec.to(device)

    one_ofvec0to1[0][0] = av_vec_1[0]
    # ベクトル一つずつを1の特徴ベクトルに変えていく
    for i in range(1,9):
        one_ofvec0to1[i] = one_ofvec0to1[i-1]
        one_ofvec0to1[i][i] = av_vec_1[i]
        
    
    vec0to1 = vec0to1.astype(np.float32)
    vec0to1 = torch.from_numpy(vec0to1).clone()
    vec0to1 = vec0to1.to(device)

    

    one_ofvec0to1 = one_ofvec0to1.astype(np.float32)
    one_ofvec0to1 = torch.from_numpy(one_ofvec0to1).clone()
    one_ofvec0to1 = one_ofvec0to1.to(device)

    diff = diff.astype(np.float32)
    diff = torch.from_numpy(diff).clone()
    diff = diff.to(device)


    model.eval()    # modelを推論モードに切り替える
    with torch.no_grad():   # 重み更新をしない
        for i in range(10):
            for j in range(10):
                if diff[j] > 0:
                    vec0to1[j] = vec0to1[j] - diff[j]
                else:
                    vec0to1[j] = vec0to1[j] + diff[j]
            

            out = model.decode(vec0to1).detach().cpu() # 特徴ベクトルを取り出す
            out = to_img(out)
            save_image(out, './log/feat_vec_mnist/get_closer/decode{}_image.png'.format(i))

            _out = model.decode(one_ofvec0to1[i]).detach().cpu() # 特徴ベクトルを取り出す
            _out = to_img(_out)
            save_image(_out, './log/feat_vec_mnist/onebyone/decode{}_image.png'.format(i))
            

            
            _outs = model.decode(av_vec[i]).detach().cpu() # 特徴ベクトルを取り出す
            _outs = to_img(_outs)
            save_image(_outs, './log/feat_vec_mnist/average/decode{}_image.png'.format(i))

        print(vec0to1)
        print(av_vec_1) 
            
    
    
    # t-SNEで2次元に変換
    vec_2d = TSNE(n_components=2).fit_transform(vec)

    plt.figure()
    plt.title("feat vec")
    plt.xlabel("x")
    plt.ylabel("y")

    uniq_lavel = np.unique(lab) # 0~9のラベルが入っているリスト
    # 全test dataに対してグラフ作成
    plt.figure()
    plt.title('feat vec')
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(uniq_lavel.size):
        _vec = vec_2d[uniq_lavel[i] == lab]
        plt.scatter(_vec[:1024, 0], _vec[:1024, 1], label=uniq_lavel[i])
    plt.legend()
    plt.savefig('./log/feat_vec/all_feat_vec.png')


    # ラベルごとにグラフ作成
    for i in range(uniq_lavel.size):
        plt.figure()
        plt.title('{}_feat vec'.format(i))
        plt.xlabel("x")
        plt.ylabel("y")

        _vec = vec_2d[uniq_lavel[i] == lab]
        plt.scatter(_vec[:1024, 0], _vec[:1024, 1], label=uniq_lavel[i])
        plt.legend()
        plt.savefig('./log/feat_vec/{}_feat_vec.png'.format(i))

    """
    uniq_lavel = np.unique(lab) # 0~9のラベルが入っているリスト
    # 全test dataに対してグラフ作成
    plt.figure()
    plt.title('feat vec')
    plt.xlabel("x")
    plt.ylabel("y")
    for i in range(uniq_lavel.size):
        _vec = vec[uniq_lavel[i] == lab]
        plt.scatter(_vec[:1024, 0], _vec[:1024, 1], label=uniq_lavel[i])
    plt.legend()
    plt.savefig('./log/feat_vec/all_feat_vec.png')
   
if __name__ == '__main__':

    args = sys.argv
    main(args[1], args[2])


    