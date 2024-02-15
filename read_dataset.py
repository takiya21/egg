import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

#PytorchのDatasetクラスを継承
class read_dataset(Dataset):
    def __init__(self, df,transform=None):
        
        image_path = df['path']
        score = {}
        for i in range(1,8):
            tmp = f"factor{i}"
            score[tmp] = df[tmp]

        self.df = df
        self.image_paths = image_path
        self.score = score
        self.transform = transform


    def __getitem__(self, index):#引数のindexの画像の情報を返す
        path = "/dataset/dataset/taki/data/bento/bento_dataset1000/" +  self.image_paths[index] 
        #画像読み込み。
        img = Image.open(path)

        #transform事前処理実施
        if self.transform is not None:
            img = self.transform(img)
        
        img = torch.Tensor(img)
        score = self.df.loc[index]
        score = score.drop("path")
        score = torch.Tensor(score)

        return img, score, index


    def __len__(self):
        #データ数を返す
        return len(self.image_paths)

if __name__ == '__main__':
    #transformで32x32画素に変換して、テンソル化。
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    #データセット作成
    dataset = read_dataset("./score_ml_promax_7.csv",transform)
    #dataloader化
    dataloader = DataLoader(dataset, batch_size=32)

    #データローダの中身確認
    for img,label ,image_path in dataloader:
        print('label=',label)
        print('image_path=',image_path)
        print('img.shape=',img.shape)