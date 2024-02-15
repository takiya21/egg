from IPython.utils import io
import torch
import PIL
import pickle
import os
import numpy as np
import pandas as pd
import random
import ipywidgets as widgets
import matplotlib.pyplot as plt
from PIL import Image
from ipywidgets import fixed

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

import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器


linear_bottleneck = 256
b = 0.001
manualSeed = 0
decode_type = 'upsamp'

data_dir = "/dataset/dataset/egg/nomal_egg"
#data_dir = '/home/taki/egg/data/egg/nomal_egg'
print(f"linear_bottleneck:{linear_bottleneck}")
log_path = f'/home/taki/egg/log/vae/{decode_type}_linear{linear_bottleneck}_b{b}_nomalegg_dim128_8_8'


# 再現性のためにrandom seedを設定する
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Transform を作成する
image_size = 256
transform =     torchvision.transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Dataset を作成する。
test_dataset = ImageFolder.ImageFolder(data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)


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
        original = batch
        original = original.to(device, non_blocking=True)
        decoded, _ , _, z = model(original)
        diff = torch.abs(original - decoded)
        score = torch.mean(diff.view(diff.shape[0], -1), dim=1)
        
        original_imgs.extend(original.detach().cpu().numpy())
        decoded_imgs.extend(decoded.detach().cpu().numpy())
        diff_imgs.extend(diff.detach().cpu().numpy())
        scores.extend(score.detach().cpu().numpy())
        latent_dirs.extend(z.detach().cpu().numpy())

original_imgs = np.array(original_imgs).transpose(0, 2, 3, 1)
decoded_imgs = np.array(decoded_imgs).transpose(0, 2, 3, 1)
diff_imgs = np.array(diff_imgs).transpose(0, 2, 3, 1)
scores = np.array(scores)


decoded = model.decode(z)
print(decoded)