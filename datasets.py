import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import pprint
import math
import collections

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms
from PIL import Image

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


class ImageDataset(Dataset):

    def __init__(self,
                 image_dir
                 # transform
                 # extension
                 ):
        # self.image_files = [p for p in image_dir.iterdir() if p.suffix == '.jpg'] # f->path
        # self.image_files = [f for f in image_dir.glob('*.jpg')] #
        self.image_files = list(image_dir.glob('*.jpg'))
        # ↓消す^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # image = pil_loader(self.image_files[idx]) # モノクロ画像に使える？ convert to RGB?
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        image = self.transforms(image)

        return image, image_file.stem


# celeba

# chairs


class ImageScatDataset(Dataset):
    def __init__(self,
                 image_dir,
                 # image_dir = Path('/Users/araki/OutSide/OutSide_Python/Datasets/CelebA/64_rgb/image/train')
                 scat_dir,  # images
                 scat_suffix,  # scat_suffix = '_scat_pcs.npy'
                 # num_channels=3
                 check_scat_exist=True  # 【コードレビュー】
                 ):
        # image_dir
        self.image_files = list(image_dir.glob('*.jpg'))
        # self.image_stems = [f.stem for f in image_dir.glob('*.jpg')]
        # /Users/araki/OutSide/OutSide_Python/Datasets/CelebA/64_rgb/Scat_J4/pca/train_pcs
        #         self.scat_files = list(scat_dir.glob('*.npy'))
        # print(self.image_files[2].stem)
        self.scat_files = [scat_dir / (f.stem + scat_suffix) for f in self.image_files]  # f"{f.stem}"

        # データがきちんと想定通り入ってるかチェック 【コードレビュー】
        if not all(map(lambda x: x.exists(), self.scat_files)):
            raise FileNotFoundError('The scattering coefficients data file corresponding to the image not found')
            # return False

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # image_file = self.image_files[idx]
        image = Image.open(self.image_files[idx])
        image = image_transforms(image)

        # scat_file = self.scat_files[idx]
        scat = np.load(self.scat_files[idx])
        scat = torch.from_numpy(scat).float()
        # torch.from_numpy(scat).clone()　＃メモリを共有させない

        return image, scat
