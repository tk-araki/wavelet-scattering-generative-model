import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

class ImageDataset(Dataset):

    def __init__(self, image_dir):
        self.image_files = list(image_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file)
        image = image_transforms(image)

        return image, image_file.stem


# celeba

# chairs


class ImageScatDataset(Dataset):
    def __init__(self,
                 image_dir,
                 scat_dir,
                 scat_suffix  # scat_suffix = '_scat_pcs.npy'
                 # num_channels=3
                 ):

        self.image_files = list(image_dir.glob('*.jpg'))
        self.scat_files = [scat_dir / (f.stem + scat_suffix) for f in self.image_files]  # f"{f.stem}"

        # データチェック
        if not all(map(lambda x: x.exists(), self.scat_files)):
            raise FileNotFoundError('The scattering coefficients data file corresponding to the image not found')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        image = image_transforms(image)

        scat = np.load(self.scat_files[idx])
        scat = torch.from_numpy(scat).float()

        return image, scat
