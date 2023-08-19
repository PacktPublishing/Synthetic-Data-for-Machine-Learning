import os
import glob
import torch
import warnings
from PIL import Image
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class SynthDataset_nat(Dataset):
    """Undamaged synthetic cars"""

    def __init__(self, root_dir, transform=None):
        self.images =  glob.glob(root_dir+'/*.*')
        self.classId = 0#no accident cars
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        sample = {'image': image, 'class': torch.tensor(self.classId, dtype=torch.int8)}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

class SynthDataset_acc(Dataset):
    """Damaged synthetic cars"""

    def __init__(self, root_dir, transform=None):
        self.images = glob.glob(root_dir+'/*.*')
        self.classId = 1#accident cars
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        sample = {'image': image, 'class': torch.tensor(self.classId, dtype=torch.int8)}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

class RealDataset_nat(Dataset):
    """Undamaged real cars"""

    def __init__(self, root_dir, transform=None):
        self.images =  glob.glob(root_dir+'/*.*')
        self.classId = 0#no accident cars
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        sample = {'image': image, 'class': torch.tensor(self.classId, dtype=torch.int8)}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

class RealDataset_acc(Dataset):
    """Damaged real cars"""

    def __init__(self, root_dir, transform=None):
        self.images = glob.glob(root_dir+'/*.*')
        self.classId = 1#accident cars
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)

        sample = {'image': image, 'class': torch.tensor(self.classId, dtype=torch.int8)}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
