import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess
from . import listflowfile as lt
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path, 'ascii') ## need encoding ???


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader, calib=None):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.calib = calib

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, _ = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        # unnormalize dexter disp (dexter disp has normalized value which is actual disparity divided by width)
        # see https://github.com/victech-dev/unstereo/blob/a545c4cdcae045716511601b329f9df4742a5b37/dataloader/dataloader.py#L106
        w, h = left_img.size
        dataL *= w

        # if calib(fx*b) is not None, convert disparity to depth
        if self.calib:
            depth = np.zeros_like(dataL)
            mask = dataL > 0
            depth[mask] = calib / dataL[mask]
            dataL = depth

        if self.training:
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            left_img = left_img.crop((w - 640, h - 480, w, h))
            right_img = right_img.crop((w - 640, h - 480, w, h))
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
