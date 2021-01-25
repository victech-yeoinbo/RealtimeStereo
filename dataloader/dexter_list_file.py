import torch.utils.data as data

from PIL import Image
import os
import os.path
from pathlib import Path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    filepath = Path(filepath)
    train_all_list_path = filepath / 'dexter_train_all.txt'
    val_list_path = filepath / 'dexter_val.txt'

    with open(train_all_list_path) as f:
        train_names = [l.split() for l in f.readlines()]

    left_train = [str(filepath / names[0]) for names in train_names]
    right_train = [str(filepath / names[1]) for names in train_names]
    disp_train_L = [str(filepath / names[2]) for names in train_names]
    #disp_train_R = [str(filepath / names[3]) for names in train_names]

    with open(val_list_path) as f:
        val_names = [l.split() for l in f.readlines()]

    left_val = [str(filepath / names[0]) for names in val_names]
    right_val = [str(filepath / names[1]) for names in val_names]
    disp_val_L = [str(filepath / names[2]) for names in val_names]
    #disp_val_R = [str(filepath / names[3]) for names in val_names]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
