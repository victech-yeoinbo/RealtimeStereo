import os
import time
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision as tv
import torchvision.transforms.functional as F

from models import *
from dataloader import preprocess
import dataloader.readpfm as rp
from utils.pcl import query_intrinsic

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')

parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='RTStereoNet',
                    help='select model')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

from dataloader import dexter_list_file as lt
from dataloader import dexter_loader as DA

# python3 test.py --loadmodel result/checkpoint_249.tar

def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = np.abs(disp - gt)
    err3 = np.bitwise_and((errmap[mask] > 3.), (errmap[mask] / gt[mask] > 0.05)).sum()
    return np.float64(err3) / np.float64(mask.sum())

def predict(model, imgnameL, imgnameR, cat='withrobot'):
    imgL = Image.open(imgnameL).convert('RGB')
    imgR = Image.open(imgnameR).convert('RGB')
    W, H = imgL.size
    K, baseline = query_intrinsic(cat)

    ## img preprocessing
    processed = preprocess.get_transform(augment=False)
    inpL = processed(imgL).float().cuda().unsqueeze(0)
    inpR = processed(imgR).float().cuda().unsqueeze(0)

    if inpL.shape[2] % 16 != 0:
        times = inpL.shape[2]//16
        top_pad = (times+1)*16 -inpL.shape[2]
    else:
        top_pad = 0

    if inpL.shape[3] % 16 != 0:
        times = inpL.shape[3]//16
        right_pad = (times+1)*16-inpL.shape[3]
    else:
        right_pad = 0

    inpL = F.pad(inpL, (0, right_pad, top_pad, 0))
    inpR = F.pad(inpR, (0, right_pad, top_pad, 0))

    ## disp preprocessing
    imgnameL_path = Path(imgnameL)
    dispL_gt_path = imgnameL_path.parent.parent / 'dispL_occ_pfm' / (imgnameL_path.stem + '.pfm')
    print(f'dispL_gt_path={str(dispL_gt_path)}')
    dispL_gt, _ = rp.readPFM(str(dispL_gt_path))
    dispL_gt = np.ascontiguousarray(dispL_gt, dtype=np.float32)
    dispL_gt *= W # for dexter normalizing
    depth_gt = K[0,0] * baseline / dispL_gt

    model.eval()

    t0 = time.time()
    with torch.no_grad():
        disp = model(inpL, inpR)
        disp = torch.squeeze(disp, 1)

    if top_pad != 0:
        disp = disp[:,top_pad:,:]

    t1 = time.time()
    print("*", Path(imgnameL).stem, "elspaed:", t1 - t0)

    disp = disp[0].cpu().numpy()
    #oH, oW = disp.shape
    #disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_LINEAR) * (W / oW)
    #disp = np.clip(disp, 0, 192)
    depth = K[0,0] * baseline / disp

    err = error_estimating(disp, dispL_gt)
    print(f'* err={err}')

    return np.array(imgL), depth, K

if __name__ == '__main__':
    from utils.nav3d import NavScene

    for key, value in sorted(vars(args).items()):
        print(str(key) + ': ' + str(value))

    model = RTStereoNet(args.maxdisp)
    model = nn.DataParallel(model)
    model.cuda()

    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])
    print("=> loaded pretrained model '{}'".format(args.loadmodel))

    # detect resource root
    res_root = None
    for cand in ['M:/', '/media/vicnas', '/media/devshare']:
        if Path(cand).is_dir():
            res_root = Path(cand)
            break
    assert res_root is not None

    ''' point cloud test of dexter data '''
    #data_dir = res_root / 'datasets/dexter/suntemple_850'
    data_dir = res_root / 'datasets/dexter/ModularPrison'
    imgnamesL = [f for f in (data_dir/'imL').glob('*.png') if not f.stem.startswith('gray')]
    imgnamesL = sorted(imgnamesL, key=lambda v: int(v.stem))
    imgnamesL = imgnamesL[200:]
    def ns_feeder(index):
        imgnameL = imgnamesL[(index * 10) % len(imgnamesL)]
        imgnameR = (data_dir/'imR'/imgnameL.stem).with_suffix('.png')
        img, depth, K = predict(model, str(imgnameL), str(imgnameR), cat='dexter')
        return img, np.clip(depth, 0, 30), K

    scene = NavScene(ns_feeder)
    scene.run()
    scene.clear()
