from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import math
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--maxdepth', type=int, default=96,
                    help='maxium depth')
parser.add_argument('--model', default='RTStereoDepthNet',
                    help='select model')
parser.add_argument('--datatype', default='dexter',
                    help='datapath')
parser.add_argument('--datapath', default='/workspace/AnyNet/dataset_dexter',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./result',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dexterdepth', action='store_true', default=True,
                    help='enables depth prediction instead of disparity')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as lt
    from dataloader import KITTILoader as DA
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as lt
    from dataloader import KITTILoader as DA
elif args.datatype == 'dexter':
    from dataloader import dexter_list_file as lt
    from dataloader import dexter_loader as DA
else:
    from dataloader import listflowfile as lt
    from dataloader import SecenFlowLoader as DA

if args.datatype == 'dexter' and args.dexterdepth:
    from utils.pcl import query_intrinsic
    K, baseline = query_intrinsic('dexter')
    fxb = K[0,0] * baseline
else:
    fxb = None

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

trainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, fxb=fxb),
        batch_size=12, shuffle=True, num_workers=8, drop_last=False)

testImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, fxb=fxb),
        batch_size=8, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
elif args.model == 'RTStereoNet':
    model = RTStereoNet(args.maxdisp, fxb)
elif args.model == 'RTStereoDepthNet':
    model = RTStereoDepthNet(args.maxdepth, fxb)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

def train(imgL, imgR, gtL):
    model.train()

    if args.cuda:
        imgL, imgR, gtL = imgL.cuda(), imgR.cuda(), gtL.cuda()

    #---------
    mask = gtL < (args.maxdepth if fxb else args.maxdisp)
    mask.detach_()
    #---------
    optimizer.zero_grad()
    
    if args.model in ['stackhourglass', 'RTStereoNet', 'RTStereoDepthNet']:
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = (0.25*F.smooth_l1_loss(output1[mask], gtL[mask], size_average=True)
            + 0.5*F.smooth_l1_loss(output2[mask], gtL[mask], size_average=True)
            + F.smooth_l1_loss(output3[mask], gtL[mask], size_average=True))
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], dispgtL_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data

def test(imgL, imgR, gtL):
    model.eval()

    if args.cuda:
        imgL, imgR, gtL = imgL.cuda(), imgR.cuda(), gtL.cuda()
    #---------
    mask = gtL < (args.maxdepth if fxb else args.maxdisp)
    #---------

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16
        top_pad = (times+1)*16 -imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3]//16
        right_pad = (times+1)*16-imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3, 1)
    
    if top_pad != 0:
        img = output3[:,top_pad:,:]
    else:
        img = output3

    if len(gtL[mask]) == 0:
        loss = 0
    else:
        loss = F.l1_loss(img[mask], gtL[mask])
        #loss = torch.mean(torch.abs(img[mask]-gtL[mask]))  # end-point-error

    return loss.data.cpu()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 150:
        lr = args.lr
    elif epoch <= 200:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    print(f'lr = {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def epe_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        epe = np.mean(np.abs(d_est - d_gt))
    else:
        epe = torch.mean(torch.abs(d_est - d_gt))
    return epe

def error_estimating(disp, ground_truth, maxdisp=192, print_val=False):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    return epe_metric(disp, gt, mask).float()

    # errmap = torch.abs(disp - gt)
    # if print_val:
    #     print(f'disp.shape={disp.shape}') # (16, 480, 640)
    #     print(f'errmap={errmap[0][300]}')
    #     print(f'disp={disp[0][300]}')
    #     print(f'gt={gt[0][300]}')

    # err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    # return err3.float() / mask.sum().float()

def main():
    writer = SummaryWriter()
    start_full_time = time.time()

    if not os.path.isdir(args.savemodel):
        os.makedirs(args.savemodel)

    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        start_epoch_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        ## training
        total_train_loss = 0
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(trainImgLoader):
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('epoch %d : [%d/%d] training loss = %.3f' % (epoch, batch_idx, len(trainImgLoader), loss))
            total_train_loss += loss
        avg_train_loss = total_train_loss/len(trainImgLoader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        print('epoch %d : total training loss = %.3f, time = %.2f'
             % (epoch, avg_train_loss, time.time() - start_epoch_time))

        ## test
        total_test_loss = 0
        for batch_idx, (imgL, imgR, disp_L) in enumerate(testImgLoader):
            test_loss = test(imgL, imgR, disp_L)
            total_test_loss += test_loss
            print('epoch %d : [%d/%d] test loss = %.3f' % (epoch, batch_idx, len(testImgLoader), test_loss))
        avg_test_loss = total_test_loss/len(testImgLoader)
        writer.add_scalar("Epe/val", avg_test_loss, epoch)
        print('epoch %d : total test loss = %.3f' % (epoch, avg_test_loss))

        ## save
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(trainImgLoader),
        }, savefilename)

        ## time
        print('epoch %d : total time = %.2f' % (epoch, time.time() - start_epoch_time))

        writer.flush()

    print('full training time = %.2f HR' % ((time.time() - start_full_time)/3600))
    writer.close()

if __name__ == '__main__':
    main()

'''
python3 main.py --model RTStereoNet
'''