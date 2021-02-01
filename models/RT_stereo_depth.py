from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np

norm_layer2d = nn.BatchNorm2d 
norm_layer3d = nn.BatchNorm3d 

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=dilation if dilation > 1 else pad, dilation=dilation, groups=groups, bias=False),
        norm_layer2d(out_planes))

class featexchange(nn.Module):
    def __init__(self):
        super(featexchange, self).__init__()

        self.x2_fusion = nn.Sequential(nn.ReLU(),convbn(4, 4, 3, 1, 1, 1, 1),
                                       nn.ReLU(),nn.Conv2d(4, 4, 3, 1, 1, bias=False))
        self.upconv4 = nn.Sequential(nn.Conv2d(8, 4, 1, 1, 0, bias=False),
                                     norm_layer2d(4)) 
        self.upconv8 = nn.Sequential(nn.Conv2d(20, 4, 1, 1, 0, bias=False),
                                     norm_layer2d(4)) 

        self.x4_fusion = nn.Sequential(nn.ReLU(),convbn(8, 8, 3, 1, 1, 1, 1),
                                       nn.ReLU(),nn.Conv2d(8, 8, 3, 1, 1, bias=False))
        self.downconv4 = nn.Sequential(nn.Conv2d(4, 8, 3, 2, 1, bias=False),
                                       norm_layer2d(8))
        self.upconv8_2 = nn.Sequential(nn.Conv2d(20, 8, 1, 1, 0, bias=False),
                                       norm_layer2d(8))

        self.x8_fusion = nn.Sequential(nn.ReLU(),convbn(20, 20, 3, 1, 1, 1, 1),
                                       nn.ReLU(),nn.Conv2d(20, 20, 3, 1, 1, bias=False))
        self.downconv81 = nn.Sequential(nn.Conv2d(8, 20, 3, 2, 1, bias=False),
                                        norm_layer2d(20))
        self.downconv82 = nn.Sequential(nn.Conv2d(8, 20, 3, 2, 1, bias=False),
                                        norm_layer2d(20))

    def forward(self, x2, x4, x8, attention):
        A = torch.split(attention,[4,8,20],dim=1)

        x4tox2 = self.upconv4(F.upsample(x4, (x2.size()[2],x2.size()[3])))
        x8tox2 = self.upconv8(F.upsample(x8, (x2.size()[2],x2.size()[3])))
        fusx2  = x2 + x4tox2 + x8tox2
        fusx2  = self.x2_fusion(fusx2)*A[0].contiguous()+fusx2

        x2tox4 = self.downconv4(x2)
        x8tox4 = self.upconv8_2(F.upsample(x8, (x4.size()[2],x4.size()[3])))
        fusx4  = x4 + x2tox4 + x8tox4 
        fusx4  = self.x4_fusion(fusx4)*A[1].contiguous()+fusx4

        x2tox8 = self.downconv81(x2tox4)
        x4tox8 = self.downconv82(x4)
        fusx8  = x8 + x2tox8 + x4tox8
        fusx8  = self.x8_fusion(fusx8)*A[2].contiguous()+fusx8

        return fusx2, fusx4, fusx8

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()

        self.inplanes = 1
        self.firstconv = nn.Sequential(nn.Conv2d(3, 3, 3, 2, 1, bias=False),
                                       nn.Conv2d(3, 3, 3, 2, 1, bias=False),
                                       nn.BatchNorm2d(3),
                                       nn.ReLU(),
                                       nn.Conv2d(3, 4, 1, 1, 0, bias=False),
                                       convbn(4, 4, 3, 1, 1, 1, 4),
                                       nn.ReLU(),
                                       nn.Conv2d(4, 4, 1, 1, 0, bias=False),
                                       convbn(4, 4, 3, 1, 1, 1, 4)) # 1/4

        self.stage2 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(4, 8, 1, 1, 0, bias=False),
                                    convbn(8, 8, 3, 2, 1, 1, 8),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 8, 1, 1, 0, bias=False),
                                    convbn(8, 8, 3, 1, 1, 1, 8)) # 1/8

        self.stage3 = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(8, 20, 1, 1, 0, bias=False),
                                    convbn(20, 20, 3, 2, 1, 1, 20),
                                    nn.ReLU(),
                                    nn.Conv2d(20, 20, 1, 1, 0, bias=False),
                                    convbn(20, 20, 3, 1, 1, 1,20)) #1/16
                
        self.attention = nn.Sequential(nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(20, 10, 1, 1, 0, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(10, 32, 1, 1, 0, bias=True),
                                    nn.Sigmoid(),
                                    )
        
        self.fusion = featexchange()

    def forward(self, x):
        #stage 1# 1x
        out_s1 = self.firstconv(x)
        out_s2 = self.stage2(out_s1)
        out_s3 = self.stage3(out_s2)
        attention = self.attention(out_s3)
        out_s1, out_s2, out_s3 = self.fusion(out_s1, out_s2, out_s3, attention)
        return [out_s3, out_s2, out_s1]

def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(
            norm_layer3d(in_planes),
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))

def post_3dconvs(layers, channels):
    net  = [nn.Conv3d(1, channels, kernel_size=3, padding=1, stride=1, bias=False)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)

class RTStereoDepthNet(nn.Module):
    def __init__(self, maxdepth, fxb):
        super(RTStereoDepthNet, self).__init__()

        self.feature_extraction = feature_extraction()
        self.maxdepth = maxdepth
        self.fxb = fxb
        self.volume_postprocess = []
        
        layer_setting = [8, 4, 4]
        for i in range(3):
            net3d = post_3dconvs(3, layer_setting[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _warp_3d(self, x, disp):
        """
        warp an tensor x, according to the disp
        args:
            x: [B, C, D, H, W]
            disp: [B, 1, D, H, W]
        returns:
            x_warped: [B, C, D, H, W]
        """
        b, _, d, h, w = x.size()
        disp = disp.squeeze(1) # [b, d, h, w]

        # mesh grid [b, d, h, w, 3]
        xx = torch.arange(0, w).cuda().float() # torch.Size([12, 6, 16, 32])
        xx = xx.view(1, 1, 1, w).expand(b, d, h, w) - disp
        xx = 2 * (xx / (w - 1)) - 1
        xx = xx.unsqueeze(-1)
        
        yy = torch.arange(0, h).cuda().float()
        yy = yy.view(1, 1, h, 1).expand(b, d, h, w)
        yy = 2 * (yy / (h - 1)) - 1
        yy = yy.unsqueeze(-1)
        
        dd = torch.arange(0, d).cuda().float()
        dd = dd.view(1, d, 1, 1).expand(b, d, h, w)
        dd = 2 * (dd / (d - 1)) - 1
        dd = dd.unsqueeze(-1)

        grid = torch.cat((xx, yy, dd), dim=-1)

        # sampling from x [b, c, d, h, w]
        x_warped = nn.functional.grid_sample(x, grid, align_corners=True)
        return x_warped

    def _build_volume(self, feat_l, feat_r, fxb, depth, depth_offsets):
        b, _, h, w = feat_l.size()
        d = len(depth_offsets)

        # [b, 1, h, w] -> [b, 1, d, h, w]
        depth = depth.unsqueeze(dim=2).repeat(1, 1, d, 1, 1)
        depth_offset = torch.tensor(depth_offsets).cuda().float() # [d]
        depth_offset = depth_offset.view(1, 1, -1, 1, 1).repeat(b, 1, 1, h, w).requires_grad_(False) # [b, 1, d, h, w]
        disp = fxb / (depth + depth_offset).clip(0.1, 9999)

        batch_feat_l = feat_l[:, :, None, :, :].repeat(1, 1, d, 1, 1) # [b, c, d, h, w]
        batch_feat_r = feat_r[:, :, None, :, :].repeat(1, 1, d, 1, 1) # [b, c, d, h, w]
        batch_feat_r = self._warp_3d(batch_feat_r, disp)

        # final L1 norm cost [b, 1, d, h, w]
        cost = torch.norm(batch_feat_l - batch_feat_r, p=1, dim=1, keepdim=True)
        return cost.contiguous()

    def forward(self, left, right):
        b, _, h, w = left.size()

        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)

        pred = []
        for stage in range(len(feats_l)):
            fh, fw = feats_l[stage].size(-2), feats_l[stage].size(-1)
            inv_scale = w // fw # image inv scale per stage. ex) 4, 8, 16
            assert self.maxdepth % inv_scale == 0 # Assume maxdepth is multiple of inv_scale
            fxb_scaled = self.fxb / inv_scale

            detph_offset_strides = [8, 4, 2]
            if stage > 0:
                depth_offsets = [d * detph_offset_strides[stage] for d in range(-2, 3)]
                depth = F.upsample(pred[stage-1], (fh, fw), mode='bilinear')
                cost = self._build_volume(feats_l[stage], feats_r[stage], fxb_scaled,
                                          depth, depth_offsets)
            else:
                depth_offsets = [d for d in range(1, 1 + self.maxdepth, detph_offset_strides[stage])]
                depth = torch.zeros(b, 1, fh, fw).cuda().requires_grad_(False)
                cost = self._build_volume(feats_l[stage], feats_r[stage], fxb_scaled,
                                          depth, depth_offsets)

            cost = self.volume_postprocess[stage](cost)
            cost = cost.squeeze(1)
            if stage == 0:
                pred_low_res = depthregression(depth_offsets)(F.softmax(cost, dim=1))
                depth_up = F.upsample(pred_low_res, (h, w), mode='bilinear')
                pred.append(depth_up)
            else:
                pred_low_res = depthregression(depth_offsets)(F.softmax(cost, dim=1))
                depth_up = F.upsample(pred_low_res, (h, w), mode='bilinear')
                pred.append(depth_up + pred[stage-1])
        if self.training:
            return pred[0], pred[1], pred[2]
        else:
            return pred[-1]

class depthregression(nn.Module):
    def __init__(self, depth):
        super(depthregression, self).__init__()
        self.depth = torch.tensor(depth).cuda().requires_grad_(False).float().view(1, -1, 1, 1)

    def forward(self, x):
        out = torch.sum(x * self.depth, 1, keepdim=True)
        return out