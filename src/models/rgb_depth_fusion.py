# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch.nn as nn
import  torch
from src.models.model_utils import SqueezeAndExcitation
from src.models.model_utils import Swish


def convblock(in_, out_, ks, st=1, pad=1,dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class Fuse_module(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(Fuse_module, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)
        self.se_thermal = SqueezeAndExcitation(channels_in,
                                             activation=activation)

        self.CBR = convblock(256,channels_in,1,1,0)
        self.conv_r = convblock(channels_in, 128, 3, 1, 1)
        self.conv_t = convblock(channels_in, 128, 3, 1, 1)
        self.conv_d = convblock(channels_in, 128, 3, 1, 1)

        # self.sa = SpatialAttention()

    def forward(self, rgb, thermal,  depth):
        rgb_ = self.conv_r(self.se_rgb(rgb))
        depth_ = self.conv_d(self.se_depth(depth))
        thermal_ = self.conv_t(self.se_thermal(thermal))

        d_rgb = torch.mul(rgb_,depth_)+rgb_
        t_rgb = torch.mul(rgb_,thermal_)+thermal_

        out = torch.concat((t_rgb,d_rgb),dim=1)
        out = self.CBR(out)
        # out_sa = self.sa(out)
        # out = torch.mul(out, out_sa)
        return out


class S_fusion(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(S_fusion, self).__init__()
        self.ca = ChannelAttention(in_planes=20)
        self.sa = SpatialAttention()
        self.conv = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.sig = nn.Sigmoid()
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,s2,s3,x3):
        s3 = self.conv1(self.Upsample(s3))
        s2_3 = s3+s2
        mean23 = self.conv(torch.mean(s2_3, dim=1, keepdim=True))
        mean23 = self.sig(mean23)
        s2_3 = torch.mul(s2_3, mean23)
        x3 = torch.mul(x3, mean23)
        out_put = s2_3+x3
        spatial_weight = self.sa(out_put)
        out_put = torch.mul(out_put,spatial_weight)
        return out_put,mean23

