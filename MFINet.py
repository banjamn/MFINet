import torch
from torch import nn
import torch.nn.functional as F
import vgg
from torch.nn import Conv2d, Parameter, Softmax
import numpy as np

import matplotlib.pyplot as plt
from src.models.model_utils import SqueezeAndExcitation


def convblock(in_, out_, ks, st=1, pad=1, dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias=False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class SCSC(nn.Module):
    def __init__(self):
        super(SCSC, self).__init__()

    def forward(self, x):
        x_ = self.conv(x)
        x_1 = self.conv_d3(x_)
        x_2 = self.conv_d6(x_)
        x_3 = self.conv_d9(x_)
        x_4 = self.conv_d12(x_)
        x_ = self.Sig(self.conv(x_))




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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

        self.CBR = convblock(384,channels_in,1,1,0)
        self.conv_r = convblock(channels_in, 128, 3, 1, 1)
        self.conv_t = convblock(channels_in, 128, 3, 1, 1)
        self.conv_d = convblock(channels_in, 128, 3, 1, 1)
        self.conv = convblock(128,channels_in,3,1,1,1)

        self.fuse_dconv = nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(channels_in, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, rgb, thermal,  depth):
        rgb_ = self.conv_r(self.se_rgb(rgb))
        depth_ = self.conv_d(self.se_depth(depth))
        thermal_ = self.conv_t(self.se_thermal(thermal))
        fusion = self.CBR(torch.cat((rgb_, depth_, thermal_),dim=1))
        return fusion


class Fuse_module_aspp(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(Fuse_module_aspp, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)
        self.se_thermal = SqueezeAndExcitation(channels_in,
                                             activation=activation)

        self.CBR = convblock(384,channels_in,1,1,0)
        self.conv_r = convblock(channels_in, 128, 3, 1, 1)
        self.conv_t = convblock(channels_in, 128, 3, 1, 1)
        self.conv_d = convblock(channels_in, 128, 3, 1, 1)

        self.conv = convblock(channels_in,channels_in,3,1,1,1)
        self.Sigm = nn.Sigmoid()

        self.dconv1 = BasicConv2d(channels_in, int(channels_in / 4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(channels_in, int(channels_in / 4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(channels_in, int(channels_in / 4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(channels_in, int(channels_in / 4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(channels_in, 2, kernel_size=3, padding=1, bias=True)

        # self.sa = SpatialAttention()

    def forward(self, rgb, thermal,  depth):
        rgb_ = self.conv_r(self.se_rgb(rgb))
        depth_ = self.conv_d(self.se_depth(depth))
        thermal_ = self.conv_t(self.se_thermal(thermal))
        fusion = self.CBR(torch.cat((rgb_, depth_, thermal_),dim=1))
        fusion2 = self.Sigm(self.conv(fusion))
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out_cat = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.fuse_dconv(out_cat*fusion2+out_cat+rgb)
        return out


class S_fusion(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(S_fusion, self).__init__()
        self.ca = ChannelAttention(in_planes=20)
        self.sa = SpatialAttention()
        self.msa = SpatialAttention()
        self.conv = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.sig = nn.Sigmoid()
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,s2,s3,x3):
        s3 = self.conv1(self.Upsample(s3))
        s2_3 = s3+s2
        mean23 = self.conv(torch.mean(s2_3, dim=1, keepdim=True))
        mean23 = self.sig(mean23)
        # mean23 = self.msa(s2_3)
        s2_3 = torch.mul(s2_3, mean23)
        x3 = torch.mul(x3, mean23)
        out_put = s2_3+x3
        spatial_weight = self.sa(out_put)
        out_put = torch.mul(out_put,spatial_weight)
        return out_put,mean23


class Guide_flow(nn.Module):
    def __init__(self, x_channel, y_channel):
        super(Guide_flow, self).__init__()
        self.guidemap = nn.Conv2d(x_channel, 1, 1)
        self.gateconv = GatedSpatailConv2d(y_channel)

    def forward(self, x, y):  # x guide y
        guide = self.guidemap(x)
        guide_flow = F.interpolate(guide, size=y.size()[2:], mode='bilinear')
        y = self.gateconv(y, guide_flow)
        return y


class GatedSpatailConv2d(nn.Module):
    def __init__(self, channels=32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias_attr=False):
        super(GatedSpatailConv2d, self).__init__()
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(channels + 1),
            nn.Conv2d(channels + 1, channels + 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels + 1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=groups)

    def forward(self, input_features, gating_features):
        cat = torch.cat((input_features, gating_features), 1)
        attention = self._gate_conv(cat)
        x = input_features * (attention + 1)
        x = self.conv(x)
        return x


def d_module(channel1=512, channel2=256):
    return nn.Sequential(
            nn.Dropout2d(p=0.2),
            BasicConv2d(channel1, channel1, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel1, channel2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )


class Decoder(nn.Module):
    def __init__(self, channel5=512, channel4=256, channel3=128, channel2=64, channel1=1, n_classes=1):
        super(Decoder, self).__init__()
        self.S5 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S4 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.Conv_s23 = nn.Conv2d(128,64,3,1,1)
        self.Conv_s45_4 = nn.Conv2d(512, 256, 3, 1, 1)
        self.Conv_s45_3 = nn.Conv2d(512, 128, 3, 1, 1)
        self.Conv_s45_2 = nn.Conv2d(512, 64, 3, 1, 1)
        self.conv = nn.Conv2d(128, 64,3,1,1)

        self.s1_s2 = S_fusion(128,64)
        self.s2_s3 = S_fusion(256,128)
        self.s3_s4 = S_fusion(512, 256)
        self.s4_s5 = S_fusion(512, 512)

        self.guideflow_sh2 = Guide_flow(channel2, channel2)
        self.guideflow_sh3 = Guide_flow(channel3, channel2)
        self.guideflow_sh4 = Guide_flow(channel4, channel2)
        self.guideflow_sh5 = Guide_flow(channel5, channel2)

        # 22 22
        self.decoder5 = d_module(channel5, channel5)
        # 44 44
        self.decoder4 = d_module(channel5, channel4)
        # 88 88
        self.decoder3 = d_module(2*channel4, channel3)
        # 176 176
        self.decoder2 = d_module(2*channel3, channel2)
        self.semantic_pred2 = nn.Conv2d(channel2, n_classes, kernel_size=3, padding=1)
        # 352 352
        self.decoder1 = nn.Sequential(
            nn.Dropout2d(p=0.2),
            BasicConv2d(channel2, channel1, kernel_size=3, padding=3, dilation=3),
            BasicConv2d(channel1, channel1, kernel_size=3, padding=1),
            nn.Conv2d(channel1, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, rgb,t,d, T1, s2, s3, s4, s5):
        x_size = rgb[0].size()[2:]
        x5_decoder = self.decoder5(s5)

        s4_5_fuse,w45 = self.s4_s5(s4, s5, x5_decoder)                                               # ,s4和s5、x5融合
        x4_decoder = self.decoder4(s4_5_fuse)                                                    # 融合后进行解码

        s4_5_fuse_4 = F.interpolate(self.Conv_s45_4(s4_5_fuse), scale_factor=2, mode='bilinear', align_corners=True)  # 特征融合特征s45x5的大小，以适应跳跃连接
        s4_5_fuse_3 = F.interpolate(self.Conv_s45_3(s4_5_fuse), scale_factor=4, mode='bilinear', align_corners=True)
        s4_5_fuse_2 = F.interpolate(self.Conv_s45_2(s4_5_fuse), scale_factor=8, mode='bilinear', align_corners=True)

        s3_4_fuse,w34 = self.s3_s4(s3, s4, x4_decoder)
        x3_decoder = self.decoder3(torch.cat((s3_4_fuse, s4_5_fuse_4), dim=1))

        # ********改版s2和s3融合
        s2_3_fuse,w23 = self.s2_s3(s2, s3, x3_decoder)
        x2_decoder = self.decoder2(torch.cat((s2_3_fuse,s4_5_fuse_3),dim=1))

        # ***************
        # ****skip connection need to change s23fuse
        # s2_3_fuse = F.interpolate(self.Conv_s23(s2_3_fuse), scale_factor=2, mode='bilinear', align_corners=True)

        s1_2_fuse,w12 = self.s1_s2(T1,s2,x2_decoder)
        s1_4 = self.conv(torch.cat((s1_2_fuse, s4_5_fuse_2),dim=1))
        sM1 = self.guideflow_sh5(x5_decoder, s1_4)
        sM2 = self.guideflow_sh4(x4_decoder, sM1)
        sM3 = self.guideflow_sh3(x3_decoder, sM2)
        sM4 = self.guideflow_sh2(x2_decoder, sM3)

        semantic_pred = self.decoder1(sM4)

        x5 = self.S5(x5_decoder)
        x4 = self.S4(x4_decoder)
        x3 = self.S3(x3_decoder)
        semantic_pred2 = self.semantic_pred2(x2_decoder)

        x5_ = F.interpolate(x5, x_size, mode='bilinear', align_corners=True)
        x4_ = F.interpolate(x4, x_size, mode='bilinear', align_corners=True)
        x3_ = F.interpolate(x3, x_size, mode='bilinear', align_corners=True)

        return semantic_pred, semantic_pred2, x3_, x4_, x5_


class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.rgb_net = vgg.a_vgg16()
        self.t_net = vgg.a_vgg16()
        self.d_net = vgg.a_vgg16()
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.w1 = nn.Parameter(torch.Tensor(3))
        self.w2 = nn.Parameter(torch.Tensor(3))
        self.w3 = nn.Parameter(torch.Tensor(3))
        self.w4 = nn.Parameter(torch.Tensor(3))
        self.w5 = nn.Parameter(torch.Tensor(3))

        self.activation = nn.ReLU()

        self.se_layer0 = Fuse_module_aspp(
            64, activation=self.activation)
        self.se_layer1 = Fuse_module_aspp(
            128, activation=self.activation)
        self.se_layer2 = Fuse_module_aspp(
            256, activation=self.activation)
        self.se_layer3 = Fuse_module_aspp(
            512, activation=self.activation)
        self.se_layer4 = Fuse_module_aspp(
            512, activation=self.activation)

    def forward(self, rgb, t, d):
        rgb = self.rgb_net(rgb)
        t = self.t_net(t)
        d = self.d_net(d)

        s5 = self.se_layer4(rgb[4], t[4], d[4])
        s4 = self.se_layer3(rgb[3], t[3], d[3])
        s3 = self.se_layer2(rgb[2], t[2], d[2])
        s2 = self.se_layer1(rgb[1], t[1], d[1])
        s1 = self.se_layer0(rgb[0], t[0], d[0])

        score1,score2,  score3, score4, score5= self.decoder(rgb,t,d, s1, s2, s3, s4, s5)
    
        return score1, score2,  score3, score4, score5

    def load_pretrained_model(self):
        st = torch.load("/home/cuifengyu/Instance/HWSI_finalcode/vgg16.pth")
        st2 = {}
        for key in st.keys():
            st2['base.' + key] = st[key]
        self.rgb_net.load_state_dict(st2)
        self.t_net.load_state_dict(st2)
        self.d_net.load_state_dict(st2)
        print('loading pretrained model success!')
