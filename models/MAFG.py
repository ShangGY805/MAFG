import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.nn import init
from functools import partial
from models.Basenetworks import *
from models.help_funcs import TwoLayerConv2d, save_to_mat
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.helpers import load_pretrained
import types
import math
import pdb
from scipy.io import savemat
import numpy as np
from torch.autograd import Variable
from torchvision import models
from abc import ABCMeta, abstractmethod
from models.pixel_shuffel_up import PS_UP
import sys, os
from models.unitv2 import *
from models.acsp import *
from models.gcn import *

torch.backends.cudnn.enabled = False
BatchNorm = nn.BatchNorm2d


# Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )

# Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )

class MAFG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.att_f = Attention_fusion(in_ch=3, out_ch=64)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        x1 = self.att_f(x)
        return x1

    def forward(self, x):
        x = self.forward_features(x)
        return x

class MAFG_Decoder(nn.Module):

    def __init__(self, embedding_dim=64, output_nc=2,decoder_softmax=False):
        super(MAFG_Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_nc = output_nc

        self.acsp = ACSP(self.embedding_dim)
        self.gcn_basic = DualGCN_Spatial_fist(64)

        # convolutional Difference Modules
        self.diff = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        # Final predction head
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def forward(self, inputs1, inputs2):
        outputs = []

        f_diff = self.diff(torch.cat((inputs1, inputs2), dim=1))

        f_acsp = self.acsp(f_diff)
        f_g = self.gcn_basic(f_acsp)
        f_c = self.change_probability(f_g)

        outputs.append(f_c)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class MAFG(nn.Module):

    def __init__(self, output_nc=2, decoder_softmax=False, embed_dim=64):
        super(MAFG, self).__init__()

        self.embedding_dim = embed_dim
        self.encoder = MAFG_Encoder()

        self.decoder = MAFG_Decoder(embedding_dim=self.embedding_dim,
                                    output_nc=output_nc,
                                    decoder_softmax=decoder_softmax)

    def forward(self, x1, x2):
        output = []

        [x1_, x2_] = [self.encoder(x1), self.encoder(x2)]
        out = self.decoder(x1_[-1], x2_[-1])
        output.append(out[-1])

        return output

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        # if 'data_format' in kwargs:
        #     data_format = kwargs['data_format']
        # else:
        #     data_format = 'NCHW'
        self._batch_norm = nn.BatchNorm2d(out_channels)    # , data_format=data_format
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2)    # , bias_attr=False
        self.conv_out = ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1)    # , bias_attr=False
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 act_type=None,
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        # if 'data_format' in kwargs:
        #     data_format = kwargs['data_format']
        # else:
        #     data_format = 'NCHW'
        self._batch_norm = nn.BatchNorm2d(out_channels)    # , data_format=data_format

        self._act_type = act_type
        if act_type is not None:
            self._act = nn.LeakyReLU()

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        if self._act_type is not None:
            x = self._act(x)
        return x

class ConvBN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=1,
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        # if 'data_format' in kwargs:
        #     data_format = kwargs['data_format']
        # else:
        #     data_format = 'NCHW'
        self._batch_norm = nn.BatchNorm2d(out_channels)    # , data_format=data_format

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x

###################################
def avg_max_reduce_channel_helper(x, use_concat=True):
    # Reduce hw by avg and max, only support single input
    assert not isinstance(x, (list, tuple))
    # max_pool = F.adaptive_max_pool2d(x, 1)
    # mean_pool = F.g
    mean_value = torch.mean(x, axis=1, keepdim=True)
    max_value = torch.max(x, axis=1, keepdim=True)

    if use_concat:
        res = torch.cat([mean_value, max_value], axis=1)
    else:
        res = [mean_value, max_value.values]
    return res

def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return torch.cat(res, axis=1)

def avg_max_reduce_hw_helper(x, is_training, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    # TODO(pjc): when axis=[2, 3], the paddle.max api has bug for training.
    # if is_training:
    max_pool = F.adaptive_max_pool2d(x, 1)
    # else:
        # max_pool = torch.max(x, [2, 3], keepdim=True)
        # max_pool = torch.max(x, 2, keepdim=True)

    if use_concat:
        res = torch.cat([avg_pool, max_pool], axis=1)
    else:
        res = [avg_pool, max_pool]
    return res

def avg_max_reduce_hw(x, is_training):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x, is_training)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0], is_training)
    else:
        res_avg = []
        res_max = []
        # gcp_l = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, is_training, False)
            res_avg.append(avg)
            res_max.append(max)
            # gcp = avg + max
            # gcp_l.append(gcp)
        res = res_avg + res_max
        return torch.cat(res, axis=1)

##AFM##
class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNAct(
                4 * y_ch,
                # 2 * y_ch,
                # y_ch // 2,
                y_ch,
                kernel_size=3,
                # bias_attr=False,
                act_type="leakyrelu"),)
            # ConvBN(
            #     y_ch // 2, y_ch, kernel_size=3))    # , bias_attr=False

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_hw([x, y], self.training)
        atten = torch.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(
                4, 2, kernel_size=3, padding=1, ),
            ConvBN(
                2, 1, kernel_size=3, padding=1,))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class Attention_fusion(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super(Attention_fusion, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block(filters[0], filters[1], filters[1])
        self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block(filters[1], filters[2], filters[2])
        self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])

        self.conv_final = nn.Conv2d(filters[0] , out_ch, kernel_size=1)
        self.uafm0_1 = UAFM_ChAtten(32, 64, 32)
        self.uafm1_1 = UAFM_ChAtten(64, 128, 64)
        self.uafm2_1 = UAFM_ChAtten(128, 256, 128)
        self.uafm0_2 = UAFM_ChAtten(32, 64, 32)
        self.uafm1_2 = UAFM_ChAtten(64, 128, 64)
        self.uafm0_3 = UAFM_ChAtten(32, 64, 32)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self,x):
        x0_0 = self.conv0_0(x)        #32,256,256
        x1_0 = self.conv1_0(self.pool(x0_0))      #64,128,128
        x2_0 = self.conv2_0(self.pool(x1_0))      #128,64,64
        x3_0 = self.conv3_0(self.pool(x2_0))      #256,32,32
        x0_1 = self.uafm0_1(x0_0,x1_0)
        x1_1 = self.uafm1_1(x1_0,x2_0)
        x0_2 = self.uafm0_2(x0_1,x1_1)
        x2_1 = self.uafm2_1(x2_0,x3_0)
        x1_2 = self.uafm1_2(x1_1,x2_1)
        x0_3 = self.uafm0_3(x0_2,x1_2)
        out = x0_3
        out = self.conv_final(out)
        output = []
        output.append(out)
        return output

class conv_block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x

