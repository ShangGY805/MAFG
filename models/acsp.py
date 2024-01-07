import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from cupy_layers.aggregation_zeropad import LocalConvolution
from .layers import get_act_layer


class ACSP(nn.Module):
    def __init__(self, inplanes, output_stride=8):
        super(ACSP, self).__init__()
        self.inplanes = inplanes

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.acsp1 = CotLayer(dim=self.inplanes, kernel_size=3, padding=1, dilation=dilations[0])
        self.acsp2 = CotLayer(dim=self.inplanes, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.acsp3 = CotLayer(dim=self.inplanes, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.acsp4 = CotLayer(dim=self.inplanes, kernel_size=3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 64, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.acsp1(x)
        x2 = self.acsp2(x)
        x3 = self.acsp3(x)
        x4 = self.acsp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size,padding,dilation):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.padding, dilation=self.dilation, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        act = get_act_layer('swish')
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.t = attn_chs
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)  #v
        x = self.local_conv(x, w) #qk和V的合并
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        # print(x_gap.shape)   #8, 64, 256, 256]
        # print(self.t)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        # print(x_gap.shape)  #[8, 64, 1, 1]
        x_attn = self.se(x_gap)
        # print(x_attn.shape)
        x_attn = x_attn.view(B, C, self.radix)
        # print(x_attn.shape)
        x_attn = F.softmax(x_attn, dim=2)
        # print(x_attn.shape)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out.contiguous()

