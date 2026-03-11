import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from basicsr.models.archs_stereo.freq import FDM, AFM
from basicsr.models.archs_stereo.sim import SIM
from basicsr.models.archs_stereo.lce import LCE

class FCIB(nn.Module):

    def __init__(self, chan=128):
        super(FCIB, self).__init__()
        self.fdm_left = FDM(chan)
        self.fdm_right = FDM(chan)

        self.conv_down = nn.Conv2d(chan * 2, chan, kernel_size=1)

        self.proj_left = nn.Conv2d(chan, chan, kernel_size=1)
        self.proj_right = nn.Conv2d(chan, chan, kernel_size=1)

        # Feature weight ratio
        self.beta = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, chan, 1, 1)), requires_grad=True)

        self.lce = LCE(chan)

        self.deform_conv_left = DeformConv2d(chan, chan, kernel_size=3, padding=1)
        self.deform_conv_right = DeformConv2d(chan, chan, kernel_size=3, padding=1)

        self.offset_conv_left = nn.Conv2d(chan, 18, kernel_size=3, padding=1)
        self.offset_conv_right = nn.Conv2d(chan, 18, kernel_size=3, padding=1)

        self.sim = SIM(c=chan)

        # 动态合频时的模块
        self.afm_left = AFM(chan)
        self.afm_right = AFM(chan)

    def forward(self, inputs_l, inputs_r):
        # 动态分频代码
        l_low, l_high = self.fdm_left(inputs_l)
        r_low, r_high = self.fdm_right(inputs_r)

        concat_high = torch.cat((l_high, r_high), dim=1)
        # 通道还原
        concat_high = self.conv_down(concat_high)

        concat_high = self.lce(concat_high)

        offset_left = self.offset_conv_left(l_high)
        offset_right = self.offset_conv_right(r_high)

        concat_high_left = self.deform_conv_left(concat_high.contiguous(), offset_left.contiguous())
        concat_high_right = self.deform_conv_right(concat_high.contiguous(), offset_right.contiguous())

        proj_l_high = self.proj_left(l_high)
        proj_r_high = self.proj_right(r_high)

        cross_left_high = torch.mul(proj_l_high, concat_high_left)
        cross_right_high = torch.mul(proj_r_high, concat_high_right)

        l_high_out = l_high + cross_left_high * self.beta
        r_high_out = r_high + cross_right_high * self.gamma

        # SIM 处理低频
        l_low_out, r_low_out = self.sim(l_low, r_low)

        # AFM 动态合频机制
        out_l = self.afm_left(l_low_out, l_high_out) + l_high_out + l_low_out
        out_r = self.afm_right(r_low_out, r_high_out) + r_high_out + r_low_out

        return out_l, out_r
