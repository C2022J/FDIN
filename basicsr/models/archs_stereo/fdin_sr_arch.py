import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from basicsr.models.archs_stereo.stereo_fdin_arch import FEB
import torch
from mmcv.ops import DeformConv2d
from basicsr.models.archs_stereo.freq import FDM, AFM
from basicsr.models.archs_stereo.sim import SIM
from basicsr.models.archs_stereo.lce import LCE

class FDINSR(BaseModule):
    def __init__(self,
                 img_channels=3,
                 mid_channels=64,
                 num_stages=4,
                 blocks_per_stage=8,
                 scale=2):

        super().__init__()
        self.scale = scale
        c = mid_channels

        self.shallow_conv = nn.Conv2d(img_channels, c, 3, 1, 1)

        self.body = nn.ModuleList()
        for _ in range(num_stages):
            # 每个阶段包含: N个单视图模块 + 1个跨视图模块

            # (A) N个单视图模块
            intra_blocks = nn.Sequential(*[FEB(in_channels=c) for _ in range(blocks_per_stage)])
            self.body.append(intra_blocks)

            # (B) 1个跨视图模块
            self.body.append(FCIB(c))

        self.final_conv = nn.Conv2d(c, c, 3, 1, 1)

        self.final_upsampler = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=img_channels * (self.scale ** 2),
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True
            ),
            nn.PixelShuffle(self.scale)
        )

    def forward(self, left, right):
        original_lr_left = left
        original_lr_right = right

        x_left = self.shallow_conv(left)
        x_right = self.shallow_conv(right)

        for module in self.body:
            if isinstance(module, FCIB):
                x_left, x_right = module(x_left, x_right)
            else:
                x_left = module(x_left)
                x_right = module(x_right)

        x_left = self.final_conv(x_left)
        x_right = self.final_conv(x_right)

        pred_residual_hr_left = self.final_upsampler(x_left)
        pred_residual_hr_right = self.final_upsampler(x_right)

        base_hr_left = F.interpolate(original_lr_left, scale_factor=self.scale, mode='bicubic', align_corners=False)
        base_hr_right = F.interpolate(original_lr_right, scale_factor=self.scale, mode='bicubic', align_corners=False)

        final_left = pred_residual_hr_left + base_hr_left
        final_right = pred_residual_hr_right + base_hr_right

        return final_left, final_right

class FCIB(nn.Module):

    def __init__(self, chan=128):
        super(FCIB, self).__init__()
        self.fdm_left = FDM(chan)
        self.fdm_right = FDM(chan)

        self.conv_down = nn.Conv2d(chan * 2, chan, kernel_size=1)

        self.norm_high_l = LayerNorm2d(chan)
        self.norm_high_r = LayerNorm2d(chan)

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

        l_high_norm = self.norm_high_l(l_high)
        r_high_norm = self.norm_high_r(r_high)

        offset_left = self.offset_conv_left(l_high_norm)
        offset_right = self.offset_conv_right(r_high_norm)

        concat_high_left = self.deform_conv_left(concat_high.contiguous(), offset_left.contiguous())
        concat_high_right = self.deform_conv_right(concat_high.contiguous(), offset_right.contiguous())

        proj_l_high = self.proj_left(l_high_norm)
        proj_r_high = self.proj_right(r_high_norm)

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


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)