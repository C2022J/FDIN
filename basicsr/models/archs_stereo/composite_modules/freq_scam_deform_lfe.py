import torch
import torch.nn as nn

# 从 MMCV 导入可变形卷积
from mmcv.ops import DeformConv2d
# 导入 SCAM 模块
from basicsr.models.archs_stereo.base_modules.fft import FFTFrequencySplit
from basicsr.models.archs_stereo.base_modules.freq_new import FrequencyDecomposition, SFconv
from basicsr.models.archs_stereo.base_modules.scam import SCAM
from basicsr.models.archs_stereo.base_modules.lfe import LFE

# 低频SCAM 高频LFE
class FreqDEBlockSCAM(nn.Module):

    def __init__(self,
                 chan=128,
                 ):
        super(FreqDEBlockSCAM, self).__init__()

        # self.freq_split_left = FrequencyDecomposition(chan)
        # self.freq_split_right = FrequencyDecomposition(chan)

        self.freq_split = FFTFrequencySplit()

        self.conv_down = nn.Conv2d(chan * 2,  chan, kernel_size=1)
        self.proj_left = nn.Conv2d(chan, chan, kernel_size=1)
        self.proj_right = nn.Conv2d(chan, chan, kernel_size=1)

        # Feature weight ratio
        self.beta = nn.Parameter(
            torch.zeros((1, chan, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(
            torch.zeros((1, chan, 1, 1)), requires_grad=True)

        # 定义 ConvMod 模块
        self.lfe1 = LFE(chan)

        # 增加两个 Deformable Conv 层
        self.deform_conv_left = DeformConv2d(chan, chan, kernel_size=3, padding=1)
        self.deform_conv_right = DeformConv2d(chan, chan, kernel_size=3, padding=1)

        # 用于生成偏移量的卷积层，分别用于左、右特征
        self.offset_conv_left = nn.Conv2d(chan, 18, kernel_size=3, padding=1)
        self.offset_conv_right = nn.Conv2d(chan, 18, kernel_size=3, padding=1)

        self.scam = SCAM(c=chan)

        # self.left_agg = SFconv(chan)
        # self.right_agg = SFconv(chan)

    def init_weights(self):
        pass

    def forward(self, inputs_l, inputs_r):

        # l_low, l_high = self.freq_split_left(inputs_l)
        # r_low, r_high = self.freq_split_right(inputs_r)

        l_low, l_high = self.freq_split(inputs_l)
        r_low, r_high = self.freq_split(inputs_r)

        # 合并通道
        concat_high = torch.cat((l_high, r_high), dim=1)
        # 通道还原
        concat_high = self.conv_down(concat_high)

        # 经过 2 × LFE
        concat_high = self.lfe1(concat_high)

        # 分别计算左右特征的偏移量
        offset_left = self.offset_conv_left(l_high)
        offset_right = self.offset_conv_right(r_high)

        # 新增的 Deformable Conv，分别处理左右特征并传入各自的偏移量
        concat_high_left = self.deform_conv_left(concat_high.contiguous(), offset_left.contiguous())
        concat_high_right = self.deform_conv_right(concat_high.contiguous(), offset_right.contiguous())

        # projection
        proj_l_high = self.proj_left(l_high)
        proj_r_high = self.proj_right(r_high)

        cross_left_high = torch.mul(proj_l_high, concat_high_left)
        cross_right_high = torch.mul(proj_r_high, concat_high_right)

        l_high = l_high + cross_left_high * self.beta
        r_high = r_high + cross_right_high * self.gamma

        # SCAM 处理低频
        l_low, r_low = self.scam(l_low, r_low)

        out_l = l_high + l_low
        out_r = r_high + r_low

        # out_l = self.left(l_high + l_low)
        # out_r = self.right(r_high + r_low)

        # out_l = self.left_agg(l_low, l_high) + l_high + l_low
        # out_r = self.right_agg(r_low, r_high) + r_high + r_low

        return out_l, out_r
