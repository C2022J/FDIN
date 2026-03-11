import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from basicsr.models.archs_stereo.freq import FDM, AFM
from basicsr.models.archs.sim_mono import SIM_Mono
from basicsr.models.archs_stereo.lce import LCE

class FCIB_Mono(nn.Module):

    def __init__(self,
                 chan=128,
                 ):
        super(FCIB_Mono, self).__init__()

        # 只需要一个 FDM
        self.fdm = FDM(chan)

        self.proj = nn.Conv2d(chan, chan, kernel_size=1)

        self.beta = nn.Parameter(
            torch.zeros((1, chan, 1, 1)), requires_grad=True)

        self.lce = LCE(chan)

        self.deform_conv = DeformConv2d(chan, chan, kernel_size=3, padding=1)

        self.offset_conv = nn.Conv2d(chan, 18, kernel_size=3, padding=1)

        self.sim = SIM_Mono(c=chan)

        self.afm = AFM(chan)

    def init_weights(self):
        pass

    def forward(self, inputs):

        low, high = self.fdm(inputs)

        processed_high = self.lce(high)

        offset = self.offset_conv(high)

        processed_high_warped = self.deform_conv(processed_high.contiguous(), offset.contiguous())

        proj_high = self.proj(high)

        cross_high = torch.mul(proj_high, processed_high_warped)

        # 融合
        high = high + cross_high * self.beta

        # --- 低频 SIM-Mono 自注意力 ---
        low = self.sim(low)  # 调用 SIM_Mono

        out = self.afm(low, high) + high + low

        return out