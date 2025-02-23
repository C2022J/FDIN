# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from basicsr.models.archs_stereo.composite_modules.freq_scam_deform_lfe import FreqDEBlockSCAM
from basicsr.models.nafnet.naf_avgpool2d import Local_Base
from basicsr.models.nafnet.naf_layerNorm2d import LayerNorm2d

class StereoNAFNetFreqDEBlockSCAM(BaseModule):
    # """NAFNet.
    #
    # The original version of NAFNet in "Simple Baseline for Image Restoration".
    #
    # Args:
    #     img_channels (int): Channel number of inputs.
    #     mid_channels (int): Channel number of intermediate features.
    #     middle_blk_num (int): Number of middle blocks.
    #     enc_blk_nums (List of int): Number of blocks for each encoder.
    #     dec_blk_nums (List of int): Number of blocks for each decoder.
    # """

    def __init__(self,
                 img_channels=3,
                 mid_channels=16,
                 middle_blk_num=1,
                 enc_blk_freq_deblock_scam_idx=[0, 1, 2, 3],
                 enc_blk_nums=[],
                 dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True)
        self.ending = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=img_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True)
        # 新增一个单独的上采样模块
        # self.first_up_left = nn.Sequential(
        #     nn.Conv2d(img_channels, img_channels * 4, kernel_size=3, padding=1),  # 增加通道数
        #     nn.PixelShuffle(2),  # 上采样两倍
        # )
        # self.first_up_right = nn.Sequential(
        #     nn.Conv2d(img_channels, img_channels * 4, kernel_size=3, padding=1),  # 增加通道数
        #     nn.PixelShuffle(2),  # 上采样两倍
        # )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # 定义 FreqDEBlockSCAM 层的 ModuleList
        self.freq_deblock_scam = nn.ModuleList()


        chan = mid_channels
        self.enc_blk_freq_deblock_scam_idx = enc_blk_freq_deblock_scam_idx

        for idx, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

            if idx in enc_blk_freq_deblock_scam_idx:
                self.freq_deblock_scam.append(FreqDEBlockSCAM(chan))  # FreqDEBlockSCAM 模块
            else:
                self.freq_deblock_scam.append(nn.Identity())  # 没有 FreqDEBlockSCAM 模块时保持原样

            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))

            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2**len(self.encoders)
        # self.mod_pad_h = 0
        # self.mod_pad_w = 0

    def forward(self, left, right):
        pred_left, pred_right = self.forward_stereo(left, right)
        return pred_left, pred_right

    def forward_stereo(self, left, right):
        """Forward function for stereo input (left and right images)."""
        B, C, H, W = left.shape

        # left = self.first_up_left(left)
        # right = self.first_up_right(right)

        left = self.check_image_size(left)
        right = self.check_image_size(right)

        x_left = self.intro(left)
        x_right = self.intro(right)

        encs_left = []
        encs_right = []

        for idx, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x_left = encoder(x_left)
            x_right = encoder(x_right)

            # 使用 FreqDEBlockSCAM 细节增强卷积
            if idx in self.enc_blk_freq_deblock_scam_idx:
                x_left, x_right = self.freq_deblock_scam[idx](x_left, x_right)

            encs_left.append(x_left)
            encs_right.append(x_right)

            x_left = down(x_left)
            x_right = down(x_right)

        x_left = self.middle_blks(x_left)
        x_right = self.middle_blks(x_right)

        for decoder, up, enc_skip_left, enc_skip_right in zip(self.decoders, self.ups, encs_left[::-1],
                                                              encs_right[::-1]):
            x_left = up(x_left)
            x_right = up(x_right)

            x_left = x_left + enc_skip_left
            x_right = x_right + enc_skip_right

            x_left = decoder(x_left)
            x_right = decoder(x_right)

        x_left = self.ending(x_left)
        x_right = self.ending(x_right)

        pred_left = x_left + left[:, :, :x_left.shape[2], :x_left.shape[3]]
        pred_right = x_right + right[:, :, :x_right.shape[2], :x_right.shape[3]]

        # pred_left = self.restore_image_size(pred_left)
        # pred_right = self.restore_image_size(pred_right)

        return pred_left, pred_right


    def check_image_size(self, x):
        """Check image size and pad images so that it has enough dimension do
        downsample.

        args:
            x: input tensor image with (B, C, H, W) shape.
        """
        _, _, h, w = x.size()
        self.mod_pad_h = (self.padder_size -
                     h % self.padder_size) % self.padder_size
        self.mod_pad_w = (self.padder_size -
                     w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, self.mod_pad_w, 0, self.mod_pad_h))
        return x

    # def restore_image_size(self, x):
    #     """Check image size and pad images so that it has enough dimension do
    #     downsample.
    #
    #     args:
    #         x: input tensor image with (B, C, H, W) shape.
    #     """
    #     _, _, h, w = x.size()
    #     x = x[:, :, 0:h - self.mod_pad_h, 0:w - self.mod_pad_w]
    #     return x


class NAFNetLocal(Local_Base, StereoNAFNetFreqDEBlockSCAM):
    """The original version of NAFNetLocal in "Simple Baseline for Image
    Restoration".

    NAFNetLocal uses local average pooling modules than NAFNet.

    Args:
        img_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
        middle_blk_num (int): Number of middle blocks.
        enc_blk_nums (List of int): Number of blocks for each encoder.
        dec_blk_nums (List of int): Number of blocks for each decoder.
    """

    def __init__(self,
                 *args,
                 train_size=(1, 3, 256, 256),
                 fast_imp=False,
                 **kwargs):
        Local_Base.__init__(self)
        StereoNAFNetFreqDEBlockSCAM.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(
                base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# Components for NAFNet


class NAFBlock(BaseModule):
    """NAFNet's Block in paper.

    Simple gate will shrink the channel to a half.
    To keep the number of channels,
    it expands the channels first.

    Args:
        in_channels (int): number of channels
        DW_Expand (int): channel expansion factor for part 1
        FFN_Expand (int): channel expansion factor for part 2
        drop_out_rate (float): drop out ratio
    """

    def __init__(self,
                 in_channels,
                 DW_Expand=2,
                 FFN_Expand=2,
                 drop_out_rate=0.):
        super().__init__()

        # Part 1

        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True),
        )

        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)

        # Part 2

        ffn_channel = FFN_Expand * in_channels
        self.conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True)

        # Simple Gate
        self.sg = SimpleGate()

        # Layer Normalization
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

        # Dropout
        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Feature weight ratio
        self.beta = nn.Parameter(
            torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(
            torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        """Forward Function.

        Args:
            inp: input tensor image
        """
        x = inp
        # part 1
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        # part 2
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        out = y + x * self.gamma

        return out


class SimpleGate(BaseModule):
    """The Simple Gate in "Simple Baseline for Image Restoration".

    Args:
        x: input tensor feature map with (B, 2 * C, H, W)

    Return:
        x1 * x2
        (where x1, x2 are two separate parts by simple split x to [B, C, H, W])
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
