import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from basicsr.models.archs_stereo.fcib import FCIB

class StereoFDIN(BaseModule):
    def __init__(self,
                 img_channels=3,
                 mid_channels=8,
                 middle_blk_num=1,
                 enc_blk_fcib_idx=[],
                 enc_blk_nums=[],
                 dec_blk_nums=[],
                 scale=1):
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
        self.scale = scale

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Define FCIB layer ModuleList
        self.fcib = nn.ModuleList()

        chan = mid_channels
        self.enc_blk_fcib_idx = enc_blk_fcib_idx

        for idx, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[FEB(chan) for _ in range(num)]))
            if idx in enc_blk_fcib_idx:
                self.fcib.append(FCIB(chan))  # Use FCIB module
            else:
                self.fcib.append(nn.Identity())

            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))

            chan = chan * 2

        self.middle_blks = nn.Sequential(*[FEB(chan) for _ in range(middle_blk_num)])

        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)))
            chan = chan // 2

            self.decoders.append(nn.Sequential(*[FEB(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, left, right):
        # 
        # This diagram illustrates the U-Net like structure with encoder, middle blocks, and decoder, 
        # showing where the FCIB modules would be integrated in the encoder path.
        pred_left, pred_right = self.forward_stereo(left, right)
        return pred_left, pred_right

    def forward_stereo(self, left, right):
        B, C, H, W = left.shape

        left_padded = self.check_image_size(left)
        right_padded = self.check_image_size(right)

        x_left = self.intro(left_padded)
        x_right = self.intro(right_padded)

        encs_left = []
        encs_right = []

        for idx, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x_left = encoder(x_left)
            x_right = encoder(x_right)

            if idx in self.enc_blk_fcib_idx:
                # Call FCIB
                x_left, x_right = self.fcib[idx](x_left, x_right)

            encs_left.append(x_left)
            encs_right.append(x_right)

            # Downsampling
            x_left = down(x_left)
            x_right = down(x_right)

        x_left = self.middle_blks(x_left)
        x_right = self.middle_blks(x_right)

        for idx, (decoder, up, enc_skip_left, enc_skip_right) in enumerate(
                zip(self.decoders, self.ups, encs_left[::-1],
                    encs_right[::-1])):
            x_left = up(x_left)
            x_right = up(x_right)

            x_left = x_left + enc_skip_left
            x_right = x_right + enc_skip_right

            x_left = decoder(x_left)
            x_right = decoder(x_right)

        x_left = self.ending(x_left)
        x_right = self.ending(x_right)

        pred_left = x_left + left_padded
        pred_right = x_right + right_padded

        pred_left = self.restore_image_size(pred_left)
        pred_right = self.restore_image_size(pred_right)

        return pred_left, pred_right

    def check_image_size(self, x):
        _, _, h, w = x.size()
        self.mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        self.mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, self.mod_pad_w, 0, self.mod_pad_h), 'replicate')
        return x

    def restore_image_size(self, x):
        _, _, h, w = x.size()
        x = x[:, :, 0:h - self.mod_pad_h, 0:w - self.mod_pad_w]
        return x

class FEB(BaseModule):
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

        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
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
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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