import torch
import torch.nn as nn
import torch.nn.functional as F
from .res2tormer_block import TransformerBlock

class BasicLayer(nn.Module):
    def __init__(self,
                 dim, num_heads, bias=False, LayerNorm_type='WithBias', depth=2, ffn_expansion_factor=2.66, num_blocks=1, att_types='DWConv', mlp_types='MLP',
                 ):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, bias=bias, LayerNorm_type=LayerNorm_type, ffn_expansion_factor=ffn_expansion_factor,num_blocks=num_blocks, att_types=att_types, mlp_types=mlp_types)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x



class Res2tormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8],
                 depths=[2, 2, 2, 2],
                 bias=False,
                 LayerNorm_type='WithBias',
                 ffn_expansion_factor=2.66,
                 # num_blocks=[1, 2],
                 num_blocks=[[1, 1], [1, 1], [1, 2], [1, 2], [1, 2], [1, 1], [1, 1]],
                 att_types='DWConv',
                 mlp_types='MLP',
                 upscale_factor=1,
                 ):
        super(Res2tormer, self).__init__()
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        # setting
        self.patch_size = 4
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(embed_dims[0], num_heads[0], bias, LayerNorm_type, depths[0], ffn_expansion_factor, num_blocks[0], att_types, mlp_types)

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(embed_dims[1], num_heads[1], bias, LayerNorm_type, depths[1], ffn_expansion_factor, num_blocks[1], att_types, mlp_types)

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(embed_dims[2], num_heads[2], bias, LayerNorm_type, depths[2],ffn_expansion_factor, num_blocks[2], att_types, mlp_types)

        self.patch_merge3 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.skip3 = nn.Conv2d(embed_dims[2], embed_dims[2], 1)

        self.layer4 = BasicLayer(embed_dims[3], num_heads[3], bias, LayerNorm_type, depths[3],ffn_expansion_factor, num_blocks[3], att_types, mlp_types)

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        # assert embed_dims[1] == embed_dims[3]
        # self.fusion1 = SKFusion(embed_dims[3])
        self.layer5 = BasicLayer(embed_dims[4], num_heads[4], bias, LayerNorm_type, depths[4],ffn_expansion_factor, num_blocks[4], att_types, mlp_types)

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[5], embed_dim=embed_dims[4])

        # assert embed_dims[0] == embed_dims[4]
        # self.fusion2 = SKFusion(embed_dims[4])
        self.layer6 = BasicLayer(embed_dims[5], num_heads[5], bias, LayerNorm_type, depths[5],ffn_expansion_factor, num_blocks[5], att_types, mlp_types)

        self.patch_split3 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[6], embed_dim=embed_dims[5])

        self.layer7 = BasicLayer(embed_dims[6], num_heads[6], bias, LayerNorm_type, depths[6],ffn_expansion_factor, num_blocks[6], att_types, mlp_types)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[6], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        skip3 = x

        x = self.patch_merge3(x)
        x = self.layer4(x)
        # skip3 = x

        x = self.patch_split1(x)

        x = x + self.skip3(skip3)
        x = self.layer5(x)
        x = self.patch_split2(x)

        x = x + self.skip2(skip2)
        x = self.layer6(x)
        x = self.patch_split3(x)

        # x = self.fusion2([x, self.skip1(skip1)]) + x
        x = x + self.skip1(skip1)
        x = self.layer7(x)
        x = self.patch_unembed(x)
        return x


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.forward_features(x)
        # K, B = torch.split(feat, (1, 3), dim=1)
        # x = K * x - B + x
        # y = x + feat
        out = x[:, :, :H, :W]
        return out
