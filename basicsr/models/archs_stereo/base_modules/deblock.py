from torch import nn
# 导入 DEConv 模块
from basicsr.models.archs_stereo.base_modules.deconv import DEConv

class DEBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DEBlock, self).__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res