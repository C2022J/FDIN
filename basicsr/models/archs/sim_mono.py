import torch
import torch.nn as nn


# 假设 LayerNorm2d 已经定义在别处
from basicsr.models.archs_stereo.sim import LayerNorm2d

class SIM_Mono(nn.Module):
    '''
    Monocular Self-Attention Module (修改自 SCAM)
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm = LayerNorm2d(c)  # 只需要一个 norm

        # 将 l_proj1, r_proj1 替换为 q_proj, k_proj
        self.q_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.k_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        # 将 l_proj2 替换为 v_proj (r_proj2 移除)
        self.v_proj = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        # 只需要一个 beta (gamma 移除)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):  # 输入从 (x_l, x_r) 变为 (x)

        # Q, K, V 都来自同一个 x
        Q = self.q_proj(self.norm(x)).permute(0, 2, 3, 1)  # B, H, W, c
        K_T = self.k_proj(self.norm(x)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)
        V = self.v_proj(x).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q, K_T) * self.scale  # 自注意力

        F_self = torch.matmul(torch.softmax(attention, dim=-1), V)  # B, H, W, c

        # scale
        F_self = F_self.permute(0, 3, 1, 2) * self.beta

        # 返回单个输出
        return x + F_self