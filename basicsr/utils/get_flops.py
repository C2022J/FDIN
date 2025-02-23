import torch
from basicsr.models.archs_stereo.stereo_bbnet_arch import StereoBBNet
from basicsr.models.archs_stereo.stereo_nafnet_freq_deblock_scam_arch import StereoNAFNetFreqDEBlockSCAM
from utils_modelsummary import get_model_activation, get_model_flops

# 创建 StereoBBNet 模型
model = StereoNAFNetFreqDEBlockSCAM(
    img_channels=3,
    mid_channels=32,
    middle_blk_num=2,
    enc_blk_nums=[2, 2, 2, 2],
    enc_blk_freq_deblock_scam_idx=[0, 1, 2, 3],
    dec_blk_nums=[2, 2, 2, 2]).cuda()

# 输入维度设置
input_dim = (3, 128, 128)  # 输入的图像维度

# 创建左图和右图的输入
left_input = torch.randn(1, 3, 128, 128).cuda()  # batch size = 1, 形状 (3, 128, 128)
right_input = torch.randn(1, 3, 128, 128).cuda()

# 定义输入构造器
def input_constructor(input_res):
    channels, height, width = input_res  # 只解包 3 个维度
    batch_size = 1  # 设置默认的 batch_size，或者可以根据需要动态调整
    left_input = torch.randn(batch_size, 3, 128, 128).cuda()
    right_input = torch.randn(batch_size, 3, 128, 128).cuda()
    return {'left': left_input, 'right': right_input}

# 计算模型激活和 FLOPs
with torch.no_grad():

    activations, num_conv2d = get_model_activation(model, input_dim, input_constructor)
    print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations/10**6))
    print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

    # 计算 FLOPs
    # flops = get_model_flops(model, input_constructor, False)
    # print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops/10**9))

    # 计算参数量
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters/10**6))
