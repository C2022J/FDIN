# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import time
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

# <--- MODIFIED 1: 导入你的单目模型
from basicsr.models.archs.fdin_mono_arch import MonoNAFNetFreqDEBlockSCAM

import utils
from natsort import natsorted
from glob import glob
from torch.cuda.amp import autocast
import lpips
import yaml

# =============================================================================
# (可视化库导入保持不变)
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================

parser = argparse.ArgumentParser(description='Single Image Restoration using MonoNAFNet')  # <--- MODIFIED

parser.add_argument('--input_dir', default='../data/StereoWaterdrop/test/', type=str,help='Directory of validation images')
# parser.add_argument('--input_dir', default='../data/huawei/evaluation_crop/', type=str,help='Directory of validation images')
parser.add_argument('--result_dir',default='../experiments/Monoholopix50_Baseline_C_Mono-Interact_Left/results/test_mynt/left', type=str, help='Directory for results')
# parser.add_argument('--result_dir',default='../experiments/huawei/results/val', type=str, help='Directory for results')
parser.add_argument('--weights',default='../experiments_247/MonoStereoWaterdrop_Baseline_C_Mono-Interact_Left/models/net_g_77000.pth',type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--visualize_feature', action='store_true', help='Visualize the high-frequency feature map of the first image.')

args = parser.parse_args()


# =============================================================================
# (可视化辅助函数 'plot_3d_surface', 'visualize_deep_feature_map' 保持不变)
def plot_3d_surface(image_patch, title, z_lim_max=255):
    """绘制图像块的3D像素表面图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sh_x, sh_y = image_patch.shape
    x = np.arange(0, sh_y)
    y = np.arange(0, sh_x)
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, image_patch, cmap='viridis', rstride=1, cstride=1, antialiased=False)
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_zlabel('Value')
    ax.set_zlim(0, z_lim_max)
    ax.set_title(title, fontsize=16)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    print(f"正在显示 '{title}' 的3D可视化图... 请关闭图像窗口以继续。")
    plt.show()


def visualize_deep_feature_map(feature_map, target_size_hw):
    """将深层特征图处理成与原图尺寸相同的可视化灰度图"""
    aggregated_map = np.mean(feature_map, axis=0)
    target_size_wh = (target_size_hw[1], target_size_hw[0])
    upsampled_map = cv2.resize(aggregated_map, target_size_wh, interpolation=cv2.INTER_CUBIC)
    normalized_map = cv2.normalize(upsampled_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_map


# =============================================================================


# =============================================================================
# (Hook 函数 'get_features_hook' 保持不变)
captured_features = {}


def get_features_hook(module_name):
    def hook(module, input, output):
        low_freq_component = output[0]
        high_freq_component = output[1]
        captured_features[f'{module_name}_low'] = low_freq_component.detach()
        captured_features[f'{module_name}_high'] = high_freq_component.detach()

    return hook


# =============================================================================


####### Load yaml #######
yaml_file = 'Options/Train_Mono_FDIN.yml'
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
x = yaml.load(open(yaml_file, mode='r', encoding='utf-8'), Loader=Loader)
s = x['network_g'].pop('type')

####### Load Model #######
# <--- MODIFIED 4: 实例化你的单目模型
model_restoration = MonoNAFNetFreqDEBlockSCAM(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# =============================================================================
# 4. 注册Hook
# =============================================================================
if args.visualize_feature:
    # <--- MODIFIED 5: 更改Hook的目标模块路径和名称
    #
    # !! 关键注意 !!
    # 我假设你的 FreqDEBlockSCAM_Mono 模块内部有一个名为 'freq_split' 的子模块
    # (对应于旧的 'freq_split_left')。
    # 如果你给它取了别的名字，请在这里修改 'freq_split'
    #
    try:
        target_module = model_restoration.module.freq_deblock_scam[0].freq_split

        # 将 'encoder_0_left' 改为 'encoder_0'
        handle = target_module.register_forward_hook(get_features_hook('encoder_0'))
        print(f"Hook 已成功注册到模块: {target_module}")
    except AttributeError:
        print("\n\n!! Hook 注册失败 !!")
        print(f"无法在 {model_restoration.module.freq_deblock_scam[0]} 中找到名为 'freq_split' 的子模块。")
        print("请检查你的 FreqDEBlockSCAM_Mono 类的代码，确认频率分解模块的变量名并在此处更新。\n\n")
        args.visualize_feature = False  # 禁用可视化以允许测试继续
# =============================================================================

alex = lpips.LPIPS(net='alex').cuda()
result_dir = args.result_dir

# <--- MODIFIED 6: 移除 left/right 子目录，直接保存到 result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

# <--- MODIFIED 7: 你的YML配置显示测试数据在 '.../low/left' 和 '.../gt/left'
# 脚本已经在使用这些路径了，我们只重命名变量，移除 'right' 的逻辑
files_lq = natsorted(glob(os.path.join(args.input_dir, 'low', 'left', '*')))
files_gt = natsorted(glob(os.path.join(args.input_dir, 'gt', 'left', '*')))

# <--- MODIFIED 8: 移除 'right' 的指标列表
psnr_list, ssim_list, pips_list = [], [], []
total_inference_time = 0

with torch.no_grad():
    # <--- MODIFIED 9: 修改循环，只处理 lq 和 gt
    for file_lq, file_gt in tqdm(zip(files_lq, files_gt), total=len(files_gt)):

        # <--- MODIFIED 10: 移除 'right' 图像的加载
        img_lq = np.float32(utils.load_img(file_lq)) / 255.
        img_gt = np.float32(utils.load_img(file_gt)) / 255.

        # <--- MODIFIED 11: 移除 'right' 张量
        img_lq_tensor = torch.from_numpy(img_lq).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        img_gt_tensor = torch.from_numpy(img_gt).unsqueeze(0).permute(0, 3, 1, 2).cuda()

        # with autocast():
        start_time = time.time()
        # <--- MODIFIED 12: 使用单目模型进行推理 (单输入，单输出)
        restored = model_restoration(img_lq_tensor)
        end_time = time.time()
        total_inference_time += end_time - start_time

        # =============================================================================
        # 5. 触发可视化并退出
        # =============================================================================
        if args.visualize_feature:
            print("\n正在提取并可视化特征图...")

            # <--- MODIFIED 13: 更改捕获的特征名称
            low_freq_tensor = captured_features.get('encoder_0_low')
            high_freq_tensor = captured_features.get('encoder_0_high')

            # 1. 可视化原始输入图像
            original_gray_patch = cv2.cvtColor(np.uint8(img_lq * 255), cv2.COLOR_RGB2GRAY)  # 使用 img_lq
            plot_3d_surface(original_gray_patch, 'Original Input Patch (Grayscale)')

            # 2. 可视化捕获到的低频分量
            if low_freq_tensor is not None:
                feature_map_numpy = low_freq_tensor.cpu().numpy()[0]
                visualized_map = visualize_deep_feature_map(
                    feature_map=feature_map_numpy,
                    target_size_hw=(img_lq_tensor.shape[2], img_lq_tensor.shape[3])  # 使用 img_lq_tensor
                )
                plot_3d_surface(visualized_map, 'Low-Frequency Component (from Hook)')
            else:
                print("错误：未能捕获到低频特征！")

            # 3. 可视化捕获到的高频分量
            if high_freq_tensor is not None:
                feature_map_numpy = high_freq_tensor.cpu().numpy()[0]
                visualized_map = visualize_deep_feature_map(
                    feature_map=feature_map_numpy,
                    target_size_hw=(img_lq_tensor.shape[2], img_lq_tensor.shape[3])  # 使用 img_lq_tensor
                )
                plot_3d_surface(visualized_map, 'High-Frequency Component (from Hook)')
            else:
                print("错误：未能捕获到高频特征！")

            # 清理Hook并退出循环
            if 'handle' in locals():  # 检查 handle 是否成功创建
                handle.remove()
            print("可视化完成，已移除Hook并退出测试循环。")
            break  # 退出 for 循环
        # =============================================================================

        # <--- MODIFIED 14: 处理单个输出张量
        restored = torch.clamp(restored, 0, 1)

        # <--- MODIFIED 15: 计算单个输出的指标
        pips_list.append(alex(img_gt_tensor, restored, normalize=True).item())
        ssim_list.append(utils.SSIM(img_gt_tensor, restored).item())

        restored_np = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        psnr_list.append(utils.PSNR(img_gt, restored_np))

        if args.save_images:
            # <--- MODIFIED 16: 保存单个输出图像
            save_file = os.path.join(result_dir, os.path.split(file_lq)[-1])
            utils.save_img_16bit(save_file, restored_np)

        torch.cuda.empty_cache()
        # <--- MODIFIED 17: 删除单个张量
        del restored, img_lq_tensor, img_gt_tensor

# 如果不是因为可视化而提前退出，则正常打印指标
if not args.visualize_feature:
    # <--- MODIFIED 18: 转换和打印单个指标
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    pips_list = np.array(pips_list)

    print(
        "Metrics: PSNR {:2f} SSIM {:4f} LPIPS {:4f}".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(pips_list)))
    print("Total inference time: {:.2f} seconds".format(total_inference_time))