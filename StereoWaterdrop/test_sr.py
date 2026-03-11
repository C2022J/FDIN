import time
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from basicsr.models.archs_stereo.fdin_sr_arch import FDINSR
from natsort import natsorted
from glob import glob
from torch.cuda.amp import autocast
import cv2
import yaml

# ======================= 核心：导入与训练时一致的指标计算函数 =======================
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
# ==============================================================================


def load_img(filepath):
    """一个简单的图像加载函数"""
    img = cv2.imread(filepath)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_img_16bit(filepath, img_float):
    """保存16-bit图像的辅助函数"""
    img_float = np.clip(img_float, 0.0, 1.0)
    img_16bit = (img_float * 65535).astype(np.uint16)
    img_bgr = cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)


# --- 1. 修改命令行参数 ---
parser = argparse.ArgumentParser(description='Automated Multi-Dataset Testing Script')
parser.add_argument('--input_root', default='../../stereo-derain/data/Flickr1024_Middlebury/test', type=str, help='Root directory of validation images')
parser.add_argument('--result_root', default='../experiments/Stereo_FDINSR_2x/results/test_x2', type=str, help='Root directory for results')
parser.add_argument('--weights', default='../experiments/Stereo_FDINSR_x4/models/stereo_fdinsr_x4.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save output images')
parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation')
args = parser.parse_args()


# --- 2. 在循环外加载模型 (提高效率) ---
####### Load yaml #######
yaml_file = 'Options/Train_Stereo_FDIN_SR.yml'
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
x = yaml.load(open(yaml_file, mode='r', encoding='utf-8'), Loader=Loader)
s = x['network_g'].pop('type')
##########################

model_restoration = FDINSR(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===> Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

if args.tta:
    # 打印消息更新为 4 种增强
    print("===> Test-Time Augmentation (TTA) with 4 flips is ENABLED.")
else:
    print("===> Test-Time Augmentation (TTA) is DISABLED.")

# --- 3. 定义要测试的数据集列表 ---
datasets = ['KITTI2015', 'KITTI2012', 'Middlebury', 'Flickr1024']

# --- 4. 开始主循环，遍历每个数据集 ---
for dataset_name in datasets:
    print("\n" + "="*100 + "\n")
    print(f"===== Evaluating Dataset: {dataset_name} =====")

    # --- 动态构建当前数据集的路径 ---
    current_input_dir = os.path.join(args.input_root, dataset_name)
    current_result_dir = os.path.join(args.result_root, dataset_name)

    left_result_dir = os.path.join(current_result_dir, 'left')
    right_result_dir = os.path.join(current_result_dir, 'right')

    if args.save_images:
        os.makedirs(left_result_dir, exist_ok=True)
        os.makedirs(right_result_dir, exist_ok=True)

    files_left = natsorted(glob(os.path.join(current_input_dir, 'lr_x4', 'left', '*')))
    files_gt_left = natsorted(glob(os.path.join(current_input_dir, 'GT', 'left', '*')))

    # --- 在每个数据集开始时重置指标列表 ---
    psnr_left, psnr_right, ssim_left, ssim_right = [], [], [], []
    total_inference_time = 0

    with torch.no_grad():
        for file_left, file_gt_left in tqdm(zip(files_left, files_gt_left), total=len(files_gt_left), desc=f'{dataset_name}'):

            # --- 内部评估逻辑保持不变 ---
            file_right = file_left.replace('left', 'right')
            file_gt_right = file_gt_left.replace('left', 'right')

            img_left = np.float32(load_img(file_left)) / 255.
            img_right = np.float32(load_img(file_right)) / 255.
            img_gt_left = np.float32(load_img(file_gt_left)) / 255.
            img_gt_right = np.float32(load_img(file_gt_right)) / 255.

            img_left_tensor = torch.from_numpy(img_left).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            img_right_tensor = torch.from_numpy(img_right).unsqueeze(0).permute(0, 3, 1, 2).cuda()

            with autocast():
                start_time = time.time()
                if args.tta:
                    # 初始化结果累加器
                    restored_left_sum = 0
                    restored_right_sum = 0

                    # ------------------- 4种翻转 (原有逻辑) -------------------
                    # 1. 原始 (Identity)
                    restored_left_v1, restored_right_v1 = model_restoration(img_left_tensor, img_right_tensor)
                    restored_left_sum += restored_left_v1
                    restored_right_sum += restored_right_v1

                    # 2. 水平翻转 (Horizontal Flip)
                    lq_l_hflip, lq_r_hflip = torch.flip(img_left_tensor, dims=[3]), torch.flip(img_right_tensor, dims=[3])
                    sr_l_v2, sr_r_v2 = model_restoration(lq_l_hflip, lq_r_hflip)
                    sr_l_v2, sr_r_v2 = torch.flip(sr_l_v2, dims=[3]), torch.flip(sr_r_v2, dims=[3]) # 反向翻转
                    restored_left_sum += sr_l_v2
                    restored_right_sum += sr_r_v2

                    # 3. 垂直翻转 (Vertical Flip)
                    lq_l_vflip, lq_r_vflip = torch.flip(img_left_tensor, dims=[2]), torch.flip(img_right_tensor, dims=[2])
                    sr_l_v3, sr_r_v3 = model_restoration(lq_l_vflip, lq_r_vflip)
                    sr_l_v3, sr_r_v3 = torch.flip(sr_l_v3, dims=[2]), torch.flip(sr_r_v3, dims=[2]) # 反向翻转
                    restored_left_sum += sr_l_v3
                    restored_right_sum += sr_r_v3

                    # 4. 水平+垂直翻转 (Horizontal + Vertical Flip)
                    lq_l_hvflip, lq_r_hvflip = torch.flip(img_left_tensor, dims=[2, 3]), torch.flip(img_right_tensor, dims=[2, 3])
                    sr_l_v4, sr_r_v4 = model_restoration(lq_l_hvflip, lq_r_hvflip)
                    sr_l_v4, sr_r_v4 = torch.flip(sr_l_v4, dims=[2, 3]), torch.flip(sr_r_v4, dims=[2, 3]) # 反向翻转
                    restored_left_sum += sr_l_v4
                    restored_right_sum += sr_r_v4

                    # 平均 4 个结果
                    restored_left = restored_left_sum / 4.0
                    restored_right = restored_right_sum / 4.0
                else:
                    # 原始非 TTA 逻辑保持不变
                    restored_left, restored_right = model_restoration(img_left_tensor, img_right_tensor)
                end_time = time.time()
                total_inference_time += end_time - start_time

            restored_left = torch.clamp(restored_left, 0, 1)
            restored_right = torch.clamp(restored_right, 0, 1)

            restored_left_np = restored_left.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored_right_np = restored_right.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

            gt_left_255 = img_gt_left * 255
            sr_left_255 = restored_left_np * 255
            gt_right_255 = img_gt_right * 255
            sr_right_255 = restored_right_np * 255

            psnr_left.append(calculate_psnr(gt_left_255, sr_left_255, crop_border=0, test_y_channel=False))
            ssim_left.append(calculate_ssim(gt_left_255, sr_left_255, crop_border=0, test_y_channel=False))
            psnr_right.append(calculate_psnr(gt_right_255, sr_right_255, crop_border=0, test_y_channel=False))
            ssim_right.append(calculate_ssim(gt_right_255, sr_right_255, crop_border=0, test_y_channel=False))

            if args.save_images:
                save_file = os.path.join(left_result_dir, os.path.split(file_left)[-1])
                save_img_16bit(save_file, restored_left_np)
                save_file = os.path.join(right_result_dir, os.path.split(file_right)[-1])
                save_img_16bit(save_file, restored_right_np)

            torch.cuda.empty_cache()
            del restored_left, restored_right

    # --- 在每个数据集循环的末尾打印当前数据集的结果 ---
    psnr_left, psnr_right, ssim_left, ssim_right = np.array(psnr_left), np.array(psnr_right), np.array(ssim_left), np.array(ssim_right)
    avg_psnr = (np.mean(psnr_left) + np.mean(psnr_right)) / 2
    avg_ssim = (np.mean(ssim_left) + np.mean(ssim_right)) / 2

    print(f"\n--- Results for {dataset_name} ---\n")
    print(f"Left: PSNR/SSIM:  {np.mean(psnr_left):.2f}/{np.mean(ssim_left):.4f}       "
          f"Avg: PSNR/SSIM:  {avg_psnr:.2f}/{avg_ssim:.4f}")

print("\n" + "="*100)
print("All evaluations finished.")