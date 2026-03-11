# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import time
import numpy as np
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch.nn as nn
import torch
import torch.nn.functional as F

# 确保路径和类名正确，根据您实际的项目结构调整
from basicsr.models.archs_stereo.stereo_fdin_arch import StereoFDIN
import utils
from natsort import natsorted
from glob import glob
from torch.cuda.amp import autocast
import lpips
import yaml

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def main():
    parser = argparse.ArgumentParser(description='Stereo Image Restoration Testing')

    parser.add_argument('--input_dir', default='../../stereo-derain/data/k15/test/', type=str,
                        help='Directory of validation images')
    parser.add_argument('--result_dir', default='../experiments/Stereo_FDIN_Holopix50/results/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights',
                        # default='../experiments/Stereo_FDIN_StereoWaterdrop/models/stereo_fdin_stereowaterdrop_net_g.pth',
                        default='../experiments/Stereo_FDIN_KITTI2015/models/stereo_fdin_kitti2015_net_g.pth',
                        # default='../experiments/Stereo_FDIN_Holopix50/models/stereo_fdin_holopix50_net_g.pth',
                        type=str)
    parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = 'Options/Train_Stereo_FDIN.yml'
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    x = yaml.load(open(yaml_file, mode='r', encoding='utf-8'), Loader=Loader)
    # 移除可能不需要的 type 字段，防止传给模型报错
    if 'type' in x['network_g']:
        x['network_g'].pop('type')

    ####### Load Model #######
    model_restoration = StereoFDIN(**x['network_g'])
    checkpoint = torch.load(args.weights)
    model_restoration.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration.eval()

    # --- 正常指标评估模式 ---
    alex = lpips.LPIPS(net='alex').cuda()
    result_dir = args.result_dir
    left_result_dir = os.path.join(result_dir, 'left')
    right_result_dir = os.path.join(result_dir, 'right')

    if args.save_images:
        os.makedirs(left_result_dir, exist_ok=True)
        os.makedirs(right_result_dir, exist_ok=True)

    files_left = natsorted(glob(os.path.join(args.input_dir, 'low', 'left', '*')))
    files_gt_left = natsorted(glob(os.path.join(args.input_dir, 'gt', 'left', '*')))

    psnr_left, psnr_right, ssim_left, ssim_right, pips_left, pips_right = [], [], [], [], [], []
    total_inference_time = 0

    with torch.no_grad():
        for file_left, file_gt_left in tqdm(zip(files_left, files_gt_left), total=len(files_gt_left),
                                            desc="Evaluating metrics"):
            file_right = file_left.replace('left', 'right')
            file_gt_right = file_gt_left.replace('left', 'right')

            img_left = np.float32(utils.load_img(file_left)) / 255.
            img_right = np.float32(utils.load_img(file_right)) / 255.
            img_gt_left = np.float32(utils.load_img(file_gt_left)) / 255.
            img_gt_right = np.float32(utils.load_img(file_gt_right)) / 255.

            img_left_tensor = torch.from_numpy(img_left).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            img_right_tensor = torch.from_numpy(img_right).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            img_left_gt_tensor = torch.from_numpy(img_gt_left).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            img_right_gt_tensor = torch.from_numpy(img_gt_right).unsqueeze(0).permute(0, 3, 1, 2).cuda()

            with autocast():
                start_time = time.time()
                # 注意：这里移除了 profile=True 参数
                restored_left, restored_right = model_restoration(img_left_tensor, img_right_tensor)
            end_time = time.time()
            total_inference_time += end_time - start_time

            restored_left = torch.clamp(restored_left, 0, 1)
            restored_right = torch.clamp(restored_right, 0, 1)

            pips_left.append(alex(img_left_gt_tensor, restored_left, normalize=True).item())
            pips_right.append(alex(img_right_gt_tensor, restored_right, normalize=True).item())

            restored_left_np = restored_left.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            restored_right_np = restored_right.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

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
                utils.save_img_16bit(save_file, restored_left_np)
                save_file = os.path.join(right_result_dir, os.path.split(file_right)[-1])
                utils.save_img_16bit(save_file, restored_right_np)

            torch.cuda.empty_cache()
            del restored_left, restored_right

    psnr_left, psnr_right = np.array(psnr_left), np.array(psnr_right)
    ssim_left, ssim_right = np.array(ssim_left), np.array(ssim_right)
    pips_left, pips_right = np.array(pips_left), np.array(pips_right)

    print("\n" + "=" * 50)
    print("Left:  PSNR {:4f} | SSIM {:4f} | LPIPS {:4f}".format(
        np.mean(psnr_left), np.mean(ssim_left), np.mean(pips_left)))
    print("Right: PSNR {:4f} | SSIM {:4f} | LPIPS {:4f}".format(
        np.mean(psnr_right), np.mean(ssim_right), np.mean(pips_right)))

    # 计算左右平均值
    avg_psnr = (np.mean(psnr_left) + np.mean(psnr_right)) / 2
    avg_ssim = (np.mean(ssim_left) + np.mean(ssim_right)) / 2
    avg_lpips = (np.mean(pips_left) + np.mean(pips_right)) / 2

    print("-" * 50)
    print("Avg:   PSNR {:4f} | SSIM {:4f} | LPIPS {:4f}".format(
        avg_psnr, avg_ssim, avg_lpips))

    print("Total inference time (pure model forward): {:.2f} seconds".format(total_inference_time))
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()