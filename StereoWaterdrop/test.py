"""
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
"""
import time

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch

from basicsr.models.archs_stereo.stereo_nafnet_freq_deblock_scam_arch import StereoNAFNetFreqDEBlockSCAM
import utils
from natsort import natsorted
from glob import glob

# 设置可见的GPU为2和3
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import lpips
alex = lpips.LPIPS(net='alex').cuda()


parser = argparse.ArgumentParser(description='Single Image Defocus Deblurring using Restormer')

parser.add_argument('--input_dir', default='../data/Flickr1024_Middlebury/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../experiments/StereoWaterdrop_NAFNet_Freq_LFE×1_SCAM_deform_param_L1SSIMLoss_128_midchan=32_100000_SR2/results/test', type=str, help='Directory for results')
parser.add_argument('--weights', default='../experiments/StereoWaterdrop_NAFNet_Freq_LFE×1_SCAM_deform_param_L1SSIMLoss_128_midchan=32_100000_SR2/models/net_g_70000.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/StereoWaterdrop_NAFNet_Freq_DEBlock_SCAM.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = StereoNAFNetFreqDEBlockSCAM(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = args.result_dir

# 新增：创建左、右结果目录
left_result_dir = os.path.join(result_dir, 'left')
right_result_dir = os.path.join(result_dir, 'right')

if args.save_images:
    os.makedirs(left_result_dir, exist_ok=True)
    os.makedirs(right_result_dir, exist_ok=True)

files_left = natsorted(glob(os.path.join(args.input_dir, 'input', 'left', '*.png')))
files_gt_left = natsorted(glob(os.path.join(args.input_dir, 'gt', 'left', '*.png')))


psnr_left, psnr_right, ms_ssim_left, ms_ssim_right, pips_left, pips_right = [], [],  [], [], [], []

total_inference_time = 0

with torch.no_grad():
    for file_left, file_gt_left in tqdm(zip(files_left, files_gt_left), total=len(files_gt_left)):

        file_right = file_left.replace('left', 'right')
        file_gt_right = file_gt_left.replace('left', 'right')

        img_left = np.float32(utils.load_img(file_left))/255.
        img_right = np.float32(utils.load_img(file_right))/255.
        img_gt_left = np.float32(utils.load_img(file_gt_left))/255.
        img_gt_right = np.float32(utils.load_img(file_gt_right))/255.

        img_left_tensor = torch.from_numpy(img_left).unsqueeze(0).permute(0,3,1,2).cuda()
        img_right_tensor = torch.from_numpy(img_right).unsqueeze(0).permute(0,3,1,2).cuda()
        img_left_gt_tensor = torch.from_numpy(img_gt_left).unsqueeze(0).permute(0,3,1,2).cuda()
        img_right_gt_tensor = torch.from_numpy(img_gt_right).unsqueeze(0).permute(0,3,1,2).cuda()

        # 记录整个推理过程的时间
        start_time = time.time()
        restored_left, restored_right = model_restoration(img_left_tensor, img_right_tensor)
        # 结束推理时间的记录
        end_time = time.time()

        total_inference_time += end_time - start_time

        restored_left = torch.clamp(restored_left,0,1)
        restored_right = torch.clamp(restored_right,0,1)
        pips_left.append(alex(img_left_gt_tensor, restored_left, normalize=True).item())
        pips_right.append(alex(img_right_gt_tensor, restored_right, normalize=True).item())

        ms_ssim_left.append(utils.MS_SSIM(img_left_gt_tensor, restored_left).item())
        ms_ssim_right.append(utils.MS_SSIM(img_right_gt_tensor, restored_right).item())

        restored_left = restored_left.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_right = restored_right.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr_left.append(utils.PSNR(img_gt_left, restored_left))
        psnr_right.append(utils.PSNR(img_gt_right, restored_right))

        if args.save_images:
            save_file = os.path.join(left_result_dir, os.path.split(file_left)[-1])
            restored_left = np.uint8((restored_left*255).round())
            utils.save_img(save_file, restored_left)

            save_file = os.path.join(right_result_dir, os.path.split(file_right)[-1])
            restored_right = np.uint8((restored_right * 255).round())
            utils.save_img(save_file, restored_right)


psnr_left, psnr_right, ms_ssim_left, ms_ssim_right, pips_left, pips_right = np.array(psnr_left), np.array(psnr_right), np.array(ms_ssim_left), np.array(ms_ssim_right), np.array(pips_left), np.array(pips_right)


print("Left: PSNR {:4f} MS_SSIM {:4f} LPIPS {:4f}".format(np.mean(psnr_left), np.mean(ms_ssim_left), np.mean(pips_left)))
print("Right: PSNR {:4f} MS_SSIM {:4f} LPIPS {:4f}".format(np.mean(psnr_right), np.mean(ms_ssim_right), np.mean(pips_right)))
# 打印出整个推理过程的时间
print("Total inference time: {:.2f} seconds".format(total_inference_time))
