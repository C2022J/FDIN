## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import cv2
import math
from pytorch_msssim import ms_ssim,ssim
from skimage import metrics
from sklearn.metrics import mean_absolute_error
# from
def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def PSNR(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

# def SSIM(img1, img2):
#     return metrics.structural_similarity(img1, img2, data_range=1, multichannel=True)

def SSIM(img1, img2):
    return ssim(img1, img2, data_range=1)

def MS_SSIM(img1, img2):
    return ms_ssim(img1, img2, data_range=1)

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_img_16bit(filepath, img_float):
    img_float = np.clip(img_float, 0.0, 1.0)
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.png', '.tiff', '.tif']:
        # 16位保存
        img_16bit = (img_float * 65535).astype(np.uint16)
    else:
        # 8位保存 (包括.jpg, .jpeg等)
        img_16bit = (img_float * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img_bgr)
