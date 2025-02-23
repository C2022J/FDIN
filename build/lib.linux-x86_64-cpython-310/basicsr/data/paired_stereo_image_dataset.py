from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop_DP, random_augmentation, paired_stereo_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP

import random
import numpy as np
import torch
import cv2


class Dataset_PairedStereoImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedStereoImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
            'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path_left = self.paths[index]['gt_path']
        gt_path_right = self.paths[index]['gt_path'].replace('left', 'right')
        img_bytes_left = self.file_client.get(gt_path_left, 'gt_left')
        img_bytes_right = self.file_client.get(gt_path_right, 'gt_right')
        try:
            img_gt_left = imfrombytes(img_bytes_left, float32=True)
            img_gt_right = imfrombytes(img_bytes_right, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_left))

        lq_path_left = self.paths[index]['lq_path']
        lq_path_right = self.paths[index]['lq_path'].replace('left', 'right')
        img_bytes_left = self.file_client.get(lq_path_left, 'lq_left')
        img_bytes_right = self.file_client.get(lq_path_right, 'lq_right')
        try:
            img_lq_left = imfrombytes(img_bytes_left, float32=True)
            img_lq_right = imfrombytes(img_bytes_right, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_left))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt_left, img_lq_left = padding(img_gt_left, img_lq_left, gt_size)
            img_gt_right, img_lq_right = padding(img_gt_right, img_lq_right, gt_size)

            # random crop
            img_gt_left, img_lq_left, img_gt_right, img_lq_right = paired_stereo_random_crop(img_gt_left, img_lq_left,
                                                                                             img_gt_right, img_lq_right,
                                                                                             gt_size, scale,
                                                                                             gt_path_left)

            # flip, rotation augmentations
            # if self.geometric_augs:
            #     img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_left, img_lq_left = img2tensor([img_gt_left, img_lq_left],
                                              bgr2rgb=True,
                                              float32=True)
        img_gt_right, img_lq_right = img2tensor([img_gt_right, img_lq_right],
                                                bgr2rgb=True,
                                                float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq_left, self.mean, self.std, inplace=True)
            normalize(img_lq_right, self.mean, self.std, inplace=True)
            normalize(img_gt_left, self.mean, self.std, inplace=True)
            normalize(img_gt_right, self.mean, self.std, inplace=True)

        return {
            'lq_left': img_lq_left,
            'lq_right': img_lq_right,
            'gt_left': img_gt_left,
            'gt_right': img_gt_right,
            'lq_path_left': lq_path_left,
            'lq_path_right': lq_path_right,
            'gt_path_left': gt_path_left,
            'gt_path_right': gt_path_right,
        }

    def __len__(self):
        return len(self.paths)
