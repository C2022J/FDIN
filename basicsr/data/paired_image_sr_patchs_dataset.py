from torchvision.transforms.functional import normalize
import torch
from PIL import Image
from torch.utils import data as data
from os import path as osp
from basicsr.data.transforms2 import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
import os
import numpy as np

def _to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

def _Mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)  # numpy float
    height = im1.shape[0]
    r_index = np.random.permutation(height)
    im1 = v * im1 + (1 - v) * im1[r_index, :]
    im2 = v * im2 + (1 - v) * im2[r_index, :]
    return im1, im2

class PairedStereoImageDatasetSRPatchs(data.Dataset):
    def __init__(self, opt):
        super(PairedStereoImageDatasetSRPatchs, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.dataroot = opt['dataroot_gt']  #

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.subfolders = sorted(os.listdir(self.dataroot))

        self.nums = len(self.subfolders)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        subfolder_name = self.subfolders[index]
        subfolder_path = os.path.join(self.dataroot, subfolder_name)

        gt_path_L = os.path.join(subfolder_path, 'hr0.png')
        gt_path_R = os.path.join(subfolder_path, 'hr1.png')
        lq_path_L = os.path.join(subfolder_path, 'lr0.png')
        lq_path_R = os.path.join(subfolder_path, 'lr1.png')

        img_bytes = self.file_client.get(gt_path_L, 'gt')
        try:
            img_gt_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_L))

        img_bytes = self.file_client.get(gt_path_R, 'gt')
        try:
            img_gt_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path_R))

        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))

        img_bytes = self.file_client.get(lq_path_R, 'lq')
        try:
            img_lq_R = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_R))

        img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        img_gt = np.ascontiguousarray(img_gt)
        img_lq = np.ascontiguousarray(img_lq)

        # augmentation for training
        if self.opt['phase'] == 'train':

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            if 'mix_up' in self.opt and self.opt['mix_up']:
                img_gt, img_lq = _Mixup(img_gt, img_lq)

            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                   self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)

            img_gt, img_lq = imgs

        img_gt_L_np = img_gt[..., :3]
        img_gt_R_np = img_gt[..., 3:]
        img_lq_L_np = img_lq[..., :3]
        img_lq_R_np = img_lq[..., 3:]

        img_gt_L, img_gt_R, img_lq_L, img_lq_R = img2tensor(
            [img_gt_L_np, img_gt_R_np, img_lq_L_np, img_lq_R_np],
            bgr2rgb=True,
            float32=True
        )

        if self.mean is not None or self.std is not None:
            normalize(img_lq_L, self.mean, self.std, inplace=True)
            normalize(img_lq_R, self.mean, self.std, inplace=True)
            normalize(img_gt_L, self.mean, self.std, inplace=True)
            normalize(img_gt_R, self.mean, self.std, inplace=True)

        return {
            'lq_left': img_lq_L,
            'lq_right': img_lq_R,
            'gt_left': img_gt_L,
            'gt_right': img_gt_R,
            'lq_path_left': lq_path_L,
            'gt_path_left': gt_path_L,
        }

    def __len__(self):
        return self.nums

class Dataset_StereoSR_Test(data.Dataset):
    def __init__(self, opt):
        super(Dataset_StereoSR_Test, self).__init__()
        self.opt = opt
        self.scale = opt['scale']
        self.phase = opt['phase']

        self.gt_root = opt['dataroot_gt']
        self.lq_root = opt['dataroot_lq']

        self.lq_left_root = osp.join(self.lq_root, 'left')
        self.lq_right_root = osp.join(self.lq_root, 'right')
        self.gt_left_root = osp.join(self.gt_root, 'left')
        self.gt_right_root = osp.join(self.gt_root, 'right')

        self.image_basenames = sorted(list(set([
            osp.splitext(fname)[0] for fname in os.listdir(self.lq_left_root)
        ])))

    def _find_image_path(self, directory, basename):
        supported_exts = ['.png', '.jpg']
        for ext in supported_exts:
            path = osp.join(directory, f"{basename}{ext}")
            if osp.exists(path):
                return path
        return None

    def __getitem__(self, index):
        basename = self.image_basenames[index]

        lq_left_path = self._find_image_path(self.lq_left_root, basename)
        lq_right_path = self._find_image_path(self.lq_right_root, basename)
        gt_left_path = self._find_image_path(self.gt_left_root, basename)
        gt_right_path = self._find_image_path(self.gt_right_root, basename)

        if not all([lq_left_path, lq_right_path, gt_left_path, gt_right_path]):
            print(f"Error: Missing one or more images for basename '{basename}'")
            if index == 0:
                raise FileNotFoundError(f"Cannot load even the first item for basename '{self.image_basenames[0]}'")
            return self.__getitem__(0)

        try:
            img_lr_left = Image.open(lq_left_path)
            img_lr_right = Image.open(lq_right_path)
            img_hr_left = Image.open(gt_left_path)
            img_hr_right = Image.open(gt_right_path)
        except Exception as e:
            print(f"Error loading images for basename {basename}: {e}")
            return self.__getitem__(0)

        img_lr_left = np.array(img_lr_left, dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left = np.array(img_hr_left, dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)

        img_lr_left_t = _to_tensor(img_lr_left)
        img_lr_right_t = _to_tensor(img_lr_right)
        img_hr_left_t = _to_tensor(img_hr_left)
        img_hr_right_t = _to_tensor(img_hr_right)

        return {
            'lq_left': img_lr_left_t,
            'lq_right': img_lr_right_t,
            'gt_left': img_hr_left_t,
            'gt_right': img_hr_right_t,
            'scale': self.scale,
            'lq_path_left': lq_left_path,
            'gt_path_left': gt_left_path
        }

    def __len__(self):
        return len(self.image_basenames)
