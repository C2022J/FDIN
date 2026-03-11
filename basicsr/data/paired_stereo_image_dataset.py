import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_stereo_random_crop, paired_stereo_center_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import numpy as np

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

            self.paths = []
            meta_info_path = os.path.join(self.gt_folder, 'meta_info.txt')

            print(f"正在手动加载路径，以修复多点号文件名问题: {meta_info_path}")
            if not os.path.exists(meta_info_path):
                raise FileNotFoundError(f"找不到 meta_info.txt: {meta_info_path}")

            with open(meta_info_path, 'r') as fin:
                for line in fin:
                    gt_key = line.strip()
                    self.paths.append({
                        'gt_path': gt_key,
                        'lq_path': gt_key
                    })

            print(f"成功加载 {len(self.paths)} 对数据。")

        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
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

        def read_img_robust(path, client_key):
            img_bytes = self.file_client.get(path, client_key)

            if img_bytes is None and not path.lower().endswith(('.png', '.jpg')):
                path_png = path + '.png'
                img_bytes = self.file_client.get(path_png, client_key)
                if img_bytes is not None:
                    return img_bytes, path_png

            if img_bytes is None and not path.lower().endswith(('.png', '.jpg')):
                path_jpg = path + '.jpg'
                img_bytes = self.file_client.get(path_jpg, client_key)
                if img_bytes is not None:
                    return img_bytes, path_jpg

            if img_bytes is None:
                if path.lower().endswith('.png'):
                    path_jpg = path[:-4] + '.jpg'
                    img_bytes = self.file_client.get(path_jpg, client_key)
                    if img_bytes is not None:
                        return img_bytes, path_jpg
                elif path.lower().endswith('.jpg'):
                    path_png = path[:-4] + '.png'
                    img_bytes = self.file_client.get(path_png, client_key)
                    if img_bytes is not None:
                        return img_bytes, path_png

            return img_bytes, path

        gt_path_left_raw = self.paths[index]['gt_path']
        img_bytes_left, gt_path_left = read_img_robust(gt_path_left_raw, 'gt')

        if img_bytes_left is None:
            raise Exception(f"GT Left 读取失败: {gt_path_left_raw} (已尝试 png/jpg)")

        gt_path_right = gt_path_left.replace('left', 'right')
        img_bytes_right = self.file_client.get(gt_path_right, 'gt')

        if img_bytes_right is None:
            img_bytes_right, gt_path_right = read_img_robust(gt_path_right, 'gt')
            if img_bytes_right is None:
                raise Exception(f"GT Right 读取失败: {gt_path_right}")

        try:
            img_gt_left = imfrombytes(img_bytes_left, float32=True)
            img_gt_right = imfrombytes(img_bytes_right, float32=True)
        except:
            raise Exception("GT 图片解码失败: {}".format(gt_path_left))

        lq_path_left_raw = self.paths[index]['lq_path']
        img_bytes_left, lq_path_left = read_img_robust(lq_path_left_raw, 'lq')

        if img_bytes_left is None:
            raise Exception(f"LQ Left 读取失败: {lq_path_left_raw} (已尝试 png/jpg)")

        lq_path_right = lq_path_left.replace('left', 'right')
        img_bytes_right = self.file_client.get(lq_path_right, 'lq')

        if img_bytes_right is None:
            img_bytes_right, lq_path_right = read_img_robust(lq_path_right, 'lq')
            if img_bytes_right is None:
                raise Exception(f"LQ Right 读取失败: {lq_path_right}")

        try:
            img_lq_left = imfrombytes(img_bytes_left, float32=True)
            img_lq_right = imfrombytes(img_bytes_right, float32=True)
        except:
            raise Exception("LQ 图片解码失败: {}".format(lq_path_left))

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            img_gt_left, img_lq_left = padding(img_gt_left, img_lq_left, gt_size)
            img_gt_right, img_lq_right = padding(img_gt_right, img_lq_right, gt_size)

            # Random Crop
            img_gt_left, img_lq_left, img_gt_right, img_lq_right = paired_stereo_random_crop(
                img_gt_left, img_lq_left,
                img_gt_right, img_lq_right,
                gt_size, scale,
                gt_path_left)

            if self.opt.get('flip_RGB', False):
                perm = np.random.permutation(3)
                img_gt_left = img_gt_left[:, :, perm]
                img_gt_right = img_gt_right[:, :, perm]
                img_lq_left = img_lq_left[:, :, perm]
                img_lq_right = img_lq_right[:, :, perm]

            if self.geometric_augs:
                img_gt_left = np.ascontiguousarray(img_gt_left)
                img_gt_right = np.ascontiguousarray(img_gt_right)
                img_lq_left = np.ascontiguousarray(img_lq_left)
                img_lq_right = np.ascontiguousarray(img_lq_right)
                img_list = [img_gt_left, img_lq_left, img_gt_right, img_lq_right]
                img_list = augment(img_list, hflip=True, rotation=True)
                img_gt_left, img_lq_left, img_gt_right, img_lq_right = img_list

        if self.opt['phase'] == 'val':
            gt_size = self.opt.get('gt_size', None)

            if gt_size is not None:
                img_gt_left, img_lq_left = padding(img_gt_left, img_lq_left, gt_size)
                img_gt_right, img_lq_right = padding(img_gt_right, img_lq_right, gt_size)

                img_gt_left, img_lq_left, img_gt_right, img_lq_right = paired_stereo_center_crop(
                    img_gt_left, img_lq_left,
                    img_gt_right, img_lq_right,
                    gt_size, scale
                )

        img_gt_left, img_lq_left = img2tensor([img_gt_left, img_lq_left], bgr2rgb=True, float32=True)
        img_gt_right, img_lq_right = img2tensor([img_gt_right, img_lq_right], bgr2rgb=True, float32=True)

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
