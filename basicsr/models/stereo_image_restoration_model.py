import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from torch.cuda.amp import autocast, GradScaler
from basicsr.models.archs_stereo import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
import os
import random
import torch.nn.functional as F
from functools import partial

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(target.size(0)).to(self.device)

        # Mixup 逻辑: 对输入和目标进行同样的线性插值
        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class StereoImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(StereoImageCleanModel, self).__init__(opt)

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.use_cwl = train_opt.get('use_cwl', False)
        if self.use_cwl and not self.use_task_loss:
            raise ValueError("Confidence-Weighted Loss (use_cwl=True) requires task_loss_opt to be configured.")
        logger = get_root_logger()
        logger.info(f"Using Confidence-Weighted Loss: {self.use_cwl}")

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq_left = data['lq_left'].to(self.device)
        self.lq_right = data['lq_right'].to(self.device)
        if 'gt_left' in data:
            self.gt_left = data['gt_left'].to(self.device)
            self.gt_right = data['gt_right'].to(self.device)

        if self.mixing_flag and 'gt_left' in data:
            gt_combined = torch.cat([self.gt_left, self.gt_right], dim=1)
            lq_combined = torch.cat([self.lq_left, self.lq_right], dim=1)

            gt_combined, lq_combined = self.mixing_augmentation(gt_combined, lq_combined)

            self.gt_left, self.gt_right = torch.chunk(gt_combined, 2, dim=1)
            self.lq_left, self.lq_right = torch.chunk(lq_combined, 2, dim=1)

    def feed_data(self, data):
        self.lq_left = data['lq_left'].to(self.device)
        self.lq_right = data['lq_right'].to(self.device)
        if 'gt_left' in data:
            self.gt_left = data['gt_left'].to(self.device)
            self.gt_right = data['gt_right'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        preds_left, preds_right = self.net_g(self.lq_left, self.lq_right)
        if not isinstance(preds_left, list):
            preds_left = [preds_left]
            preds_right = [preds_right]

        self.output_left = preds_left[-1]
        self.output_right = preds_right[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix_left = 0.
        l_pix_right = 0.
        for pred in preds_left:
            l_pix_left += self.cri_pix(pred, self.gt_left)
        for pred in preds_right:
            l_pix_right += self.cri_pix(pred, self.gt_right)
        l_pix = l_pix_left + l_pix_right
        loss_dict['l_pix'] = l_pix
        loss_dict['l_pix_left'] = l_pix_left
        loss_dict['l_pix_right'] = l_pix_right

        l_pix.backward()

        if self.opt['train']['use_grad_clip']:

            max_norm = self.opt['train'].get('grad_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq_left.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img_left = F.pad(self.lq_left, (0, mod_pad_w, 0, mod_pad_h), 'replicate')
        img_right = F.pad(self.lq_right, (0, mod_pad_w, 0, mod_pad_h), 'replicate')
        self.nonpad_test(img_left, img_right)
        _, _, h, w = self.output_left.size()
        self.output_left = self.output_left[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.output_right = self.output_right[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img_left=None, img_right=None):
        if img_left is None:
            img_left = self.lq_left
        if img_right is None:
            img_right = self.lq_right
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred_left, pred_right = self.net_g_ema(img_left, img_right)
            if isinstance(pred_left, list):
                pred_left = pred_left[-1]
                pred_right = pred_right[-1]
            self.output_left = pred_left
            self.output_right = pred_right
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred_left, pred_right = self.net_g(img_left, img_right)
            if isinstance(pred_left, list):
                pred_left = pred_left[-1]
            if isinstance(pred_right, list):
                pred_right = pred_right[-1]
            self.output_left = pred_left
            self.output_right = pred_right
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric + '_left': 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results.update({
                metric + '_right': 0
                for metric in self.opt['val']['metrics'].keys()
            })

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path_left'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img_left = tensor2img([visuals['result_left']], rgb2bgr=rgb2bgr)
            sr_img_right = tensor2img([visuals['result_right']], rgb2bgr=rgb2bgr)
            if 'gt_left' in visuals:
                gt_img_left = tensor2img([visuals['gt_left']], rgb2bgr=rgb2bgr)
                gt_img_right = tensor2img([visuals['gt_right']], rgb2bgr=rgb2bgr)
                del self.gt_left
                del self.gt_right

            del self.lq_left
            del self.lq_right
            del self.output_left
            del self.output_right
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                imwrite(sr_img_left, save_img_path)

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name + '_left'] += getattr(metric_module, metric_type)(sr_img_left,
                                                                                                   gt_img_left, **opt_)
                        self.metric_results[name + '_right'] += getattr(metric_module, metric_type)(sr_img_right,
                                                                                                    gt_img_right,
                                                                                                    **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name + '_left'] += getattr(metric_module, metric_type)(
                            visuals['result_left'], visuals['gt_left'], **opt_)
                        self.metric_results[name + '_right'] += getattr(metric_module, metric_type)(
                            visuals['result_right'], visuals['gt_right'], **opt_)
            cnt += 1

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt

            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

            return self.metric_results
        else:
            return {}

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}, '
        for metric, value in self.metric_results.items():
            log_str += f' # {metric}: {value:.4f}'

        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_left'] = self.lq_left.detach().cpu()
        out_dict['lq_right'] = self.lq_right.detach().cpu()
        out_dict['result_left'] = self.output_left.detach().cpu()
        out_dict['result_right'] = self.output_right.detach().cpu()
        if hasattr(self, 'gt_left'):
            out_dict['gt_left'] = self.gt_left.detach().cpu()
            out_dict['gt_right'] = self.gt_right.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)