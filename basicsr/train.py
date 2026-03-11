import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
from collections import OrderedDict

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
        is not None) and (opt['logger']['wandb'].get('project')
                          is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase.split('_')[0] == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
            val_loaders.append(val_loader)
        else:
            pass  # ignore other phases

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
            'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    while current_iter <= total_iters:
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            ### ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size,
                                                                                          mini_batch_size * torch.cuda.device_count()))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            ###-------------------------------------------

            # 单目数据流
            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # =============================================================================
            # VALIDATION & SAVING LOGIC (ADAPTED)
            # =============================================================================
            if opt['rank'] == 0 and opt.get('val') is not None and val_loaders and current_iter > 0:
                val_settings = opt['val']

                # 1. Check "full_val_freq" (Highest priority)
                full_val_freq = val_settings.get('full_val_freq')
                is_full_val_iter = (full_val_freq is not None and current_iter % full_val_freq == 0)

                # 2. Check "frequent" validation (Stage-based or fixed freq)
                is_frequent_val_iter = False
                if not is_full_val_iter:  # Only check frequent if not doing full validation
                    val_stages = val_settings.get('val_stages', None)
                    if val_stages:
                        # Case 1: Use val_stages logic
                        for stage in val_stages:
                            if stage['start'] <= current_iter <= stage['end']:
                                stage_freq = stage.get('freq')
                                if stage_freq is not None and current_iter % stage_freq == 0:
                                    is_frequent_val_iter = True
                    else:
                        # Case 2: Fallback to legacy val_freq
                        frequent_val_freq = val_settings.get('val_freq')
                        if frequent_val_freq is not None and current_iter % frequent_val_freq == 0:
                            is_frequent_val_iter = True

                # 3. Execute Validation
                if is_full_val_iter or is_frequent_val_iter:
                    log_format_config = val_settings.get('log_format')
                    rgb2bgr = val_settings.get('rgb2bgr', True)
                    use_image = val_settings.get('use_image', True)

                    # Helper function to format logging table
                    def format_and_log_table(results_dict, format_config):
                        if not format_config:
                            return
                        dataset_names = list(results_dict.keys())
                        if not dataset_names: return

                        COLUMN_WIDTH = 18
                        header_line = f'{"Metric":<17}'
                        for name in dataset_names:
                            header_line += f'{name:^{COLUMN_WIDTH}}'

                        if len(dataset_names) > 1:
                            header_line += f'{"Average":^{COLUMN_WIDTH}}'

                        row_lines = []
                        for row_title, metric_keys in format_config.items():
                            current_row_str = f'{row_title:<17}'
                            psnr_vals_for_avg = []
                            ssim_vals_for_avg = []

                            for name in dataset_names:
                                metrics = results_dict[name]
                                # Handle list of keys or single key
                                psnr_key = metric_keys.get('psnr')
                                if isinstance(psnr_key, list):
                                    psnr_val = sum(metrics.get(k, 0) for k in psnr_key) / len(
                                        psnr_key) if psnr_key else 0
                                else:
                                    psnr_val = metrics.get(psnr_key, 0)

                                ssim_key = metric_keys.get('ssim')
                                if isinstance(ssim_key, list):
                                    ssim_val = sum(metrics.get(k, 0) for k in ssim_key) / len(
                                        ssim_key) if ssim_key else 0
                                else:
                                    ssim_val = metrics.get(ssim_key, 0)

                                psnr_vals_for_avg.append(psnr_val)
                                ssim_vals_for_avg.append(ssim_val)
                                cell_str = f'{psnr_val:.2f}/{ssim_val:.4f}'
                                current_row_str += f'{cell_str:^{COLUMN_WIDTH}}'

                            if len(dataset_names) > 1:
                                avg_psnr = sum(psnr_vals_for_avg) / len(psnr_vals_for_avg)
                                avg_ssim = sum(ssim_vals_for_avg) / len(ssim_vals_for_avg)
                                avg_cell_str = f'{avg_psnr:.2f}/{avg_ssim:.4f}'
                                current_row_str += f'{avg_cell_str:^{COLUMN_WIDTH}}'

                            row_lines.append(current_row_str)

                        separator = '-' * len(header_line)
                        logger.info(separator)
                        logger.info(header_line)
                        for line in row_lines:
                            logger.info(line)
                        logger.info(separator)

                    if is_full_val_iter:
                        logger.info(
                            f'---------- Iter {current_iter}, Starting FULL validation on {len(val_loaders)} datasets... ----------')
                        full_results = OrderedDict()
                        for val_loader in val_loaders:
                            dataset_name = val_loader.dataset.opt['name']
                            metric_results = model.validation(val_loader, current_iter, tb_logger,
                                                              val_settings['save_img'], rgb2bgr, use_image)
                            full_results[dataset_name] = metric_results

                        if log_format_config:
                            format_and_log_table(full_results, log_format_config)

                    elif is_frequent_val_iter:
                        # Only validate the first dataset for frequent validation
                        val_loader = val_loaders[0]
                        dataset_name = val_loader.dataset.opt['name']
                        logger.info(
                            f'---------- Iter {current_iter}, Starting FREQUENT/STAGE validation on {dataset_name}... ----------')
                        metric_results = model.validation(val_loader, current_iter, tb_logger,
                                                          val_settings['save_img'], rgb2bgr, use_image)
                        if log_format_config:
                            format_and_log_table({dataset_name: metric_results}, log_format_config)

                    # Save checkpoint immediately after validation
                    logger.info(f'Saving models and training states at iter {current_iter}.')
                    model.save(epoch, current_iter)

            # =============================================================================

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter
        epoch += 1

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)

    # Final validation
    if opt.get('val') is not None and val_loaders:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()