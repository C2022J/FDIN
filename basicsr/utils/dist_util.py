# Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py  # noqa: E501
import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta  # <--- 1. 新增：导入 timedelta


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)

    # --- 2. 新增：强制设置超时时间 ---
    if 'timeout' not in kwargs:
        # 设置为 3 小时 (10800 秒)，足够跑完任何验证集了
        kwargs['timeout'] = timedelta(hours=3)
    # ------------------------------

    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None, **kwargs):  # <--- 3. 修改：加上 **kwargs 接收可能的参数
    """Initialize slurm distributed training environment.
    ... (注释省略) ...
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)

    # --- 4. 新增：Slurm 模式也加上超时设置 ---
    timeout = kwargs.get('timeout', timedelta(hours=3))

    dist.init_process_group(backend=backend, timeout=timeout)


def get_dist_info():
    # ... (保持不变) ...
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    # ... (保持不变) ...
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper