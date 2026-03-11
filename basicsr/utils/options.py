import yaml
from collections import OrderedDict
from os import path as osp


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def parse(opt_path, is_train=True):
    """Parse option file.

    Args:
        opt_path (str): Option file path.
        is_train (str): Indicate whether in training or not. Default: True.

    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train

    # <<< MODIFIED START: 添加了对列表和字典的兼容处理 >>>
    # datasets
    for phase, dataset_opt in opt['datasets'].items():
        # 检查 dataset_opt 的类型
        if isinstance(dataset_opt, list):
            # 如果是列表 (例如我们的 val)，则遍历列表中的每个数据集
            for sub_dataset_opt in dataset_opt:
                current_phase = phase.split('_')[0]
                sub_dataset_opt['phase'] = current_phase
                if 'scale' in opt:
                    sub_dataset_opt['scale'] = opt['scale']
                if sub_dataset_opt.get('dataroot_gt') is not None:
                    sub_dataset_opt['dataroot_gt'] = osp.expanduser(sub_dataset_opt['dataroot_gt'])
                if sub_dataset_opt.get('dataroot_lq') is not None:
                    sub_dataset_opt['dataroot_lq'] = osp.expanduser(sub_dataset_opt['dataroot_lq'])
        else:
            # 如果是字典 (例如 train)，则按原样处理
            current_phase = phase.split('_')[0]
            dataset_opt['phase'] = current_phase
            if 'scale' in opt:
                dataset_opt['scale'] = opt['scale']
            if dataset_opt.get('dataroot_gt') is not None:
                dataset_opt['dataroot_gt'] = osp.expanduser(dataset_opt['dataroot_gt'])
            if dataset_opt.get('dataroot_lq') is not None:
                dataset_opt['dataroot_lq'] = osp.expanduser(dataset_opt['dataroot_lq'])
    # <<< MODIFIED END >>>


    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key
                                  or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    opt['path']['root'] = osp.abspath(
        osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments',
                                    opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root,
                                                  'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root,
                                                'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test_V2
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg