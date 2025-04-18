import os
import sys
# import cv2
import subprocess
import logging
from collections import defaultdict, OrderedDict

import torch
import torchvision
from fvcore.nn import FlopCountAnalysis, flop_count_table

def collect_env_info():

    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name


    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['TorchVision'] = torchvision.__version__
    # env_info['OpenCV'] = cv2.__version__


    return env_info


def collect_method_info(config, method, device):
    """Plot the basic infomation of supported methods"""
    T, C, H, W = config['in_seq_length'], config['channels'], config['img_height'], config['img_width']
    if config['method'].lower() in ['convlstm']:
        Hp, Wp = H // config['patch_size'], W // config['patch_size']
        Cp = config['patch_size'] ** 2 * C

        _tmp_input = torch.ones(1, int(config['in_seq_length'] + config['out_seq_length']), Hp, Wp, Cp).to(device, dtype=torch.float32)
        _tmp_flag = torch.ones(1,  config['out_seq_length'] - 1, Hp, Wp, Cp).to(device, dtype=torch.float32)
        input_dummy = (_tmp_input, _tmp_flag)

        # gpu = torch.cuda.get_device_properties(device)
        # print(f"GPU Name: {gpu.name}")
        # print(f"GPU Memory Total: {gpu.total_memory / 1024**3} GB")
        # print(f"GPU Memory Free: {torch.cuda.memory_allocated(device) / 1024**3} GB")
        # print(f"GPU Memory Used: {torch.cuda.memory_reserved(device) / 1024**3} GB")
    else:
        raise ValueError(f"Invalid method name {config['method']}")

    info = method.model.__repr__()
    flops = FlopCountAnalysis(method.model, input_dummy)
    flops = flop_count_table(flops)

    return info, flops

def print_log(message):
    print(message)
    logging.info(message)



def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu