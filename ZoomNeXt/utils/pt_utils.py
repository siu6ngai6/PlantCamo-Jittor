import logging
import os
import random
import numpy as np
import jittor as jt
from jittor import nn

LOGGER = logging.getLogger("main")

def customized_worker_init_fn(worker_id):
    worker_seed = jt.get_seed() % 2**32
    np.random.seed(worker_seed)

def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    jt.set_seed(seed)

def initialize_seed_cudnn(seed, deterministic):
    assert isinstance(deterministic, bool) and isinstance(seed, int)
    if seed >= 0:
        LOGGER.info(f"We will use a fixed seed {seed}")
    else:
        seed = np.random.randint(2**32)
        LOGGER.info(f"We will use a random seed {seed}")
    set_seed_for_lib(seed)

# def to_device(data, device="cuda"):
#     if isinstance(data, (tuple, list)):
#         return [to_device(item, device) for item in data]
#     elif isinstance(data, dict):
#         return {name: to_device(item, device) for name, item in data.items()}
#     elif isinstance(data, jt.Var):
#         return data
#     else:
#         raise TypeError(f"Unsupported type {type(data)}. Only support Tensor or tuple/list/dict containing Tensors.")

def frozen_bn_stats(model, freeze_affine=False):
    """
    将模型中所有的 BN 层设为 eval 模式（冻结均值和方差）。
    Args:
        model (nn.Module): 目标模型
        freeze_affine (bool): 是否同时冻结权重 (weight) 和偏置 (bias)
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm):
            m.eval()
            if freeze_affine:
                m.requires_grad_(False)