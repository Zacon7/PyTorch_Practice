# -*- coding: utf-8 -*-
"""
通用函数
"""

import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def transform_invert(img_tensor, train_transforms):
    """
    将data 进行反transforms操作
    :param img_tensor: tensor
    :param train_transforms: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(train_transforms):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), train_transforms.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(train_transforms):
        img_tensor = img_tensor.detach().numpy() * 255

    if img_tensor.shape[2] == 3:
        img_tensor = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img_tensor = Image.fromarray(img_tensor.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img_tensor


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
