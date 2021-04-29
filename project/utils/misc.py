import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/0c67dce524b2eb94dc3587ff2832e28f11440cae/utils/utils.py#L26
def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
