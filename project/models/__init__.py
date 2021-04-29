import torch.nn as nn
from torchvision import models

from .toy import *

DICT = {
    # model
    # - classification
    "toy_conv": toy_conv,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    # - segmentation
    "fcn_resnet50": models.segmentation.fcn_resnet50,
    "fcn_resnet101": models.segmentation.fcn_resnet101,
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101,
    # loss
    "cross_entropy": nn.CrossEntropyLoss,
}


def get_model(name):
    assert name in DICT, f"model {name} is not implemented!"
    return DICT[name]
