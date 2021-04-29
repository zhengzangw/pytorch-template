import torch.nn as nn
from torchvision import models

from .preact_resnet import PreActResNet18
from .toy import toy_conv

DICT = {
    # === classification ===
    "toy_conv": toy_conv,
    "resnet18": models.resnet18,
    "preact_resnet18_cifar10": PreActResNet18,
    "resnet50": models.resnet50,
    # === segmentation ===
    "fcn_resnet50": models.segmentation.fcn_resnet50,
    "fcn_resnet101": models.segmentation.fcn_resnet101,
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101,
    # === loss ===
    "cross_entropy": nn.CrossEntropyLoss,
}


def get_model(name):
    assert name in DICT, f"model {name} is not implemented!"
    return DICT[name]
