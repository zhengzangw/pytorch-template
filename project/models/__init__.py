import torch.nn as nn
from torchvision import models

DICT = {
    # model_backbone
    "fcn_resnet50": models.segmentation.fcn_resnet50,
    "fcn_resnet101": models.segmentation.fcn_resnet101,
    "deeplabv3_resnet50": models.segmentation.deeplabv3_resnet50,
    "deeplabv3_resnet101": models.segmentation.deeplabv3_resnet101,
    # loss
    "cross_entropy": nn.CrossEntropyLoss,
}


def get_model(name):
    return DICT[name]
