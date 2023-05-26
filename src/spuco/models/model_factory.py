import random
from enum import Enum
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50

from spuco.models import MLP, Bert, DistilBert, LeNet, SpuCoModel
from spuco.utils.random_seed import seed_randomness


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class SupportedModels(Enum):
    """
    Enum listing all supported models.
    """
    MLP = "mlp"
    LeNet = "lenet"
    BERT = "bert"
    DistilBERT = "distilbert"
    ResNet50 = "resnet50"

def model_factory(arch: str, input_shape: Tuple[int, int, int], num_classes: int):
    """
    Factory function to create a SpuCoModel based on the specified architecture.

    :param arch: The architecture name.
    :type arch: str
    :param input_shape: The shape of the input data in the format (channels, height, width).
    :type input_shape: Tuple[int, int, int]
    :param num_classes: The number of output classes.
    :type num_classes: int
    :return: A SpuCoModel instance.
    :rtype: SpuCoModel
    :raises NotImplementedError: If the specified architecture is not supported.
    """
    seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
    arch = SupportedModels(arch)
    channel = input_shape[0]
    image_size = (input_shape[1], input_shape[2])
    backbone = None
    representation_dim = -1 
    
    if arch == SupportedModels.MLP:
        backbone = MLP(channel * image_size[0] * image_size[1])
        representation_dim = backbone.representation_dim
    elif arch == SupportedModels.LeNet: 
        backbone = LeNet(channel=channel)
        representation_dim = backbone.representation_dim
    elif arch == SupportedModels.BERT:
        backbone = Bert.from_pretrained('bert-base-uncased')
        representation_dim = backbone.d_out
    elif arch == SupportedModels.DistilBERT:
        backbone = DistilBert.from_pretrained('distilbert-base-uncased')
        representation_dim = backbone.d_out
    elif arch == SupportedModels.ResNet50:
        backbone = resnet50(pretrained=True)
        representation_dim = backbone.fc.in_features
        backbone.fc = Identity()
    else:
        raise NotImplemented(f"Model {arch} not supported currently")
    return SpuCoModel(
        backbone=backbone, 
        representation_dim=representation_dim, 
        num_classes=num_classes
    )

