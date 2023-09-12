import random as random

import numpy as np
import torch as torch
import torch.nn.functional as F
from torch import nn

from spuco.utils.random_seed import seed_randomness

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


class LeNet(nn.Module):
    """
    LeNet implementation.
    """
    def __init__(self, channel: int):
        """
        Initializes LeNet.

        :param channel: Number of input channels.
        :type channel: int
        """
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.representation_dim = 84

    def forward(self, x):       
        """
        Forward pass of the LeNet model.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return x