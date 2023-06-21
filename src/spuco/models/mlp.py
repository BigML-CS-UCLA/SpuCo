import random

import numpy as np
import torch
import torch.nn as nn

from spuco.utils.random_seed import seed_randomness


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
    ):
        """
        Initializes the model.

        :param input_dim: Dimensionality of the input.
        :type input_dim: int
        """
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = 256
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, self.representation_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        x = x.view(-1, self.input_dim)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x
