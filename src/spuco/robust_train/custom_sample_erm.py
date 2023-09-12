import random
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from spuco.evaluate import Evaluator
from spuco.robust_train import BaseRobustTrain
from spuco.utils import CustomIndicesSampler, Trainer
from spuco.utils.random_seed import seed_randomness


class CustomSampleERM(BaseRobustTrain):
    """
    CustomSampleERM class for training a model using custom sampling of the dataset
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        indices: List[int],
        criterion=nn.CrossEntropyLoss(), 
        device: torch.device = torch.device("cpu"),
        val_evaluator: Evaluator = None,
        verbose=False
    ):  
        """
        Initializes CustomSampleERM.

        :param model: The PyTorch model to be trained.
        :type model: nn.Module
        :param trainset: The training dataset.
        :type trainset: Dataset
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param optimizer: The optimizer used for training.
        :type optimizer: optim.Optimizer
        :param num_epochs: The number of training epochs.
        :type num_epochs: int
        :param indices: A list of indices specifying the samples to be shown to the model in 1 epoch.
        :type indices: List[int]
        :param criterion: The loss criterion used for training (default: CrossEntropyLoss).
        :type criterion: nn.Module
        :param device: The device to be used for training (default: CPU).
        :type device: torch.device
        :param verbose: Whether to print training progress (default: False).
        :type verbose: bool
        """
         
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        
        super().__init__(val_evaluator=val_evaluator, verbose=verbose)
        
        self.num_epochs = num_epochs
        self.indices = indices
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            sampler=CustomIndicesSampler(indices=self.indices, shuffle=True),
            verbose=verbose,
            device=device
        )
