import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import MultiStepLR

from spuco.evaluate import Evaluator
from spuco.robust_train import BaseRobustTrain
from spuco.utils import CustomIndicesSampler, Trainer
from spuco.utils.random_seed import seed_randomness


class PDE(BaseRobustTrain):
    """
    Progressive Data Expansion (PDE): https://arxiv.org/abs/2306.04949
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        group_partition: Dict[Tuple[int, int], List[int]],
        criterion=nn.CrossEntropyLoss(), 
        device: torch.device = torch.device("cpu"),
        val_evaluator: Evaluator = None,
        verbose=False,
        use_wandb=False,
        warmup_epochs=15,
        expansion_size=10,
        expansion_interval=10,
        subsample_cap=-1,
        gamma=1,
    ):  
        """
        Initializes PDE.

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
        :param warmup_epochs: The number of epochs to train on the group-balanced downsampled dataset before starting progressive data expansion (default: 15).
        :type warmup_epochs: int
        :param expansion_size: The number of samples to add to the training set in each expansion interval (default: 10).
        :type expansion_size: int
        :param expansion_interval: The number of epochs between each expansion (default: 10).
        :type expansion_interval: int
        :param subsample_cap: The minimum number of samples to keep from each group (default: -1).
        :type subsample_cap: int
        :param gamma: The learning rate decay factor (default: 1).
        :type gamma: float
        """
        
        seed_randomness(torch_module=torch, random_module=random, numpy_module=np)
        
        super().__init__(val_evaluator=val_evaluator, verbose=verbose, use_wandb=use_wandb)

        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.expansion_size = expansion_size
        self.expansion_interval = expansion_interval
        self.subsample_cap = subsample_cap

        len_min_group = max(min([len(group_partition[key]) for key in group_partition.keys()]), self.subsample_cap)
        self.indices = []
        for key in group_partition.keys():
            if len(group_partition[key]) < len_min_group:
                self.indices.extend(group_partition[key])
            else:
                group_indices = torch.randperm(len(group_partition[key]))[:len_min_group].tolist()
                self.indices.extend([group_partition[key][i] for i in group_indices])

        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            sampler=CustomIndicesSampler(indices=self.indices, shuffle=True),
            lr_scheduler=MultiStepLR(optimizer, milestones=[warmup_epochs], gamma=gamma) if gamma < 1 else None,
            verbose=verbose,
            device=device
        )
        
    def train_epoch(self, epoch):
        """
        Trains the model for a single epoch with a group balanced batch (in expectation)

        :param epoch: The current epoch number.
        :type epoch: int
        """
        if (epoch >= self.warmup_epochs) and ((epoch - self.warmup_epochs) % self.expansion_interval == 0):
            print(f"Expanding training set with {self.expansion_size} samples at epoch {epoch}")
            unused_indices = list(set(range(len(self.trainer.trainset))) - set(self.indices))
            self.indices.extend(random.choices(unused_indices, k=self.expansion_size))
            self.trainer.sampler.indices = self.indices
            print(f"Training set size: {len(self.indices)}")
        return self.trainer.train_epoch(epoch)
        