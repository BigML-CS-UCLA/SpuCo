import random
from typing import Callable

import numpy as np
import torch
from torch import nn, optim

from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.evaluate import Evaluator 
from spuco.robust_train import BaseRobustTrain
from spuco.utils import CustomIndicesSampler, Trainer
from spuco.utils.random_seed import seed_randomness


class GroupWeightedLoss(nn.Module):
    """
    A module for computing group-weighted loss.
    """
    def __init__(
        self, 
        criterion: Callable[[torch.tensor, torch.tensor], torch.tensor],
        num_groups: int,
        group_weight_lr: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes GroupWeightedLoss.

        :param criterion: The loss criterion function.
        :type criterion: Callable[[torch.tensor, torch.tensor], torch.tensor]
        :param num_groups: The number of groups to consider.
        :type num_groups: int
        :param group_weight_lr: The learning rate for updating group weights (default: 0.01).
        :type group_weight_lr: float
        :param device: The device on which to perform computations. Defaults to CPU.
        :type device: torch.device
        """
        super(GroupWeightedLoss, self).__init__()
        self.criterion = criterion
        self.device = device
        self.num_groups = num_groups
        self.group_weights = torch.ones(self.num_groups).to(self.device)
        self.group_weights.data = self.group_weights.data / self.group_weights.data.sum()
        self.group_weight_lr = group_weight_lr

    def forward(self, outputs, labels, groups):
        """
        Computes the group-weighted loss.
        """
        # compute loss for different groups and update group weights
        loss = self.criterion(outputs, labels)
        group_loss = torch.zeros(self.num_groups).to(self.device)
        for i in range(self.num_groups):
            if (groups==i).sum() > 0:
                group_loss[i] += loss[groups==i].mean()
        self.update_group_weights(group_loss)

        # compute weighted loss
        loss = group_loss * self.group_weights
        loss = loss.sum()
        
        return loss

    def update_group_weights(self, group_loss):
        group_weights = self.group_weights
        group_weights = group_weights * torch.exp(self.group_weight_lr * group_loss)
        group_weights = group_weights / group_weights.sum()
        self.group_weights.data = group_weights.data

class GroupDRO(BaseRobustTrain):
    """
    Group DRO (https://arxiv.org/abs/1911.08731)
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: GroupLabeledDatasetWrapper,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device = torch.device("cpu"),
        val_evaluator: Evaluator = None,
        verbose=False,
    ):
        """
        Initializes GroupDRO.

        :param model: The PyTorch model to be trained.
        :type model: nn.Module
        :param trainset: The training dataset containing group-labeled samples.
        :type trainset: GroupLabeledDatasetWrapper
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param optimizer: The optimizer used for training.
        :type optimizer: optim.Optimizer
        :param num_epochs: The number of training epochs.
        :type num_epochs: int
        :param device: The device to be used for training (default: CPU).
        :type device: torch.device
        :param verbose: Whether to print training progress (default: False).
        :type verbose: bool
        """

        seed_randomness(torch_module=torch, random_module=random, numpy_module=np)
    
        super().__init__(val_evaluator=val_evaluator, verbose=verbose)
    
        assert batch_size >= len(trainset.group_partition), "batch_size must be >= number of groups (Group DRO requires at least 1 example from each group)"

        def forward_pass(self, batch):
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels, groups)
            return loss, outputs, labels
        
        self.num_epochs = num_epochs
        self.group_partition = trainset.group_partition
        self.group_weighted_loss = GroupWeightedLoss(criterion=nn.CrossEntropyLoss(reduction="none"), num_groups=len(self.group_partition), device=device)
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=self.group_weighted_loss,
            forward_pass=forward_pass,
            sampler=CustomIndicesSampler(indices=[]),
            verbose=verbose,
            device=device
        )

        max_group_len = max([len(self.group_partition[key]) for key in self.group_partition.keys()])
        self.base_indices = []
        self.sampling_weights = []
        for key in self.group_partition.keys():
            self.base_indices.extend(self.group_partition[key])
            self.sampling_weights.extend([max_group_len / len(self.group_partition[key])] * len(self.group_partition[key]))
        
    def train_epoch(self, epoch):
        """
        Trains the model for a single epoch with a group balanced batch (in expectation)

        :param epoch: The current epoch number.
        :type epoch: int
        """
        self.trainer.sampler.indices = random.choices(
            population=self.base_indices,
            weights=self.sampling_weights, 
            k=len(self.trainer.trainset)
        )
        return self.trainer.train_epoch(epoch)
