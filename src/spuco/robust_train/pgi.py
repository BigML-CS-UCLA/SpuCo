
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


class PGILoss(nn.Module):
    """
    A module for computing predictive group invariance loss.
    """
    def __init__(
        self, 
        num_classes: int,
        criterion: Callable[[torch.tensor, torch.tensor], torch.tensor],
        penalty_weight: float = 0.01,
        rampup_epochs: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes PGILoss.

        :param criterion: The loss criterion function.
        :type criterion: Callable[[torch.tensor, torch.tensor], torch.tensor]
        :param penalty_weight: The weight for the KL divergence penalty (default: 0.01).
        :type penalty_weight: float
        :param device: The device on which to perform computations. Defaults to CPU.
        :type device: torch.device
        """
        super(PGILoss, self).__init__()
        self.criterion = criterion
        self.num_classes = num_classes
        self.device = device
        self.penalty_weight = penalty_weight
        self.rampup_epochs = rampup_epochs
        self.current_epoch = 0

    def forward(self, outputs, labels, groups):
        """
        Computes the group-weighted loss.
        """
        # compute loss
        loss = self.criterion(outputs, labels)

        for c in range(self.num_classes):
            # compute the KL divergence between the output distributions of the two groups
            group_0_outputs = outputs[labels == c][groups[labels == c] == 0]
            group_1_outputs = outputs[labels == c][groups[labels == c] == 1]

            group_0_probs = nn.functional.softmax(group_0_outputs, dim=1)
            group_1_probs = nn.functional.softmax(group_1_outputs, dim=1)

            # compute the KL divergence between the output distributions of the two groups in the same class
            kl_divergence = nn.functional.kl_div(group_0_probs.mean(dim=0), group_1_probs.mean(dim=0), reduction="batchmean")

        # compute the final loss as the sum of the loss and the KL divergence
        loss = loss.mean() + kl_divergence * self.penalty_weight * min(self.current_epoch / self.rampup_epochs, 1.0)
        
        return loss


class PGI(BaseRobustTrain):
    """
    Predictive Group Invariance (PGI)
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: GroupLabeledDatasetWrapper,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        penalty_weight: float = 0.01,
        rampup_epochs: int = 10,
        device: torch.device = torch.device("cpu"),
        val_evaluator: Evaluator = None,
        verbose=False,
    ):
        """
        Initializes PGI.

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
        self.penalty_weight = penalty_weight
        self.group_partition = trainset.group_partition
        self.num_classes = trainset.dataset.num_classes
        # require the number of groups to be 2
        assert len(self.group_partition) == 2, "PGI requires exactly 2 groups"
        self.pgi_loss = PGILoss(criterion=nn.CrossEntropyLoss(reduction="none"), num_classes=self.num_classes, penalty_weight=penalty_weight, rampup_epochs=rampup_epochs, device=device)
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=self.pgi_loss,
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
        self.pgi_loss.current_epoch = epoch
        return self.trainer.train_epoch(epoch)
