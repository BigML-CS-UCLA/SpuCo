import random
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from spuco.evaluate import Evaluator
from spuco.robust_train import GroupBalanceBatchERM
from spuco.utils import convert_labels_to_partition
from spuco.utils.random_seed import seed_randomness


class SpareTrain(GroupBalanceBatchERM):
    """
    SpareTrain class for training a model using group balanced sampling
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        group_partition: Dict,
        sampling_powers: List,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device = torch.device("cpu"),
        val_evaluator: Evaluator = None,
        verbose=False
    ):
        """
        Initializes GroupBalanceBatchERM.

        :param model: The PyTorch model to be trained.
        :type model: nn.Module
        :param trainset: The training dataset.
        :type trainset: Dataset
        :param group_partition: A dictionary mapping group labels to the indices of examples belonging to each group.
        :type group_partition: Dict
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
        
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)

        super().__init__(model=model, trainset=trainset, group_partition=group_partition, batch_size=batch_size, optimizer=optimizer, num_epochs=num_epochs, device=device, val_evaluator=val_evaluator, verbose=verbose)
        
        assert batch_size >= len(trainset.group_partition), "batch_size must be >= number of groups (Group DRO requires at least 1 example from each group)"
        
        self.class_partition = convert_labels_to_partition(trainset.labels)

        self.sampling_weights = []
        for key in self.group_partition.keys():
            self.sampling_weights.extend([1 / len(self.group_partition[key]) ** sampling_powers[key[0]]] * len(self.group_partition[key]))

        self.sampling_weights = np.array(self.sampling_weights)
        for key in self.class_partition.keys():
            # finds where class_partition[key] are in base indices
            indices = [i for i, x in enumerate(self.base_indices) if x in self.class_partition[key]]
            indices = np.array(indices)
            
            # normalize the sampling weights so that each class has the same total weight
            self.sampling_weights[indices] = self.sampling_weights[indices] / sum(self.sampling_weights[indices])
        
        self.sampling_weights = list(self.sampling_weights)

        
    def train_epoch(self, epoch: int):
        """
        Trains the model for a single epoch with a custom upsampled batch

        :param epoch: The current epoch number.
        :type epoch: int
        """
        self.trainer.sampler.indices = random.choices(
            population=self.base_indices,
            weights=self.sampling_weights, 
            k=len(self.trainer.trainset)
        )

        self.trainer.train_epoch(epoch)
