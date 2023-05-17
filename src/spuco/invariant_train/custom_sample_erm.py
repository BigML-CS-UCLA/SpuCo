from typing import List

import torch
from torch import nn, optim
from torch.utils.data import Dataset

from spuco.utils import CustomIndicesSampler, Trainer


class CustomSampleERM():
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
        
    def train(self):
        """
        Trains the model using the given hyperparameters.
        """
        for epoch in range(self.num_epochs):
            self.trainer.train_epoch(epoch)