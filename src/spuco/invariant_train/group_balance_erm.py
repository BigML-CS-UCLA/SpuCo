import torch 
from torch import nn, optim
from spuco.utils import Trainer, GroupLabeledDataset, CustomIndicesSampler
import random  # TODO: Do we want to control the randomness here?
from torch.utils.data import Dataset
from typing import Dict

class GroupBalanceERM():
    """
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        group_partition: Dict,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device = torch.device("cpu"),
        verbose=False
    ):
        """
        Initializes GroupBalanceERM
        """

        assert batch_size >= len(trainset.group_partition), "batch_size must be >= number of groups (Group DRO requires at least 1 example from each group)"
        
        self.num_epochs = num_epochs
        self.group_partition = group_partition
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
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
        
    def train(self):
        """
        Trains the model using the given hyperparameters.
        """
        for epoch in range(self.num_epochs):
            self.trainer.sampler.indices = random.choices(
                population=self.base_indices,
                weights=self.sampling_weights, 
                k=len(self.trainer.trainset)
            )
            self.trainer.train_epoch(epoch)
