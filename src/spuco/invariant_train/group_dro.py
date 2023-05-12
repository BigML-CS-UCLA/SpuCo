from typing import Callable, Dict, List, Tuple
import torch 
from torch import nn, optim
import torch.nn.functional as F
from spuco.invariant_train import InvariantTrainSampler, InvariantTrainsetWrapper
from spuco.util.trainer import Trainer
import random  # TODO: Do we want to control the randomness here?

class GroupWeightedLoss(nn.Module):
    def __init__(
        self, 
        criterion: Callable[[torch.tensor, torch.tensor], torch.tensor],
        num_groups: int,
        group_weight_lr: float = 0.01,
        device: torch.device = torch.device("cpu"),

    ):
        """
        A module for computing group-weighted loss.
        
        :param num_groups: The number of groups to consider.
        :type num_groups: int
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

class GroupDRO():
    """
    Group DRO (https://arxiv.org/abs/1911.08731)
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: InvariantTrainsetWrapper,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        group_partition: Dict[Tuple[int, int], List[int]],
        criterion=nn.CrossEntropyLoss(reduction='none'), 
        device: torch.device = torch.device("cpu"),
        verbose=False
    ):
        """
        Initializes GroupDRO
        """

        assert batch_size >= len(group_partition), "batch_size must be >= number of groups (Group DRO requires at least 1 example from each group)"

        def forward_pass(self, batch):
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels, groups)
            return loss, outputs, labels
        
        self.num_epochs = num_epochs
        self.group_partition = group_partition
        self.group_weighted_loss = GroupWeightedLoss(criterion=criterion, num_groups=len(self.group_partition), device=device)
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=self.group_weighted_loss,
            forward_pass=forward_pass,
            sampler=InvariantTrainSampler(indices=[]),
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
            self.trainer.train(epoch)
