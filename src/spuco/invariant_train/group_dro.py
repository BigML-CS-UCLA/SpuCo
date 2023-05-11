from typing import Callable, List, Tuple
import torch 
from torch import nn, optim
import torch.nn.functional as F

from spuco.invariant_train.invariant_trainset_wrapper import InvariantTrainsetWrapper
from spuco.util.trainer import Trainer

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
        self.group_weights = F.normalize(torch.ones(self.num_groups).to(device), p=1, dim=0)
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
        """
        Updates the group weights based on the group losses.
        
        :param group_loss: The group losses.
        :type group_loss: torch.Tensor
        """
        self.group_weights = F.normalize(self.group_weights * torch.exp(self.group_weight_lr * group_loss), p=1, dim=0)

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
        num_groups: int,
        criterion=nn.CrossEntropyLoss(reduction='none'), 
        device: torch.device = torch.device("cpu"),
        verbose=False
    ):
        """
        Initializes a ERM instance.

        :param model: The neural network model to train.
        :type model: nn.Module
        :param trainset: The trainset to use for training.
        :type trainset: InvariantTrainsetWrapper
        :param batch_size: The batch size to use during training.
        :type batch_size: int
        :param optimizer: The optimizer to use during training.
        :type optimizer: torch.optim.Optimizer
        :param num_epochs: The number of epochs to train for.
        :type num_epochs: int
        :param criterion: The loss function to use. Default is nn.CrossEntropyLoss().
        :type criterion: nn.Module, optional
        :param device: The device to use for training. Default is torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: If True, prints verbose training information. Default is False.
        :type verbose: bool, optional
        """
        def forward_pass(self, batch):
            inputs, labels, groups = batch
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels, groups)
            return loss, outputs, labels
        
        self.group_weighted_loss = GroupWeightedLoss(criterion=criterion, num_groups=num_groups, device=device)
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            num_epochs=num_epochs,
            criterion=self.group_weighted_loss,
            forward_pass=forward_pass,
            verbose=verbose,
            device=device
        )

    def train(self):
        """
        Trains the model using the given hyperparameters.
        """
        self.trainer.train()