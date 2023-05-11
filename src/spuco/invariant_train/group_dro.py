import torch 
import torch.nn as nn
import torch.nn.functional as F

class GroupWeightedLoss(nn.Module):
    def __init__(
        self, 
        num_groups: int,
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
        self.device = device
        self.num_groups = num_groups
        self.group_weights = F.normalize(torch.ones(self.num_groups).to(device), p=1)

    def forward(self, loss, groups):
        """
        Computes the group-weighted loss.
        
        :param loss: The loss tensor.
        :type loss: torch.Tensor
        :param groups: The group tensor. Should be a tensor of integers between 0 and num_groups-1.
        :type groups: torch.Tensor
        :return: The group-weighted loss.
        :rtype: torch.Tensor
        """
        # compute loss for different groups and update group weights
        group_loss = torch.zeros(self.num_groups).to(self.args.device)
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
        self.group_weights = F.normalize(self.group_weights * torch.exp(self.group_weight_lr * group_loss))