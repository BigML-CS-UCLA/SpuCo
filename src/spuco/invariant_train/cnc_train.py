from torch import nn, optim
from spuco.utils import GroupLabeledDataset
from spuco.models import BaseEncoder
import torch 

class CorrectNContrastTrain():
    def __init__(
        self,
        trainset: GroupLabeledDataset,
        model: BaseEncoder,
        batch_size: int,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False  
    ):
        def forward_pass(self, batch):
            inputs, labels, groups = batch 
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)
            cross_entropy_loss = self.criterion(self.model(inputs), labels)

            sup_cl_loss = self.model.encode(inputs)