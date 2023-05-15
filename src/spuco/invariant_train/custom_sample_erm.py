from typing import List
import torch 
from torch import nn, optim
from torch.utils.data import Dataset
from spuco.utils import Trainer, CustomIndicesSampler

class CustomSampleERM():
    """
    CustomSampleTrain - specify sampling the dataset in any way by passing in indices: List[int]
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