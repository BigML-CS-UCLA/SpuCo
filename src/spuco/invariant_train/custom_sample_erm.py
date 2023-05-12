from typing import List
import torch 
from torch import nn, optim
from torch.utils.data import Dataset
from spuco.invariant_train import InvariantTrainSampler
from spuco.util.trainer import Trainer
import random 

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
            sampler=InvariantTrainSampler(indices=self.indices),
            verbose=verbose,
            device=device
        )
        
    def train(self):
        """
        Trains the model using the given hyperparameters.
        """
        for epoch in range(self.num_epochs):
            random.shuffle(self.indices)
            self.trainer.sampler.indices = self.indices
            self.trainer.train(epoch)