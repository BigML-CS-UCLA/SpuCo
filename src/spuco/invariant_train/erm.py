import torch
from torch import nn, optim 
from torch.utils.data import Dataset
from spuco.utils import Trainer

class ERM():
    """
    Empirical Risk Minimization (ERM) Trainer.
    """
    def __init__(
        self,
        model: nn.Module,
        trainset: Dataset,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int,
        criterion=nn.CrossEntropyLoss(), 
        device: torch.device = torch.device("cpu"),
        verbose=False
    ):
        """
        Initializes a ERM instance.

        :param model: The neural network model to train.
        :type model: nn.Module
        :param trainset: The trainset to use for training.
        :type trainset: Dataset
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

        self.num_epochs = num_epochs
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            verbose=verbose,
            device=device
        )

    def train(self):
        """
        Trains the model using the given hyperparameters.
        """
        for epoch in range(self.num_epochs):
            self.trainer.train(epoch)
