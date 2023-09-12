import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from spuco.evaluate import Evaluator
from spuco.robust_train import BaseRobustTrain
from spuco.utils import Trainer
from spuco.utils.random_seed import seed_randomness


class ERM(BaseRobustTrain):
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
        lr_scheduler=None,
        max_grad_norm=None,
        val_evaluator: Evaluator = None,
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
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(val_evaluator=val_evaluator, verbose=verbose)

        self.num_epochs = num_epochs
        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            criterion=criterion,
            verbose=verbose,
            device=device
        )
