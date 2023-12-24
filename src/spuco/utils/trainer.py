import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from spuco.utils.random_seed import seed_randomness
from spuco.utils.misc import get_model_outputs

try:
    import wandb
except ImportError:
    pass

class Trainer:
    def __init__(
            self,
            trainset: Dataset,
            model: nn.Module,
            batch_size: int,
            optimizer: optim.Optimizer,
            lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            max_grad_norm: Optional[float] = None,
            criterion: nn.Module = nn.CrossEntropyLoss(),
            forward_pass: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
            sampler: Sampler = None,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False,
            name: str = "",
            use_wandb: bool = False
    ) -> None:
        """
        Initializes an instance of the Trainer class.

        :param trainset: The training set.
        :type trainset: torch.utils.data.Dataset
        :param model: The PyTorch model to train.
        :type model: torch.nn.Module
        :param batch_size: The batch size to use during training.
        :type batch_size: int
        :param optimizer: The optimizer to use for training.
        :type optimizer: torch.optim.Optimizer
        :param criterion: The loss function to use during training. Default is nn.CrossEntropyLoss().
        :type criterion: torch.nn.Module, optional
        :param forward_pass: The forward pass function to use during training. Default is None.
        :type forward_pass: Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], optional
        :param sampler: The sampler to use for creating batches. Default is None.
        :type sampler: torch.utils.data.Sampler, optional
        :param device: The device to use for computations. Default is torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to print training progress. Default is False.
        :type verbose: bool, optional
        """
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        self.trainset = trainset
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.criterion = criterion
        self.batch_size = batch_size
        self.sampler = sampler
        self.verbose = verbose
        self.device = device
        self.name = name
        self.use_wandb = use_wandb
        
        if forward_pass is None:
            def forward_pass(self, batch):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                return loss, outputs, labels
            self.forward_pass = forward_pass 
        else:
            self.forward_pass = forward_pass

        self.trainloader = DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            shuffle=(self.sampler is None), 
            sampler=self.sampler,
            num_workers=4, 
            pin_memory=True
        )

    def train(self, num_epochs: int):
        """
        Trains for given number of epochs 

        :param num_epochs: Number of epochs to train for
        :type num_epochs: int
        """
        for epoch in range(num_epochs):
            self.train_epoch(epoch) 
            
    def train_epoch(self, epoch: int) -> None:
        """
        Trains the PyTorch model for 1 epoch

        :param epoch: epoch number that is being trained (only used by logging)
        :type epoch: int
        """
        self.model.train()
        batch_idx = 0
        with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            average_accuracy = 0.
            average_loss = 0.
            for batch in pbar:
                loss, outputs, labels = self.forward_pass(self, batch)
                accuracy = Trainer.compute_accuracy(outputs, labels)

                # backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.lr_scheduler is not None and isinstance(self.optimizer, optim.AdamW):
                    self.lr_scheduler.step()
                self.optimizer.step()

                pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")
                if self.verbose:
                    print(f"{self.name} | Epoch {epoch} | Loss: {loss.item()} | Accuracy: {accuracy}%")
                if self.use_wandb:
                    wandb.log({f"{self.name}_train_loss": loss.item(), f"{self.name}_train_acc": accuracy})
                average_accuracy += accuracy
                average_loss += loss.item()

                batch_idx += 1
                            
            if self.lr_scheduler is not None and not isinstance(self.optimizer, optim.AdamW):
                self.lr_scheduler.step()
                
            return average_accuracy / len(pbar), average_loss / len(pbar)
    
    @staticmethod
    def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Computes the accuracy of the PyTorch model.

        :param outputs: The predicted outputs of the model.
        :type outputs: torch.Tensor
        :param labels: The ground truth labels.
        :type labels: torch.Tensor
        :return: The accuracy of the model.
        :rtype: float
        """
        predicted = torch.argmax(outputs, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return 100. * correct / total
    
    def get_trainset_outputs(self, features=False):
        """
        Gets output of model on trainset
        """
        return get_model_outputs(self.model, self.trainset, self.device, features, self.verbose)
            
