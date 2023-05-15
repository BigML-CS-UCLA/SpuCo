from typing import Any, Callable, Optional, Tuple
import torch
from torch.utils.data import Sampler
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            trainset: Dataset,
            model: nn.Module,
            batch_size: int,
            optimizer: optim.Optimizer,
            criterion: nn.Module = nn.CrossEntropyLoss(),
            forward_pass: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
            sampler: Sampler = None,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False
    ) -> None:
        """
        Constructor for the Trainer class.

        :param trainset: The training set.
        :type trainset: torch.utils.data.Dataset
        :param model: The PyTorch model to train.
        :type model: torch.nn.Module
        :param batch_size: The batch size to use during training.
        :type batch_size: int
        :param optimizer: The optimizer to use for training.
        :type optimizer: torch.optim.Optimizer
        :param criterion: The loss function to use during training.
        :type criterion: torch.nn.Module, optional
        :param forward_pass: The forward pass function to use during training.
        :type forward_pass: Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], optional
        :param device: The device to use for computations.
        :type device: torch.device, optional
        :param verbose: Whether to print training progress.
        :type verbose: bool, optional
        """
        self.trainset = trainset
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.sampler = sampler
        self.verbose = verbose
        self.device = device
        
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
            sampler=self.sampler
        )

    def train(self, epoch: int) -> None:
        """
        Trains the PyTorch model for 1 epoch
        """
        self.model.train()
        with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for batch in pbar:
                loss, outputs, labels = self.forward_pass(self, batch)
                accuracy = Trainer.compute_accuracy(outputs, labels)

                # backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")
    
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
    
    def get_trainset_outputs(self):
        with torch.no_grad():
            self.model.eval()
            eval_trainloader = DataLoader(
                dataset=self.trainset,
                batch_size=self.batch_size,
                shuffle=False
            )
            with tqdm(eval_trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                outputs = []
                for input, _ in pbar:
                    outputs.append(self.model(input.to(self.device)))
                return torch.cat(outputs, dim=0)