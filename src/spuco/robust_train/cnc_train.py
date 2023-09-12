import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.evaluate import Evaluator
from spuco.robust_train import BaseRobustTrain
from spuco.models import SpuCoModel
from spuco.utils.random_seed import seed_randomness


class CorrectNContrastTrain(BaseRobustTrain):
    """
    CorrectNContrastTrain class for training a model using CNC's 
    Cross Entropy + modified Supervised Contrastive Learning loss.
    """
    def __init__(
        self,
        trainset: GroupLabeledDatasetWrapper,
        model: SpuCoModel,
        batch_size: int,
        optimizer_encoder: optim.Optimizer,
        optimizer_classifier: optim.Optimizer,
        num_pos: int, 
        num_neg: int,
        num_epochs: int,
        lambda_ce: float,
        temp: float,
        device: torch.device = torch.device("cpu"),
        accum: int = 32, 
        val_evaluator: Evaluator = None,
        verbose: bool = False  
    ):
        """
        Initializes CorrectNContrastTrain.

        :param trainset: The training dataset containing group-labeled samples.
        :type trainset: GroupLabeledDatasetWrapper
        :param model: The SpuCoModel to be trained.
        :type model: SpuCoModel
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param optimizer: The optimizer used for training.
        :type optimizer: optim.Optimizer
        :param num_pos: The number of positive examples for contrastive loss.
        :type num_pos: int
        :param num_neg: The number of negative examples for contrastive loss.
        :type num_neg: int
        :param num_epochs: The number of training epochs.
        :type num_epochs: int
        :param lambda_ce: The weight of the regular cross-entropy loss.
        :type lambda_ce: float
        :param temp: The temperature the regular cross-entropy loss.
        :type temp: float
        :param device: The device to be used for training (default: CPU).
        :type device: torch.device
        :param verbose: Whether to print training progress (default: False).
        :type verbose: bool
        """
        
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        
        super().__init__(val_evaluator=val_evaluator, verbose=verbose)
    
        self.num_epochs = num_epochs 

        def forward_pass(self, batch):
            """
            Custom forward pass function for Correct & Contrast training.

            :param batch: A batch of input samples.
            :return: The loss, model outputs, and labels.
            """
            # Unpack inputs and move to correct device
            inputs, labels, groups = batch 
            inputs, labels, groups = inputs.to(self.device), labels.to(self.device), groups.to(self.device)

            # Regular Cross Entropy Loss
            outputs = self.model(inputs)
            loss = lambda_ce * self.criterion(outputs, labels)

            # Contrastive Loss
            for anchor_idx in range(len(batch)):
                # Only computed with majority examples as anchor
                if labels[anchor_idx] != groups[anchor_idx]:
                    continue 
                pos_idx = []
                neg_idx = []
                for i in range(len(inputs)):
                    if len(pos_idx) < num_pos: # Positives = same label, but different group
                        if labels[i] == labels[anchor_idx] and groups[i] != groups[anchor_idx]:
                            pos_idx.append(i)

                    if len(neg_idx) < num_neg: # Negatives = different label, but same spurious attribute
                        if labels[i] != labels[anchor_idx] and groups[i] == groups[anchor_idx]:
                            neg_idx.append(i)
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    anchor = self.model.backbone(torch.unsqueeze(inputs[anchor_idx], dim=0))
                    pos = self.model.backbone(inputs[pos_idx])
                    neg = self.model.backbone(inputs[neg_idx])
                    pos_sim = torch.exp(torch.cosine_similarity(anchor, pos)/temp)
                    pos_sum_sim = torch.sum(pos_sim) 
                    neg_sum_sim = torch.sum(torch.exp(torch.cosine_similarity(anchor, neg)/temp))
                    sup_cl_loss = torch.sum(torch.log(pos_sim / (pos_sum_sim + neg_sum_sim)))
                    loss += (1-lambda_ce) * sup_cl_loss / len(pos_idx)

            return loss, outputs, labels

        self.trainer = CNCTrainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer_1=optimizer_encoder,
            optimizer_2=optimizer_classifier,
            accum_1=1,
            accum_2=accum,
            forward_pass=forward_pass,
            verbose=verbose,
            device=device
        )

class CNCTrainer:
    def __init__(
            self,
            trainset: Dataset,
            model: nn.Module,
            batch_size: int,
            optimizer_1: optim.Optimizer,
            optimizer_2: optim.Optimizer,
            accum_1: int,
            accum_2: int,
            lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            max_grad_norm: Optional[float] = None,
            criterion: nn.Module = nn.CrossEntropyLoss(),
            forward_pass: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
            sampler: Sampler = None,
            device: torch.device = torch.device("cpu"),
            verbose: bool = False
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
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2
        self.accum_1 = accum_1
        self.accum_2 = accum_2
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
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
        Trains the model for 1 epoch using CNC method

        :param epoch: epoch number that is being trained (only used by logging)
        :type epoch: int
        """
        self.model.train()
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        with tqdm(self.trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
            pbar.set_description(f"Epoch {epoch}")
            average_accuracy = 0.
            idx_batch = 1
            for batch in pbar:
                loss, outputs, labels = self.forward_pass(self, batch)
                accuracy = CNCTrainer.compute_accuracy(outputs, labels)

                # backward pass and optimization
                
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.lr_scheduler is not None and isinstance(self.optimizer_1, optim.AdamW):
                    self.lr_scheduler.step()
                if idx_batch % self.accum_1 == 0 or idx_batch == (len(self.trainloader)):
                    self.optimizer_1.step()
                    self.optimizer_1.zero_grad()
                if idx_batch % self.accum_2 == 0 or idx_batch == (len(self.trainloader)):
                    self.optimizer_2.step()
                    self.optimizer_2.zero_grad()

                # TODO: check if step should be called every batch or every epoch
                if self.lr_scheduler is not None and not isinstance(self.optimizer_1, optim.AdamW):
                    self.lr_scheduler.step()

                pbar.set_postfix(loss=loss.item(), accuracy=f"{accuracy}%")
                average_accuracy += accuracy
                idx_batch += 1
            return average_accuracy / len(pbar)
    
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
        """
        Gets output of model on trainset
        """
        with torch.no_grad():
            self.model.eval()
            eval_trainloader = DataLoader(
                dataset=self.trainset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4, 
                pin_memory=True
            )
            with tqdm(eval_trainloader, unit="batch", total=len(self.trainloader), disable=not self.verbose) as pbar:
                outputs = []
                pbar.set_description("Getting Trainset Outputs")
                for input, _ in pbar:
                    outputs.append(self.model(input.to(self.device)))
                return torch.cat(outputs, dim=0)