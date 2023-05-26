import torch
from torch import optim

from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.invariant_train import BaseInvariantTrain
from spuco.models import SpuCoModel
from spuco.utils import Trainer
from spuco.utils.random_seed import seed_randomness

import random
import numpy as np

class CorrectNContrastTrain(BaseInvariantTrain):
    """
    CorrectNContrastTrain class for training a model using CNC's 
    Cross Entropy + modified Supervised Contrastive Learning loss.
    """
    def __init__(
        self,
        trainset: GroupLabeledDatasetWrapper,
        model: SpuCoModel,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_pos: int, 
        num_neg: int,
        num_epochs: int,
        lambda_ce: float,
        device: torch.device = torch.device("cpu"),
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
        :param device: The device to be used for training (default: CPU).
        :type device: torch.device
        :param verbose: Whether to print training progress (default: False).
        :type verbose: bool
        """
        super().__init__()

        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

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
                    pos_sim = torch.exp(torch.cosine_similarity(anchor, pos))
                    pos_sum_sim = torch.sum(pos_sim) 
                    neg_sum_sim = torch.sum(torch.exp(torch.cosine_similarity(anchor, neg)))
                    sup_cl_loss = torch.sum(torch.log(pos_sim / (pos_sum_sim + neg_sum_sim)))
                    loss += sup_cl_loss / len(pos_idx)

            return loss, outputs, labels

        self.trainer = Trainer(
            trainset=trainset,
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            forward_pass=forward_pass,
            verbose=verbose,
            device=device
        )

    def train(self):
        """
        Trains the model using the given hyperparameters and the Correct & Contrast training approach.
        """
        self.trainer.train(self.num_epochs)