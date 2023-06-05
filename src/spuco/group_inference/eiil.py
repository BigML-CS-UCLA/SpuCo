import random
from typing import List

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from spuco.group_inference import BaseGroupInference
from spuco.utils.random_seed import seed_randomness


class EIIL(BaseGroupInference):
    """
    Environment Inference for Invariant Learning: https://arxiv.org/abs/2010.07249
    """
    def __init__(
        self, 
        logits: torch.Tensor, 
        class_labels: List[int],
        num_steps: int, 
        lr: float,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False

    ):
        """
        Initializes the EIILInference object.

        :param logits: The logits output of the model.
        :type logits: torch.Tensor
        :param class_labels: The class labels for each sample.
        :type class_labels: List[int]
        :param num_steps: The number of steps for training the soft environment assignment.
        :type num_steps: int
        :param lr: The learning rate for training.
        :type lr: float
        :param device: The device to use for training. Defaults to CPU.
        :type device: torch.device, optional
        :param verbose: Whether to print training progress. Defaults to False.
        :type verbose: bool, optional
        """
         
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        
        super().__init__()

        self.logits = logits 
        self.class_labels = class_labels
        self.num_steps = num_steps
        self.lr = lr
        self.device = device
        self.verbose = verbose

    def infer_groups(self):
        """
        Performs EIIL inference to infer group partitions.

        :return: The group partition based on EIIL inference.
        :rtype: Dict[Tuple[int, int], List[int]]
        """

        # Initialize
        scale = torch.tensor(1.).to(self.device).requires_grad_()
        train_criterion = nn.CrossEntropyLoss(reduction='none')
        loss = train_criterion(self.logits.to(self.device) * scale, torch.tensor(self.class_labels).long().to(self.device))
        env_w = torch.randn(len(self.logits)).to(self.device).requires_grad_()
        optimizer = optim.Adam([env_w], lr=self.lr)

        # Train assignment
        for i in tqdm(range(self.num_steps), disable=not self.verbose, desc="EIIL Inferring Groups"):
            # penalty for env a
            lossa = (loss.squeeze() * env_w.sigmoid()).mean()
            grada = torch.autograd.grad(lossa, [scale], create_graph=True)[0]
            penaltya = torch.sum(grada**2)
            # penalty for env b
            lossb = (loss.squeeze() * (1-env_w.sigmoid())).mean()
            gradb = torch.autograd.grad(lossb, [scale], create_graph=True)[0]
            penaltyb = torch.sum(gradb**2)
            # negate
            npenalty = - torch.stack([penaltya, penaltyb]).mean()
            optimizer.zero_grad()
            npenalty.backward(retain_graph=True)
            optimizer.step()

        # Sigmoid to get env assignment
        spurious_labels = env_w.sigmoid() > .5
        spurious_labels = spurious_labels.int().detach().cpu().numpy()
        
        # Partition using group labels to get group partition 
        group_partition = {}
        for i in range(len(spurious_labels)):
            group_label = (0, spurious_labels[i])
            if group_label not in group_partition:
                group_partition[group_label] = []
            group_partition[group_label].append(i)

        return group_partition
