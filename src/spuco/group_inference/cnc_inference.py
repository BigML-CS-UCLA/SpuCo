import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

from spuco.group_inference import BaseGroupInference
from spuco.utils import Trainer
from spuco.utils.random_seed import seed_randomness


class CorrectNContrastInference(BaseGroupInference):
    """
    Correct-n-Contrast Inference: https://proceedings.mlr.press/v162/zhang22z.html
    """
    def __init__(
        self,
        trainset: Dataset,
        model: nn.Module,
        batch_size: int,
        optimizer: optim.Optimizer,
        num_epochs: int, 
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ):
        """
        Initializes the CorrectNContrastInference object.

        :param trainset: The training dataset.
        :type trainset: Dataset
        :param model: The model for training.
        :type model: nn.Module
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param optimizer: The optimizer for training.
        :type optimizer: optim.Optimizer
        :param num_epochs: The number of epochs for training.
        :type num_epochs: int
        :param device: The device to use for training. Defaults to CPU.
        :type device: torch.device, optional
        :param verbose: Whether to print training progress. Defaults to False.
        :type verbose: bool, optional
        """
         
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        
        super().__init__()
        
        self.trainer = Trainer(
            trainset=trainset,
            model=model, 
            batch_size=batch_size,
            optimizer=optimizer,
            device=device,
            verbose=verbose
        )
        self.num_epochs = num_epochs

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Performs Correct-n-Contrast inference to infer group partitions.

        :return: The group partition based on Correct-n-Contrast inference.
        :rtype: Dict[Tuple[int, int], List[int]]
        """
        
        self.trainer.train(self.num_epochs)
        
        spurious = torch.argmax(self.trainer.get_trainset_outputs(), dim=-1).cpu().tolist()

        # CNC only wants to group based on spurious attribute, but for consistency in API
        # we need a tuple as key, hence group_label set to (0, spurious) 
        group_partition = {}
        for i in range(len(self.trainer.trainset)):
            group_label = (0, spurious[i])
            if group_label not in group_partition:
                group_partition[group_label] = []
            group_partition[group_label].append(i)
            
        return group_partition

