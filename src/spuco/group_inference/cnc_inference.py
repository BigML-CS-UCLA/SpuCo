from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset

from spuco.group_inference import BaseGroupInference
from spuco.utils import Trainer

class CorrectNContrastInference(BaseGroupInference):
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

