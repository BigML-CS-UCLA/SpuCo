from typing import Dict, List, Tuple
from torch import nn, optim
from torch.utils.data import Dataset
from spuco.group_inference import BaseGroupInference
from spuco.utils import Trainer
import torch 

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
        for epoch in range(self.num_epochs):
            self.trainer.train(epoch)

        spurious = torch.argmax(self.trainer.get_trainset_outputs(), dim=-1)

        group_partition = {}
        for i, (_, y) in enumerate(self.trainer.trainset):
            group_label = (y, spurious[i])
            if group_label not in group_partition:
                group_partition[group_label] = []
            group_partition[group_label].append(i)
            
        return group_partition

