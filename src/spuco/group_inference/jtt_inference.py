from typing import Dict, List, Tuple, Optional
from spuco.group_inference import BaseGroupInference 
from torch.utils.data import Dataset 
import torch

from spuco.utils.trainer import Trainer 

class JTTInference(BaseGroupInference):
    def __init__(
        self,
        trainset: Dataset, 
        predictions: Optional[torch.Tensor] = None,
        trainer: Optional[Trainer] = None,
        num_epochs: Optional[int] = None
    ):
        self.trainset = trainset 
        self.trainer = trainer
        self.num_epochs = num_epochs
        self.predictions = predictions 

        if self.predictions is None:
            if self.trainer is None or self.num_epochs is None:
                raise ValueError("Either predictions or trainer and num_epochs must be non-None") 

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        if self.predictions is None:
            for epoch in range(self.num_epochs):
                self.trainer.train(epoch)
            self.predictions = torch.argmax(self.trainer.get_trainset_outputs(), dim=1)

        # Tuple as key to be consistent (0, 0) - correct set, (0, 1) - error set
        group_partition = {(0, 0): [], (0, 1): []} 

        for i, (_, label) in enumerate(self.trainset):
            if label == self.predictions[i].item():
                group_partition[(0, 0)].append(i)
            else:
                group_partition[(0, 1)].append(i)

        return group_partition

