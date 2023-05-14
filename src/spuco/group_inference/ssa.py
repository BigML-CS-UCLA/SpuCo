
from typing import Dict, List, Tuple
from torch import nn, optim
from torch.utils.data import Dataset
from spuco.group_inference import BaseGroupInference
from spuco.invariant_train import InvariantTrainset
import torch

class SSA(BaseGroupInference):
    def __init__(
        self, 
        group_unlabled_dataset: Dataset, 
        group_labeled_dataset: InvariantTrainset,
        model: nn.Module, 
        optimizer: optim.Optimizer,
        batch_size: int = 64,
        num_folds: int = 3, 
    ):
        # Create Splits 
        self.splits = []
        indices = range(len(group_unlabled_dataset))

        # Define samplers for all the splits (train, val)

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        spurious_labels = torch.argmax(self.best_trainer.get(), dim=-1)

        return super().infer_groups()

    def ssa_kth_fold(k: int):
        pass 

