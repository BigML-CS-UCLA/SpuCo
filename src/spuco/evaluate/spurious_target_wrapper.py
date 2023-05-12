from typing import Any
from torch.utils.data import Dataset 
import torch 

class SpuriousTargetWrapper(Dataset):
    """
    Wrapper class that takes a Dataset and the spurious labels of the data
    and returns a dataset where the labels are the spurious labels. 
    """
    def __init__(
        self,
        dataset: Dataset,
        spurious_labels: torch.Tensor
    ):
        self.dataset = dataset
        self.spurious_labels = spurious_labels

    def __getitem__(self, index):
        return (self.dataset.__getitem__(index)[0], self.spurious_labels[index])
    
    def __len__(self):
        return len(self.dataset)
        