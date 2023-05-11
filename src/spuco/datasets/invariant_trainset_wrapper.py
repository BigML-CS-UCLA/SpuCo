from typing import List
from torch.utils.data import Dataset

class InvariantTrainsetWrapper(Dataset):
    def __init__(
        self, 
        dataset: Dataset,
        spurious: List[int],
    ):
        self.dataset = dataset
        self.spurious = spurious

    def __getitem__(self, index):
        return self.dataset.__getitem__(index) + (self.spurious)
    
