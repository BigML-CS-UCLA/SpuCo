from typing import Dict, Iterator, Tuple, List
from torch.utils.data import Sampler 

class InvariantTrainSampler(Sampler[int]):
    """
    Samples from the specified indices (pass indices - upsampled, downsampled, group balanced etc. to this class)
    """
    def __init__(
        self, 
        indices: List[int]
    ):
        self.indices = indices 

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)