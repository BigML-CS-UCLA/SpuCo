from typing import Iterator, List

from torch.utils.data import Sampler
import random 

class CustomIndicesSampler(Sampler[int]):
    """
    Samples from the specified indices (pass indices - upsampled, downsampled, group balanced etc. to this class)
    Default is no shuffle.
    """
    def __init__(
        self, 
        indices: List[int],
        shuffle: bool = False,
    ):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self) -> int:
        return len(self.indices)