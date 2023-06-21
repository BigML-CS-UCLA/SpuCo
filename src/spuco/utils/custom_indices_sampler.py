import random
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler

from spuco.utils.random_seed import seed_randomness


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
        """
        Samples elements from the specified indices.

        :param indices: The list of indices to sample from.
        :type indices: list[int]
        :param shuffle: Whether to shuffle the indices. Default is False.
        :type shuffle: bool, optional
        """
         
        seed_randomness(random_module=random, torch_module=torch, numpy_module=np)

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the sampled indices.

        :return: An iterator over the sampled indices.
        :rtype: iterator[int]
        """
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self) -> int:
        """
        Returns the number of sampled indices.

        :return: The number of sampled indices.
        :rtype: int
        """
        return len(self.indices)