from typing import Callable, Optional
from torchvision import datasets
from torchvision.transforms import v2 as T
import random
import numpy as np
import torch

from spuco.datasets import (TRAIN_SPLIT, SpuCoImageFolder)
from spuco.datasets.base_spuco_dataset import MASK_SPURIOUS
from spuco.utils.random_seed import seed_randomness

SUN397_BACKGROUND_SIZE = 224
    
class SpuCoSun(SpuCoImageFolder):
    """
    
    """
    def __init__(
        self,
        root: str,
        label_noise: float = 0.0,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
        mask_type: Optional[str] = None,
        sun397_root_dir: Optional[str] = "/data",
        verbose: bool = False
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str
        
        :param label_noise: The amount of label noise to apply.
        :type label_noise: float, optional

        :param split: The split of the dataset.
        :type split: str, optional

        :param transform: Optional transform to be applied to the data.
        :type transform: Callable, optional

        :param verbose: Whether to print verbose information during dataset initialization.
        :type verbose: bool, optional
        """
        super().__init__(root=root, label_noise=label_noise, split=split, transform=transform, verbose=verbose)
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        self.mask_type = mask_type
        
        if self.mask_type == MASK_SPURIOUS:
            sun397_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(SUN397_BACKGROUND_SIZE)
            ])
            self.sun397 = datasets.SUN397(root=sun397_root_dir, download=True, transform=sun397_transform)
            
    def load_image(self, filename: str):
        if self.mask_type != MASK_SPURIOUS:
            return super().load_image(filename=filename)
        else:
            sun397_idx = int(filename.split('-')[2])
            return self.sun397[sun397_idx][0]