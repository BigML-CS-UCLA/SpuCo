import os
import random
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from spuco.datasets import (TRAIN_SPLIT, BaseSpuCoDataset, SourceData)
from spuco.utils.random_seed import seed_randomness

# Constants 
MAJORITY = "majority"
MINORITY = "minority"

class SpuCoImageFolder(BaseSpuCoDataset):
    """
        Expects a folder structure of the following type:
        For 2 class setting: (note for K classes, each class must have K subfolders, 1 for a spurious 
        corresponding to each class)
        - train
            - 0 (class idx)
                - 0
                - 1
                - ...
            - 1
            ....
        - val
            ....
        - test 
            ....
    """
    def __init__(
        self,
        root: str,
        label_noise: float = 0.0,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
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

        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        
        num_classes = self._count_directories(os.path.join(root, TRAIN_SPLIT))
        super().__init__(
            root=root, 
            split=split,
            transform=transform,
            num_classes=num_classes,
            verbose=verbose
        )
        self.label_noise = label_noise
        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])
        
    def _count_directories(self, path):
        # Get a list of all items in the directory
        items = os.listdir(path)

        # Filter the list to include only directories
        directories = [item for item in items if os.path.isdir(os.path.join(path, item))]

        # Count the number of directories
        num_directories = len(directories)

        return num_directories

        
    def load_data(self) -> SourceData:
        """
        Loads data from expected image folder structure and sets spurious labels, label noise.

        :return: The spurious correlation dataset.
        :rtype: SourceData, List[int], List[int]
        """
        self.dset_dir = os.path.join(self.root, self.split)
 
        if not os.path.exists(self.dset_dir):
            raise RuntimeError(f"Dataset not found {self.dset_dir}")
        try:
            self.data = SourceData(verbose=False)
            
            # Iterate through folder structure, load file names of images and core and spurious labels
            class_dirs = [item for item in os.listdir(self.dset_dir) if os.path.isdir(os.path.join(self.dset_dir, item))]
            for class_idx, class_dir in tqdm(enumerate(class_dirs), desc="Loading classes", disable=not self.verbose, total=len(class_dirs)):
                class_dir = os.path.join(self.dset_dir, class_dir)
                spurious_dirs = [item for item in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, item))]
                for spurious_idx, spurious_dir in enumerate(spurious_dirs):
                    spurious_dir = os.path.join(class_dir, spurious_dir)
                    for f in os.listdir(spurious_dir):
                        self.data.X.append(os.path.join(spurious_dir, f))
                        self.data.labels.append(class_idx)
                        self.data.spurious.append(spurious_idx)
            
            # Noisy Labels implementation 
            if self.label_noise > 0.0:
                self.data.clean_labels = deepcopy(self.data.labels)
                self.is_noisy_label = torch.zeros(len(self.data.X))
                self.is_noisy_label[torch.randperm(len(self.data.X))[:int(self.label_noise * len(self.data.X))]] = 1
                self.data.labels = [1 - label if self.is_noisy_label[i] else label for i, label in enumerate(self.data.clean_labels)]
        except:
            raise RuntimeError(f"Dataset corrupted, please fix directory structure.")
            
        # Skip validation 
        self.skip_group_validation = True 
        
        return self.data, list(range(self.num_classes)), list(range(self.num_classes))
    
    def load_image(self, filename: str):
        image = Image.open(filename).convert("RGB")
        return image 
    
    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        :param index: Index of the item to get.
        :type index: int
        :return: A tuple of (sample, target) where target is class_index of the target class.
        :rtype: tuple
        """
        
        image = self.base_transform(self.load_image(self.data.X[index]))
        label = self.data.labels[index]
        if self.transform is None:
            return image, label
        else:
            return self.transform(image), label