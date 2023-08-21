import os
import pickle
import random
from typing import Callable, Optional

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from spuco.datasets import (TRAIN_SPLIT, MASK_CORE, MASK_SPURIOUS, 
                            BaseSpuCoDataset, SourceData,
                            SpuCoBirds, SpuCoDogs)
from spuco.utils.random_seed import seed_randomness

MASK_URL = "https://ucla.box.com/shared/static/ebgwn3rb6es45rubfa2k68zhsam7qeis"
MASK_FNAME = "spuco_animals_masks.pkl"

class SpuCoAnimals(BaseSpuCoDataset):
    """
    Next, we introduce SpuCoAnimals, a large-scale vision dataset curated from ImageNet with two realistic spurious correlations. 

    SpuCoAnimals has 4 classes: 

    - landbirds
    - waterbirds
    - small dog breeds
    - big dog breeds.

    Waterbirds and Landbirds are spuriously correlated with *water* and *land* backgrounds, respectively. Small dogs and big dogs are spuriously correlated with *indoor* and *outdoor* backgrounds, respectively.
    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        label_noise: float = 0.0,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
        mask_type: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str

        :param download: Whether to download the dataset. Defaults to True.
        :type download: bool, optional

        :param label_noise: The amount of label noise to apply. Defaults to 0.0.
        :type label_noise: float, optional

        :param split: The split of the dataset. Defaults to TRAIN_SPLIT.
        :type split: str, optional

        :param transform: Optional transform to be applied to the data. Defaults to None.
        :type transform: Callable, optional
        
        :param mask_type: Optionally mask out the spurious or core feature
        :type mask_type: str, optional
        
        :param verbose: Whether to print verbose information during dataset initialization. Defaults to False.
        :type verbose: bool, optional
        """

        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(
            root=root, 
            split=split,
            transform=transform,
            num_classes=4,
            verbose=verbose
        )
        self.download = download
        self.label_noise = label_noise
        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])
        
        self.mask_type = mask_type
        
        # Don't have examples in "all" groups, skip validation
        self.skip_group_validation = True 
        
    def load_data(self) -> SourceData:
        """
        Loads SpuCoAnimals and sets spurious labels, label noise.

        :return: The spurious correlation dataset.
        :rtype: SourceData, List[int], List[int]
        """
        if self.verbose:
            print("Loading SpuCoBirds")
        self.data = SpuCoBirds(
            root=self.root,
            download=self.download,
            label_noise=self.label_noise,
            split=self.split,
            transform=self.transform,
            verbose=self.verbose   
        ).load_data()[0]
        
        if self.verbose:
            print("Loading SpuCoDogs")
        self.dogs_data = SpuCoDogs(
            root=self.root,
            download=self.download,
            label_noise=self.label_noise,
            split=self.split,
            transform=self.transform,
            verbose=self.verbose   
        ).load_data()[0]
        
        # If masks required, download and load
        if self.mask_type is not None:
            self.mask_fname = os.path.join(self.root, MASK_FNAME)
            
            # Download masks (if needed)
            if not os.path.exists(self.mask_fname):
                if self.verbose:
                    print(f"{self.mask_fname} not found, downloading instead.")
                self._download_masks()
                
            if self.verbose: 
                print("Loading Masks")
                
            with open(self.mask_fname, "rb") as f:
                self.masks = pickle.load(f)
        
        self.data.core_feature_noise = None
        self.data.X.extend(self.dogs_data.X)
        self.data.labels.extend([label + 2 for label in self.dogs_data.labels])
        self.data.spurious.extend([label + 2 for label in self.dogs_data.spurious])
        if self.data.clean_labels is not None:   
            self.data.clean_labels.extend([label + 2 for label in self.dogs_data.clean_labels])   
        
        return self.data, list(range(4)), list(range(4))
    
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
        
    def load_image(self, filename: str):
        image = Image.open(filename).convert("RGB")
        
        if self.mask_type is None:
            return image 
        
        image = np.array(image)
        mask_index = int(filename.split("/")[-1].split(".")[0])
        spurious_mask = np.moveaxis(np.stack([self.masks[mask_index]] * 3), 0, 2)
        if self.mask_type == MASK_SPURIOUS:
            image = image * spurious_mask
        elif self.mask_type == MASK_CORE:
            image = image * np.invert(spurious_mask)
        else:
            raise ValueError(f"Mask Type must be one of None, {MASK_CORE}, {MASK_SPURIOUS}")
        
        return Image.fromarray(image)    
    
    def _download_masks(self):
        response = requests.get(MASK_URL, stream=True)
        response.raise_for_status()

        with open(self.mask_fname, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=1024), total=9786292, desc="Downloading Masks for SpuCoAnimals", unit="KB"):
                file.write(chunk)