import os
import random
import tarfile
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from spuco.datasets import (TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT,
                            BaseSpuCoDataset, SourceData)
from spuco.utils.random_seed import seed_randomness

# Constants
DOWNLOAD_URL = "https://ucla.box.com/shared/static/zrsx09ik7lqad1e389w06f1c4nbajo9z"
DATASET_NAME = "spuco_birds"
LANDBIRDS = "landbirds"
WATERBIRDS = "waterbirds"
LAND = "land"
WATER = "water"
MAJORITY_SIZE = {
    TRAIN_SPLIT: 10000,
    VAL_SPLIT: 500,
    TEST_SPLIT: 500,
}
MINORITY_SIZE = {
    TRAIN_SPLIT: 500,
    VAL_SPLIT: 25,
    TEST_SPLIT: 500,
}

class SpuCoBirds(BaseSpuCoDataset):
    """
    Subset of SpuCoAnimals only including Bird classes.
    """

    def __init__(
        self,
        root: str,
        download: bool = True,
        label_noise: float = 0.0,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
        verbose: bool = False
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str

        :param download: Whether to download the dataset.
        :type download: bool, optional

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

        super().__init__(
            root=root, 
            split=split,
            transform=transform,
            num_classes=2,
            verbose=verbose
        )
        self.download = download
        self.label_noise = label_noise
        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor()
        ])
        
    def load_data(self) -> SourceData:
        """
        Loads SpuCoBirds and sets spurious labels, label noise.

        :return: The spurious correlation dataset.
        :rtype: SourceData, List[int], List[int]
        """
        
        self.dset_dir = os.path.join(self.root, DATASET_NAME, self.split)
        if not os.path.exists(self.dset_dir):
            if not self.download:
                raise RuntimeError(f"Dataset not found {self.dset_dir}, run again with download=True")
            self._download_data()
            self._untar_data()
            os.remove(self.filename)
            
        try:
            self.data = SourceData()
            
            # Landbirds Land 
            landbirds_land = os.listdir(os.path.join(self.dset_dir, f"{LANDBIRDS}/{LAND}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{LANDBIRDS}/{LAND}", x)) for x in landbirds_land])
            self.data.labels.extend([0] * len(landbirds_land))
            self.data.spurious.extend([0] * len(landbirds_land))
            assert len(landbirds_land) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_land]. Expected {MAJORITY_SIZE[self.split]} files got {len(landbirds_land)}"
            
            # Landbirds Water 
            landbirds_water = os.listdir(os.path.join(self.dset_dir, f"{LANDBIRDS}/{WATER}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
            self.data.labels.extend([0] * len(landbirds_water))
            self.data.spurious.extend([1] * len(landbirds_water))   
            assert len(landbirds_water) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [landbirds_water]. Expected {MINORITY_SIZE[self.split]} files got {len(landbirds_water)}"
            
            # Waterbirds Land
            waterbirds_land = os.listdir(os.path.join(self.dset_dir, f"{WATERBIRDS}/{LAND}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
            self.data.labels.extend([1] * len(waterbirds_land))
            self.data.spurious.extend([0] * len(waterbirds_land))
            assert len(waterbirds_land) == MINORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_land]. Expected {MINORITY_SIZE[self.split]} files got {len(waterbirds_land)}"
            
            # Waterbirds Water
            waterbirds_water = os.listdir(os.path.join(self.dset_dir, f"{WATERBIRDS}/{WATER}"))
            self.data.X.extend([str(os.path.join(self.dset_dir, f"{WATERBIRDS}/{WATER}", x)) for x in waterbirds_water])
            self.data.labels.extend([1] * len(waterbirds_water))
            self.data.spurious.extend([1] * len(waterbirds_water)) 
            assert len(waterbirds_water) == MAJORITY_SIZE[self.split], f"Dataset corrupted or missing files [waterbirds_water]. Expected {MAJORITY_SIZE[self.split]} files got {len(waterbirds_water)}"
            
            if self.label_noise > 0.0:
                self.data.clean_labels = deepcopy(self.data.labels)
                self.is_noisy_label = torch.zeros(len(self.data.X))
                self.is_noisy_label[torch.randperm(len(self.data.X))[:int(self.label_noise * len(self.data.X))]] = 1
                self.data.labels = [1 - label if self.is_noisy_label[i] else label for i, label in enumerate(self.data.clean_labels)]
        except:
            raise RuntimeError(f"Dataset corrupted, please delete {self.dset_dir} and run again with download=True")
            
        return self.data, list(range(2)), list(range(2))

    def _download_data(self):
        self.filename = f"{self.root}/{DATASET_NAME}.tar.gz"

        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status()

        with open(self.filename, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=1024), total=2952065, desc="Downloading SpuCoBirds", unit="KB"):
                file.write(chunk)
    
    def _untar_data(self):
        # Open the tar.gz file
        with tarfile.open(self.filename, "r:gz") as tar:
            # Extract all files to the specified output directory
            tar.extractall(self.root)
            
    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        :param index: Index of the item to get.
        :type index: int
        :return: A tuple of (sample, target) where target is class_index of the target class.
        :rtype: tuple
        """
        
        image = self.base_transform(Image.open(self.load_image(self.data.X[index])))
        label = self.data.labels[index]
        if self.transform is None:
            return image, label
        else:
            return self.transform(image), label