import csv
import os
import random
from glob import glob
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
DATASET_NAME = "corrected_waterbirds"
LANDBIRDS = "landbirds"
WATERBIRDS = "waterbirds"
LAND = "land"
WATER = "water"

data_split = {
    0: 'train',
    1: 'val',
    2: 'test'
}


class SpuCoCorrectedWaterbirds(BaseSpuCoDataset):
    """
   Correlated Waterbirds dataset wrapper.
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
        :param download: If True, downloads the dataset. Default is True.
        :type download: bool, optional
        :param label_noise: The amount of label noise to add to the dataset. Default is 0.0.
        :type label_noise: float, optional
        :param split: The dataset split to use. Default is TRAIN_SPLIT.
        :type split: str, optional
        :param transform: The transform to apply to the dataset. Default is None.
        :type transform: Optional[Callable], optional
        :param verbose: If True, prints verbose information. Default is False.
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
        self.dset_dir = os.path.join(self.root, DATASET_NAME, 'images', self.split)
        self.data = SourceData()
        self.places = {}

        with open(os.path.join(self.root, DATASET_NAME, 'metadata.csv')) as meta_file:
            csv_reader = csv.reader(meta_file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                img_id,	img_filename, y, split_index, place, place_filename = row
                self.places[img_filename.split('/')[-1]] = int(place)

        data_classes = sorted(os.listdir(self.dset_dir))
        print("-"*10, f"indexing {self.split} data", "-"*10)
        for data_class in tqdm(data_classes):
            target = int(data_class)
            class_image_file_paths = glob(
                os.path.join(self.dset_dir, data_class, '*'))
            self.data.X.extend(class_image_file_paths)
            self.data.labels.extend([target] * len(class_image_file_paths))
            self.data.spurious.extend([self.places[img.split('/')[-1]] for img in class_image_file_paths])

        return self.data, list(range(2)), list(range(2))

    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        :param index: Index of the item to get.
        :type index: int
        :return: A tuple of (sample, target) where target is class_index of the target class.
        :rtype: tuple
        """
        
        image = self.base_transform(Image.open(self.data.X[index]).convert('RGB'))
        label = self.data.labels[index]
        if self.transform is None:
            return image, label
        else:
            return self.transform(image), label