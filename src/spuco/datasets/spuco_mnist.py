import itertools
import random
from copy import deepcopy
from enum import Enum
from typing import Callable, List, Optional

import matplotlib.cm as cm
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset
from tqdm import tqdm

from spuco.datasets import (TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT,
                            BaseSpuCoDataset, SourceData,
                            SpuriousFeatureDifficulty)
from spuco.datasets.spuco_mnist_config import config
from spuco.utils.random_seed import seed_randomness


class ColourMap(Enum):
    HSV = "hsv"
    
class SpuCoMNIST(BaseSpuCoDataset):
    """
    A dataset consisting of images from the MNIST dataset
    with added spurious features to create a spurious MNIST dataset.
    """

    def __init__(
        self,
        root: str,
        spurious_feature_difficulty: SpuriousFeatureDifficulty,
        classes: List[List[int]],
        spurious_correlation_strength = 0.0,
        label_noise: float = 0.0,
        core_feature_noise: float = 0.0,
        color_map: ColourMap = ColourMap.HSV,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
        verbose: bool = False
    ):
        """
        Initializes the SpuCoMNIST dataset.

        :param root: The root directory of the dataset.
        :type root: str
        :param spurious_feature_difficulty: The difficulty level of the spurious feature.
        :type spurious_feature_difficulty: SpuriousFeatureDifficulty
        :param classes: The list of class labels for each digit.
        :type classes: List[List[int]]
        :param spurious_correlation_strength: The strength of the spurious feature correlation. Default is 0.
        :param color_map: The color map to use. Default is ColourMap.HSV.
        :type color_map: ColourMap
        :param split: The dataset split to load. Default is "train".
        :type split: str
        :param transform: The data transformation function. Default is None.
        :type transform: Optional[Callable]
        """

        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(
            root=root, 
            spurious_correlation_strength=spurious_correlation_strength,
            spurious_feature_difficulty=spurious_feature_difficulty,
            split=split,
            label_noise=label_noise,
            core_feature_noise=core_feature_noise,
            transform=transform,
            num_classes=len(classes),
            verbose=verbose
        )

        self.classes = classes
        self.colors = self.init_colors(color_map)

    def validate_data(self):
        """
        Validates that the generated dataset has been loaded correctly.
        """
        pass 

    def load_data(self) -> SourceData:
        """
        Loads the MNIST dataset and generates the spurious correlation dataset.

        :return: The spurious correlation dataset.
        :rtype: SourceData
        """
        self.mnist = torchvision.datasets.MNIST(
            root=self.root, 
            train=self.split != TEST_SPLIT, 
            download=self.download,
            transform=T.Compose([
                T.ToTensor(),
                T.Lambda(lambda x: torch.cat([x, x, x], dim=0))  # convert grayscale to RGB
            ])
        )

        # Return predetermined train / val split
        if self.split == TRAIN_SPLIT:
            self.mnist = Subset(
                dataset=self.mnist, 
                indices=[i for i in range(len(self.mnist)) if i not in config[VAL_SPLIT]]
            )
        elif self.split == VAL_SPLIT:
            self.mnist = Subset(
                dataset=self.mnist, 
                indices=config[VAL_SPLIT]
            )

        self.data = SourceData(self.mnist)
        
        # Validate Classes
        assert SpuCoMNIST.validate_classes(self.classes), "Classes should be disjoint and only contain elements 0<= label <= 9"

        # Get New Labels
        kept = []
        for i, label in enumerate(self.data.labels):
            for class_idx, latent_class in enumerate(self.classes):
                if label in latent_class:
                    self.data.labels[i] = class_idx
                    kept.append(i)
        
        self.data.X = [self.data.X[i] for i in kept]
        self.data.labels = [self.data.labels[i] for i in kept]
        
        # Partition indices by new labels
        self.partition = {}
        for i, label in enumerate(self.data.labels):
            if label not in self.partition:
                self.partition[label] = []
            self.partition[label].append(i)
        
        # Train / Val: Add spurious correlation iteratively for each class
        self.data.spurious = [-1] * len(self.data.X)
        if self.split == TRAIN_SPLIT or (self.split == VAL_SPLIT and self.spurious_correlation_strength != 0):
            assert self.spurious_correlation_strength != 0, f"spurious correlation strength must be specified and > 0 for split={TRAIN_SPLIT}"

            # Determine label noise idx
            if self.label_noise > 0:
                self.data.clean_labels = deepcopy(self.data.labels)
                self.is_noisy_label = torch.zeros(len(self.data.X))
                self.is_noisy_label[torch.randperm(len(self.data.X))[:int(self.label_noise * len(self.data.X))]] = 1

            # Determine feature noise idx
            if self.core_feature_noise > 0:
                self.data.core_feature_noise = torch.zeros(len(self.data.X))
                self.data.core_feature_noise[torch.randperm(len(self.data.X))[:int(self.core_feature_noise * len(self.data.X))]] = 1

            for label in tqdm(self.partition.keys(), desc="Adding spurious feature", disable=not self.verbose):

                # Get spurious correlation strength for this class
                spurious_correlation_strength = self.spurious_correlation_strength
                if type(self.spurious_correlation_strength) == list:
                    spurious_correlation_strength = self.spurious_correlation_strength[label]

                # Randomly permute and choose which points will have spurious feature (avoids issue of sampling leading to 
                # too many examples having spurious ---> no examples for some groups)
                is_spurious = torch.zeros(len(self.partition[label]))
                is_spurious[torch.randperm(len(self.partition[label]))[:int(spurious_correlation_strength * len(self.partition[label]))]] = 1

                # Get what the other labels could be for this class (all but spurious)
                other_labels = [x for x in range(len(self.classes)) if x != label]

                # Determine background of every example 
                background_label = torch.tensor([label if is_spurious[i] else other_labels[i % len(other_labels)] for i in range(len(self.partition[label]))])
                background_label = background_label[torch.randperm(len(background_label))]

                # Create and apply background for all examples
                for i, idx in enumerate(self.partition[label]):
                    self.data.spurious[idx] = background_label[i].item()
                    background = SpuCoMNIST.create_background(self.spurious_feature_difficulty, self.colors[self.data.spurious[idx]])

                    # Feature noise is a random mask applied to core feature
                    core_feature_noise = torch.ones_like(self.data.X[idx]) >= 1.0 # default noise is no noise
                    if self.data.core_feature_noise[idx]:
                        core_feature_noise = (torch.randn_like(self.data.X[idx][0, :, :]) > 0.25).unsqueeze(dim=0).repeat(3, 1, 1)
                    self.data.X[idx] = self.data.X[idx] * core_feature_noise
                    self.data.X[idx] = (background * (self.data.X[idx] == 0)) + self.data.X[idx]
                    
                    # If noisy label
                    if self.is_noisy_label[idx]:
                        self.data.labels[idx] = random.choice(other_labels)


        # Test / Val: Create spurious balanced test set
        else:
            for label in tqdm(self.partition.keys(), desc="Adding spurious feature", disable=not self.verbose):
                # Generate balanced background labels
                background_label = torch.tensor([i % len(self.classes) for i in range(len(self.partition[label]))])
                background_label = background_label[torch.randperm(len(background_label))]

                # Create and apply background for all examples
                for i, idx in enumerate(self.partition[label]):
                    self.data.spurious[idx] = background_label[i].item()
                    background = SpuCoMNIST.create_background(self.spurious_feature_difficulty, self.colors[self.data.spurious[idx]])
                    self.data.X[idx] = (background * (self.data.X[idx] == 0)) + self.data.X[idx]   

        # Return data, list containing all class labels, list containing all spurious labels
        return self.data, range(len(self.classes)), range(len(self.classes))

    def init_colors(self, color_map: ColourMap) -> List[List[float]]:
        """
        Initializes the color values for the spurious features.

        :param color_map: The color map to use for the spurious features. Should be a value from the `ColourMap`
            enum class.
        :type color_map: ColourMap
        
        :return: The color values for the spurious features.
        :rtype: List[List[float]]
        """
        color_map = cm.get_cmap(color_map.value)
        cmap_vals = np.arange(0, 1, step=1 / len(self.classes))
        colors = []
        for i in range(len(self.classes)):
            rgb = color_map(cmap_vals[i])[:3]
            rgb = [float(x) for x in np.array(rgb)]
            colors.append(rgb)
        # Append black as no-spurious background
        colors.append([0., 0., 0.])
        return colors
    
    @staticmethod
    def validate_classes(classes: List[List[int]]) -> bool:
        """
        Validates that the classes provided to the `SpuCoMNIST` dataset are disjoint and only contain integers
        between 0 and 9.

        :param classes: The classes to be included in the dataset, where each element is a list of integers
            representing the digits to be included in a single class.
        :type classes: List[List[int]]

        :return: Whether the classes are valid.
        :rtype: bool
        """
        sets = [set(latent_class) for latent_class in classes]

        for i in range(len(sets)):
            if any([x < 0 or x > 9 for x in sets[i]]):
                return False
            for j in range(i + 1, len(sets)):
                if sets[i].intersection(sets[j]):
                    return False
        return True

    @staticmethod
    def create_background(spurious_feature_difficulty: SpuriousFeatureDifficulty, hex_code: str) -> torch.Tensor:
        """
        Generates a tensor representing a background image with a specified spurious feature difficulty and hex code color.

        :param spurious_feature_difficulty: The difficulty level of the spurious feature to add to the background image.
        :type spurious_feature_difficulty: SpuriousFeatureDifficulty

        :param hex_code: The hex code of the color to use for the background image.
        :type hex_code: str

        :return: A tensor representing the generated background image.
        :rtype: torch.Tensor
        """
        background = SpuCoMNIST.rgb_to_mnist_background(hex_code)
        if spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_EASY:
            return background
        elif spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_MEDIUM:
            unmask_points = torch.tensor(list(itertools.product(range(4), range(4))))
            mask = SpuCoMNIST.compute_mask(unmask_points)
        elif spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_HARD:
            unmask_points = torch.tensor(list(itertools.product(range(2), range(2))))
            mask = SpuCoMNIST.compute_mask(unmask_points)
        elif spurious_feature_difficulty == SpuriousFeatureDifficulty.VARIANCE_EASY:
            unmask_points = torch.tensor(list(itertools.product(range(7), range(7))))
            mask = SpuCoMNIST.compute_mask(unmask_points)
        elif spurious_feature_difficulty == SpuriousFeatureDifficulty.VARIANCE_MEDIUM:
            all_points = torch.tensor(list(itertools.product(range(14), range(14))))
            unmask_points = all_points[torch.randperm(len(all_points))[:49]]
            mask = SpuCoMNIST.compute_mask(unmask_points)
        elif spurious_feature_difficulty == SpuriousFeatureDifficulty.VARIANCE_HARD:
            all_points = torch.tensor(list(itertools.product(range(28), range(28))))
            unmask_points = all_points[torch.randperm(len(all_points))[:49]]
            mask = SpuCoMNIST.compute_mask(unmask_points)
        return background * mask
    
    @staticmethod
    def compute_mask(unmask_points: torch.Tensor) -> torch.Tensor:
        """
        Computes a binary mask based on the unmasked points.

        :param unmask_points: The coordinates of the unmasked points.
        :type unmask_points: torch.Tensor
        :return: The binary mask with 1s at the unmasked points and 0s elsewhere.
        :rtype: torch.Tensor
        """
        rows = torch.tensor([point[0] for point in unmask_points])
        cols = torch.tensor([point[1] for point in unmask_points])
        mask = torch.zeros((3,28,28))
        mask[:, rows, cols] = 1.
        return mask
    
    @staticmethod
    def rgb_to_mnist_background(rgb: List[float]) -> torch.Tensor:
        """
        Converts an RGB color to a MNIST background tensor.

        :param rgb: The RGB color values.
        :type rgb: List[float]
        :return: The MNIST background tensor with the specified RGB color.
        :rtype: torch.Tensor
        """
        return torch.tensor(rgb).unsqueeze(1).unsqueeze(2).repeat(1, 28, 28)  