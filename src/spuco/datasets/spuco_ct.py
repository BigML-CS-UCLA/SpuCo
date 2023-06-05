import os
import random
from copy import deepcopy
from enum import Enum
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from spuco.datasets import (TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT,
                            BaseSpuCoDataset, SourceData,
                            SpuriousCorrelationStrength,
                            SpuriousFeatureDifficulty)
from spuco.utils import convert_labels_to_partition
from spuco.utils.random_seed import seed_randomness


class Shape(Enum):
    CIRCLE = 0
    TRIANGLE = 1
    RECTANGLE = 2
    NONE = 3

SPURIOUS_UNIFORM = [0.9, 0.9, 0.9, 0.9]
SPURIOUS_LINEAR = [0.99, 0.95, 0.9, 0.85]

class SpuCoCT(BaseSpuCoDataset):
    """
    A dataset consisting of images from CT/lungs from RadImageNet
    """
    def __init__(
        self,
        root: str,
        spurious_feature_difficulty: SpuriousFeatureDifficulty,
        spurious_correlation_strength: SpuriousCorrelationStrength,
        label_noise: float = 0.0,
        core_feature_noise: float = 0.0,
        split: str = TRAIN_SPLIT,
        transform: Optional[Callable] = None,
        save: bool = True,
        verbose: bool = False
    ):
        """
        Initializes SpuCoCT object
        """

        assert (spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_EASY 
                or spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_HARD), "SpuCoCT only supports MAGNITUDE EASY and MAGNITUDE HARD spurious feature difficulty"

        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__(
            root=root, 
            split=split,
            num_classes=4,
            transform=transform,
            verbose=verbose
        )

        if spurious_correlation_strength == SpuriousCorrelationStrength.UNIFORM:
            self.spurious_correlation_strength = SPURIOUS_UNIFORM
        elif spurious_correlation_strength == SpuriousCorrelationStrength.LINEAR:
            self.spurious_correlation_strength = SPURIOUS_LINEAR
        else:
            raise ValueError("Invalid argument value, unsupported correlation strength")
        self.spurious_feature_difficulty = spurious_feature_difficulty
        self.label_noise = label_noise
        self.core_feature_noise = core_feature_noise
        self.save = save
        
        self.skip_group_validation = True
        
        if split != TRAIN_SPLIT:
            assert (label_noise > 0 or core_feature_noise > 0), "Label noise and feature noise not allowed if validation or test data"

    def load_data(self) -> SourceData:
        """
        Loads base lung CT scan data (adapted from RadImageNet https://www.radimagenet.com/)
        and adds spurious features based on parameters 

        :return: The spurious correlation dataset.
        :rtype: Tuple[SourceData, int, int]
        """

        self.data = None
        try:
            self.data = torch.load(f"\
                {self.root}\
                /{self.spurious_feature_difficulty}-\
                {self.spurious_correlation_strength}-\
                {self.label_noise}-\
                {self.core_feature_noise}\
            .pt")
        except:
            if self.verbose:
                print("This version of SpuCoCT doesn't exist. Assembling SpuCoCT.")
            
        # Load data into source data
        self.data = SourceData()
        to_tensor = T.ToTensor()
        self.root = os.path.join(self.root, self.split)
        for class_label, class_dir in enumerate(os.listdir(self.root)):
            for file in tqdm(list(os.listdir(os.path.join(self.root, class_dir))), desc=f"loading {class_dir}", disable=not self.verbose):
                self.data.X.append(to_tensor(Image.open(os.path.join(self.root, class_dir, file)))[0])
                self.data.labels.append(class_label)
        self.class_partition = convert_labels_to_partition(self.data.labels)

        # Create spurious patches
        self.spurious_patch = []
        for i in range(3):
            self.spurious_patch.append(SpuCoCT.create_spurious_patch(i, spurious_feature_difficulty=SpuriousFeatureDifficulty))
        if self.verbose:
            print("Spurious patches created")

        # Train / Val: Add spurious correlation iteratively for each class
        self.data.spurious = [-1] * len(self.data.X)
        if self.split == TRAIN_SPLIT or self.split == VAL_SPLIT:
            # Determine label noise idx
            if self.label_noise > 0:
                self.data.clean_labels = deepcopy(self.data.labels)
                self.is_noisy_label = torch.zeros(len(self.data.X))
                self.is_noisy_label[torch.randperm(len(self.data.X))[:int(self.label_noise * len(self.data.X))]] = 1
                if self.verbose:
                        print("Label noise determined.")

            # Determine feature noise idx
            if self.core_feature_noise > 0:
                self.data.core_feature_noise = torch.zeros(len(self.data.X))
                self.data.core_feature_noise[torch.randperm(len(self.data.X))[:int(self.core_feature_noise * len(self.data.X))]] = 1
                if self.verbose:
                        print("Feature noise determined.")

            for label in self.class_partition.keys():
                # Randomly permute and choose which points will have spurious feature (avoids issue of sampling leading to 
                # too many examples having spurious ---> no examples for some groups)
                is_spurious = torch.zeros(len(self.class_partition[label]))
                is_spurious[torch.randperm(len(self.class_partition[label]))[:int(self.spurious_correlation_strength[label] * len(self.class_partition[label]))]] = 1

                # Get what the other labels could be for this class (all but spurious)
                other_labels = [x for x in range(self.num_classes) if x != label]

                # Determine background of every example 
                spurious_label = torch.tensor([label if is_spurious[i] else other_labels[i % len(other_labels)] for i in range(len(self.class_partition[label]))])
                spurious_label = spurious_label[torch.randperm(len(spurious_label))]

                self.process_data(label, spurious_label, other_labels)

        # Test / Val: Create spurious balanced test set
        else:
            for label in self.class_partition.keys():
                # Generate balanced background labels
                spurious_label = torch.tensor([i % self.num_classes for i in range(len(self.class_partition[label]))])
                spurious_label = spurious_label[torch.randperm(len(spurious_label))]

                self.process_data(label, spurious_label)
        
        # Save data
        if self.save:
            print("".join(f"\
                {self.root}/\
                {self.spurious_feature_difficulty}-\
                {self.spurious_correlation_strength}-\
                {self.label_noise}-\
                {self.core_feature_noise}\
            .pt".split()))
            torch.save(self.data, "".join(f"\
                {self.root}/\
                {self.spurious_feature_difficulty}-\
                {self.spurious_correlation_strength}-\
                {self.label_noise}-\
                {self.core_feature_noise}\
            .pt".split()))
        
        # Return data, list containing all class labels, list containing all spurious labels
        return self.data, range(self.num_classes), range(self.num_classes)

    def process_data(self, label: int, background_label: torch.Tensor, other_labels: List[int] = None):
        # Create and apply background for all examples
        mask_start = (150, 20)
        mask_end = (200, 200)
        for i, idx in tqdm(
            enumerate(self.class_partition[label]), 
            desc=f"Adding spurious feature to class {label}", 
            total=len(self.class_partition[label]), 
            disable=not self.verbose
        ):
            self.data.spurious[idx] = background_label[i].item()
            
            # Feature noise is a random mask applied to core feature
            core_feature_noise = torch.ones_like(self.data.X[idx]) >= 1.0 # default noise is no noise
            if self.split == TRAIN_SPLIT and self.data.core_feature_noise[idx]:
                core_feature_noise = (torch.randn_like(self.data.X[idx]) >= -0.5)   

            # Try finding position for spurious, retry and modify mask start
            if self.data.spurious[idx] != Shape.NONE.value:
                num_retries = 0
                spurious_pos = None 
                while (spurious_pos is None and num_retries < 4):
                    spurious_pos = SpuCoCT.place_spurious(
                        self.data.X[idx],
                        self.spurious_patch[self.data.spurious[idx]].shape,
                        mask_start,
                        mask_end                    
                    )
                    self.debugparams = (
                        self.data.X[idx],
                        self.spurious_patch[self.data.spurious[idx]].shape,
                        mask_start,
                        mask_end                    
                    )
                    mask_start = (mask_start[0]-50, mask_start[1])
                    num_retries += 1
                self.data.X[idx] = SpuCoCT.add_spurious_and_noise(self.data.X[idx], self.spurious_patch[self.data.spurious[idx]], spurious_pos, core_feature_noise=core_feature_noise)

            # If TRAIN_SPLIT and noisy label
            if self.split == TRAIN_SPLIT and self.is_noisy_label[idx]:
                self.data.labels[idx] = random.choice(other_labels)
    
    @staticmethod
    def create_spurious_patch(shape: int, spurious_feature_difficulty: SpuriousFeatureDifficulty):
        shape = Shape(shape)
        patch_size = None 
        patch = None 
        if shape == Shape.NONE:
            return None 
        elif shape == Shape.CIRCLE:
            if spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_EASY:
                patch_size = (20, 20)
            else:
                patch_size = (10, 10)
            patch = np.zeros(patch_size)
            def draw_circle(image, center, radius, value):
                x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
                distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                image[distance <= radius] = value
                return image 
            radius = patch_size[0]/2
            center = (patch_size[0]/2, patch_size[1]/2)
            patch = draw_circle(patch, center, radius, 1.0)
            patch = draw_circle(patch, center, radius-2, 0.75)
        elif shape == Shape.TRIANGLE:
            if spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_EASY:
                patch_size = (20, 20)
            else:
                patch_size = (10, 10)
            patch = np.zeros(patch_size)
            def draw_triangle(image, vertices, value):
                vertices = np.array(vertices, dtype=np.int32)
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.fillPoly(mask, [vertices], value)
                image[mask == value] = value
                return image 
            vertices = [(0,0), (patch_size[0]/2, patch_size[1]), (0, patch_size[1])]
            patch = draw_triangle(patch, vertices, 1.)
            vertices = [(0,0), (patch_size[0]/2, patch_size[1]), (0, patch_size[1])]
            patch = draw_triangle(patch, vertices, 0.75)
        elif shape == Shape.RECTANGLE:
            if spurious_feature_difficulty == SpuriousFeatureDifficulty.MAGNITUDE_EASY:
                patch_size = (10, 20)
            else:
                patch_size = (5, 10)
            patch = np.ones(patch_size)
            patch[1:patch_size[0] - 1, 2:patch_size[1] - 2] = 0.75
        else:
            raise ValueError("Invalid spurious label")
        return torch.tensor(patch)
    
    @staticmethod
    def add_spurious_and_noise(tensor_img: torch.Tensor, patch: torch.Tensor, pos: Tuple[int, int], core_feature_noise: torch.Tensor):
        spurious_overlay = torch.zeros_like(tensor_img)
        spurious_overlay[pos[0]:pos[0]+patch.shape[0], pos[1]:pos[1]+patch.shape[1]] = patch
        tensor_img = (tensor_img * (spurious_overlay == 0)) * (core_feature_noise * (spurious_overlay == 0)) + spurious_overlay 
        return tensor_img

    @staticmethod
    def place_spurious(
        tensor_img: torch.Tensor, 
        patch_size: Tuple[int, int], 
        mask_start: Tuple[int, int], 
        mask_end: Tuple[int, int], 
        pixel_brightness: int = 150
    ):
        mask = torch.zeros_like(tensor_img)
        mask[mask_start[0]:mask_end[0], mask_start[1]:mask_end[1]] = 1.
        placement = mask * (tensor_img > pixel_brightness / 255)
        candidates = {}
        SpuCoCT.image = tensor_img
        # Iterate backwards to place spurious feature
        for i in range(mask_end[0], mask_start[0] + patch_size[0], -1):
            for j in range(mask_end[1], mask_start[1] + patch_size[1], -1):
                candidates[(i-patch_size[0], j-patch_size[1])] = torch.mean(placement[i-patch_size[0]:i, j-patch_size[1]:j])
                if torch.mean(placement[i-patch_size[0]:i, j-patch_size[1]:j])  > 0.8:
                    return i-patch_size[0], j-patch_size[1]
        return None