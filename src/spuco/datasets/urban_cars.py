from enum import Enum
from tqdm import tqdm 
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
from PIL import Image
import glob
import os

from spuco.datasets import BaseSpuCoCompatibleDataset

class UrbanCarsSpuriousLabel(Enum):
    BOTH = "both"
    BG = "bg"
    CO_OCCUR = "co_occur"
    
class UrbanCars(BaseSpuCoCompatibleDataset):
    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    def __init__(
        self,
        root: str,
        split: str,
        spurious_label_type: UrbanCarsSpuriousLabel = UrbanCarsSpuriousLabel.BOTH,
        verbose: bool = False,
        transform=None,
    ):
        super().__init__()
        self.verbose = verbose
        self._spurious_label_type = spurious_label_type
        

        # Load dataset
        if self.verbose:
            print("Loading dataset")
        self.split = split
        if self.split == "train":
            bg_ratio = 0.95
            co_occur_obj_ratio = 0.95
        elif self.split in ["val", "test"]:
            bg_ratio = 0.5
            co_occur_obj_ratio = 0.5
        else:
            raise NotImplementedError

        self._bg_ratio = bg_ratio
        self._co_occur_obj_ratio = co_occur_obj_ratio
        
        assert os.path.exists(os.path.join(root))
        
        self.transform = transform

        ratio_combination_folder_name = (
            f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        )
        img_root = os.path.join(
            root, ratio_combination_folder_name, split
        )

        self._img_fpath_list = []
        self._obj_bg_co_occur_obj_label_list = []

        for obj_id, obj_name in enumerate(self.obj_name_list):
            for bg_id, bg_name in enumerate(self.bg_name_list):
                for co_occur_obj_id, co_occur_obj_name in enumerate(
                    self.co_occur_obj_name_list
                ):
                    dir_name = (
                        f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    )
                    dir_path = os.path.join(img_root, dir_name)
                    assert os.path.exists(dir_path)

                    img_fpath_list = glob.glob(os.path.join(dir_path, "*.jpg"))
                    self._img_fpath_list += img_fpath_list

                    self._obj_bg_co_occur_obj_label_list += [
                        (obj_id, bg_id, co_occur_obj_id)
                    ] * len(img_fpath_list)

        # Set up required properties for SpuCo
        self._labels = [x[0] for x in self._obj_bg_co_occur_obj_label_list]
        self._num_classes = len(set(self._labels))
        self._both_spurious = [tuple(x[1:]) for x in self._obj_bg_co_occur_obj_label_list]
        self._bg_spurious = [x[0] for x in self._both_spurious]
        self._co_occur_spurious = [x[1] for x in self._both_spurious]
        
        # Create group partitions
        self._both_group_partition = {}
        self._bg_group_partition = {}
        self._co_occur_group_partition = {}
        for label in self._labels:
            for spurious_label in self._both_spurious:
                self._both_group_partition[(label, spurious_label)] = []
                self._bg_group_partition[(label, spurious_label[0])] = []
                self._co_occur_group_partition[(label, spurious_label[1])] = []
        for i in tqdm(range(len(self._img_fpath_list)), desc="Creating group partitions", disable=not(self.verbose)):
            label = self._labels[i]
            spurious_label = self._both_spurious[i]
            self._both_group_partition[(label, spurious_label)].append(i)
            self._bg_group_partition[(label, spurious_label[0])].append(i)
            self._co_occur_group_partition[(label, spurious_label[1])].append(i)

        if self.verbose:
            print("Computing group weights")
        self._both_group_weights = self._compute_group_weights(self._both_group_partition)
        self._bg_group_weights = self._compute_group_weights(self._bg_group_partition)
        self._co_occur_group_weights = self._compute_group_weights(self._co_occur_group_partition)

        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if self.verbose:
            print("Done!")
            
    def _compute_group_weights(self, group_partition):
        group_weights = {}
        for group_label in group_partition.keys():
            group_weights[group_label] = len(group_partition[group_label]) / len(self._img_fpath_list)
        return group_weights
     
    @property           
    def group_partition(self) -> Dict[Tuple[int, any], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        if self._spurious_label_type == UrbanCarsSpuriousLabel.BOTH:
            return self._both_group_partition 
        elif self._spurious_label_type == UrbanCarsSpuriousLabel.BG:
            return self._bg_group_partition
        elif self._spurious_label_type == UrbanCarsSpuriousLabel.CO_OCCUR:
            return self._co_occur_group_partition
        else:
            raise ValueError("Invalid spurious feature type")
    @property
    def group_weights(self) -> Dict[Tuple[int, any], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        if self._spurious_label_type == UrbanCarsSpuriousLabel.BOTH:
            return self._both_group_weights
        elif self._spurious_label_type == UrbanCarsSpuriousLabel.BG:
            return self._bg_group_weights
        elif self._spurious_label_type == UrbanCarsSpuriousLabel.CO_OCCUR:
            return self._co_occur_group_weights
        else:
            raise ValueError("Invalid spurious feature type")
    
    @property
    def spurious(self) -> List[any]:
        """
        List containing spurious labels for each example
        """
        if self._spurious_label_type == UrbanCarsSpuriousLabel.BOTH:
            return self._both_spurious
        elif self._spurious_label_type == UrbanCarsSpuriousLabel.BG:
            return self._bg_spurious
        elif self._spurious_label_type == UrbanCarsSpuriousLabel.CO_OCCUR:
            return self._co_occur_spurious
        else:
            raise ValueError("Invalid spurious feature type")

    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        return self._labels
    
    @property
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return self._num_classes
    
    def __len__(self):
        return len(self._img_fpath_list)

    def __getitem__(self, index):
        img_fpath = self._img_fpath_list[index]
        label = self._labels[index]

        img = Image.open(img_fpath)
        img = self.base_transform(img.convert("RGB"))
        if self.transform is not None:
            img = self.transform(img)
        return img, label