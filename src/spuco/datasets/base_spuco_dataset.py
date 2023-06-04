from abc import ABC
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm 
from spuco.datasets import BaseSpuCoCompatibleDataset

TRAIN_SPLIT = "train"
VAL_SPLIT= "val"
TEST_SPLIT = "test"

class SpuriousFeatureDifficulty(Enum):
    """
    Enumeration class for spurious feature difficulty levels.

    Each level corresponds to a combination of the magnitude and variance
    of the spurious feature.

    Magnitude definition of difficulty:
        Easy <-> Large Magnitude

        Medium <-> Medium Magnitude

        Hard <-> Small Magnitude

    Variance definition of difficulty:
        Easy <-> Small Variance

        Medium <-> Medium Variance

        Hard <-> Large Variance
    """

    MAGNITUDE_EASY = "magnitude_easy"
    MAGNITUDE_MEDIUM = "magnitude_medium"
    MAGNITUDE_HARD = "magnitude_hard"
    VARIANCE_EASY = "variance_easy"
    VARIANCE_MEDIUM = "variance_medium"
    VARIANCE_HARD = "variance_hard"

class SpuriousCorrelationStrength(Enum):
    UNIFORM = "unform"
    LINEAR = "linear"
    
class SourceData():
    """
    Class representing the source data.

    This class contains the input data and corresponding labels.
    """

    def __init__(self, data=None):
        """
        Initialize the SourceData object.

        Args:
            data (list of tuple, optional): The input data and labels.
        """
        self.X = []
        self.labels = []
        self.spurious = []
        self.clean_labels = None
        self.core_feature_noise = None
        if data is not None:
            for x, label in tqdm(data):
                self.X.append(x)
                self.labels.append(label)

class BaseSpuCoDataset(BaseSpuCoCompatibleDataset, ABC):
    def __init__(
        self,
        root: str,
        num_classes: int,
        split: str = "train",
        transform: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str
        :type spurious_feature_difficulty: SpuriousFeatureDifficulty
        """
        super().__init__()
        self.root = root 
        self._num_classes = num_classes
        assert split == TRAIN_SPLIT or split == VAL_SPLIT or split == TEST_SPLIT, f"split must be one of {TRAIN_SPLIT}, {VAL_SPLIT}, {TEST_SPLIT}"
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.skip_group_validation = False

    def initialize(self):
        """
        Initializes the dataset.
        """
        # Load Data
        self.data, classes, spurious_classes = self.load_data()
        self.num_spurious = len(spurious_classes)
        
        # Group Partition
        self._group_partition = {}
        for i, group_label in enumerate(zip(self.data.labels, self.spurious)):
            if group_label not in self._group_partition:
                self._group_partition[group_label] = []
            self._group_partition[group_label].append(i)

        self._clean_group_partition = None
        if self.data.clean_labels is not None:
            self._clean_group_partition = {}
            for i, group_label in enumerate(zip(self.data.clean_labels, self.spurious)):
                if group_label not in self._clean_group_partition:
                    self._clean_group_partition[group_label] = []
                self._clean_group_partition[group_label].append(i)
            
        # Validate partition sizes
        if not self.skip_group_validation:
            for class_label in classes:
                for spurious_label in spurious_classes:
                    group_label = (class_label, spurious_label)
                    assert group_label in self._group_partition and len(self._group_partition[group_label]) > 0, f"No examples in {group_label}, considering reducing spurious correlation strength"

        # Group Weights
        self._group_weights = {}
        for key in self._group_partition.keys():
            self._group_weights[key] = len(self._group_partition[key]) / len(self.data.X)
                
    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        return self._group_partition 

    @property
    def clean_group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups based on clean labels
        """
        if self._clean_group_partition is None:
            return self._group_partition
        else:
            return self._clean_group_partition 
     
    @property
    def group_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        return self._group_weights
    
    @property
    def spurious(self) -> List[int]:
        """
        List containing spurious labels for each example
        """
        return self.data.spurious

    @property
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        return self.data.labels
    
    @property
    def num_classes(self) -> int:
        """
        Number of classes
        """
        return self._num_classes
    
    def __getitem__(self, index):
        """
        Gets an item from the dataset.

        :param index: Index of the item to get.
        :type index: int
        :return: A tuple of (sample, target) where target is class_index of the target class.
        :rtype: tuple
        """
        if self.transform is None:
            return self.data.X[index], self.data.labels[index]
        else:
            return self.transform(self.data.X[index]), self.data.labels[index]
        
    def __len__(self):
        """
        Gets the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.data.X)
