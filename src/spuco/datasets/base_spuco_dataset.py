from typing import Callable, Optional
from torch.utils.data import Dataset
from enum import Enum
from abc import ABC, abstractmethod

class SpuriousFeatureDifficulty(Enum):
    """
    Enumeration class for spurious feature difficulty levels.

    Each level corresponds to a combination of the magnitude and variance
    of the spurious feature.
    """
    MAGNITUDE_EASY = (
        "magnitude_easy",
        "Magnitude definition of difficulty. Easy <-> Large Magnitude"
    )
    MAGNITUDE_MEDIUM = (
        "magnitude_medium",
        "Magnitude definition of difficulty. Medium <-> Medium Magnitude"
    )
    MAGNITUDE_HARD = (
        "magnitude_hard",
        "Magnitude definition of difficulty. Hard <-> Small Magnitude"
    )
    VARIANCE_EASY = (
        "variance_easy",
        "Variance definition of difficulty. Easy <-> Small Variance"
    )
    VARIANCE_MEDIUM = (
        "variance_medium",
        "Variance definition of difficulty. Medium <-> Medium Variance"
    )
    VARIANCE_HARD = (
        "variance_hard",
        "Variance definition of difficulty. Hard <-> Large Variance"
    )

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
        self.labels =[]

        if data is not None:
            for x, label in data:
                self.X.append(x)
                self.labels.append(label)

class BaseSpuCoDataset(Dataset, ABC):
    def __init__(
        self,
        root: str,
        spurious_correlation_strength: float,
        spurious_feature_difficulty: SpuriousFeatureDifficulty,
        num_classes: int,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        """
        Initializes the dataset.

        :param root: Root directory of the dataset.
        :type root: str
        :param spurious_correlation_strength: Strength of spurious correlation.
        :type spurious_correlation_strength: float
        :param spurious_feature_difficulty: Difficulty of spurious features.
        :type spurious_feature_difficulty: SpuriousFeatureDifficulty
        :param train: If True, returns the training dataset. Otherwise, returns the test dataset. Default is True.
        :type train: bool, optional
        :param transform: A function/transform that takes in a sample and returns a transformed version. Default is None.
        :type transform: callable, optional
        :param download: If True, downloads the dataset from the internet and puts it in the root directory. If the dataset is already downloaded, it is not downloaded again. Default is False.
        :type download: bool, optional
        """
        self.root = root 
        self.spurious_correlation_strength = spurious_correlation_strength
        self.spurious_feature_difficulty = spurious_feature_difficulty
        self.num_classes = num_classes
        self.train = train
        self.transform = transform
        self.download = download

    @abstractmethod
    def validate_data(self):
        """
        Validates the dataset.
        """
        pass

    def initialize(self):
        """
        Initializes the dataset.
        """
        # Validate Data
        if not self.download:
            self.validate_data()

        # Load Data
        self.data = self.load_data()

        # Group Partition
        self.group_partition = {}
        for i, group_label in enumerate(zip(self.data.labels, self.spurious)):
            if group_label not in self.group_partition:
                self.group_partition[group_label] = []
            self.group_partition[group_label].append(i)

        # Group Weights
        self.group_weights = None
        if self.train:
            self.group_weights = {}
            for key in self.group_partition.keys():
                self.group_weights[key] = len(self.group_partition[key]) / len(self.data.X)

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
