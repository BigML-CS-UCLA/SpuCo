from abc import abstractmethod
from typing import Dict, List, Tuple

from torch.utils.data import Dataset


class BaseSpuCoCompatibleDataset(Dataset):
    """
    Base class for all SpuCo Compatible Datasets
    """
    def __init__(self):
        super().__init__()

    def initialize(self):
        """
        Dummy method to ensure compatibility with other spuco datasets
        """
        return 
    
    @property
    @abstractmethod
    def group_partition(self) -> Dict[Tuple[int, any], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        pass 
    
    @property
    @abstractmethod
    def group_weights(self) -> Dict[Tuple[int, any], float]:
        """
        Dictionary containing the fractional weights of each group
        """
        pass 
    
    @property
    @abstractmethod
    def spurious(self) -> List[any]:
        """
        List containing spurious labels for each example
        """
        pass 

    @property
    @abstractmethod
    def labels(self) -> List[int]:
        """
        List containing class labels for each example
        """
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """
        Number of classes
        """
        pass