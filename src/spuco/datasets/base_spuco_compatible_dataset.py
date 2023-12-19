from abc import abstractmethod
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BaseSpuCoCompatibleDataset(Dataset):
    """
    Base class for all SpuCo Compatible Datasets
    """
    def __init__(self):
        super().__init__()
        self._group_partition = None 
        self._group_weights = None 
        self._labels = None 
        self._spurious = None
        self._num_classes = None 
        self.base_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
    def initialize(self):
        """
        Dummy method to ensure compatibility with other spuco datasets
        """
        return 
    
    @property
    def group_partition(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Dictionary partitioning indices into groups
        """
        return self._group_partition 
    
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
        return self._spurious

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
        return len(self.data)
    
    def __getitem__(self, index):
        img_fpath = self.data[index]
        label = self._labels[index]

        img = Image.open(img_fpath)
        img = self.base_transform(img.convert("RGB"))
        if self.transform is not None:
            img = self.transform(img)
        return img, label
