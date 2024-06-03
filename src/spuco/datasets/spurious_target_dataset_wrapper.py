from torch.utils.data import Dataset 
from typing import List, Optional

import torch 

class SpuriousTargetDatasetWrapper(Dataset):
    """
    Wrapper class that takes a Dataset and the spurious labels of the data
    and returns a dataset where the labels are the spurious labels. 
    """
    def __init__(
        self,
        dataset: Dataset,
        spurious_labels: List[int],
        num_classes: Optional[int] = None
    ):
        """
        Initialize an instance of SpuriousTargetDatasetWrapper.
        
        If num_classes specified, doesn't not iterate over examples with "no spurious feature" -> indicated by spurious label >= num_classes. 

        :param dataset: The original dataset.
        :type dataset: Dataset
        :param spurious_labels: The spurious labels corresponding to the data.
        :type spurious_labels: List[int]
        """
        
        self.dataset = dataset
        self.spurious_labels = spurious_labels
        self.num_classes = num_classes
        self.idx = range(len(self.dataset))
        
        if self.num_classes is not None:
            self.idx = []
            for i, spurious_label in enumerate(self.spurious_labels):
                if spurious_label < self.num_classes:
                    self.idx.append(i)
            
    def __getitem__(self, index):
        """
        Get an item from the dataset.

        :param index: The index of the item to retrieve.
        :type index: int
        :return: A tuple containing the input data and the spurious label.
        :rtype: Tuple[Any, int]
        """
        index = self.idx[index]
        try:
            return (self.dataset.__getitem__(index)[0], torch.tensor(self.spurious_labels[index]))
        except:
            raise ValueError("Spurious attribute prediction not supported with non-int spurious labels")
        
    
    def __len__(self):
        """
        Get the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.idx)
        