from torch.utils.data import Dataset 
from typing import List

class SpuriousTargetDatasetWrapper(Dataset):
    """
    Wrapper class that takes a Dataset and the spurious labels of the data
    and returns a dataset where the labels are the spurious labels. 
    """
    def __init__(
        self,
        dataset: Dataset,
        spurious_labels: List[int]
    ):
        """
        Initialize an instance of SpuriousTargetDatasetWrapper.

        :param dataset: The original dataset.
        :type dataset: Dataset
        :param spurious_labels: The spurious labels corresponding to the data.
        :type spurious_labels: List[int]
        """
        self.dataset = dataset
        self.spurious_labels = spurious_labels

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        :param index: The index of the item to retrieve.
        :type index: int
        :return: A tuple containing the input data and the spurious label.
        :rtype: Tuple[Any, int]
        """
        return (self.dataset.__getitem__(index)[0], self.spurious_labels[index])
    
    def __len__(self):
        """
        Get the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.dataset)
        