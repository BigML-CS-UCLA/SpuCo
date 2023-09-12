from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

class GroupLabeledDatasetWrapper(Dataset):
    def __init__(
        self, 
        dataset: Dataset,
        group_partition: Dict[Tuple[int, int], int],
        subset_indices: Optional[List[int]] = None
    ):
        """
        Initializes a GroupLabeledDataset.

        :param dataset: The underlying dataset.
        :type dataset: torch.utils.data.Dataset

        :param group_partition: The group partition dictionary mapping indices to group labels.
        :type group_partition: Dict[Tuple[int, int], int]

        :param subset_indices: Optional list of subset indices to consider from the dataset. Defaults to None.
        :type subset_indices: Optional[List[int]]
        """
        self.dataset = dataset
        self.group = torch.zeros(len(self.dataset))
        self.group_partition = group_partition
        
        group_idx = 0
        for key in sorted(group_partition.keys()):
            self.group[group_partition[key]] = group_idx
            group_idx += 1 
        self.num_groups = len(group_partition.keys())
        self.group = self.group.long().tolist()
        
        # Subset if needed
        self.indices = range(len(dataset))
        if subset_indices is not None:
            self.indices = subset_indices
        
    def __getitem__(self, index: int):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        index = self.indices[index]
        source_tuple = self.dataset.__getitem__(index)
        return (source_tuple[0], source_tuple[1], self.group[index])
    
    def __len__(self):
        """
        Returns the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.indices)