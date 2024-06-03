from typing import Dict, List, Tuple, Optional

from tqdm import tqdm
from wilds.datasets.wilds_dataset import WILDSDataset

from spuco.datasets import BaseSpuCoCompatibleDataset


class WILDSDatasetWrapper(BaseSpuCoCompatibleDataset):
    """
    Wrapper class that wraps WILDSDataset into a Dataset to be compatible with SpuCo.
    """
    def __init__(
        self,
        dataset: WILDSDataset,
        metadata_spurious_label: str,
        verbose=False,
        subset_indices: Optional[List[int]] = None
    ):
        """
        Wraps  WILDS Dataset into a Dataset object. 

        :param dataset: The source WILDS dataset
        :type dataset: WILDDataset
        :param metadata_spurious_label: String name of property in metadata_map corresponding to spurious target
        :type metadata_spurious_label: str 
        :param verbose: Show logs
        :type verbose: bool
        """
        super().__init__()

        self.dataset = dataset
        self._num_classes = dataset.n_classes 
        
        # Subset if needed
        self.indices = range(len(dataset))
        if subset_indices is not None:
            self.indices = subset_indices

        # Get index in meta data array corresponding to spurious target 
        spurious_target_idx = dataset.metadata_fields.index(metadata_spurious_label)

        # Get labels 
        self._labels = dataset.y_array.long()[self.indices].tolist()

        # Get spurious labels
        self._spurious = dataset.metadata_array[:, spurious_target_idx].long()[self.indices].tolist()

            
        # Create group partition using labels and spurious labels
        self._group_partition = {}
        for i, group_label in tqdm(
            enumerate(zip(self._labels, self._spurious)),
            desc="Partitioning data indices into groups",
            disable=not verbose,
            total=len(self.dataset)
        ):
            if group_label not in self._group_partition:
                self._group_partition[group_label] = []
            self._group_partition[group_label].append(i)
        
        # Set group weights based on group sizes
        self._group_weights = {}
        for group_label in self._group_partition.keys():
            self._group_weights[group_label] = len(self._group_partition[group_label]) / len(self.dataset)
    
    def __getitem__(self, index):
        """
        Retrieves an item from the dataset.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        """
        index = self.indices[index]
        source_tuple = self.dataset.__getitem__(index)
        return (source_tuple[0], source_tuple[1])
    
    def __len__(self):
        """
        Returns the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.indices)
