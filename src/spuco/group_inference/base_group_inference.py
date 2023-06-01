from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pickle 

class BaseGroupInference(ABC):
    """
    BaseGroupInference abstract class for inferring group partitions.
    """
    def __init__(self):
        """
        Initializes BaseGroupInference.
        """
        pass 

    @abstractmethod
    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Abstract method for inferring group partitions.

        :return: Dictionary mapping group tuples to indices of examples belonging to each group.
        """
        pass 

    def process_cluster_partition(self, cluster_partition: Dict, class_index: int):
        """
        Processes cluster partition:
        - Converts keys from clusters into (class, spurious) format
        - Converts class indices from class-wise clustering into global (actual trainset) indices

        :param cluster_partition: Dictionary mapping cluster labels to indices of examples.
        :param class_index: Index of the class being processed.
        :return: Processed group partition mapping group tuples to indices of examples belonging to each group.
        """
        assert self.class_partition is not None, "self.class_partition must be defined for processing"
        group_partition = {}
        for new_cluster_label, cluster_label in enumerate(sorted(cluster_partition.keys())):
            group_partition[(class_index, new_cluster_label)] = [self.class_partition[class_index][i] for i in cluster_partition[cluster_label]]
        return group_partition
    
    @staticmethod
    def save_group_partition(group_partition: Dict[Tuple[int, int], List[int]], prefix: str):
        with open(f"{prefix}_group_partition.pkl", "wb") as f:
            pickle.dump(group_partition, f)
