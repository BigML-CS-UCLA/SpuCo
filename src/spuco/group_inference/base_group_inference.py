from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class BaseGroupInference(ABC):
    def __init__(self):
        pass 

    @abstractmethod
    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        pass 

    def process_cluster_partition(self, cluster_partition: Dict, class_index: int):
        """
        Processes cluster partition:
        - Converts keys from clusters into (class, spurious) format
        - Converts class indices from class-wise clustering into global (actual trainset) indices
        """
        assert self.class_partition is not None, "self.class_partition must be defined for processing"
        group_partition = {}
        for i, cluster_label in sorted(cluster_partition.keys()):
            group_partition[(class_index, i)] = [self.class_partition[class_index][i] for i in cluster_partition[cluster_label]]
        return group_partition