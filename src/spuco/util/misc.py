from typing import Dict, List 
import numpy as np 

def convert_labels_to_partition(labels: List[int]) -> Dict[int, List[int]]:
    partition = {}
    for i, label in enumerate(labels):
        if label not in partition:
            partition[label] = []
        partition[label].append(i)
    return partition

def convert_partition_to_labels(partition: Dict[int, List[int]]) -> List[int]:
    labels = np.array([-1] * sum([len(partition[key]) for key in partition.keys()]))
    for key in partition.keys():
        labels[partition[key]] = key 
    return labels.tolist()