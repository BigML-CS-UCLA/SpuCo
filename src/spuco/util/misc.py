import torch 
from torch import nn 
from torch.utils.data import DataLoader
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

def label_examples(unlabled_dataloader: DataLoader, model: nn.Module, device: torch.device):
    labels = []
    for X in unlabled_dataloader:
        labels.append(torch.argmax(model(X.to(device)), dim=-1))
    labels = torch.cat(labels, dim=0)
    return labels.detach().cpu().tolist()
