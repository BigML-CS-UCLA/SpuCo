import torch 
from torch import nn 
from torch.utils.data import DataLoader, Dataset
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

def pairwise_similarity(Z1: torch.tensor, Z2: torch.tensor, block_size: int = 1024):
        similarity_matrices = []
        for i in range(Z1.shape[0] // block_size + 1):
            similarity_matrices_i = []
            e = Z1[i*block_size:(i+1)*block_size]
            for j in range(Z2.shape[0] // block_size + 1):
                e_t = Z2[j*block_size:(j+1)*block_size].t()
                similarity_matrices_i.append(
                    np.array(
                    torch.cosine_similarity(e[:, :, None], e_t[None, :, :]).detach().cpu()
                    )
                )
            similarity_matrices.append(similarity_matrices_i)
        similarity_matrix = np.block(similarity_matrices)

        return similarity_matrix

def get_class_labels(dataset: Dataset) -> List[int]:
    labels = []
    for _, y in dataset:
        labels.append(y)
    return labels