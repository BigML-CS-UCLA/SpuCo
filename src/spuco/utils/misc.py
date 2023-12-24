from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def convert_labels_to_partition(labels: List[int]) -> Dict[int, List[int]]:
    """
    Converts a list of labels into a partition dictionary.

    :param labels: List of labels.
    :type labels: List[int]
    :return: Partition dictionary mapping labels to their corresponding indices.
    :rtype: Dict[int, List[int]]
    """
    partition = {}
    for i, label in enumerate(labels):
        if label not in partition:
            partition[label] = []
        partition[label].append(i)
    return partition

def convert_partition_to_labels(partition: Dict[int, List[int]]) -> List[int]:
    """
    Converts a partition dictionary into a list of labels.

    :param partition: Partition dictionary mapping labels to their corresponding indices.
    :type partition: Dict[int, List[int]]
    :return: List of labels.
    :rtype: List[int]
    """
    labels = np.array([-1] * sum([len(partition[key]) for key in partition.keys()]))
    for key in partition.keys():
        labels[partition[key]] = key 
    return labels.tolist()

def label_examples(unlabled_dataloader: DataLoader, model: nn.Module, device: torch.device):
    """
    Labels examples using a trained model.

    :param unlabeled_dataloader: Dataloader containing unlabeled examples.
    :type unlabeled_dataloader: torch.utils.data.DataLoader
    :param model: Trained model for labeling examples.
    :type model: torch.nn.Module
    :param device: Device to use for computations.
    :type device: torch.device
    :return: List of predicted labels.
    :rtype: List[int]
    """
    labels = []
    for X in unlabled_dataloader:
        labels.append(torch.argmax(model(X.to(device)), dim=-1))
    labels = torch.cat(labels, dim=0)
    return labels.detach().cpu().tolist()

def pairwise_similarity(Z1: torch.tensor, Z2: torch.tensor, block_size: int = 1024):
    """
    Computes pairwise similarity between two sets of embeddings.

    :param Z1: Tensor containing the first set of embeddings.
    :type Z1: torch.tensor
    :param Z2: Tensor containing the second set of embeddings.
    :type Z2: torch.tensor
    :param block_size: Size of the blocks for computing similarity. Default is 1024.
    :type block_size: int
    :return: Pairwise similarity matrix.
    :rtype: np.array
    """
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

def get_group_ratios(indices: List[int], group_partition: Dict[Tuple[int, int], List[int]]):
    """
    Returns the ratio of each group found in the given indices

    :param Z1: Tensor containing the first set of embeddings.
    :type Z1: torch.tensor
    :param Z2: Tensor containing the second set of embeddings.
    :type Z2: torch.tensor  
    """
    group_ratio = {}
    for key in group_partition.keys():
        group_ratio[key] = len([i for i in indices if i in group_partition[key]]) / len(group_partition[key])
    return group_ratio

def get_model_outputs(model, dataset, device=torch.device("cpu"), features=False, verbose=False):
    """
    Gets output of model on a dataset
    """
    with torch.no_grad():
        model.eval()
        eval_trainloader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4, 
            pin_memory=True
        )
        with tqdm(eval_trainloader, unit="batch", total=len(eval_trainloader), disable=not verbose) as pbar:
            outputs = []
            pbar.set_description("Getting model outputs")
            for input, _ in pbar:
                if features:
                    outputs.append(model.backbone(input.to(device)))
                else:
                    outputs.append(model(input.to(device)))
            return torch.cat(outputs, dim=0)