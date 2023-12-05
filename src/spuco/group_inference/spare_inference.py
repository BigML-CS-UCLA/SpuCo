import random
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from spuco.group_inference.cluster import ClusterAlg
from spuco.group_inference import Cluster
from spuco.utils.random_seed import seed_randomness, get_seed


class SpareInference(Cluster):
    """
    SPARE Inference: https://arxiv.org/abs/2305.18761
    """
    def __init__(
        self,
        Z: torch.Tensor,
        class_labels: Optional[List[int]] = None,
        cluster_alg: ClusterAlg = ClusterAlg.KMEANS,
        num_clusters: int = -1,
        max_clusters: int = -1,
        silhoutte_threshold: float = 0.9,
        high_sampling_power: int = 2,
        device: torch.device = torch.device("cpu"), 
        verbose: bool = False
    ):
        """
        Initializes Spare Inference.

        :param Z: The output of the network.
        :type Z: torch.Tensor
        :param class_labels: Optional list of class labels for class-wise clustering. Defaults to None.
        :type class_labels: Optional[List[int]], optional
        :param cluster_alg: The clustering algorithm to use. Defaults to ClusterAlg.KMEANS.
        :type cluster_alg: ClusterAlg, optional
        :param num_clusters: The number of clusters to create. Defaults to -1.
        :type num_clusters: int, optional
        :param max_clusters: The maximum number of clusters to consider. Defaults to -1.
        :type max_clusters: int, optional
        :param silhoutte_threshold: The silhouette threshold for determining the sampling powers. Defaults to 0.9.
        :type silhoutte_threshold: float, optional
        :param high_sampling_power: The sampling power for the low-silhouette clusters. Defaults to 2.
        :type high_sampling_power: int, optional
        :param device: The device to run the clustering on. Defaults to torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to display progress and logging information. Defaults to False.
        :type verbose: bool, optional
        """
        
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__(Z=Z, class_labels=class_labels, cluster_alg=cluster_alg, num_clusters=num_clusters, max_clusters=max_clusters, device=device, verbose=verbose)

        self.silhouette_threshold = silhoutte_threshold
        self.high_sampling_power = high_sampling_power
    
    def infer_groups(self) -> Dict[int, List[int]]:
        """
        Infers the group partition based on the clustering results.

        :return: The group partition.
        :rtype: Dict[int, List[int]]

        :return: The sampling powers for each group.
        :rtype: List[int]
        """ 
        # Get class-wise group partitions
        cluster_partitions = [] 
        sampling_powers = []
        for class_label in tqdm(self.class_partition.keys(), disable=not self.verbose, desc="Clustering class-wise"):
            Z = self.Z[self.class_partition[class_label]]
            if self.num_clusters == -1:
                partition, silhouette = self.silhouette(Z)
            else:
                cluster_labels, partition = self.kmeans(Z, num_clusters=self.num_clusters)
                silhouette = silhouette_score(Z, cluster_labels)
            cluster_partitions.append(partition)
            if silhouette < self.silhouette_threshold:
                sampling_powers.append(self.high_sampling_power)
            else:
                sampling_powers.append(1)

        # Merge class-wise group partitions into one dictionary
        group_partition = {}
        for class_index, partition in zip(self.class_partition.keys(), cluster_partitions):
            group_partition.update(self.process_cluster_partition(partition, class_index))
        return group_partition, sampling_powers