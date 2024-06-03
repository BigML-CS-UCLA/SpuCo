import random
from sklearn.preprocessing import StandardScaler

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
import umap 

from spuco.group_inference.cluster import ClusterAlg
from spuco.group_inference import Cluster
from spuco.group_inference.george_utils.cluster import OverclusterModel
from spuco.utils.misc import convert_labels_to_partition
from spuco.utils.random_seed import seed_randomness


class GeorgeInference(Cluster):
    """
    George Inference: https://arxiv.org/abs/2011.12945
    """
    def __init__(
        self,
        Z: torch.Tensor,
        class_labels: Optional[List[int]] = None,
        cluster_alg: ClusterAlg = ClusterAlg.GMM,
        max_clusters: int = -1,
        umap_n_components: int = 2,
        umap_n_neighbors: int = 10,
        device: torch.device = torch.device("cpu"), 
        verbose: bool = False
    ):
        """
        Initializes George Inference

        :param Z: The features learnt by the network (output of penultimate layer)
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
        :param umap_n_components: # of components to reduce features to using umap to
        :type umap_n_components: int
        :param umap_n_neighbors: # of neighbors to use for umap
        :type umap_n_neighbors: int
        :param device: The device to run the clustering on. Defaults to torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to display progress and logging information. Defaults to False.
        :type verbose: bool, optional
        """
        
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)
        super().__init__(
            Z=Z,
            class_labels=class_labels,
            cluster_alg=cluster_alg,
            num_clusters=-1,
            max_clusters=max_clusters,
            device=device,
            verbose=verbose
        )
        self.Z = self.Z.cpu().numpy()
        if self.cluster_alg == ClusterAlg.KMEDOIDS:
            raise NotImplementedError("George doesn't support k-medoids clustering.")
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
    
    def infer_groups(self) -> Dict[Tuple[int,int], List[int]]:
        """
        Infers the group partition based on the clustering results.

        :return: The group partition.
        :rtype: Dict[Tuple[int,int], List[int]]
        """ 
        
        # UMAP 
        scaler = StandardScaler()
        scaled_Z = scaler.fit_transform(self.Z)
        umap_model = umap.UMAP(n_components=self.umap_n_components, n_neighbors=self.umap_n_neighbors)
        self.Z = umap_model.fit_transform(scaled_Z)
        
        cluster_partitions = []
        for class_label in tqdm(self.class_partition.keys(), disable=not self.verbose, desc="Clustering class-wise"):
            Z = self.Z[self.class_partition[class_label]]
            overcluster = OverclusterModel(self.cluster_alg.value, max_k=self.max_clusters)
            overcluster.fit(Z)
            cluster_partitions.append(convert_labels_to_partition(overcluster.predict(Z)))
            
        # Merge class-wise group partitions into one dictionary
        group_partition = {}
        for class_index, partition in zip(self.class_partition.keys(), cluster_partitions):
            group_partition.update(self.process_cluster_partition(partition, class_index))
        
        return group_partition
