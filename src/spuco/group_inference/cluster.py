import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from spuco.group_inference import BaseGroupInference
from spuco.utils import (cluster_by_exemplars, convert_labels_to_partition,
                         convert_partition_to_labels, pairwise_similarity)
from spuco.utils.random_seed import seed_randomness, get_seed


class ClusterAlg(Enum):
    KMEANS = "kmeans"
    GMM = "gmm"
    KMEDOIDS = "kmedoids"

class Cluster(BaseGroupInference):
    """
    Clustering-based Group Inference
    """
    def __init__(
        self,
        Z: torch.Tensor,
        class_labels: Optional[List[int]] = None,
        cluster_alg: ClusterAlg = ClusterAlg.KMEANS,
        num_clusters: int = -1,
        max_clusters: int = -1,
        device: torch.device = torch.device("cpu"), 
        verbose: bool = False
    ):
        """
        Initializes the Cluster object.

        :param Z: The input tensor for clustering.
        :type Z: torch.Tensor
        :param class_labels: Optional list of class labels for class-wise clustering. Defaults to None.
        :type class_labels: Optional[List[int]], optional
        :param cluster_alg: The clustering algorithm to use. Defaults to ClusterAlg.KMEANS.
        :type cluster_alg: ClusterAlg, optional
        :param num_clusters: The number of clusters to create. Defaults to -1.
        :type num_clusters: int, optional
        :param max_clusters: The maximum number of clusters to consider. Defaults to -1.
        :type max_clusters: int, optional
        :param device: The device to run the clustering on. Defaults to torch.device("cpu").
        :type device: torch.device, optional
        :param verbose: Whether to display progress and logging information. Defaults to False.
        :type verbose: bool, optional
        """
         
        seed_randomness(torch_module=torch, numpy_module=np, random_module=random)

        super().__init__()

        # Argument Validation 
        if num_clusters < 2:
            if max_clusters < 2:
                raise ValueError("At least one of num_clusters and max_clusters must be valid i.e. >= 2")
        elif max_clusters >= 2:
            raise ValueError("num_clusters and max_clusters are mutually exclusive")

        # Partition into classes if given labels to compute clusters class-wise
        self.class_partition = {}
        if class_labels is None:
            self.class_partition[-1] = range(len(Z))
        else:
            self.class_partition = convert_labels_to_partition(class_labels)
        
        # Initialize variables
        self.cluster_alg = cluster_alg
        self.num_clusters = num_clusters 
        self.max_clusters = max_clusters
        self.device = device
        self.verbose = verbose

        # Processing Z
        self.Z = Z
        if cluster_alg == ClusterAlg.KMEANS:
            if type(self.Z) == torch.Tensor:
                self.Z = self.Z.detach().cpu().numpy()

    def infer_groups(self) -> Dict[Tuple[int,int], List[int]]:
        """
        Infers the group partition based on the clustering results.

        :return: The group partition.
        :rtype: Dict[Tuple[int,int], List[int]]
        """ 
        # Get class-wise group partitions
        cluster_partitions = [] 
        for class_label in tqdm(self.class_partition.keys(), disable=not self.verbose, desc="Clustering class-wise"):
            Z = self.Z[self.class_partition[class_label]]
            if self.num_clusters < 0:
                partition, scores = self.silhouette(Z)
            elif self.cluster_alg == ClusterAlg.KMEANS:
                _, partition = self.kmeans(Z, num_clusters=self.num_clusters)
            else:
                similarity_matrix = pairwise_similarity(Z.to(self.device), Z.to(self.device))
                _, partition = self.kmedoids(Z, similiarity_matrix=similarity_matrix)
            cluster_partitions.append(partition)

        # Merge class-wise group partitions into one dictionary
        group_partition = {}
        for class_index, partition in zip(self.class_partition.keys(), cluster_partitions):
            group_partition.update(self.process_cluster_partition(partition, class_index))
        return group_partition
    
    def silhouette(self, Z):
        """
        Uses the silhouette score to determine the optimal number of clusters and perform clustering based on self.cluster_alg.

        :param Z: The input data for clustering.
        :type Z: torch.Tensor

        :return: The cluster partition based on the optimal number of clusters.
        :rtype: List[int]
        """

        silhouette_scores = []
        partitions = []

        similarity_matrix = None 
        if self.cluster_alg == ClusterAlg.KMEDOIDS:
            similarity_matrix = pairwise_similarity(Z.to(self.device), Z.to(self.device))
        
        # Iterate through possible num_clusters
        for num_clusters in range(2, self.max_clusters+1):
            # Cluster using num_clusters
            cluster_labels, cluster_partition = None, None 
            if self.cluster_alg == ClusterAlg.KMEANS:
                cluster_labels, cluster_partition = self.kmeans(Z, num_clusters=num_clusters)
                silhouette_scores.append(silhouette_score(Z, cluster_labels)) 
            else: 
                cluster_labels, cluster_partition = self.kmedoids(Z, similiarity_matrix=similarity_matrix, num_clusters=num_clusters)
                silhouette_scores.append(silhouette_score(Z.cpu().numpy(), cluster_labels)) 
            partitions.append(cluster_partition)
             
            if self.verbose:
                print("For n_clusters =", num_clusters,
                    "The average silhouette_score is :", silhouette_scores[-1])
        # Pick best num_clusters
        best_partition_idx = np.argmax(silhouette_scores)
        return partitions[best_partition_idx], silhouette_scores[best_partition_idx]
    
    def kmeans(self, Z, num_clusters: int =-1):
        """
        Performs K-means clustering on the input data.

        :param Z: The input data for clustering.
        :type Z: torch.Tensor
        :param num_clusters: The number of clusters to create. If not specified, the value from the object will be used.
        :type num_clusters: int, optional

        :return: The cluster labels and partition based on the K-means clustering.
        :rtype: Tuple[np.ndarray, List[List[int]]]
        """
        # if num_clusters not passed, use value from object
        if num_clusters < 0:
            num_clusters = self.num_clusters

        # K-Means 
        clusterer = KMeans(n_clusters=num_clusters,
                            random_state=get_seed(),
                            n_init=10)
        cluster_labels = clusterer.fit_predict(Z)
        return cluster_labels, convert_labels_to_partition(cluster_labels.astype(int).tolist())

    def kmedoids(self, Z, similiarity_matrix: torch.Tensor, num_clusters=-1):
        """
        Performs K-medoids clustering on the input data.

        :param Z: The input data for clustering.
        :type Z: torch.Tensor
        :param similiarity_matrix: The similarity matrix for pairwise similarities between data points.
        :type similiarity_matrix: torch.Tensor
        :param num_clusters: The number of clusters to create. If not specified, the value from the object will be used.
        :type num_clusters: int, optional

        :return: The cluster labels and partition based on the K-medoids clustering.
        :rtype: Tuple[np.ndarray, List[List[int]]]
        """
        # if num_clusters not passed, use value from object
        if num_clusters < 0:
            num_clusters = self.num_clusters

        # K-Medoids greedy (cluster by exemplars)
        cluster_partition = cluster_by_exemplars(
            similarity_matrix=similiarity_matrix, 
            num_exemplars=num_clusters, 
            verbose=self.verbose
        )

        return convert_partition_to_labels(cluster_partition), cluster_partition