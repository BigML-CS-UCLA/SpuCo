from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from spuco.group_inference import BaseGroupInference
from spuco.util import pairwise_similarity, cluster_by_exemplars


class ClusterAlg(Enum):
    KMEANS = "kmeans"
    KMEDOIDS = "kmedoids"

class Cluster(BaseGroupInference):
    def __init__(
        self,
        Z: torch.Tensor,
        cluster_alg: ClusterAlg = ClusterAlg.KMEANS,
        num_clusters: int = -1,
        max_clusters: int = -1,
        random_seed: int = 0,
        device: torch.device = torch.device("cpu"), 
        verbose: bool = False
    ):
        """
        num_clusters and max_cluster should be mutuallly exclusive
        """
        self.Z = Z
        if cluster_alg == ClusterAlg.KMEANS:
            self.Z = self.Z.detach().cpu().numpy()
        if num_clusters < 2:
            if max_clusters < 2:
                raise ValueError("At least one of num_clusters and max_clusters must be valid i.e. >= 2")
        elif max_clusters >= 2:
            raise ValueError("num_clusters and max_clusters are mutually exclusive")
        self.cluster_alg = cluster_alg
        self.num_clusters = num_clusters 
        self.max_clusters = max_clusters
        self.random_seed = random_seed
        self.device = device
        self.verbose = verbose

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        cluster_partition = None
        if self.num_clusters < 2:
            return self.silhouette()
        elif self.cluster_alg == ClusterAlg.KMEANS:
            _, cluster_partition = self.kmeans()
        else:
            similarity_matrix = pairwise_similarity(self.Z.to(self.device), self.Z.to(self.device))
            _, cluster_partition = self.kmedoids(similiarity_matrix=similarity_matrix)
        return cluster_partition
    
    def silhouette(self):
        silhouette_scores = []
        partitions = []

        similarity_matrix = None 
        if self.cluster_alg == ClusterAlg.KMEDOIDS:
            similarity_matrix = pairwise_similarity(self.Z.to(self.device), self.Z.to(self.device))
        
        for i in range(2, self.max_clusters+1):
            cluster_labels, cluster_partition = None, None 
            if self.cluster_alg == ClusterAlg.KMEANS:
                cluster_labels, cluster_partition = self.kmeans(num_clusters=i)
            else: 
                cluster_labels, cluster_partition = self.kmedoids(similiarity_matrix=similarity_matrix, num_clusters=i)
            silhouette_scores.append(silhouette_score(self.Z, cluster_labels))  
            if self.verbose:
                print("For n_clusters =", i,
                    "The average silhouette_score is :", silhouette_scores[-1])
            partitions.append(cluster_partition)
        best_partition_idx = np.argmax(silhouette_scores)
        return partitions[best_partition_idx]
    
    def kmeans(self, num_clusters: int =-1):
        # if num_clusters not passed, use value from object
        if num_clusters < 0:
            num_clusters = self.num_clusters

        clusterer = KMeans(n_clusters=num_clusters,
                            random_state=self.random_seed,
                            n_init=10)
        cluster_labels = clusterer.fit_predict(self.Z)
        cluster_partition = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in cluster_partition:
                cluster_partition[cluster_label] = []
            cluster_partition[cluster_label].append(i)
        return cluster_labels, cluster_partition

    def kmedoids(self, similiarity_matrix: torch.Tensor, num_clusters=-1):
        # if num_clusters not passed, use value from object
        if num_clusters < 0:
            num_clusters = self.num_clusters
        cluster_partition = cluster_by_exemplars(
            similarity_matrix=similiarity_matrix, 
            num_exemplars=num_clusters, 
            verbose=self.verbose
        )
        cluster_labels = [-1] * len(self.Z)
        for cluster_label in cluster_partition.keys():
            for i in cluster_partition[cluster_label]:
                cluster_labels[i] = cluster_label
        return cluster_labels, cluster_partition