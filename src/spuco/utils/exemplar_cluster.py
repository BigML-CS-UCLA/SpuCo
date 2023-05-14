import numpy as np
import torch
from tqdm import tqdm

from spuco.utils.submodular import FacilityLocation, lazy_greedy

def cluster_by_exemplars(similarity_matrix, num_exemplars, verbose=False):
    """
    Returns dictionary mapping exemplar index to list of tuples (index, similarity_to_exemplar)
    exemplar_index found in exemplar list too
    """
    submodular_function = FacilityLocation(D=similarity_matrix, V=range(len(similarity_matrix)))
    exemplar_indices, _ = lazy_greedy(F=submodular_function, V=range(len(similarity_matrix)), B=num_exemplars)
    clusters = {}

    for exemplar_index in exemplar_indices:
        clusters[exemplar_index] = []

    for index in tqdm(range(len(similarity_matrix)), desc="Sorting samples by exemplar", disable=not verbose):
        exemplar_index, similarity = closest_exemplar(index, exemplar_indices, similarity_matrix)
        clusters[exemplar_index].append((index, similarity))

    return clusters

def closest_exemplar(sample_index, exemplar_indices, similarity_matrix):
    max_similarity = np.NINF
    best_exemplar_index = -1

    for curr_exemplar_index in exemplar_indices:
        if similarity_matrix[sample_index][curr_exemplar_index] > max_similarity:
            max_similarity = similarity_matrix[sample_index][curr_exemplar_index]
            best_exemplar_index = curr_exemplar_index 
    
    return best_exemplar_index, max_similarity

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

