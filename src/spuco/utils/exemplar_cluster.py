from typing import Dict, List

import numpy as np
from tqdm import tqdm

from spuco.utils.submodular import FacilityLocation, lazy_greedy

def cluster_by_exemplars(similarity_matrix, num_exemplars, verbose=False) -> Dict[int, List[int]]:
    """
    Returns a dictionary mapping exemplar index to a list of indices.

    :param similarity_matrix: The similarity matrix.
    :type similarity_matrix: numpy.ndarray
    :param num_exemplars: The number of exemplars to select.
    :type num_exemplars: int
    :param verbose: Whether to print progress information.
    :type verbose: bool, optional
    :return: A dictionary mapping exemplar index to a list of indices.
    :rtype: dict[int, list[int]]]
    """
    submodular_function = FacilityLocation(D=similarity_matrix, V=range(len(similarity_matrix)))
    exemplar_indices, _ = lazy_greedy(F=submodular_function, V=range(len(similarity_matrix)), B=num_exemplars, verbose=verbose)
    clusters = {}

    for exemplar_index in exemplar_indices:
        clusters[exemplar_index] = []

    for index in tqdm(range(len(similarity_matrix)), desc="Sorting samples by exemplar", disable=not verbose):
        exemplar_index, _  = closest_exemplar(index, exemplar_indices, similarity_matrix)
        clusters[exemplar_index].append(index)

    return clusters

def closest_exemplar(sample_index, exemplar_indices, similarity_matrix):
    """
    Finds the closest exemplar to a given sample index.

    :param sample_index: The index of the sample.
    :type sample_index: int
    :param exemplar_indices: The indices of the exemplars.
    :type exemplar_indices: list[int]
    :param similarity_matrix: The similarity matrix.
    :type similarity_matrix: numpy.ndarray
    :return: The index of the closest exemplar and the similarity score.
    :rtype: tuple[int, float]
    """

    max_similarity = np.NINF
    best_exemplar_index = -1

    for curr_exemplar_index in exemplar_indices:
        if similarity_matrix[sample_index][curr_exemplar_index] > max_similarity:
            max_similarity = similarity_matrix[sample_index][curr_exemplar_index]
            best_exemplar_index = curr_exemplar_index 
    
    return best_exemplar_index, max_similarity
