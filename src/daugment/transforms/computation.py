"""Module with constructs to compute similarity of key terms across questions"""

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=0)


def compute_similarity(
    terms: list[str], embedding_matrix: np.ndarray
) -> dict[tuple[str, str], float]:
    """
    Compute similarity value of each pair of terms using term embeddings
    Output dictionary where keys are pairs of terms and values denote similarity of the 2 terms
    """
    normalized = normalize(embedding_matrix)
    similarities = np.dot(normalized.T, normalized)
    return {
        (terms[i], terms[j]): np.abs(similarities[i, j])
        for i in range(len(terms))
        for j in range(len(terms))
    }


def propose_terms(
    terms: list[str],
    embedding_matrix: np.ndarray,
    range_min: int = 0.3,
    range_max: int = 0.6,
) -> list[str]:
    """Select and propose terms with an embedding similarity score falling in a specified range for new combinations"""
    pairs = compute_similarity(terms, embedding_matrix)
    new_terms = [
        element
        for pair in pairs
        if range_min <= pairs[pair] <= range_max
        for element in pair
    ]
    return new_terms
