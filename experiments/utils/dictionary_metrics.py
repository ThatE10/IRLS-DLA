import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Union, Tuple


class DictionaryMetrics:
    """
    Metrics for evaluating dictionary recovery quality.
    
    Implements:
    - Hungarian algorithm with cosine similarity for atom matching
    - Average coherence (similarity between recovered and true atoms)
    - Detection rate (percentage of atoms recovered above threshold)
    """
    
    @staticmethod
    def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
        
        Returns:
            Cosine similarity in [0, 1] (absolute value for sign invariance)
        """
        dot_product = torch.abs(torch.dot(v1, v2))
        norm_product = torch.norm(v1) * torch.norm(v2)
        return (dot_product / (norm_product + 1e-10)).item()
    
    @staticmethod
    def hungarian_cosine_matching(D_true: Union[np.ndarray, torch.Tensor],
                                  D_learned: Union[np.ndarray, torch.Tensor]) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Use Hungarian algorithm to find best atom matching based on cosine similarity.
        
        The Hungarian algorithm solves the assignment problem: given a cost matrix,
        find the optimal one-to-one matching that minimizes total cost. Here we use
        negative cosine similarity as the cost (to maximize similarity).
        
        Args:
            D_true: True dictionary (n_features, n_components)
            D_learned: Learned dictionary (n_features, n_components)
        
        Returns:
            avg_similarity: Average cosine similarity of matched atoms
            true_indices: Indices of true atoms in optimal matching
            learned_indices: Indices of learned atoms in optimal matching
        """
        # Convert to torch tensors
        if isinstance(D_true, np.ndarray):
            D_true = torch.as_tensor(D_true, dtype=torch.float32)
        if isinstance(D_learned, np.ndarray):
            D_learned = torch.as_tensor(D_learned, dtype=torch.float32)
        
        n_features, n_components_true = D_true.shape
        _, n_components_learned = D_learned.shape
        
        # Compute pairwise cosine similarity matrix
        # Use absolute value to handle sign ambiguity (atoms can be ±d)
        similarity_matrix = torch.abs(D_true.T @ D_learned)  # (n_components_true, n_components_learned)
        
        # Convert to numpy for scipy's Hungarian algorithm
        similarity_np = similarity_matrix.cpu().numpy()
        
        # Hungarian algorithm minimizes cost, so use negative similarity
        cost_matrix = -similarity_np
        
        # Solve assignment problem
        true_indices, learned_indices = linear_sum_assignment(cost_matrix)
        
        # Compute average similarity of matched pairs
        matched_similarities = similarity_np[true_indices, learned_indices]
        avg_similarity = np.mean(matched_similarities)
        
        return avg_similarity, true_indices, learned_indices
    
    @staticmethod
    def detection_rate(D_true: Union[np.ndarray, torch.Tensor],
                      D_learned: Union[np.ndarray, torch.Tensor],
                      threshold: float = 0.99) -> Tuple[float, int, np.ndarray]:
        """
        Compute detection rate: percentage of true atoms recovered above threshold.
        
        Args:
            D_true: True dictionary (n_features, n_components)
            D_learned: Learned dictionary (n_features, n_components)
            threshold: Cosine similarity threshold for "detected" atoms
        
        Returns:
            detection_rate: Percentage of atoms detected [0, 1]
            n_detected: Number of atoms detected
            detected_mask: Boolean mask of detected atoms
        """
        avg_similarity, true_indices, learned_indices = DictionaryMetrics.hungarian_cosine_matching(
            D_true, D_learned
        )
        
        # Convert to torch for similarity computation
        if isinstance(D_true, np.ndarray):
            D_true = torch.as_tensor(D_true, dtype=torch.float32)
        if isinstance(D_learned, np.ndarray):
            D_learned = torch.as_tensor(D_learned, dtype=torch.float32)
        
        # Compute similarity for each matched pair
        similarities = []
        for i_true, i_learned in zip(true_indices, learned_indices):
            sim = DictionaryMetrics.cosine_similarity(
                D_true[:, i_true], 
                D_learned[:, i_learned]
            )
            similarities.append(sim)
        
        similarities = np.array(similarities)
        detected_mask = similarities >= threshold
        n_detected = np.sum(detected_mask)
        detection_rate = n_detected / len(similarities)
        
        return detection_rate, n_detected, detected_mask
    
    @staticmethod
    def average_coherence(D_true: Union[np.ndarray, torch.Tensor],
                         D_learned: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute average coherence (maximum cosine similarity for each true atom).
        
        For each true atom, finds the most similar learned atom. This doesn't
        enforce one-to-one matching like Hungarian algorithm.
        
        Args:
            D_true: True dictionary (n_features, n_components)
            D_learned: Learned dictionary (n_features, n_components)
        
        Returns:
            avg_coherence: Average maximum similarity across true atoms
        """
        if isinstance(D_true, np.ndarray):
            D_true = torch.as_tensor(D_true, dtype=torch.float32)
        if isinstance(D_learned, np.ndarray):
            D_learned = torch.as_tensor(D_learned, dtype=torch.float32)
        
        # Compute pairwise similarities
        similarity_matrix = torch.abs(D_true.T @ D_learned)
        
        # For each true atom, find maximum similarity with any learned atom
        max_similarities = torch.max(similarity_matrix, dim=1)[0]
        
        return torch.mean(max_similarities).item()
    
    @staticmethod
    def compute_all_metrics(D_true: Union[np.ndarray, torch.Tensor],
                           D_learned: Union[np.ndarray, torch.Tensor],
                           threshold: float = 0.99,
                           verbose: bool = True) -> dict:
        """
        Compute all dictionary recovery metrics.
        
        Args:
            D_true: True dictionary (n_features, n_components)
            D_learned: Learned dictionary (n_features, n_components)
            threshold: Detection threshold for detection rate
            verbose: Print results
        
        Returns:
            Dictionary of metrics
        """
        # Hungarian matching
        hungarian_sim, true_idx, learned_idx = DictionaryMetrics.hungarian_cosine_matching(
            D_true, D_learned
        )
        
        # Detection rate
        det_rate, n_detected, detected_mask = DictionaryMetrics.detection_rate(
            D_true, D_learned, threshold
        )
        
        # Average coherence
        avg_coh = DictionaryMetrics.average_coherence(D_true, D_learned)
        
        metrics = {
            'hungarian_similarity': hungarian_sim,
            'detection_rate': det_rate,
            'n_detected': n_detected,
            'total_atoms': len(true_idx),
            'average_coherence': avg_coh,
            'threshold': threshold,
            'true_indices': true_idx,
            'learned_indices': learned_idx,
            'detected_mask': detected_mask
        }
        
        if verbose:
            print("=" * 60)
            print("DICTIONARY RECOVERY METRICS")
            print("=" * 60)
            print(f"Hungarian Matching Similarity: {hungarian_sim:.4f}")
            print(f"Average Coherence:             {avg_coh:.4f}")
            print(f"Detection Rate (>={threshold}):     {det_rate:.2%} ({n_detected}/{metrics['total_atoms']} atoms)")
            print("=" * 60)
        
        return metrics
    
    @staticmethod
    def visualize_matching(D_true: Union[np.ndarray, torch.Tensor],
                          D_learned: Union[np.ndarray, torch.Tensor],
                          n_show: int = 10,
                          sort_by_similarity: bool = True):
        """
        Print detailed matching information for visual inspection.
        
        Args:
            D_true: True dictionary
            D_learned: Learned dictionary
            n_show: Number of matches to show
            sort_by_similarity: Sort by similarity (best first)
        """
        hungarian_sim, true_idx, learned_idx = DictionaryMetrics.hungarian_cosine_matching(
            D_true, D_learned
        )
        
        # Convert to torch
        if isinstance(D_true, np.ndarray):
            D_true = torch.as_tensor(D_true, dtype=torch.float32)
        if isinstance(D_learned, np.ndarray):
            D_learned = torch.as_tensor(D_learned, dtype=torch.float32)
        
        # Compute similarities for each match
        matches = []
        for i_true, i_learned in zip(true_idx, learned_idx):
            sim = DictionaryMetrics.cosine_similarity(
                D_true[:, i_true],
                D_learned[:, i_learned]
            )
            matches.append((i_true, i_learned, sim))
        
        # Sort by similarity if requested
        if sort_by_similarity:
            matches.sort(key=lambda x: x[2], reverse=True)
        
        print("\n" + "=" * 70)
        print("ATOM MATCHING DETAILS (Hungarian Algorithm)")
        print("=" * 70)
        print(f"{'True Idx':<12} {'Learned Idx':<15} {'Cosine Similarity':<20}")
        print("-" * 70)
        
        for i, (i_true, i_learned, sim) in enumerate(matches[:n_show]):
            status = "✓" if sim >= 0.99 else "○" if sim >= 0.9 else "✗"
            print(f"{i_true:<12} {i_learned:<15} {sim:<20.6f} {status}")
        
        if len(matches) > n_show:
            print(f"... ({len(matches) - n_show} more matches)")
        
        print("=" * 70)
        print("Legend: ✓ = excellent (≥0.99), ○ = good (≥0.9), ✗ = poor (<0.9)")
        print("=" * 70)


# Convenience functions
def evaluate_recovery(D_true: Union[np.ndarray, torch.Tensor],
                     D_learned: Union[np.ndarray, torch.Tensor],
                     threshold: float = 0.99,
                     show_matching: bool = False,
                     n_show: int = 10) -> dict:
    """
    Convenience function to evaluate dictionary recovery.
    
    Args:
        D_true: True dictionary
        D_learned: Learned dictionary
        threshold: Detection threshold
        show_matching: Whether to show detailed matching
        n_show: Number of matches to show if show_matching=True
    
    Returns:
        Dictionary of metrics
    """
    metrics = DictionaryMetrics.compute_all_metrics(D_true, D_learned, threshold, verbose=True)
    
    if show_matching:
        DictionaryMetrics.visualize_matching(D_true, D_learned, n_show=n_show)
    
    return metrics