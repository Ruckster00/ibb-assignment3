"""
Face recognition evaluation using CMC curves and rank-k accuracy.
"""
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt


class FaceRecognitionEvaluator:
    """Evaluator for face recognition systems."""
    
    def __init__(self, distance_metric: str = 'cosine'):
        """
        Initialize evaluator.
        
        Args:
            distance_metric: Distance metric to use ('cosine' or 'euclidean')
        """
        self.distance_metric = distance_metric
        
    def compute_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute distance between two feature vectors.
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Distance value
        """
        if self.distance_metric == 'cosine':
            return cosine(feat1, feat2)
        elif self.distance_metric == 'euclidean':
            return euclidean(feat1, feat2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def compute_distances(self, query_feature: np.ndarray, 
                         gallery_features: List[np.ndarray]) -> List[float]:
        """
        Compute distances between a query and all gallery features.
        
        Args:
            query_feature: Query feature vector
            gallery_features: List of gallery feature vectors
            
        Returns:
            List of distances
        """
        distances = []
        for gallery_feat in gallery_features:
            dist = self.compute_distance(query_feature, gallery_feat)
            distances.append(dist)
        return distances
    
    def rank_gallery(self, query_feature: np.ndarray,
                    gallery_features: List[np.ndarray],
                    gallery_identities: List[int]) -> List[int]:
        """
        Rank gallery images by similarity to query.
        
        Args:
            query_feature: Query feature vector
            gallery_features: List of gallery feature vectors
            gallery_identities: List of gallery identities
            
        Returns:
            Ranked list of gallery identities (closest first)
        """
        distances = self.compute_distances(query_feature, gallery_features)
        
        # Sort by distance (ascending)
        sorted_indices = np.argsort(distances)
        ranked_identities = [gallery_identities[i] for i in sorted_indices]
        
        return ranked_identities
    
    def compute_cmc(self, query_features: List[np.ndarray],
                   query_identities: List[int],
                   gallery_features: List[np.ndarray],
                   gallery_identities: List[int],
                   max_rank: int = 20) -> np.ndarray:
        """
        Compute Cumulative Match Characteristic (CMC) curve.
        
        Args:
            query_features: List of query feature vectors
            query_identities: List of query identities
            gallery_features: List of gallery feature vectors
            gallery_identities: List of gallery identities
            max_rank: Maximum rank to compute
            
        Returns:
            CMC curve (array of recognition rates at each rank)
        """
        cmc = np.zeros(max_rank)
        
        for query_feat, query_id in zip(query_features, query_identities):
            # Rank gallery by similarity
            ranked_ids = self.rank_gallery(query_feat, gallery_features, gallery_identities)
            
            # Find the rank of the correct identity
            try:
                # Find first occurrence of correct identity
                correct_rank = ranked_ids.index(query_id)
                
                # Update CMC (all ranks >= correct_rank get a match)
                if correct_rank < max_rank:
                    cmc[correct_rank:] += 1
            except ValueError:
                # Identity not found in gallery (shouldn't happen in closed-set)
                pass
        
        # Normalize by number of queries
        cmc = cmc / len(query_features)
        
        return cmc
    
    def compute_rank_k_accuracy(self, cmc: np.ndarray, k: int) -> float:
        """
        Compute Rank-k recognition accuracy.
        
        Args:
            cmc: CMC curve
            k: Rank (1-indexed)
            
        Returns:
            Rank-k accuracy
        """
        if k <= 0 or k > len(cmc):
            raise ValueError(f"Invalid rank k={k}, must be in [1, {len(cmc)}]")
        
        return cmc[k - 1]  # Convert to 0-indexed
    
    def evaluate(self, query_features: List[np.ndarray],
                query_identities: List[int],
                gallery_features: List[np.ndarray],
                gallery_identities: List[int],
                max_rank: int = 20) -> Dict:
        """
        Full evaluation: compute CMC and rank-k accuracies.
        
        Args:
            query_features: List of query feature vectors
            query_identities: List of query identities
            gallery_features: List of gallery feature vectors
            gallery_identities: List of gallery identities
            max_rank: Maximum rank to compute
            
        Returns:
            Dictionary with evaluation results
        """
        cmc = self.compute_cmc(query_features, query_identities,
                              gallery_features, gallery_identities,
                              max_rank)
        
        rank1 = self.compute_rank_k_accuracy(cmc, 1)
        rank5 = self.compute_rank_k_accuracy(cmc, 5) if max_rank >= 5 else None
        
        return {
            'cmc': cmc,
            'rank1': rank1,
            'rank5': rank5,
            'max_rank': max_rank
        }


def plot_cmc_curves(results: Dict[str, Dict], save_path: str = None):
    """
    Plot CMC curves for multiple methods.
    
    Args:
        results: Dictionary mapping method names to evaluation results
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    for method_name, result in results.items():
        cmc = result['cmc']
        ranks = np.arange(1, len(cmc) + 1)
        plt.plot(ranks, cmc * 100, marker='o', markersize=4, label=method_name)
    
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Recognition Rate (%)', fontsize=12)
    plt.title('Cumulative Match Characteristic (CMC) Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, len(list(results.values())[0]['cmc']))
    plt.ylim(0, 105)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"CMC curve saved to {save_path}")
    
    plt.tight_layout()
    return plt
