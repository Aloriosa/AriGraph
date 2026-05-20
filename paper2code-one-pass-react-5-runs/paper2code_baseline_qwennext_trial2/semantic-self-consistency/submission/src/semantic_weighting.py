import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticWeighting:
    """Base class for semantic weighting methods"""
    
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.n_samples = len(embeddings)
    
    def compute_weights(self) -> np.ndarray:
        """Compute weights for each sample"""
        raise NotImplementedError

class CentroidProximityWeighting(SemanticWeighting):
    """Centroid Proximity Weighting (CPW)"""
    
    def compute_weights(self) -> np.ndarray:
        # Compute centroid
        centroid = np.mean(self.embeddings, axis=0)
        
        # Compute distances from centroid
        distances = np.array([
            euclidean(e, centroid) for e in self.embeddings
        ])
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Inverse weights
        weights = 1.0 / distances
        weights = weights / np.sum(weights)
        
        return weights

class SemanticConsensusWeighting(SemanticWeighting):
    """Semantic Consensus Weighting (SCW)"""
    
    def compute_weights(self) -> np.ndarray:
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(self.embeddings)
        
        # Sum similarities for each sample
        consensus_scores = np.sum(similarities, axis=1)
        
        # Normalize
        weights = consensus_scores / np.sum(consensus_scores)
        
        return weights

# Example usage
if __name__ == "__main__":
    # Mock embeddings
    np.random.seed(42)
    embeddings = np.random.randn(10, 768)
    
    # CPW
    cpw = CentroidProximityWeighting(embeddings)
    weights_cpw = cpw.compute_weights()
    print("CPW weights:", weights_cpw[:5]
    
    # SCW
    scw = SemanticConsensusWeighting(embeddings)
    weights_scw = scw.compute_weights()
    print("SCW weights:", weights_scw[:5]