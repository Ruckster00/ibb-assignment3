"""
Linear Discriminant Analysis for face recognition (Fisherfaces approach).
PCA followed by LDA: reduces dimensionality while maximizing class separability.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.decomposition import PCA
from typing import List


class LDAProjection:
    """Wrapper for PCA+LDA projection (Fisherfaces) in face recognition pipeline."""
    
    def __init__(self, n_components: int = None, pca_components: int = 150):
        """
        Initialize PCA+LDA projection.
        
        Args:
            n_components: Number of LDA components to keep (default: n_classes - 1)
            pca_components: Number of PCA components before LDA (default: 150)
        """
        self.n_components = n_components
        self.pca_components = pca_components
        self.pca = None
        self.lda = None
        self.is_fitted = False
    
    def fit(self, features: List[np.ndarray], labels: List[int]):
        """
        Train PCA then LDA on features and labels.
        
        Args:
            features: List of feature vectors
            labels: List of corresponding identity labels
        """
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        features_pca = self.pca.fit_transform(features_array)
        
        n_classes = len(np.unique(labels_array))
        if self.n_components is None:
            self.n_components = n_classes - 1
        else:
            self.n_components = min(self.n_components, n_classes - 1)
        
        self.lda = SklearnLDA(n_components=self.n_components)
        self.lda.fit(features_pca, labels_array)
        self.is_fitted = True
        
        return self
    
    def transform(self, features: List[np.ndarray]) -> List[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("LDA must be fitted before transform")
        
        features_array = np.array(features)
        features_pca = self.pca.transform(features_array)
        projected = self.lda.transform(features_pca)
        
        return [projected[i] for i in range(len(projected))]
    
    def fit_transform(self, features: List[np.ndarray], labels: List[int]) -> List[np.ndarray]:
        self.fit(features, labels)
        return self.transform(features)
    
    def get_n_components(self) -> int:
        if not self.is_fitted:
            return 0
        return self.lda.n_components
    
    def get_pca_components(self) -> int:
        if not self.is_fitted:
            return 0
        return self.pca.n_components_
