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
        
        # Step 1: Apply PCA to reduce dimensionality
        self.pca = PCA(n_components=self.pca_components, whiten=True)
        features_pca = self.pca.fit_transform(features_array)
        
        # Step 2: Determine number of LDA components
        n_classes = len(np.unique(labels_array))
        if self.n_components is None:
            self.n_components = n_classes - 1
        else:
            self.n_components = min(self.n_components, n_classes - 1)
        
        # Step 3: Train LDA on PCA-reduced features
        self.lda = SklearnLDA(n_components=self.n_components)
        self.lda.fit(features_pca, labels_array)
        self.is_fitted = True
        
        return self
    
    def transform(self, features: List[np.ndarray]) -> List[np.ndarray]:
        """
        Project features into PCA+LDA space.
        
        Args:
            features: List of feature vectors
            
        Returns:
            List of projected feature vectors
        """
        if not self.is_fitted:
            raise ValueError("LDA must be fitted before transform")
        
        features_array = np.array(features)
        # First apply PCA, then LDA
        features_pca = self.pca.transform(features_array)
        projected = self.lda.transform(features_pca)
        
        return [projected[i] for i in range(len(projected))]
    
    def fit_transform(self, features: List[np.ndarray], labels: List[int]) -> List[np.ndarray]:
        """
        Fit PCA+LDA and transform features in one step.
        
        Args:
            features: List of feature vectors
            labels: List of corresponding identity labels
            
        Returns:
            List of projected feature vectors
        """
        self.fit(features, labels)
        return self.transform(features)
    
    def get_n_components(self) -> int:
        """Get the number of LDA components."""
        if not self.is_fitted:
            return 0
        return self.lda.n_components
    
    def get_pca_components(self) -> int:
        """Get the number of PCA components actually used."""
        if not self.is_fitted:
            return 0
        return self.pca.n_components_
