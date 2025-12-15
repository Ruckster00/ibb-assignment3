"""
Feature extraction methods for face recognition.
Implements LBP, HOG, and Dense SIFT.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from typing import Tuple


class FeatureExtractor:
    """Base class for feature extractors."""
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Get the name of the feature extractor."""
        raise NotImplementedError


class LBPExtractor(FeatureExtractor):
    """Local Binary Pattern feature extractor with multi-scale approach."""
    
    def __init__(self, n_points: int = 24, radius: int = 3, grid_size: Tuple[int, int] = (4, 4)):
        """
        Initialize LBP extractor.
        
        Args:
            n_points: Number of circularly symmetric neighbor points
            radius: Radius of circle
            grid_size: Grid for spatial histogram (divide image into regions)
        """
        self.n_points = n_points
        self.radius = radius
        self.grid_size = grid_size
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features with spatial histograms.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            LBP feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to larger standard size for better features
        gray = cv2.resize(gray, (256, 256))
        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Calculate LBP
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        
        # Divide into spatial grid and compute histogram for each region
        h, w = lbp.shape
        grid_h, grid_w = self.grid_size
        cell_h, cell_w = h // grid_h, w // grid_w
        
        n_bins = self.n_points + 2  # uniform LBP bins
        histograms = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=n_bins, 
                                      range=(0, n_bins), density=True)
                histograms.append(hist)
        
        # Concatenate all histograms
        features = np.concatenate(histograms)
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def get_name(self) -> str:
        return "LBP"


class HOGExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients feature extractor."""
    
    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (16, 16),
                 cells_per_block: Tuple[int, int] = (2, 2)):
        """
        Initialize HOG extractor.
        
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell (16x16 for smaller descriptor)
            cells_per_block: Number of cells in each block (2x2 for efficiency)
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to 128x128 for reasonable feature dimension
        # (256x256 produces 72,900 dims which is too slow)
        gray = cv2.resize(gray, (128, 128))
        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Calculate HOG with better normalization
        features = hog(gray, 
                      orientations=self.orientations,
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block,
                      block_norm='L2-Hys',
                      transform_sqrt=True,  # Power law compression for illumination invariance
                      visualize=False)
        
        # Additional L2 normalization
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def get_name(self) -> str:
        return "HOG"


class DenseSIFTExtractor(FeatureExtractor):
    """Dense SIFT feature extractor using a fixed grid with improved aggregation."""
    
    def __init__(self, step_size: int = 6, patch_size: int = 16):
        """
        Initialize Dense SIFT extractor.
        
        Args:
            step_size: Step size for the dense grid
            patch_size: Size of patches for SIFT computation
        """
        self.step_size = step_size
        self.patch_size = patch_size
        self.sift = cv2.SIFT_create()
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Dense SIFT features with improved aggregation.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Dense SIFT feature vector (aggregated)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to larger standard size
        gray = cv2.resize(gray, (256, 256))
        # Histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Generate dense grid of keypoints
        keypoints = []
        height, width = gray.shape
        
        for y in range(0, height - self.patch_size, self.step_size):
            for x in range(0, width - self.patch_size, self.step_size):
                keypoints.append(cv2.KeyPoint(
                    x + self.patch_size / 2,
                    y + self.patch_size / 2,
                    self.patch_size
                ))
        
        # Compute SIFT descriptors at keypoints
        _, descriptors = self.sift.compute(gray, keypoints)
        
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector if no descriptors
            return np.zeros(256)
        
        # Improved aggregation: concatenate mean and std
        mean_desc = np.mean(descriptors, axis=0)
        std_desc = np.std(descriptors, axis=0)
        feature_vector = np.concatenate([mean_desc, std_desc])
        
        # L2 normalize
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-7)
        
        return feature_vector
    
    def get_name(self) -> str:
        return "Dense_SIFT"


def get_all_extractors():
    """Get all feature extractors."""
    return [
        LBPExtractor(),
        HOGExtractor(),
        DenseSIFTExtractor()
    ]
