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
    
    def __init__(self, n_points: int = 24, radius: int = 8, grid_size: Tuple[int, int] = (6, 4)):
        self.n_points = n_points
        self.radius = radius
        self.grid_size = grid_size
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (112, 144))
        gray = cv2.equalizeHist(gray)
        
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        
        h, w = lbp.shape
        grid_h, grid_w = self.grid_size
        cell_h, cell_w = h // grid_h, w // grid_w
        
        weights = np.array([
            [0, 1, 1, 0],
            [1, 2, 2, 1],
            [2, 4, 4, 2],
            [2, 4, 4, 2],
            [1, 2, 2, 1],
            [0, 1, 1, 0]
        ])
        
        n_bins = self.n_points + 2
        histograms = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                hist, _ = np.histogram(cell.ravel(), bins=n_bins, 
                                      range=(0, n_bins), density=True)
                hist = hist * weights[i, j]
                histograms.append(hist)
        
        features = np.concatenate(histograms)
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def get_name(self) -> str:
        return "LBP"


class HOGExtractor(FeatureExtractor):
    
    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (16, 16),
                 cells_per_block: Tuple[int, int] = (2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (112, 144))
        gray = cv2.equalizeHist(gray)
        
        features = hog(gray, 
                      orientations=self.orientations,
                      pixels_per_cell=self.pixels_per_cell,
                      cells_per_block=self.cells_per_block,
                      block_norm='L2-Hys',
                      transform_sqrt=True,
                      visualize=False)
        
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def get_name(self) -> str:
        return "HOG"


class DenseSIFTExtractor(FeatureExtractor):
    
    def __init__(self, step_size: int = 4, patch_size: int = 16, grid_size: tuple = (4, 3)):
        self.step_size = step_size
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.sift = cv2.SIFT_create()
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.resize(gray, (112, 144))
        gray = cv2.equalizeHist(gray)
        
        height, width = gray.shape
        grid_h, grid_w = self.grid_size
        cell_h, cell_w = height // grid_h, width // grid_w
        
        features = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                y_start, y_end = i * cell_h, (i + 1) * cell_h
                x_start, x_end = j * cell_w, (j + 1) * cell_w
                cell = gray[y_start:y_end, x_start:x_end]
                
                keypoints = []
                for y in range(0, cell.shape[0] - self.patch_size, self.step_size):
                    for x in range(0, cell.shape[1] - self.patch_size, self.step_size):
                        keypoints.append(cv2.KeyPoint(
                            x + self.patch_size / 2,
                            y + self.patch_size / 2,
                            self.patch_size
                        ))
                
                _, descriptors = self.sift.compute(cell, keypoints)
                
                if descriptors is None or len(descriptors) == 0:
                    cell_feature = np.zeros(128)
                else:
                    cell_feature = np.mean(descriptors, axis=0)
                
                features.append(cell_feature)
        
        feature_vector = np.concatenate(features)
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-7)
        
        return feature_vector
    
    def get_name(self) -> str:
        return "Dense_SIFT"


def get_all_extractors():
    return [
        LBPExtractor(),
        HOGExtractor(),
        DenseSIFTExtractor()
    ]
