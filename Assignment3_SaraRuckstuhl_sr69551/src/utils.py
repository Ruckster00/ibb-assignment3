"""
Data loading utilities for CelebA-HQ dataset.
"""
import pandas as pd
import cv2
import os
from typing import Tuple, List, Dict
import numpy as np


class CelebAHQDataset:
    """Dataset class for CelebA-HQ small subset."""
    
    def __init__(self, csv_path: str, images_dir: str):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with annotations
            images_dir: Path to directory containing images
        """
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.df = pd.read_csv(csv_path)
        
        print(f"Loaded dataset with {len(self.df)} images")
        print(f"Train images: {len(self.df[self.df['split'] == 'train'])}")
        print(f"Test images: {len(self.df[self.df['split'] == 'test'])}")
        print(f"Unique identities: {self.df['identity'].nunique()}")
    
    def get_split(self, split: str = 'train') -> pd.DataFrame:
        """
        Get data for a specific split.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            DataFrame with the split data
        """
        return self.df[self.df['split'] == split].copy()
    
    def load_image(self, idx: int) -> np.ndarray:
        """
        Load an image by its index.
        
        Args:
            idx: Image index
            
        Returns:
            Image as numpy array (BGR)
        """
        image_path = os.path.join(self.images_dir, f"{idx}.jpg")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def get_bounding_box(self, row: pd.Series) -> Tuple[int, int, int, int]:
        """
        Get bounding box from a DataFrame row.
        
        Args:
            row: DataFrame row
            
        Returns:
            Bounding box (x, y, width, height)
        """
        return (int(row['x_1']), int(row['y_1']), 
                int(row['width']), int(row['height']))
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                  padding: float = 0.0) -> np.ndarray:
        """
        Crop face from image using bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            padding: Padding to add around the box (as fraction of box size)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        
        if padding > 0:
            # Add padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(image.shape[1] - x, w + 2 * pad_x)
            h = min(image.shape[0] - y, h + 2 * pad_y)
        
        # Crop
        cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def load_split_data(self, split: str = 'train', 
                       crop_faces: bool = False,
                       padding: float = 0.1) -> Tuple[List[np.ndarray], 
                                                       List[int], 
                                                       List[Tuple[int, int, int, int]],
                                                       List[int]]:
        """
        Load all images and metadata for a split.
        
        Args:
            split: 'train' or 'test'
            crop_faces: Whether to crop faces using ground truth boxes
            padding: Padding for cropping (if crop_faces=True)
            
        Returns:
            Tuple of (images, identities, bounding_boxes, indices)
        """
        split_df = self.get_split(split)
        
        images = []
        identities = []
        bboxes = []
        indices = []
        
        print(f"\nLoading {split} split...")
        
        for _, row in split_df.iterrows():
            try:
                idx = int(row['idx'])
                image = self.load_image(idx)
                bbox = self.get_bounding_box(row)
                identity = int(row['identity'])
                
                if crop_faces:
                    image = self.crop_face(image, bbox, padding=padding)
                
                images.append(image)
                identities.append(identity)
                bboxes.append(bbox)
                indices.append(idx)
                
            except Exception as e:
                print(f"Warning: Failed to load image {idx}: {e}")
                continue
        
        print(f"Loaded {len(images)} images from {split} split")
        
        return images, identities, bboxes, indices


def visualize_samples(dataset: CelebAHQDataset, n_samples: int = 5, 
                     split: str = 'train', save_path: str = None):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: CelebAHQDataset instance
        n_samples: Number of samples to visualize
        split: Dataset split
        save_path: Path to save the visualization (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    split_df = dataset.get_split(split).sample(n=min(n_samples, len(dataset.get_split(split))))
    
    fig, axes = plt.subplots(1, n_samples, figsize=(3*n_samples, 4))
    
    if n_samples == 1:
        axes = [axes]
    
    for ax, (_, row) in zip(axes, split_df.iterrows()):
        idx = int(row['idx'])
        image = dataset.load_image(idx)
        bbox = dataset.get_bounding_box(row)
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ax.imshow(image_rgb)
        
        # Draw bounding box
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_title(f"ID: {int(row['identity'])}\nIdx: {idx}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return plt
