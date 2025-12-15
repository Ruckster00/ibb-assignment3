"""
Viola-Jones face detection implementation with parameter optimization.
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class ViolaJonesDetector:
    """Viola-Jones face detector with parameter optimization."""
    
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, 
                 min_size: Tuple[int, int] = (30, 30)):
        """
        Initialize Viola-Jones detector.
        
        Args:
            scale_factor: Parameter specifying how much the image size is reduced 
                         at each image scale (default: 1.1)
            min_neighbors: Parameter specifying how many neighbors each candidate 
                          rectangle should have to retain it (default: 5)
            min_size: Minimum possible object size (default: (30, 30))
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load the Haar Cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return [tuple(face) for face in faces]
    
    def detect_best_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in an image (most likely to be the primary face).
        
        Args:
            image: Input image
            
        Returns:
            Bounding box of the largest detected face or None
        """
        faces = self.detect(image)
        
        if len(faces) == 0:
            return None
            
        # Return the largest face (by area)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return largest_face


def calculate_iou(box1: Tuple[int, int, int, int], 
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box (x, y, width, height)
        box2: Second bounding box (x, y, width, height)
        
    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def optimize_parameters(images: List[np.ndarray], 
                       ground_truth_boxes: List[Tuple[int, int, int, int]],
                       scale_factors: List[float] = None,
                       min_neighbors_list: List[int] = None) -> Dict:
    """
    Optimize Viola-Jones parameters on training data.
    
    Args:
        images: List of training images
        ground_truth_boxes: List of ground truth bounding boxes
        scale_factors: List of scale factors to try
        min_neighbors_list: List of min_neighbors values to try
        
    Returns:
        Dictionary with best parameters and their performance
    """
    if scale_factors is None:
        scale_factors = [1.05, 1.1, 1.15, 1.2, 1.3]
    
    if min_neighbors_list is None:
        min_neighbors_list = [3, 4, 5, 6, 7]
    
    best_params = None
    best_iou = 0.0
    results = []
    
    print("Optimizing Viola-Jones parameters...")
    
    for scale_factor in scale_factors:
        for min_neighbors in min_neighbors_list:
            detector = ViolaJonesDetector(
                scale_factor=scale_factor,
                min_neighbors=min_neighbors
            )
            
            ious = []
            detections = 0
            
            for img, gt_box in zip(images, ground_truth_boxes):
                detected_face = detector.detect_best_face(img)
                
                if detected_face is not None:
                    detections += 1
                    iou = calculate_iou(detected_face, gt_box)
                    ious.append(iou)
                else:
                    ious.append(0.0)
            
            avg_iou = np.mean(ious)
            detection_rate = detections / len(images)
            
            results.append({
                'scale_factor': scale_factor,
                'min_neighbors': min_neighbors,
                'avg_iou': avg_iou,
                'detection_rate': detection_rate
            })
            
            print(f"Scale: {scale_factor:.2f}, MinNeighbors: {min_neighbors}, "
                  f"Avg IoU: {avg_iou:.4f}, Detection Rate: {detection_rate:.2%}")
            
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_params = {
                    'scale_factor': scale_factor,
                    'min_neighbors': min_neighbors,
                    'avg_iou': avg_iou,
                    'detection_rate': detection_rate
                }
    
    print(f"\nBest parameters: Scale Factor={best_params['scale_factor']}, "
          f"Min Neighbors={best_params['min_neighbors']}, "
          f"Avg IoU={best_params['avg_iou']:.4f}")
    
    return best_params, results
