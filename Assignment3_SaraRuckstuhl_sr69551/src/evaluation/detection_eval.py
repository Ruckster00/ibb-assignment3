"""
Evaluation metrics for face detection.
"""
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def evaluate_detection(predicted_boxes: List[Tuple[int, int, int, int]],
                       ground_truth_boxes: List[Tuple[int, int, int, int]],
                       iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate face detection performance.
    
    Args:
        predicted_boxes: List of predicted bounding boxes
        ground_truth_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Dictionary with evaluation metrics
    """
    from ..detection.viola_jones import calculate_iou
    
    ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_box, gt_box in zip(predicted_boxes, ground_truth_boxes):
        if pred_box is not None and gt_box is not None:
            iou = calculate_iou(pred_box, gt_box)
            ious.append(iou)
            
            if iou >= iou_threshold:
                true_positives += 1
            else:
                false_positives += 1
        elif pred_box is None and gt_box is not None:
            false_negatives += 1
            ious.append(0.0)
        elif pred_box is not None and gt_box is None:
            false_positives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    avg_iou = np.mean(ious) if len(ious) > 0 else 0.0
    detection_rate = sum(1 for box in predicted_boxes if box is not None) / len(predicted_boxes)
    
    return {
        'avg_iou': avg_iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detection_rate': detection_rate,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'ious': ious
    }


def plot_iou_distribution(ious: List[float], save_path: str = None):
    """
    Plot IoU distribution.
    
    Args:
        ious: List of IoU values
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 5))
    plt.hist(ious, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('IoU', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(np.mean(ious), color='red', linestyle='--', 
                label=f'Mean IoU: {np.mean(ious):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"IoU distribution plot saved to {save_path}")
    
    plt.tight_layout()
    return plt
