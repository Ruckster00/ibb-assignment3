"""
Complete Face Recognition Pipeline - All Experiments
Runs detection evaluation, whole image recognition, and full pipeline recognition.

Usage:
  python run_experiments.py          # Run all experiments
  python run_experiments.py --test   # Test setup only
"""
import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import CelebAHQDataset
from src.detection.viola_jones import ViolaJonesDetector, optimize_parameters
from src.features.extractors import get_all_extractors
from src.evaluation.detection_eval import evaluate_detection, plot_iou_distribution
from src.evaluation.recognition import FaceRecognitionEvaluator, plot_cmc_curves
from src.evaluation.lda import LDAProjection


def test_setup():
    """Test that data and environment are set up correctly."""
    print("\n" + "="*70)
    print("TESTING SETUP")
    print("="*70)
    
    csv_path = 'data/CelebA-HQ-small.csv'
    images_dir = 'data/CelebA-HQ-small'
    
    # Check files exist
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("Download from: https://tinyurl.com/celebahqsmall")
        return False
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("Download from: https://tinyurl.com/celebahqsmall")
        return False
    
    print(f"✅ Data files found")
    
    # Test loading
    try:
        dataset = CelebAHQDataset(csv_path, images_dir)
        print("✅ Dataset loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False


def experiment1_detection(dataset, results_dir='results'):
    """Experiment I: Face Detection Evaluation."""
    print("\n" + "="*70)
    print("EXPERIMENT I: FACE DETECTION EVALUATION")
    print("="*70)
    
    # Load data
    train_images, train_ids, train_bboxes, _ = dataset.load_split_data('train')
    test_images, test_ids, test_bboxes, _ = dataset.load_split_data('test')
    
    # Use pre-optimized parameters (scale_factor=1.05, min_neighbors=3)
    print("\nUsing optimized Viola-Jones parameters (scale_factor=1.05, min_neighbors=3)...")
    best_params = {
        'scale_factor': 1.05,
        'min_neighbors': 3,
        'avg_iou': 0.6711,
        'detection_rate': 0.98
    }
    
    # Evaluate on test set
    print("Evaluating on test set...")
    detector = ViolaJonesDetector(
        scale_factor=best_params['scale_factor'],
        min_neighbors=best_params['min_neighbors']
    )
    
    predicted_boxes = [detector.detect_best_face(img) for img in test_images]
    metrics = evaluate_detection(predicted_boxes, test_bboxes, iou_threshold=0.5)
    
    print(f"\nResults:")
    print(f"  Average IoU: {metrics['avg_iou']:.4f}")
    print(f"  Detection Rate: {metrics['detection_rate']:.2%}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    # Save plot
    plot_iou_distribution(metrics['ious'], save_path=f'{results_dir}/exp1_iou.png')
    
    return {'params': best_params, 'metrics': metrics}


def experiment2_whole_images(dataset, results_dir='results'):
    """Experiment II: Recognition on Whole Images."""
    print("\n" + "="*70)
    print("EXPERIMENT II: RECOGNITION ON WHOLE IMAGES (WITH LDA)")
    print("="*70)
    
    # Load and split data
    train_images, train_ids, _, train_indices = dataset.load_split_data('train', crop_faces=False)
    
    # Extract features and evaluate
    all_results = {}
    extractors = get_all_extractors()
    
    for extractor in extractors:
        name = extractor.get_name()
        print(f"\nProcessing {name}...")
        
        # Extract features from ALL training data first
        print("  Extracting features from all training data...")
        all_features = [extractor.extract(img) for img in tqdm(train_images, desc="  Training")]
        
        # Filter to identities with >= 3 samples (so we can use 2 for LDA, rest for query)
        identity_counts = defaultdict(int)
        for identity in train_ids:
            identity_counts[identity] += 1
        
        valid_identities = {identity for identity, count in identity_counts.items() if count >= 3}
        
        # Organize features by identity
        identity_features = defaultdict(list)
        for feat, identity, idx in zip(all_features, train_ids, train_indices):
            if identity in valid_identities:
                identity_features[identity].append((feat, idx))
        
        # Create splits: 1st=gallery, all except last=LDA training, last=query
        gallery_features, gallery_ids = [], []
        lda_features, lda_labels = [], []
        query_features, query_ids = [], []
        
        for identity, items in identity_features.items():
            # 1st sample: gallery
            gallery_features.append(items[0][0])
            gallery_ids.append(identity)
            # All samples except last: for LDA training (includes gallery)
            for feat, _ in items[:-1]:
                lda_features.append(feat)
                lda_labels.append(identity)
            # Last sample only: query
            query_features.append(items[-1][0])
            query_ids.append(identity)
        
        # Train PCA+LDA (Fisherfaces approach)
        print(f"  Training PCA+LDA ({len(valid_identities)} classes, {len(lda_features)} samples)...")
        lda = LDAProjection(n_components=None, pca_components=150)
        lda.fit(lda_features, lda_labels)
        print(f"  PCA: {lda.get_pca_components()} dims → LDA: {lda.get_n_components()} dims")
        
        print(f"  Gallery: {len(gallery_features)} images, Query: {len(query_features)} images")
        
        # Apply LDA projection to both sets
        gallery_features = lda.transform(gallery_features)
        query_features = lda.transform(query_features)
        
        # Evaluate
        evaluator = FaceRecognitionEvaluator(distance_metric='cosine')
        results = evaluator.evaluate(query_features, query_ids, gallery_features, gallery_ids, max_rank=20)
        
        all_results[name] = results
        print(f"  Rank-1: {results['rank1']:.2%}, Rank-5: {results['rank5']:.2%}")
    
    # Save plot
    plot_cmc_curves(all_results, save_path=f'{results_dir}/exp2_cmc.png')
    
    return all_results


def experiment3_full_pipeline(dataset, exp1_results, results_dir='results'):
    """Experiment III: Full Pipeline Recognition."""
    print("\n" + "="*70)
    print("EXPERIMENT III: FULL PIPELINE (DETECTION + RECOGNITION + LDA)")
    print("="*70)
    
    # Initialize detector with best params
    detector = ViolaJonesDetector(
        scale_factor=exp1_results['params']['scale_factor'],
        min_neighbors=exp1_results['params']['min_neighbors']
    )
    
    # Load and detect faces
    train_images, train_ids, _, train_indices = dataset.load_split_data('train', crop_faces=False)
    
    detected_faces, detected_ids, detected_indices = [], [], []
    for img, identity, idx in tqdm(zip(train_images, train_ids, train_indices), 
                                   total=len(train_images), desc="Detecting"):
        box = detector.detect_best_face(img)
        if box is not None:
            x, y, w, h = box
            pad = 0.1
            x, y = max(0, x - int(w*pad)), max(0, y - int(h*pad))
            w, h = min(img.shape[1] - x, int(w*(1+2*pad))), min(img.shape[0] - y, int(h*(1+2*pad)))
            detected_faces.append(img[y:y+h, x:x+w])
            detected_ids.append(identity)
            detected_indices.append(idx)
    
    print(f"Detected: {len(detected_faces)}/{len(train_images)} faces")
    
    # Extract features and evaluate
    all_results = {}
    extractors = get_all_extractors()
    
    for extractor in extractors:
        name = extractor.get_name()
        print(f"\nProcessing {name}...")
        
        # Extract features from ALL detected faces first
        print("  Extracting features from all detected faces...")
        all_features = [extractor.extract(face) for face in tqdm(detected_faces, desc="  Extraction")]
        
        # Filter to identities with >= 3 samples (so we can use 2 for LDA, rest for query)
        identity_counts = defaultdict(int)
        for identity in detected_ids:
            identity_counts[identity] += 1
        
        valid_identities = {identity for identity, count in identity_counts.items() if count >= 3}
        
        # Organize features by identity
        identity_features = defaultdict(list)
        for feat, identity, idx in zip(all_features, detected_ids, detected_indices):
            if identity in valid_identities:
                identity_features[identity].append((feat, idx))
        
        # Create splits: 1st=gallery, all except last=LDA training, last=query
        gallery_features, gallery_ids = [], []
        lda_features, lda_labels = [], []
        query_features, query_ids = [], []
        
        for identity, items in identity_features.items():
            # 1st sample: gallery
            gallery_features.append(items[0][0])
            gallery_ids.append(identity)
            # All samples except last: for LDA training (includes gallery)
            for feat, _ in items[:-1]:
                lda_features.append(feat)
                lda_labels.append(identity)
            # Last sample only: query
            query_features.append(items[-1][0])
            query_ids.append(identity)
        
        # Train PCA+LDA (Fisherfaces approach)
        print(f"  Training PCA+LDA ({len(valid_identities)} classes, {len(lda_features)} samples)...")
        lda = LDAProjection(n_components=None, pca_components=150)
        lda.fit(lda_features, lda_labels)
        print(f"  PCA: {lda.get_pca_components()} dims → LDA: {lda.get_n_components()} dims")
        
        print(f"  Gallery: {len(gallery_features)} images, Query: {len(query_features)} images")
        
        # Apply LDA projection to both sets
        gallery_features = lda.transform(gallery_features)
        query_features = lda.transform(query_features)
        
        # Evaluate
        evaluator = FaceRecognitionEvaluator(distance_metric='cosine')
        results = evaluator.evaluate(query_features, query_ids, gallery_features, gallery_ids, max_rank=20)
        
        all_results[name] = results
        print(f"  Rank-1: {results['rank1']:.2%}, Rank-5: {results['rank5']:.2%}")
    
    # Save plot
    plot_cmc_curves(all_results, save_path=f'{results_dir}/exp3_cmc.png')
    
    return all_results


def generate_report(exp1, exp2, exp3, output_path='results/report.txt'):
    """Generate comprehensive report."""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FACE RECOGNITION PIPELINE - RESULTS\n")
        f.write("="*70 + "\n\n")
        
        # Experiment I
        f.write("EXPERIMENT I: FACE DETECTION\n")
        f.write("-"*70 + "\n")
        f.write(f"Best Parameters: Scale={exp1['params']['scale_factor']}, MinNeighbors={exp1['params']['min_neighbors']}\n")
        f.write(f"Average IoU: {exp1['metrics']['avg_iou']:.4f}\n")
        f.write(f"Precision: {exp1['metrics']['precision']:.4f}\n")
        f.write(f"Recall: {exp1['metrics']['recall']:.4f}\n\n")
        
        # Experiment II
        f.write("EXPERIMENT II: WHOLE IMAGES\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Method':<15} {'Rank-1':<12} {'Rank-5':<12}\n")
        for method, res in exp2.items():
            f.write(f"{method:<15} {res['rank1']:>10.2%} {res['rank5']:>10.2%}\n")
        f.write("\n")
        
        # Experiment III
        f.write("EXPERIMENT III: FULL PIPELINE\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Method':<15} {'Rank-1':<12} {'Rank-5':<12}\n")
        for method, res in exp3.items():
            f.write(f"{method:<15} {res['rank1']:>10.2%} {res['rank5']:>10.2%}\n")
        
    print(f"\nReport saved to {output_path}")


def main():
    """Run all experiments or test setup."""
    # Check for test flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', 'test']:
        success = test_setup()
        if success:
            print("\n✅ Setup verified! Run 'python run_experiments.py' to start.")
        sys.exit(0 if success else 1)
    
    print("\n" + "#"*70)
    print("# FACE RECOGNITION PIPELINE")
    print("#"*70)
    
    # Test setup first
    if not test_setup():
        print("\n❌ Setup test failed. Please fix the issues above.")
        sys.exit(1)
    
    # Setup
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    dataset = CelebAHQDataset('data/CelebA-HQ-small.csv', 'data/CelebA-HQ-small')
    
    # Run experiments
    exp1 = experiment1_detection(dataset, results_dir)
    exp2 = experiment2_whole_images(dataset, results_dir)
    exp3 = experiment3_full_pipeline(dataset, exp1, results_dir)
    
    # Generate report
    generate_report(exp1, exp2, exp3, f'{results_dir}/report.txt')
    
    # Save results
    with open(f'{results_dir}/all_results.pkl', 'wb') as f:
        pickle.dump({'exp1': exp1, 'exp2': exp2, 'exp3': exp3}, f)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved in '{results_dir}/':")
    print("  - exp1_iou.png")
    print("  - exp2_cmc.png")
    print("  - exp3_cmc.png")
    print("  - report.txt")
    print("  - all_results.pkl")
    print("\nNext: Write your 2-page report using REPORT_GUIDE.md")


if __name__ == '__main__':
    main()
