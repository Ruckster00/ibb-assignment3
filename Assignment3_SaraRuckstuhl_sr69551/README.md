# Assignment 3: Face Recognition Pipeline
**Student:** Sara Ruckstuhl (sr69551@student.uni-lj.si)  
**Course:** IBB, FRI, UL

## Submission Contents

### 1. Report
- `report.pdf` - Final 2-page IEEE format report with all results

### 2. Code
- `run_experiments.py` - Main script that runs all three experiments
- `src/` - Source code directory containing:
  - `detection/viola_jones.py` - Viola-Jones face detection
  - `features/extractors.py` - LBP, HOG, Dense SIFT feature extraction
  - `evaluation/lda.py` - PCA+LDA dimensionality reduction (Fisherfaces)
  - `evaluation/recognition.py` - CMC curve computation
  - `evaluation/detection_eval.py` - Detection metrics (IoU, precision, recall)
  - `utils.py` - Dataset loading utilities

### 3. Dependencies
- `requirements.txt` - Python package dependencies

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset Setup
The code expects the CelebA-HQ-small dataset in:
```
data/
├── CelebA-HQ-small.csv
└── CelebA-HQ-small/
    └── [image files]
```

### Execute All Experiments
```bash
python run_experiments.py
```

This will run:
1. **Experiment I:** Viola-Jones detection optimization and evaluation
2. **Experiment II:** Recognition on whole images (LBP, HOG, Dense SIFT with PCA+LDA)
3. **Experiment III:** Full pipeline with detection + recognition

**Runtime:** ~10-15 minutes

### Output
Results are saved to `results/`:
- `exp1_iou.png` - IoU distribution plot
- `exp2_cmc.png` - CMC curves for whole images
- `exp3_cmc.png` - CMC curves for full pipeline
- `report.txt` - All numerical results

## Implementation Highlights

- **Detection:** Optimized Viola-Jones (scale=1.05, neighbors=3) achieving 0.686 IoU
- **Features:** LBP (416 dims), HOG (1764 dims), Dense SIFT (256 dims)
- **Dimensionality Reduction:** PCA (150 components) → LDA (49 components)
- **Best Performance:** HOG with 40% Rank-1 accuracy on full pipeline
- **Key Finding:** Dense SIFT improves from 10% to 22% Rank-1 with face detection

## Dataset Usage
- **Test set (412 images):** Used only for detection evaluation (Experiment I)
- **Training set (475 images):** Used for recognition (Experiments II & III)
  - 50 identities with ≥3 images
  - Split: 1st image → gallery, intermediate → LDA training, last → query
- **Rationale:** Train/test splits have disjoint identities, making test set unsuitable for CMC evaluation

## Results Summary

### Experiment I: Face Detection
- Average IoU: 0.686
- Detection rate: 99.76%
- Precision: 0.937, Recall: 0.997

### Experiment II: Whole Images
| Method     | Rank-1 | Rank-5 |
|------------|--------|--------|
| LBP        | 22%    | 34%    |
| HOG        | **38%**| **60%**|
| Dense SIFT | 10%    | 42%    |

### Experiment III: Full Pipeline
| Method     | Rank-1 | Rank-5 |
|------------|--------|--------|
| LBP        | 18%    | 46%    |
| HOG        | **40%**| **66%**|
| Dense SIFT | 22%    | 52%    |
