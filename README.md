# Face Recognition Pipeline - Assignment 3

## Overview
Complete face recognition pipeline with optimized feature extraction:
1. **Face Detection** using Viola-Jones algorithm  
2. **Feature Extraction** using enhanced LBP, HOG, and Dense SIFT
# Face Recognition Pipeline - Assignment 3

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset to data/CelebA-HQ-small/
# From: https://tinyurl.com/celebahqsmall

# 3. Test setup (optional)
python run_experiments.py --test

# 4. Run all experiments (~10-15 min)
python run_experiments.py
```

**Done!** Check `results/` for plots and `results/report.txt` for metrics.

## Files You Need

**Run:**
- `run_experiments.py` - Main script (does everything)

**Reference:**
- `README.md` - This file
- `REPORT_GUIDE.md` - Help for writing your 2-page report
- `requirements.txt` - Python dependencies

**Source Code:**
```
src/
├── detection/viola_jones.py      # Face detector
├── features/extractors.py        # LBP, HOG, SIFT
├── evaluation/*.py               # Metrics & CMC
└── utils.py                      # Data loading
```

## Experiments Overview

### Experiment I: Detection Performance
- Optimizes Viola-Jones (scale_factor, min_neighbors)
- Metrics: IoU, Precision, Recall, F1-Score
- **Typical Results**: IoU ~0.68, Detection Rate >99%

### Experiment II: Recognition on Whole Images
- Gallery/Query split from training set (1st image = gallery, rest = queries)
- Features extracted from full images
- **Typical Results**: HOG Rank-1 ~21%, Rank-5 ~44%

### Experiment III: Full Pipeline
- Viola-Jones detection → Feature extraction → Recognition
- **Typical Results**: HOG Rank-1 ~23%, Rank-5 ~43%
- Shows impact of detection quality on recognition

## Implementation Details

### Optimized Feature Extraction

**Local Binary Patterns (LBP)**
- 256×256 image size with histogram equalization
- 24 points, radius 3, uniform patterns
- 4×4 spatial grid for local histograms
- L2 normalization

**Histogram of Oriented Gradients (HOG)**
- 256×256 image size with histogram equalization
- 9 orientations, 8×8 pixels/cell, 3×3 cells/block
- Transform sqrt for illumination invariance
- L2-Hys + additional L2 normalization

**Dense SIFT**
- 256×256 image size with histogram equalization
- 6-pixel step, 16-pixel patches
- Mean + Std aggregation (256-dim vector)
- L2 normalization

### Recognition
- **Distance**: Cosine distance
- **Evaluation**: CMC curves, Rank-1 and Rank-5 accuracy
- **Gallery/Query Split**: First image per identity = gallery, rest = queries

## Generated Files

Results saved in `results/`:
- `exp1_iou.png` - IoU distribution plot
- `exp2_cmc.png` - CMC curves (whole images)
- `exp3_cmc.png` - CMC curves (full pipeline)
- `report.txt` - Summary with all metrics
- `all_results.pkl` - Complete results data

## For Your Report

See `REPORT_GUIDE.md` for detailed writing tips.

**Key Points to Include:**
1. **Methodology**: Viola-Jones optimization + 3 feature extractors + CMC evaluation
2. **Results**: Tables with IoU, Rank-1, Rank-5 + CMC plots
3. **Analysis**: HOG performs best (~21-23% Rank-1), LBP improved with spatial histograms
4. **Discussion**: Detection quality impacts pipeline performance; trade-offs between methods

**Generated Files for Report:**
- Use `exp1_iou.png`, `exp2_cmc.png`, `exp3_cmc.png`
- Copy metrics from `report.txt`
- Reference `REPORT_GUIDE.md` for LaTeX table templates

## Key Findings

✅ **HOG achieves best performance** (~21% Rank-1 on whole images)  
✅ **Full pipeline slightly outperforms whole images** (better localization)  
✅ **Optimized features** show significant improvement (LBP: 2.8% → 6.6%)  
✅ **High detection rate** (>99%) with optimized Viola-Jones  

## Troubleshooting

**Low accuracy?**
- Dataset has disjoint train/test identities (by design)
- We use gallery/query split from training set
- Results are expected for this challenging closed-set identification task

**Slow execution?**
- Reduce training subset in detection optimization (line ~30 in run_experiments.py)
- Features extract at ~20-50 images/second
