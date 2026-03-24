# Point-Supervised Land Cover Segmentation

Weakly supervised semantic segmentation for remote sensing, using sparse point annotations instead of dense pixel masks.

This project studies how far we can push land-cover segmentation with only a tiny fraction of labeled pixels, using the DeepGlobe Land Cover dataset and a U-Net with a ResNet34 encoder.

## Why This Project

Dense segmentation labels are expensive. This work replaces full-mask supervision with point-level supervision and trains with a **Partial Cross-Entropy** objective that only uses labeled points.

The core question is practical:
- How many points are enough?
- Which sampling strategy gives the best performance-cost tradeoff?

## Key Results

### Experiment 1: Point Density Analysis

With **class-balanced** sampling, we vary the number of labeled points per class:

| Configuration | Best mIoU | Final Accuracy | Supervision Level |
|---|---:|---:|---|
| 10 pts/class | 0.2428 | 0.5221 | ~0.1% |
| **50 pts/class** | **0.3525** | **0.6267** | **~0.5%** |
| 100 pts/class | 0.3244 | 0.6110 | ~1.0% |

**Takeaway:** 50 points per class achieves optimal cost-performance balance (45% improvement over 10 pts, no overfitting beyond 50 pts).

### Experiment 2: Sampling Strategy Comparison

With **100 points** per configuration, we compare random vs. class-balanced sampling:

| Strategy | Best mIoU | Final Accuracy | Points/Image |
|---|---:|---:|---:|
| **Random** | **0.3261** | **0.7321** | **~100** |
| Class-balanced | 0.3171 | 0.4977 | ~360 |

**Takeaway:** Random sampling slightly outperformed class-balanced while using 3.6× fewer points, suggesting efficiency wins over class-aware oversampling on this dataset.

## Method At A Glance

- **Task**: Semantic segmentation with sparse point supervision
- **Dataset**: DeepGlobe Land Cover Classification
- **Model**: U-Net + ResNet34 encoder (ImageNet pretrained)
- **Loss**: Partial Cross-Entropy (computed only on labeled points)
- **Training**:
  - Fixed point annotations per experiment
  - Freeze encoder for first 2 epochs
  - AdamW (`lr=1e-4`, `weight_decay=1e-4`)
  - ReduceLROnPlateau scheduler
- **Image size / batch / epochs**: `256x256` / `4` / `6`



## Important Data Note

Although the dataset includes a `valid/` folder, this project intentionally performs an **80/20 split from the training pool** inside the notebook due to inconsistencies observed while training however there is no issue with the provided validation split by itself.

## Setup

### 1. Clone and open

```bash
git clone <your-repo-url>
cd "Point Supervised Land Cover Segmentation"
```

### 2. Install dependencies

Use the package list from the notebook:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations tqdm matplotlib pandas numpy pillow scikit-learn
```

### 3. Dataset

Download DeepGlobe Land Cover:
- https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

Place files under `Dataset/` so the folder layout matches this repository.

## Running The Project

1. Open `weakly_supervised_segmentation.ipynb`.
2. Update `DATA_DIR` to your dataset path.
3. Run cells sequentially.

## What The Notebook Executes

- Point annotation simulation (`random`, `class_balanced`, `density_based`)
- Fixed annotation generation per experiment
- Model training and validation
- Experiment 1: point-density sweep
- Experiment 2: strategy comparison
- Metric tracking (mIoU, pixel accuracy, per-class IoU)

## Reproducibility

- Seeds are set in the notebook for deterministic behavior where possible.
- Point annotations are cached as experiment-specific files to keep supervision fixed across reruns.
- Best checkpoints are saved per experiment name.

## Limitations

- Notebook-centric workflow (no standalone `train.py`/`eval.py` script yet)
- Performance and runtime depend on available GPU resources
- Findings are dataset-specific and may vary across domains

## References

- DeepGlobe Land Cover dataset: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset
- Project report: `technical_report.md`
