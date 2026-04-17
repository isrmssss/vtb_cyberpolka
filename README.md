# VTB Cyberpolka 2026 — Multi-label Classification Solution

## Result

- **Leaderboard position**: Top 4%
- **Local Macro ROC-AUC**: 0.8142
- **Dataset**: 750K clients for training, 250K for test
- **Task**: Predict 41 financial products for each bank client

## Task Description

Data Fusion Contest 2026, task "Cyberpolka". Build a multi-label classification model to predict the probability of 41 financial products for bank clients. Data is anonymized — features named `cat_feature_N` (categorical) and `num_feature_N` (numeric), targets — `target_X_Y`.

**Metric**: Macro ROC-AUC (averaged across all 41 targets).

## Key Hypotheses and Solutions

### 1. Time-based Validation (CRITICAL)

Initially used random K-fold — results were mediocre. Switching to time-based split dramatically improved quality:
- Train: first 600K clients
- Validation: last 150K clients

**Why this matters**: Client IDs have temporal structure (1M-1.75M — train, 1.75M-2M — test). Random split leads to data leakage from future to past.

### 2. Classifier Chains

Instead of a single multi-label model, used 41 separate binary classifiers with chain dependency:
- Sort targets by frequency (from common to rare)
- Previous predictions added as features for next models
- This captures correlations between products

### 3. SVD on Extra Features

Data contains 2241 additional numeric features. Applied SVD (100 components) to compress information while preserving important patterns.

### 4. Missing Value Flags

Created binary features for top-50 features with most NaN values. Missing values often signal rare events (account blocks, premium services).

### 5. CatBoost with GPU

- 41 models with Logloss
- auto_class_weights="Balanced" for imbalanced classes
- GPU acceleration for fast iteration

## Solution Architecture

```
vtb_cyberpolka/
├── train.py              # Main training script
├── configs/
│   └── model_params.yaml # Model parameters
├── src/
│   ├── pipeline/         # Pipeline phases
│   │   ├── phase1_svd.py          # SVD on extra features
│   │   ├── phase2_adversarial.py # Feature stability analysis
│   │   ├── phase3_features.py    # NaN flags + rolling encoding
│   │   ├── phase4_training.py   # CatBoost chain training
│   │   ├── phase7_optuna_tuning.py # Optuna for problem targets
│   │   └── phase9_inference.py   # Test inference
│   ├── data/             # Data loading & preprocessing
│   ├── features/         # Feature engineering
│   └── utils/            # Metrics & utilities
└── data/
    ├── catboost_model_*.cbm  # 41 trained models
    └── submit_day2.parquet   # Final submission
```

## Challenges and Issues

- **5 problem targets** (AUC < 0.75): target_9_6 (0.679), target_9_3 (0.682), target_3_1 (0.685), target_6_1 (0.703), target_5_2 (0.705)
- Couldn't complete full ensemble and Optuna tuning for all targets
- Had to stop due to hackathon time constraints

## Running

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python train.py
```

## Dependencies

- Python 3.9+
- polars, numpy, pandas
- catboost (GPU)
- lightgbm (for adversarial validation)
- optuna (for tuning)
- pyarrow (for parquet)

## Potential Improvements

1. Full Optuna tuning for all 41 targets
2. Ensemble with LightGBM
3. Pseudo-labeling for rare targets
4. Feature interactions
5. Using target correlations