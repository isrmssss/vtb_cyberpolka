# AGENTS.md - VTB Cyberpolka Hackathon 2026

## Project Overview
Data Fusion Contest 2026 - Task 2 "Киберполка". Multi-label classification of 41 financial products for 1M bank clients using anonymized features. Metric: Macro Averaged ROC-AUC (`sklearn.metrics.roc_auc_score(average="macro")`).

## Environment & Commands
- **Python version**: 3.9.6 (also available: 3.13 in anaconda for CatBoost GPU)
- **Virtual env**: `venv` (activate with `source venv/bin/activate`)
- **Main training scripts**: 
  - `python train_day2.py` (Day 2: SVD + Features + CatBoost classifier chains)
  - `python train_day3.py` (Day 3: Optuna tuning + Inference)
- **Phase runner**: `src/pipeline/` contains phases for Day 2 and Day 3

## Data Structure
All data in `data/` directory (parquet format, excluded from git via `.opencodeignore`):
- `train_main_features.parquet` - 750K rows x 200 features (67 cat + 132 num + customer_id)
- `train_extra_features.parquet` - 750K rows x 2241 numeric features + customer_id
- `test_main_features.parquet` - 250K rows x 200 features
- `test_extra_features.parquet` - 250K rows x 2241 features
- `train_target.parquet` - 750K rows x 41 targets + customer_id
- `sample_submit.parquet` - submission format reference (250K rows x 42 cols)

### Generated Data Files (Day 2)
- `train_svd.npy`, `test_svd.npy` - SVD components (100 dims) from extra features
- `train_processed_day2.parquet`, `test_processed_day2.parquet` - Full processed datasets
- `feature_metadata_day2.json` - Feature columns and metadata
- `target_order_day2.json` - Classifier chain order (sorted by frequency)
- `catboost_model_*.cbm` - 41 saved CatBoost models
- `oof_predictions_day2.npy`, `val_true_day2.npy` - Validation predictions
- `submit_day2.parquet` - Day 2 submission

### Generated Data Files (Day 3)
- `catboost_model_opt_*.cbm` - 5 optimized CatBoost models (for problem targets)
- `oof_predictions_optimized.npy` - OOF predictions after optimization
- `optuna_study_results.json` - Optuna tuning results
- `postprocessing_results.json` - Post-processing metrics
- `oof_predictions_postprocessed.npy` - Final postprocessed OOF
- `test_predictions_day3.npy` - Test predictions
- `submit_day3.parquet` - Day 3 submission (final)

## Key Dependencies
```python
import polars as pl          # Primary data manipulation (not pandas)
import numpy as np           # Numerical operations
import yaml                  # Config file parsing
from catboost import Pool, CatBoostClassifier  # GPU classifier chains
from sklearn.metrics import roc_auc_score      # Evaluation metric
import optuna                # Hyperparameter optimization
import lightgbm as lgb       # For adversarial validation only
```
- **pandas**: Only as fallback for `to_pandas()` when CatBoost Pool has polars version conflicts
- **pyarrow**: Required for parquet I/O
- **scipy**: For rank averaging (if needed in future)
- **tqdm**: Progress bars for training loops

## Configuration (configs/model_params.yaml)
```yaml
catboost:
  task_type: GPU           # GPU acceleration
  devices: "0"             # GPU device ID
  loss_function: Logloss   # Binary classification (one model per target)
  learning_rate: 0.05
  depth: 8
  l2_leaf_reg: 3
  iterations: 1000
  early_stopping_rounds: 50
  random_seed: 42
  verbose: 100             # Print every 100 iterations

optuna:
  n_trials: 30             # Number of trials per target
  timeout: 600             # Seconds per target
  targets_to_optimize:     # Problem targets (AUC < 0.75)
    - target_9_6
    - target_9_3
    - target_3_1
    - target_6_1
    - target_5_2
  params:
    depth: [6, 8, 10]
    learning_rate: [0.03, 0.05, 0.1]
    l2_leaf_reg: [1, 3, 5]
    iterations: [800, 1000, 1500]

postprocessing:
  enabled: true
  clip_threshold: 0.99

nan_flags:
  top_n: 50                # Create binary flags for top 50 features with most NaN

validation:
  val_size: 150000         # Time-based validation split size

svd:
  n_components: 100        # SVD components from extra features

adversarial:
  importance_threshold: 0.01
  random_state: 42
  top_n_plot: 30

rolling_encoding:
  enabled: true             # Rolling target encodings by customer_id
  feature_prefix: "rolling"
```

## Pipeline Phases (Day 2)

### Phase 1: SVD on Extra Features
- Load `train_extra_features.parquet` and `test_extra_features.parquet`
- Fit IncrementalPCA with 100 components on train only
- Transform both train and test
- Save `train_svd.npy` and `test_svd.npy` as numpy memmap
- Purpose: Compress 2241 extra numeric features into 100 SVD components

### Phase 2: Adversarial Validation
- Train LightGBM to distinguish train vs test
- Identify stable features (importance < threshold)
- Result: 158 stable features, 41 shifted features
- Output: Feature importance plot

### Phase 3: Feature Engineering
- Create binary NaN flags for top 50 features with most missing values
- Compute rolling target encodings: 41 features (one per target)
- Total: 290 features (199 main + 50 nan_flags + 41 rolling)
- Save processed datasets and feature metadata JSON
- CRITICAL: Add placeholder rolling columns to test (all NaN)

### Phase 4: CatBoost GPU Classifier Chains Training
- Time-based train/val split: first 600K train, last 150K validation
- Sort targets by frequency (most common first)
- Chain: use previous predictions as features for next target
- 41 separate binary classifiers with `Logloss` + `auto_class_weights="Balanced"`
- GPU training: ~5-10 minutes per model
- Output: 41 `.cbm` models, OOF predictions, local Macro ROC-AUC: 0.8142

### Phase 5: Inference & Submission
- Load test data, apply same feature engineering
- **CRITICAL**: Add placeholder rolling columns to test to match training feature count
- Predict with classifier chain (adding previous predictions as features)
- Save `submit_day2.parquet` with 42 columns (customer_id + 41 predict_*)

## Pipeline Phases (Day 3)

### Phase 7: Optuna Hyperparameter Tuning
- **CRITICAL**: Uses SAME classifier chain logic as Phase 4
- Optimize ONLY 5 problem targets (AUC < 0.75):
  - target_9_6 (0.679), target_9_3 (0.682), target_3_1 (0.685), target_6_1 (0.703), target_5_2 (0.705)
- Parameters to optimize:
  - `depth`: [6, 8, 10]
  - `learning_rate`: [0.03, 0.05, 0.1]
  - `l2_leaf_reg`: [1, 3, 5]
  - `iterations`: [800, 1000, 1500]
- For each target:
  1. Build classifier chain features (like Phase 4)
  2. Run Optuna with n_trials=30, timeout=600s
  3. Retrain with best params
  4. Save as `catboost_model_opt_{target_idx}.cbm`
- Output: 5 optimized models, updated OOF predictions

### Phase 8: Post-processing
- Clip predictions at 0.99
- Analyze target correlations (for future use)
- Save postprocessed OOF predictions
- Output: `oof_predictions_postprocessed.npy`

### Phase 9: Inference & Submission (Day 3)
- Load test data, apply same feature engineering
- **CRITICAL**: Use optimized models for 5 problem targets, original for rest
- Use classifier chain logic (like Phase 5)
- Apply clip post-processing
- Save `submit_day3.parquet` with 42 columns

## Memory Management (CRITICAL)

### Dataset Sizes
- Train: 750K rows, 200 original features + 2241 extra features
- Test: 250K rows
- Full processed features: 390 (199 main + 50 nan + 41 rolling + 100 SVD)

### Memory Optimization Strategies
1. **Delete dataframes immediately after use**:
   ```python
   del train_main, test_main, train_target
   gc.collect()
   ```

2. **Use memmap for SVD**:
   ```python
   train_svd_mm = np.memmap(f"{data_dir}/train_svd.npy", dtype=np.float32, mode="r", shape=(750000, 100))
   X_train_svd = np.array(train_svd_mm[:n_train_split], dtype=np.float32)
   ```

3. **Process in stages**: Don't load all data into memory at once

4. **CatBoost GPU**: Uses GPU VRAM, not RAM - important for large datasets

5. **For 750K train + 250K test**: Need ~16GB RAM minimum

## Code Style Conventions

### Imports
- Group imports: standard library -> third-party -> local
- Use explicit imports (e.g., `from catboost import Pool, CatBoostClassifier`)
- Standard aliases: `pl`, `np`, `plt`, `pd`

### Data Loading
- All paths relative to project root: `'data/train_main_features.parquet'`
- Use `pl.read_parquet()` for loading
- Print shapes after loading: `print('Train shape:', train.shape)`

### Naming Conventions
- **Variables**: `snake_case` (e.g., `train_pool`, `cat_feature_names`, `test_predict`)
- **DataFrames**: `train`, `test`, `target`, `submit`, `sample_submit`
- **Feature lists**: `cat_feature_names`, `target_columns`, `predict_schema`
- **Model objects**: `model`
- **Pools**: `train_pool`, `test_pool`
- **Optimized models**: `catboost_model_opt_{idx}.cbm`

### Feature Detection
```python
cat_feature_names = [col for col in train.columns if col.startswith("cat_feature")]
target_columns = [col for col in target.columns if col.startswith("target")]
predict_schema = [col.replace("target_", "predict_") for col in target.columns if col.startswith("target_")]
```

### Type Casting
- Cast categorical features to `pl.Int32` before use:
```python
train = train.with_columns(pl.col(cat_feature_names).cast(pl.Int32))
test = test.with_columns(pl.col(cat_feature_names).cast(pl.Int32))
```

### Model Configuration

#### CatBoost GPU (Primary Model - Classifier Chains)
- 41 separate binary classifiers (one per target)
- Loss function: `'Logloss'` for binary classification
- Use `auto_class_weights="Balanced"` for imbalanced targets
- Use `task_type="GPU"` for acceleration
- **CRITICAL**: Use `predict_proba()` output (NOT `predict()` which returns RawFormulaVal)
- Early stopping: 50 rounds with validation set
- **ALWAYS use classifier chains**: append previous predictions as features for next target

#### Optuna Tuning (Day 3)
- Only for 5 problem targets (AUC < 0.75)
- **CRITICAL**: Must use same classifier chain logic as training
- Parameters:
  - `depth`: [6, 8, 10]
  - `learning_rate`: [0.03, 0.1]
  - `l2_leaf_reg`: [1, 5]
  - `iterations`: [800, 1500]

### Submission Format
- Output must be `.parquet` with exactly 42 columns: `customer_id` + `predict_1_1` through `predict_10_1`
- Column order must match `sample_submit.parquet` exactly
- Use `submit.write_parquet("path/to/submit.parquet")` to save
- Predictions are probabilities (0-1 range), NOT raw formula values

## Missing Data Strategy
- Dataset has significant missing values (some features >740K nulls out of 750K)
- Create binary NaN flags for top-50 features with most missing values: `[feature_name]_is_nan`
- CatBoost handles missing values natively for the main features
- NaN flags are powerful signals for rare events (premium services, mortgages, account blocks)
- For test set, add placeholder columns for rolling features with all NaN

## Validation Strategy (CRITICAL)
- **DO NOT use Random K-Fold** - customer_id has temporal ordering
- Train IDs: 1.0M - 1.75M, Test IDs: 1.75M - 2.0M
- Use **Time-based split**: First 600K for training, last 150K for validation
- Alternative: **Expanding Window Time Series Split**

## Classifier Chains (CRITICAL - MUST USE)
- Sort targets by frequency (most common first)
- For chain_idx > 0: append previous predictions as additional features
- Training: `X_train_chain = np.hstack([X_train_base, chain_train[:, prev_indices]])`
- Inference: Must match training feature count exactly (including rolling columns)
- **CRITICAL**: When inferencing, include all 41 rolling columns (even if all NaN for test)
- **ALWAYS use this logic in Optuna tuning** - build chain features before training

## Ensemble Strategy (Future - NOT USED IN DAY 3)
- Use **Rank Averaging** if adding multiple models
- Convert predictions to ranks using `scipy.stats.rankdata` (0-1 range)
- Then average ranks across models (LGBM + CatBoost + NN)
- ROC-AUC depends only on order, not absolute values

## Error Handling
- Wrap pyarrow import in try/except for version conflict detection
- CatBoost Pool may require `.to_pandas()` fallback if polars version incompatible
- Checkpoint system for resuming interrupted pipelines
- **Feature mismatch error**: If "Feature X is present in model but not in pool", check that test has same feature count as training
- **Optuna timeout**: Set timeout to prevent indefinite runs

## Important Notes
- Feature meanings and target meanings are **not provided** (anonymized data)
- Features named `cat_feature_N` (categorical) and `num_feature_N` (numeric)
- Targets named `target_X_Y` format; predictions named `predict_X_Y` format
- Public/Private leaderboard split: 30/70 (75K / 175K test clients)
- Up to 2 final submissions allowed for private leaderboard
- Do NOT commit `.parquet` files (in `.opencodeignore`)
- No `requirements.txt` or `pyproject.toml` present - install dependencies manually

## Key Strategic Insights

### Day 2 Results
- **Local Macro ROC-AUC**: 0.8142 (validation set)
- **Per-target AUC range**: 0.68 - 0.97 (varies significantly by target)
- **Problem targets (AUC < 0.75)**:
  - target_9_6: 0.679
  - target_9_3: 0.682
  - target_3_1: 0.685
  - target_6_1: 0.703
  - target_5_2: 0.705
- **Classifier chain order**: Most frequent first (target_10_1 has ~180K positives)

### Day 2 Weaknesses (Fixed)
1. ~~`prediction_type="RawFormulaVal"`~~ - Now uses `predict_proba()` for probabilities
2. ~~Ignored 2241 extra features~~ - Now using SVD compression (100 components)
3. ~~Single MultiLogloss model~~ - Now using 41 binary classifiers with balanced weights
4. ~~No validation~~ - Now using time-based split
5. ~~No NaN feature flags~~ - Now creates top-50 binary NaN indicators

### Winning Hypotheses Implemented (Day 2)
1. **Time-based validation**: Train [1-1.6M], Val [1.6-1.75M]
2. **NaN flags**: Top-50 features with binary indicators
3. **Classifier chains**: Predict by frequency, use previous predictions as features
4. **SVD compression**: 100 components from 2241 extra features
5. **Rolling target encodings**: 41 features from target means by customer_id order

### Day 3 Strategy (Current)
1. **Optuna tuning only for problem targets** - 5 targets with AUC < 0.75
2. **Use classifier chains in Optuna** - same logic as Phase 4
3. **Post-processing** - clip predictions, analyze correlations
4. **Day 3 target Macro ROC-AUC**: >0.82

### Post-Processing (Day 3 - Implemented)
- Clip predictions at 0.99
- Analyze target correlations (for future feature engineering)

## Project Architecture
```
vtb_cyberpolka/
├── train_day2.py              # Day 2 entry point (5-phase pipeline)
├── train_day3.py              # Day 3 entry point (3-phase pipeline)
├── configs/
│   └── model_params.yaml     # All hyperparameters (Day 2 + Day 3)
├── data/                     # Raw + processed data
├── src/
│   ├── pipeline/
│   │   ├── phase1_svd.py          # SVD on extra features
│   │   ├── phase2_adversarial.py  # Feature stability analysis
│   │   ├── phase3_features.py     # NaN flags + rolling encodings
│   │   ├── phase4_training.py     # CatBoost GPU classifier chains
│   │   ├── phase5_inference.py    # Day 2 test predictions
│   │   ├── phase7_optuna_tuning.py    # Optuna tuning (Day 3)
│   │   ├── phase8_postprocessing.py   # Clip + correlation (Day 3)
│   │   ├── phase9_inference.py       # Day 3 test predictions
│   │   └── checkpoint.py         # Resume interrupted pipeline
│   ├── data/
│   │   ├── dataset.py            # Data loading utilities
│   │   └── preprocessing.py      # Rolling target encodings
│   ├── features/
│   │   ├── extractors.py         # NaN flag creation, SVD
│   │   └── selector.py            # Feature selection
│   ├── models/
│   │   └── lgbm_multi.py         # For adversarial validation only
│   ├── training/
│   │   └── cv.py                 # Time series split
│   └── utils/
│       └── metrics.py             # Macro ROC-AUC
└── AGENTS.md
```

## Future Roadmap (After Day 3)
- **LightGBM ensemble** - if Optuna not sufficient
- **Pseudo-labeling** - for rare targets
- **More feature engineering** - interaction features
- **Target correlation exploitation** - use negative correlations

## CRITICAL RULES (MUST FOLLOW)

1. **ALWAYS use CatBoost** - Primary model, NOT LightGBM
2. **ALWAYS use classifier chains** - In training AND tuning AND inference
3. **ALWAYS use predict_proba()** - NOT predict()
4. **Optuna ONLY for problem targets** - Not all 41 targets
5. **Time-based validation** - NOT random k-fold
6. **Checkpoint resume** - For long-running phases
7. **Memory management** - Delete dataframes after use

## Current Status (Day 3)

| Phase | Status | Description |
|-------|--------|-------------|
| phase1_svd | DONE | SVD compression |
| phase2_adversarial | DONE | Feature stability |
| phase3_features | DONE | NaN flags + rolling |
| phase4_training | DONE | 41 CatBoost models, AUC=0.8142 |
| phase5_inference | DONE | Day 2 submission |
| phase7_optuna_tuning | READY | Optuna for 5 problem targets |
| phase8_postprocessing | READY | Clip + correlation |
| phase9_inference | READY | Day 3 submission |

**Target**: Macro ROC-AUC > 0.82 (current 0.8142)
