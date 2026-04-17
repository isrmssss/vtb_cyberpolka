import lightgbm as lgb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


def adversarial_validation(X_train, X_test, feature_names, importance_threshold=0.01, random_state=42, max_train_samples=200000):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if n_train > max_train_samples:
        print(f"  Subsampling train from {n_train} to {max_train_samples} for adversarial validation...")
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_train, max_train_samples, replace=False)
        X_train_sub = X_train[indices]
        del indices
        gc.collect()
    else:
        X_train_sub = X_train

    print(f"  Combining train+test for adversarial validation...")
    X_combined = np.vstack([X_train_sub, X_test])
    y_combined = np.concatenate([np.zeros(X_train_sub.shape[0]), np.ones(n_test)])

    if X_train_sub is not X_train:
        del X_train_sub
        gc.collect()

    print(f"  Adversarial dataset: {X_combined.shape[0]} samples x {X_combined.shape[1]} features")
    print(f"  Train: {n_train}, Test: {n_test}")

    pos_count = y_combined.sum()
    neg_count = len(y_combined) - pos_count
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  Class balance: pos={pos_count:.0f}, neg={neg_count:.0f}, scale_pos_weight={scale_pos_weight:.2f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "verbose": -1,
    }

    print(f"  Training adversarial classifier (up to {params['n_estimators']} rounds, early stop={params['early_stopping_rounds']})...")
    train_data = lgb.Dataset(X_combined, label=y_combined)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data],
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)],
    )

    del train_data
    gc.collect()

    feature_importance = model.feature_importance(importance_type="gain")
    total_importance = feature_importance.sum()
    if total_importance > 0:
        feature_importance_normalized = feature_importance / total_importance
    else:
        feature_importance_normalized = np.zeros_like(feature_importance)

    adversarial_auc = model.best_score["training"]["auc"]
    print(f"  Adversarial validation AUC: {adversarial_auc:.4f}")
    if adversarial_auc > 0.6:
        print(f"  WARNING: Significant distribution shift detected between train and test!")
    else:
        print(f"  Train and test distributions are similar (good).")

    shifted_mask = feature_importance_normalized > importance_threshold
    shifted_features = [feature_names[i] for i in range(len(feature_names)) if shifted_mask[i]]
    stable_features = [feature_names[i] for i in range(len(feature_names)) if not shifted_mask[i]]

    print(f"  Shifted features (importance > {importance_threshold}): {len(shifted_features)}")
    print(f"  Stable features: {len(stable_features)}")

    if len(shifted_features) > 0:
        print(f"  Top 5 shifted features:")
        shifted_indices = np.where(shifted_mask)[0]
        top5 = shifted_indices[np.argsort(feature_importance_normalized[shifted_indices])[-5:][::-1]]
        for idx in top5:
            print(f"    {feature_names[idx]}: {feature_importance_normalized[idx]:.4f}")

    del model, X_combined, y_combined
    gc.collect()

    return stable_features, shifted_features, feature_importance_normalized


def plot_adversarial_importance(feature_names, feature_importance, top_n=30, save_path="data/adversarial_feature_importance.png"):
    indices = np.argsort(feature_importance)[-top_n:]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = feature_importance[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importance, color="steelblue")
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel("Normalized Importance (Gain)")
    plt.title(f"Top {top_n} Features by Adversarial Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Adversarial feature importance plot saved to {save_path}")
