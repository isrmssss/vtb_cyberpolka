import sys
import os
import gc
import json
import yaml
import time
import numpy as np
import polars as pl
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.training.cv import TimeSeriesSplitter
from src.utils.metrics import macro_roc_auc, per_target_auc


def load_svd_as_memmap(svd_path, shape):
    return np.memmap(svd_path, dtype=np.float32, mode="r", shape=shape)


def run_training(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 4] CatBoost GPU Classifier Chains training (41 models)...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(f"{data_dir}/feature_metadata_day2.json") as f:
        metadata = json.load(f)

    target_columns = metadata["target_columns"]
    all_feature_cols = metadata["all_feature_cols"]
    cb_params = config["catboost"]

    train_df = pl.read_parquet(f"{data_dir}/train_processed_day2.parquet")

    splitter = TimeSeriesSplitter()
    train_split, val_split = splitter.simple_time_split(train_df, val_size=config["validation"]["val_size"])
    n_train_split = train_split.shape[0]
    n_val_split = val_split.shape[0]

    del train_df
    gc.collect()

    X_train_main = train_split.select(all_feature_cols).to_numpy().astype(np.float32)
    X_val_main = val_split.select(all_feature_cols).to_numpy().astype(np.float32)
    y_train = train_split.select(target_columns).to_numpy()
    y_val = val_split.select(target_columns).to_numpy()

    del train_split, val_split
    gc.collect()

    n_svd = 100
    train_svd_mm = load_svd_as_memmap(f"{data_dir}/train_svd.npy", (750000, n_svd))
    X_train_svd = np.array(train_svd_mm[:n_train_split], dtype=np.float32)
    X_val_svd = np.array(train_svd_mm[n_train_split:n_train_split + n_val_split], dtype=np.float32)
    del train_svd_mm
    gc.collect()

    X_train_base = np.hstack([X_train_main, X_train_svd])
    X_val_base = np.hstack([X_val_main, X_val_svd])
    del X_train_main, X_val_main, X_train_svd, X_val_svd
    gc.collect()

    print(f"  Base features: {X_train_base.shape[1]}")
    print(f"  Train: {X_train_base.shape}, Val: {X_val_base.shape}")

    target_order = sorted(range(len(target_columns)), key=lambda i: y_train[:, i].sum(), reverse=True)
    print(f"\n  Classifier Chains order (most frequent first):")
    for rank, idx in enumerate(target_order[:5]):
        pos = int(y_train[:, idx].sum())
        print(f"    {rank+1}. {target_columns[idx]} (pos={pos})")
    print(f"    ...")
    last = target_order[-1]
    pos = int(y_train[:, last].sum())
    print(f"    {len(target_columns)}. {target_columns[last]} (pos={pos})")

    chain_train = np.zeros((n_train_split, len(target_columns)), dtype=np.float32)
    chain_val = np.zeros((n_val_split, len(target_columns)), dtype=np.float32)

    target_aucs = {}
    best_iterations = {}
    total_start = time.time()

    for chain_idx, target_idx in enumerate(target_order):
        t_model_start = time.time()
        target_name = target_columns[target_idx]
        pos_count = int(y_train[:, target_idx].sum())

        if chain_idx == 0:
            X_train_chain = X_train_base
            X_val_chain = X_val_base
        else:
            prev_indices = target_order[:chain_idx]
            X_train_chain = np.hstack([X_train_base, chain_train[:, prev_indices]])
            X_val_chain = np.hstack([X_val_base, chain_val[:, prev_indices]])

        train_pool = Pool(X_train_chain, label=y_train[:, target_idx])
        val_pool = Pool(X_val_chain, label=y_val[:, target_idx])

        model = CatBoostClassifier(
            task_type=cb_params["task_type"],
            devices=cb_params["devices"],
            loss_function=cb_params["loss_function"],
            learning_rate=cb_params["learning_rate"],
            depth=cb_params["depth"],
            l2_leaf_reg=cb_params["l2_leaf_reg"],
            iterations=cb_params["iterations"],
            early_stopping_rounds=cb_params["early_stopping_rounds"],
            random_seed=cb_params["random_seed"],
            verbose=cb_params["verbose"],
            auto_class_weights="Balanced",
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        chain_train[:, target_idx] = model.predict_proba(X_train_chain)[:, 1]
        chain_val[:, target_idx] = model.predict_proba(X_val_chain)[:, 1]

        best_iter = model.get_best_iteration()
        auc = roc_auc_score(y_val[:, target_idx], chain_val[:, target_idx])
        target_aucs[target_name] = auc
        best_iterations[target_name] = best_iter

        model_time = time.time() - t_model_start
        elapsed = time.time() - total_start
        remaining = (elapsed / (chain_idx + 1)) * (len(target_columns) - chain_idx - 1)

        print(f"\n  [{chain_idx+1}/{len(target_columns)}] {target_name}: "
              f"AUC={auc:.4f}, best_iter={best_iter}, pos={pos_count}, "
              f"time={model_time:.0f}s, ETA={remaining/60:.0f}min")

        model.save_model(f"{data_dir}/catboost_model_{target_idx}.cbm")

        del model, train_pool, val_pool, X_train_chain, X_val_chain
        gc.collect()

    macro_auc_val = macro_roc_auc(y_val, chain_val)
    print(f"\n  Overall Macro ROC-AUC: {macro_auc_val:.5f}")

    print("\n  Per-target AUC:")
    for name, auc in sorted(target_aucs.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -1):
        if not np.isnan(auc):
            print(f"    {name}: {auc:.4f} (best_iter={best_iterations[name]})")

    np.save(f"{data_dir}/oof_predictions_day2.npy", chain_val)
    np.save(f"{data_dir}/val_true_day2.npy", y_val)

    target_order_list = [(int(idx), target_columns[idx]) for idx in target_order]
    with open(f"{data_dir}/target_order_day2.json", "w") as f:
        json.dump(target_order_list, f)

    del X_train_base, X_val_base, chain_train, chain_val, y_train, y_val
    gc.collect()

    return {
        "macro_auc": macro_auc_val,
        "n_models": len(target_columns),
        "target_aucs": {k: float(v) for k, v in target_aucs.items()},
        "target_order": target_order_list,
    }
