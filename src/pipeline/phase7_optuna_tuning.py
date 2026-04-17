import sys
import os
import gc
import json
import yaml
import numpy as np
import polars as pl
import optuna
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.training.cv import TimeSeriesSplitter
from src.utils.metrics import macro_roc_auc


OPTUNA_PROGRESS_FILE = "data/optuna_progress.json"


def load_svd_as_memmap(svd_path, shape):
    return np.memmap(svd_path, dtype=np.float32, mode="r", shape=shape)


def load_optuna_progress():
    if os.path.exists(OPTUNA_PROGRESS_FILE):
        with open(OPTUNA_PROGRESS_FILE) as f:
            return json.load(f)
    return {"targets": {}, "total": 41, "last_updated": None}


def save_optuna_progress(progress):
    progress["last_updated"] = __import__("time").strftime("%H:%M:%S")
    with open(OPTUNA_PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_optuna_tuning(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 7] Optuna hyperparameter optimization for ALL 41 targets...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(f"{data_dir}/feature_metadata_day2.json") as f:
        metadata = json.load(f)

    with open(f"{data_dir}/target_order_day2.json") as f:
        target_order = json.load(f)

    target_columns = metadata["target_columns"]
    all_feature_cols = metadata["all_feature_cols"]
    cb_params = config.get("catboost", {})

    optuna_config = config.get("optuna", {})
    n_trials = optuna_config.get("n_trials", 20)
    timeout = optuna_config.get("timeout", 900)

    progress = load_optuna_progress()
    print(f"  Optuna progress: {len([t for t in progress['targets'].values() if t.get('status') == 'completed'])}/41 completed")

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

    target_order_list = [t[1] for t in target_order]
    target_idx_map = {t: i for i, t in enumerate(target_columns)}

    print(f"  Loading validation predictions...")
    oof_preds_cb = np.load(f"{data_dir}/oof_predictions_day2.npy")
    oof_preds_optimized = np.copy(oof_preds_cb)

    print(f"  Computing training predictions from saved models for classifier chains...")
    print(f"    Building chain sequentially (like Phase 4)...")
    chain_train = np.zeros((n_train_split, len(target_columns)), dtype=np.float32)
    
    for chain_idx, (target_idx, target_name) in enumerate(target_order):
        if chain_idx == 0:
            X_train_chain = X_train_base
        else:
            prev_indices = [t[0] for t in target_order[:chain_idx]]
            X_train_chain = np.hstack([X_train_base, chain_train[:, prev_indices]])
        
        model = CatBoostClassifier()
        model.load_model(f"{data_dir}/catboost_model_{target_idx}.cbm")
        chain_train[:, target_idx] = model.predict_proba(X_train_chain)[:, 1]
        
        if chain_idx % 10 == 0:
            print(f"      [{chain_idx+1}/41] {target_name} done")
        
        del model, X_train_chain
        gc.collect()
    
    print(f"    Train chain predictions: {chain_train.shape}")

    study_results = {}

    for target_name in target_columns:
        target_progress = progress["targets"].get(target_name, {})
        
        if target_progress.get("status") == "completed":
            print(f"\n  [{target_name}] Already optimized, loading saved model...")
            
            target_idx = target_idx_map[target_name]
            y_val_col = y_val[:, target_idx]
            
            model = CatBoostClassifier()
            model.load_model(f"{data_dir}/catboost_model_opt_{target_idx}.cbm")
            
            chain_idx = target_order_list.index(target_name)
            if chain_idx == 0:
                X_val_chain = X_val_base
            else:
                prev_indices = [target_order[i][0] for i in range(chain_idx)]
                X_val_chain = np.hstack([X_val_base, oof_preds_cb[:, prev_indices]])
            
            new_val_preds = model.predict_proba(X_val_chain)[:, 1]
            oof_preds_optimized[:, target_idx] = new_val_preds
            
            study_results[target_name] = {
                "target_idx": target_idx,
                "best_params": target_progress.get("best_params"),
                "best_auc": target_progress.get("best_auc"),
            }
            
            del model, X_val_chain
            gc.collect()
            continue

        target_idx = target_idx_map[target_name]
        y_train_col = y_train[:, target_idx]
        y_val_col = y_val[:, target_idx]

        current_auc = roc_auc_score(y_val_col, oof_preds_cb[:, target_idx])
        print(f"\n  [{target_name}] Optimizing (current AUC: {current_auc:.4f})...")

        progress["targets"][target_name] = {"status": "in_progress", "current_auc": current_auc}
        save_optuna_progress(progress)

        chain_idx = target_order_list.index(target_name)

        if chain_idx == 0:
            X_train_chain = X_train_base
            X_val_chain = X_val_base
        else:
            prev_indices = [target_order[i][0] for i in range(chain_idx)]
            X_train_chain = np.hstack([X_train_base, chain_train[:, prev_indices]])
            X_val_chain = np.hstack([X_val_base, oof_preds_cb[:, prev_indices]])

        best_auc = current_auc
        best_params = None

        if target_progress.get("status") == "in_progress" and target_progress.get("best_params"):
            print(f"    Resuming with saved best params: {target_progress['best_params']}")
            best_params = target_progress["best_params"]
            best_auc = target_progress.get("best_auc", current_auc)

        if best_params is None:
            def objective(trial):
                depth = trial.suggest_int("depth", 6, 10)
                learning_rate = trial.suggest_float("learning_rate", 0.03, 0.1, log=True)
                l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 5)
                iterations = trial.suggest_int("iterations", 600, 1200)

                model = CatBoostClassifier(
                    task_type=cb_params.get("task_type", "GPU"),
                    devices=cb_params.get("devices", "0"),
                    loss_function=cb_params.get("loss_function", "Logloss"),
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    iterations=iterations,
                    early_stopping_rounds=50,
                    random_seed=cb_params.get("random_seed", 42),
                    verbose=0,
                    auto_class_weights="Balanced",
                )

                train_pool = Pool(X_train_chain, label=y_train_col)
                val_pool = Pool(X_val_chain, label=y_val_col)

                model.fit(train_pool, eval_set=val_pool, use_best_model=True)

                preds = model.predict_proba(X_val_chain)[:, 1]
                auc = roc_auc_score(y_val_col, preds)

                del model, train_pool, val_pool
                gc.collect()

                return auc

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

            best_params = study.best_params
            best_auc = study.best_value

        print(f"    Best params: {best_params}")
        print(f"    Best AUC: {best_auc:.4f} (improvement: {best_auc - current_auc:+.4f})")

        model = CatBoostClassifier(
            task_type=cb_params.get("task_type", "GPU"),
            devices=cb_params.get("devices", "0"),
            loss_function=cb_params.get("loss_function", "Logloss"),
            learning_rate=best_params.get("learning_rate", 0.05),
            depth=best_params.get("depth", 8),
            l2_leaf_reg=best_params.get("l2_leaf_reg", 3),
            iterations=best_params.get("iterations", 1000),
            early_stopping_rounds=50,
            random_seed=cb_params.get("random_seed", 42),
            verbose=100,
            auto_class_weights="Balanced",
        )

        train_pool = Pool(X_train_chain, label=y_train_col)
        val_pool = Pool(X_val_chain, label=y_val_col)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        model.save_model(f"{data_dir}/catboost_model_opt_{target_idx}.cbm")
        
        new_val_preds = model.predict_proba(X_val_chain)[:, 1]
        oof_preds_optimized[:, target_idx] = new_val_preds

        study_results[target_name] = {
            "target_idx": target_idx,
            "best_params": best_params,
            "best_auc": best_auc,
            "current_auc": current_auc,
            "improvement": best_auc - current_auc,
        }

        progress["targets"][target_name] = {
            "status": "completed",
            "best_params": best_params,
            "best_auc": best_auc,
            "current_auc": current_auc,
            "improvement": best_auc - current_auc,
        }
        save_optuna_progress(progress)

        del X_train_chain, X_val_chain, train_pool, val_pool, model
        gc.collect()

    np.save(f"{data_dir}/oof_predictions_optimized.npy", oof_preds_optimized)
    
    macro_auc_before = macro_roc_auc(y_val, oof_preds_cb)
    macro_auc_after = macro_roc_auc(y_val, oof_preds_optimized)
    print(f"\n  Macro ROC-AUC before: {macro_auc_before:.5f}")
    print(f"  Macro ROC-AUC after: {macro_auc_after:.5f}")
    print(f"  Improvement: {macro_auc_after - macro_auc_before:+.5f}")

    with open(f"{data_dir}/optuna_study_results.json", "w") as f:
        json.dump(study_results, f, indent=2)

    del X_train_base, X_val_base, y_train, y_val, oof_preds_cb, oof_preds_optimized, chain_train
    gc.collect()

    return {
        "n_optimized": len(study_results),
        "macro_auc_before": macro_auc_before,
        "macro_auc_after": macro_auc_after,
        "improvement": macro_auc_after - macro_auc_before,
        "study_results": study_results,
    }