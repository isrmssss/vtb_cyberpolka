import sys
import os
import gc
import json
import yaml
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import cast_categorical_features
from src.features.extractors import create_nan_flags


def load_svd_as_memmap(svd_path, shape):
    return np.memmap(svd_path, dtype=np.float32, mode="r", shape=shape)


def run_inference(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 9] Final Inference with optimized models...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(f"{data_dir}/feature_metadata_day2.json") as f:
        metadata = json.load(f)

    with open(f"{data_dir}/target_order_day2.json") as f:
        target_order = json.load(f)

    with open(f"{data_dir}/optuna_study_results.json") as f:
        optuna_results = json.load(f)

    optimized_targets = set(optuna_results.keys())
    print(f"  Optimized targets: {optimized_targets}")

    test_main = pl.read_parquet(f"{data_dir}/test_main_features.parquet")
    test_main = test_main.sort("customer_id")

    rolling_cols = metadata["rolling_cols"]
    feature_cols = metadata["feature_cols"]
    nan_flag_cols = metadata["nan_flag_cols"]

    if rolling_cols:
        test_main = test_main.with_columns([pl.lit(None).alias(col) for col in rolling_cols])

    test_main, _ = create_nan_flags(test_main, feature_cols, top_n=config["nan_flags"]["top_n"])
    test_main = cast_categorical_features(test_main, metadata["cat_feature_names"])

    test_customer_ids = test_main["customer_id"].to_list()
    X_test_main = test_main.select(feature_cols + nan_flag_cols).to_numpy().astype(np.float32)
    X_test_rolling = test_main.select(rolling_cols).to_numpy().astype(np.float32) if rolling_cols else np.empty((test_main.shape[0], 0), dtype=np.float32)

    del test_main
    gc.collect()

    n_svd = 100
    test_svd_mm = np.memmap(f"{data_dir}/test_svd.npy", dtype=np.float32, mode="r", shape=(250000, n_svd))
    X_test_svd = np.array(test_svd_mm, dtype=np.float32)
    del test_svd_mm
    gc.collect()

    X_test_base = np.hstack([X_test_main, X_test_rolling, X_test_svd])
    del X_test_main, X_test_svd
    gc.collect()

    target_columns = metadata["target_columns"]
    n_test = X_test_base.shape[0]
    n_targets = len(target_columns)
    test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

    print(f"  Test features: {X_test_base.shape[1]}")
    print(f"  Predicting {n_targets} models...")

    for chain_idx, (target_idx, target_name) in enumerate(tqdm(target_order, desc="  Test predict", unit="model")):
        if chain_idx == 0:
            X_test_chain = X_test_base
        else:
            prev_indices = [t[0] for t in target_order[:chain_idx]]
            X_test_chain = np.hstack([X_test_base, test_preds[:, prev_indices]])

        if target_name in optimized_targets:
            model_path = f"{data_dir}/catboost_model_opt_{target_idx}.cbm"
            print(f"    Using optimized model for {target_name}")
        else:
            model_path = f"{data_dir}/catboost_model_{target_idx}.cbm"

        model = CatBoostClassifier()
        model.load_model(model_path)
        test_preds[:, target_idx] = model.predict_proba(X_test_chain)[:, 1]

        del model, X_test_chain
        gc.collect()

    del X_test_base
    gc.collect()

    postprocessing_config = config.get("postprocessing", {})
    clip_enabled = postprocessing_config.get("enabled", True)
    clip_threshold = postprocessing_config.get("clip_threshold", 0.99)

    if clip_enabled:
        print(f"  Applying clip at {clip_threshold}...")
        test_preds = np.clip(test_preds, 0, clip_threshold)

    print(f"  Test predictions range: [{test_preds.min():.6f}, {test_preds.max():.6f}]")

    np.save(f"{data_dir}/test_predictions_day3.npy", test_preds)

    predict_schema = [col.replace("target_", "predict_") for col in target_columns]

    sample_submit = pl.read_parquet(f"{data_dir}/sample_submit.parquet")
    submit = pl.DataFrame({"customer_id": test_customer_ids})
    for i, pred_col in enumerate(predict_schema):
        submit = submit.with_columns(pl.Series(pred_col, test_preds[:, i]))

    expected_cols = sample_submit.columns
    submit = submit.select(expected_cols)
    del sample_submit
    gc.collect()

    submit.write_parquet(f"{data_dir}/submit_day3.parquet")
    print(f"  Submit saved to {data_dir}/submit_day3.parquet")

    del test_preds
    gc.collect()

    return {"submit_path": f"{data_dir}/submit_day3.parquet"}
