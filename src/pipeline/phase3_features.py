import sys
import os
import gc
import json
import yaml
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import get_feature_columns, get_target_columns, cast_categorical_features
from src.features.extractors import create_nan_flags
from src.data.preprocessing import compute_rolling_target_encodings


def run_features(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 3] Feature engineering (NaN flags + rolling encodings)...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_main = pl.read_parquet(f"{data_dir}/train_main_features.parquet")
    train_target = pl.read_parquet(f"{data_dir}/train_target.parquet")
    test_main = pl.read_parquet(f"{data_dir}/test_main_features.parquet")

    train = train_main.join(train_target, on="customer_id", how="left")
    train = train.sort("customer_id")
    test = test_main.sort("customer_id")

    del train_main, test_main, train_target
    gc.collect()

    feature_cols, cat_feature_names, num_feature_names = get_feature_columns(train)
    target_columns = get_target_columns(train)

    train, nan_flag_cols = create_nan_flags(train, feature_cols, top_n=config["nan_flags"]["top_n"])
    test, _ = create_nan_flags(test, feature_cols, top_n=config["nan_flags"]["top_n"])

    all_feature_cols = feature_cols + nan_flag_cols

    rolling_config = config["rolling_encoding"]
    rolling_cols = []
    if rolling_config["enabled"]:
        train, rolling_cols = compute_rolling_target_encodings(
            train, "customer_id", target_columns,
            feature_prefix=rolling_config["feature_prefix"],
        )
        all_feature_cols = all_feature_cols + rolling_cols
        test = test.with_columns([pl.lit(None).alias(col) for col in rolling_cols])

    train = cast_categorical_features(train, cat_feature_names)
    test = cast_categorical_features(test, cat_feature_names)

    print(f"  Total features: {len(all_feature_cols)} (main={len(feature_cols)}, nan_flags={len(nan_flag_cols)}, rolling={len(rolling_cols)})")
    print(f"  Saving processed data to disk...")

    train.write_parquet(f"{data_dir}/train_processed_day2.parquet")
    test.write_parquet(f"{data_dir}/test_processed_day2.parquet")

    metadata = {
        "feature_cols": feature_cols,
        "cat_feature_names": cat_feature_names,
        "num_feature_names": num_feature_names,
        "nan_flag_cols": nan_flag_cols,
        "rolling_cols": rolling_cols,
        "all_feature_cols": all_feature_cols,
        "target_columns": target_columns,
        "n_features": len(all_feature_cols),
        "n_targets": len(target_columns),
    }

    with open(f"{data_dir}/feature_metadata_day2.json", "w") as f:
        json.dump(metadata, f)

    del train, test
    gc.collect()

    return metadata
