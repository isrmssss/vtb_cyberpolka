import sys
import os
import gc
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.extractors import SVDCompressor, get_extra_feature_columns_from_parquet


def run_svd(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 1] SVD compression on extra features...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    extra_feature_cols = get_extra_feature_columns_from_parquet(f"{data_dir}/train_extra_features.parquet")
    print(f"  Extra feature columns: {len(extra_feature_cols)}")

    svd_config = config["svd"]
    compressor = SVDCompressor(
        n_components=svd_config["n_components"],
        random_state=svd_config["random_state"],
    )

    train_svd_path, test_svd_path = compressor.fit_transform(
        f"{data_dir}/train_extra_features.parquet",
        f"{data_dir}/test_extra_features.parquet",
        extra_feature_cols,
    )
    compressor.save(f"{data_dir}/svd_compressor.pkl")

    del compressor, extra_feature_cols
    gc.collect()

    return {
        "train_svd_path": train_svd_path,
        "test_svd_path": test_svd_path,
        "n_components": svd_config["n_components"],
        "n_train": 750000,
        "n_test": 250000,
    }
