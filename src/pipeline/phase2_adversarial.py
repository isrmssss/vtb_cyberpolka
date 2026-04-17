import sys
import os
import gc
import yaml
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import get_feature_columns
from src.features.selector import adversarial_validation, plot_adversarial_importance


def run_adversarial(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 2] Adversarial validation on main features...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_main = pl.read_parquet(f"{data_dir}/train_main_features.parquet")
    test_main = pl.read_parquet(f"{data_dir}/test_main_features.parquet")

    feature_cols, cat_feature_names, num_feature_names = get_feature_columns(train_main)

    X_train_arr = train_main.select(feature_cols).to_numpy()
    X_test_arr = test_main.select(feature_cols).to_numpy()

    del train_main, test_main
    gc.collect()

    adv_config = config["adversarial"]
    stable_features, shifted_features, feature_importance = adversarial_validation(
        X_train_arr, X_test_arr, feature_cols,
        importance_threshold=adv_config["importance_threshold"],
        random_state=adv_config["random_state"],
    )

    del X_train_arr, X_test_arr
    gc.collect()

    plot_adversarial_importance(
        feature_cols, feature_importance,
        top_n=adv_config["top_n_plot"],
        save_path=f"{data_dir}/adversarial_feature_importance.png",
    )

    del feature_importance
    gc.collect()

    return {
        "n_stable": len(stable_features),
        "n_shifted": len(shifted_features),
    }
