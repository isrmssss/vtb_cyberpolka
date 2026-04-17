import polars as pl
import numpy as np
from tqdm import tqdm
import gc


def compute_rolling_target_encodings(df, customer_id_col, target_columns, feature_prefix="rolling"):
    df_sorted = df.sort(customer_id_col)

    rolling_cols = []
    for target_col in tqdm(target_columns, desc="  Computing rolling encodings", unit="target"):
        if target_col not in df_sorted.columns:
            continue

        rolling_col = f"{feature_prefix}_{target_col}"

        cumulative_sum = df_sorted.select(pl.col(target_col).cum_sum()).to_numpy().flatten()
        cumulative_count = np.arange(1, len(cumulative_sum) + 1, dtype=np.float64)

        rolling_mean = cumulative_sum / cumulative_count

        rolling_mean_lagged = np.concatenate([[np.nan], rolling_mean[:-1]])

        df_sorted = df_sorted.with_columns(pl.Series(rolling_col, rolling_mean_lagged))
        rolling_cols.append(rolling_col)

        del cumulative_sum, cumulative_count, rolling_mean, rolling_mean_lagged
        gc.collect()

    print(f"  Created {len(rolling_cols)} rolling target encoding features")
    return df_sorted, rolling_cols


def fill_rolling_nan_with_global_mean(df, rolling_cols, target_columns, df_full):
    for i, rolling_col in enumerate(tqdm(rolling_cols, desc="  Filling rolling NaN", leave=False)):
        target_col = target_columns[i]
        global_mean = df_full.select(pl.col(target_col).mean()).item()
        df = df.with_columns(
            pl.col(rolling_col).fill_null(global_mean)
        )
    return df
