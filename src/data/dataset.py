import polars as pl


def load_data(data_dir="data"):
    train_main = pl.read_parquet(f"{data_dir}/train_main_features.parquet")
    train_extra = pl.read_parquet(f"{data_dir}/train_extra_features.parquet")
    train_target = pl.read_parquet(f"{data_dir}/train_target.parquet")
    test_main = pl.read_parquet(f"{data_dir}/test_main_features.parquet")
    test_extra = pl.read_parquet(f"{data_dir}/test_extra_features.parquet")
    sample_submit = pl.read_parquet(f"{data_dir}/sample_submit.parquet")

    print(f"Train main: {train_main.shape}")
    print(f"Train extra: {train_extra.shape}")
    print(f"Train target: {train_target.shape}")
    print(f"Test main: {test_main.shape}")
    print(f"Test extra: {test_extra.shape}")

    return train_main, train_extra, train_target, test_main, test_extra, sample_submit


def merge_datasets(train_main, train_extra, train_target):
    train = train_main.join(train_extra, on="customer_id", how="left")
    train = train.join(train_target, on="customer_id", how="left")
    train = train.sort("customer_id")
    print(f"Merged train: {train.shape}")
    return train


def get_feature_columns(df):
    cat_feature_names = [col for col in df.columns if col.startswith("cat_feature")]
    num_feature_names = [col for col in df.columns if col.startswith("num_feature")]
    feature_cols = cat_feature_names + num_feature_names
    return feature_cols, cat_feature_names, num_feature_names


def get_target_columns(df):
    return [col for col in df.columns if col.startswith("target_")]


def get_predict_schema(target_columns):
    return [col.replace("target_", "predict_") for col in target_columns]


def cast_categorical_features(df, cat_feature_names):
    return df.with_columns(pl.col(cat_feature_names).cast(pl.Int32))
