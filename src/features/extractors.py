import polars as pl
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.impute import SimpleImputer
import joblib
from tqdm import tqdm
import os
import gc


def create_nan_flags(df, feature_cols, top_n=50):
    null_counts = {}
    for col in tqdm(feature_cols, desc="  Scanning NaN counts", leave=False):
        null_counts[col] = df.select(pl.col(col).null_count()).item()

    sorted_features = sorted(null_counts.items(), key=lambda x: x[1], reverse=True)
    top_nan_features = [f[0] for f in sorted_features[:top_n] if f[1] > 0]

    print(f"  Creating NaN flags for {len(top_nan_features)} features with most missing values")

    nan_flag_cols = []
    for col in tqdm(top_nan_features, desc="  Creating NaN flags", leave=False):
        flag_col = f"{col}_is_nan"
        df = df.with_columns(pl.col(col).is_null().cast(pl.Int8).alias(flag_col))
        nan_flag_cols.append(flag_col)

    print(f"  Created {len(nan_flag_cols)} NaN flag columns")
    return df, nan_flag_cols


def get_extra_feature_columns_from_parquet(parquet_path):
    lf = pl.scan_parquet(parquet_path)
    return [name for name in lf.collect_schema().names() if name != "customer_id"]


def read_parquet_to_memmap(parquet_path, feature_cols, memmap_path, dtype=np.float32, chunk_size=50000):
    lf = pl.scan_parquet(parquet_path)
    total_rows = lf.select(pl.count()).collect().item()
    n_features = len(feature_cols)

    print(f"  Reading {parquet_path}: {total_rows} rows x {n_features} features")
    memmap_size_mb = total_rows * n_features * np.dtype(dtype).itemsize / (1024 * 1024)
    print(f"  Memmap size: {memmap_size_mb:.0f} MB")

    memmap = np.memmap(memmap_path, dtype=dtype, mode="w+", shape=(total_rows, n_features))

    n_chunks = (total_rows + chunk_size - 1) // chunk_size

    for i in tqdm(range(n_chunks), desc=f"  Reading {os.path.basename(parquet_path)}", leave=False):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)

        chunk = lf.slice(start, end - start).collect()
        arr = chunk.select(feature_cols).to_numpy().astype(dtype)
        memmap[start:end] = arr

        del chunk, arr
        gc.collect()

    memmap.flush()
    print(f"  Memmap written: {memmap.shape}, dtype={memmap.dtype}")
    return memmap


class SVDCompressor:
    def __init__(self, n_components=100, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.pca = IncrementalPCA(n_components=n_components, batch_size=50000)
        self.is_fitted = False

    def _fill_nan_chunked(self, memmap_path, shape, medians, chunk_size=50000):
        total_rows = shape[0]
        n_features = shape[1]
        n_chunks = (total_rows + chunk_size - 1) // chunk_size

        mm = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=shape)
        for i in tqdm(range(n_chunks), desc="  Filling NaN chunks", leave=False):
            start = i * chunk_size
            end = min(start + chunk_size, total_rows)
            chunk = mm[start:end].copy()
            nan_mask = np.isnan(chunk)
            if nan_mask.any():
                chunk = np.where(nan_mask, medians[np.newaxis, :], chunk)
                mm[start:end] = chunk
        mm.flush()
        del mm
        gc.collect()

    def fit_transform(self, train_extra_path, test_extra_path, extra_feature_cols,
                      memmap_dir="data"):
        os.makedirs(memmap_dir, exist_ok=True)
        train_memmap_path = os.path.join(memmap_dir, "train_extra_memmap.dat")
        test_memmap_path = os.path.join(memmap_dir, "test_extra_memmap.dat")
        train_svd_path = os.path.join(memmap_dir, "train_svd.npy")
        test_svd_path = os.path.join(memmap_dir, "test_svd.npy")

        print(f"  Reading train extra features to memmap...")
        train_memmap = read_parquet_to_memmap(
            train_extra_path, extra_feature_cols, train_memmap_path
        )
        n_train = train_memmap.shape[0]
        n_features = train_memmap.shape[1]
        del train_memmap
        gc.collect()

        print(f"  Reading test extra features to memmap...")
        test_memmap = read_parquet_to_memmap(
            test_extra_path, extra_feature_cols, test_memmap_path
        )
        n_test = test_memmap.shape[0]
        del test_memmap
        gc.collect()

        train_shape = (n_train, n_features)
        test_shape = (n_test, n_features)

        print(f"  Train memmap: {train_shape}, Test memmap: {test_shape}")
        print(f"  Computing medians in batches from parquet (memory-efficient)...")

        batch_size = 200
        medians = np.zeros(n_features, dtype=np.float32)
        n_batches = (n_features + batch_size - 1) // batch_size

        train_scan = pl.scan_parquet(train_extra_path)
        for b in tqdm(range(n_batches), desc="  Computing medians", leave=False):
            start = b * batch_size
            end = min(start + batch_size, n_features)
            batch_cols = extra_feature_cols[start:end]
            batch_medians = train_scan.select(pl.col(batch_cols).median()).collect()
            medians[start:end] = batch_medians.to_numpy().flatten().astype(np.float32)

        medians = np.nan_to_num(medians, nan=0.0)
        del train_scan
        gc.collect()

        print(f"  Filling NaN values in train memmap (chunked)...")
        chunk_size = 50000
        self._fill_nan_chunked(train_memmap_path, train_shape, medians, chunk_size)

        print(f"  Filling NaN values in test memmap (chunked)...")
        self._fill_nan_chunked(test_memmap_path, test_shape, medians, chunk_size)

        print(f"  Imputation complete")
        del medians
        gc.collect()

        train_mm = np.memmap(train_memmap_path, dtype=np.float32, mode="r", shape=train_shape)

        print(f"  Fitting IncrementalPCA with {self.n_components} components on train only (chunked)...")
        n_chunks_fit = (n_train + chunk_size - 1) // chunk_size
        for i in tqdm(range(n_chunks_fit), desc="  PCA partial_fit", leave=False):
            start = i * chunk_size
            end = min(start + chunk_size, n_train)
            self.pca.partial_fit(train_mm[start:end])
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance ratio (first {self.n_components} components): {explained_var:.4f}")

        del train_mm
        gc.collect()

        print(f"  Transforming train in chunks...")
        train_mm = np.memmap(train_memmap_path, dtype=np.float32, mode="r", shape=train_shape)
        X_train_svd = np.memmap(train_svd_path, dtype=np.float32, mode="w+", shape=(n_train, self.n_components))
        for i in tqdm(range(n_chunks_fit), desc="  PCA transform train", leave=False):
            start = i * chunk_size
            end = min(start + chunk_size, n_train)
            X_train_svd[start:end] = self.pca.transform(train_mm[start:end])
        X_train_svd.flush()
        del train_mm, X_train_svd
        gc.collect()
        try:
            os.unlink(train_memmap_path)
        except OSError:
            pass

        print(f"  Transforming test in chunks...")
        test_mm = np.memmap(test_memmap_path, dtype=np.float32, mode="r", shape=test_shape)
        n_chunks_test = (n_test + chunk_size - 1) // chunk_size
        X_test_svd = np.memmap(test_svd_path, dtype=np.float32, mode="w+", shape=(n_test, self.n_components))
        for i in tqdm(range(n_chunks_test), desc="  PCA transform test", leave=False):
            start = i * chunk_size
            end = min(start + chunk_size, n_test)
            X_test_svd[start:end] = self.pca.transform(test_mm[start:end])
        X_test_svd.flush()
        del test_mm, X_test_svd
        gc.collect()
        try:
            os.unlink(test_memmap_path)
        except OSError:
            pass

        self.is_fitted = True
        print(f"  PCA complete: train={train_svd_path}, test={test_svd_path}")

        return train_svd_path, test_svd_path

    def transform(self, extra_df, extra_feature_cols):
        if not self.is_fitted:
            raise RuntimeError("SVDCompressor must be fitted before transform")

        X = extra_df.select(extra_feature_cols).to_numpy().astype(np.float32)
        X_imputed = self.imputer.transform(X)
        X_svd = self.pca.transform(X_imputed)
        return X_svd

    def save(self, path="data/svd_compressor.pkl"):
        joblib.dump({
            "imputer": self.imputer,
            "pca": self.pca,
            "n_components": self.n_components,
            "is_fitted": self.is_fitted,
        }, path)
        print(f"  SVDCompressor saved to {path}")

    @classmethod
    def load(cls, path="data/svd_compressor.pkl"):
        data = joblib.load(path)
        compressor = cls(n_components=data["n_components"])
        compressor.imputer = data["imputer"]
        compressor.pca = data["pca"]
        compressor.is_fitted = data["is_fitted"]
        print(f"  SVDCompressor loaded from {path}")
        return compressor


def get_extra_feature_columns(df):
    return [col for col in df.columns if col != "customer_id" and not col.startswith("cat_feature") and not col.startswith("target") and not col.startswith("rolling") and not col.startswith("svd") and not col.endswith("_is_nan")]
