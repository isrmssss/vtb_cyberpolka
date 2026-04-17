import polars as pl


class TimeSeriesSplitter:
    def __init__(self, customer_id_col="customer_id"):
        self.customer_id_col = customer_id_col

    def split(self, df):
        df_sorted = df.sort(self.customer_id_col)
        min_id = df_sorted[self.customer_id_col].min()
        max_id = df_sorted[self.customer_id_col].max()
        print(f"Customer ID range: {min_id} - {max_id}")

        splits = []

        fold1_val_start = min_id + 500000
        fold1_val_end = min_id + 600000
        train_fold1 = df_sorted.filter(pl.col(self.customer_id_col) < fold1_val_start)
        val_fold1 = df_sorted.filter(
            (pl.col(self.customer_id_col) >= fold1_val_start)
            & (pl.col(self.customer_id_col) < fold1_val_end)
        )
        splits.append((train_fold1, val_fold1))
        print(f"Fold 1: Train {len(train_fold1)}, Val {len(val_fold1)}")

        fold2_val_start = min_id + 600000
        train_fold2 = df_sorted.filter(pl.col(self.customer_id_col) < fold2_val_start)
        val_fold2 = df_sorted.filter(pl.col(self.customer_id_col) >= fold2_val_start)
        splits.append((train_fold2, val_fold2))
        print(f"Fold 2: Train {len(train_fold2)}, Val {len(val_fold2)}")

        return splits

    def simple_time_split(self, df, val_size=150000):
        df_sorted = df.sort(self.customer_id_col)
        train_split = df_sorted[:-val_size]
        val_split = df_sorted[-val_size:]
        print(f"Simple split: Train {len(train_split)}, Val {len(val_split)}")
        return train_split, val_split
