import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import gc


class BinaryRelevanceLGBM:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "random_state": 42,
            "verbose": -1,
        }
        self.models = []
        self.target_names = []

    def fit(self, X_train, y_train, X_val, y_val, target_names=None):
        self.models = []
        self.target_names = target_names or [f"target_{i}" for i in range(y_train.shape[1])]

        n_targets = len(self.target_names)
        print(f"  Training {n_targets} binary classifiers...")

        train_data_base = lgb.Dataset(X_train, label=y_train[:, 0])

        for i, target_name in enumerate(tqdm(self.target_names, desc="  Training models", unit="model")):
            y_train_col = y_train[:, i]
            y_val_col = y_val[:, i]

            pos_count = int(y_train_col.sum())
            neg_count = len(y_train_col) - pos_count
            if pos_count > 0:
                pos_weight = neg_count / pos_count
                pos_weight = min(pos_weight, 50.0)
            else:
                pos_weight = 1.0

            model_params = self.params.copy()
            model_params["scale_pos_weight"] = pos_weight

            train_data = lgb.Dataset(X_train, label=y_train_col, free_raw_data=False)
            train_data.construct()
            val_data = lgb.Dataset(X_val, label=y_val_col, reference=train_data, free_raw_data=False)
            val_data.construct()

            model = lgb.train(
                model_params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(self.params["early_stopping_rounds"], verbose=False)],
            )

            self.models.append(model)

            del train_data, val_data
            gc.collect()

            val_pred = model.predict(X_val)
            try:
                auc = roc_auc_score(y_val_col, val_pred)
                tqdm.write(f"    {target_name}: Val AUC = {auc:.4f} (pos={pos_count}, best_iter={model.best_iteration})")
            except ValueError:
                tqdm.write(f"    {target_name}: No positive samples in val")

        del train_data_base
        gc.collect()

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(tqdm(self.models, desc="  Predicting", unit="model")):
            predictions[:, i] = model.predict(X)
        return predictions

    def get_best_iteration(self):
        return [m.best_iteration for m in self.models]
