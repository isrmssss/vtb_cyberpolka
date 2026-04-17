from sklearn.metrics import roc_auc_score
import numpy as np


def macro_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="macro")


def micro_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="micro")


def per_target_auc(y_true, y_pred, target_names):
    aucs = {}
    for i, name in enumerate(target_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs[name] = auc
        except ValueError:
            aucs[name] = np.nan
    return aucs
