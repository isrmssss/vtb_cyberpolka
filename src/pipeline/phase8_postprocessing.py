import sys
import os
import gc
import json
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.metrics import macro_roc_auc


def run_postprocessing(config_path="configs/model_params.yaml", data_dir="data"):
    print("\n[Phase 8] Post-processing adjustments...")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    postprocessing_config = config.get("postprocessing", {})
    clip_threshold = postprocessing_config.get("clip_threshold", 0.99)

    print(f"  Loading OOF predictions...")
    oof_preds = np.load(f"{data_dir}/oof_predictions_optimized.npy")
    y_val = np.load(f"{data_dir}/val_true_day2.npy")

    print(f"  Original shape: {oof_preds.shape}")
    print(f"  Original range: [{oof_preds.min():.6f}, {oof_preds.max():.6f}]")

    macro_before = macro_roc_auc(y_val, oof_preds)
    print(f"  Macro ROC-AUC before post-processing: {macro_before:.5f}")

    print(f"\n  Applying clip at {clip_threshold}...")
    oof_clipped = np.clip(oof_preds, 0, clip_threshold)

    macro_after_clip = macro_roc_auc(y_val, oof_clipped)
    print(f"  Macro ROC-AUC after clip: {macro_after_clip:.5f}")
    print(f"  Clip improvement: {macro_after_clip - macro_before:+.5f}")

    print(f"\n  Analyzing target correlations...")
    with open(f"{data_dir}/feature_metadata_day2.json") as f:
        metadata = json.load(f)
    target_columns = metadata["target_columns"]

    corr_matrix = np.corrcoef(y_val.T)
    print(f"  Target correlation matrix shape: {corr_matrix.shape}")

    negative_pairs = []
    for i in range(len(target_columns)):
        for j in range(i + 1, len(target_columns)):
            if corr_matrix[i, j] < -0.1:
                negative_pairs.append((target_columns[i], target_columns[j], float(corr_matrix[i, j])))

    if negative_pairs:
        print(f"  Found {len(negative_pairs)} negatively correlated pairs:")
        for t1, t2, corr in sorted(negative_pairs, key=lambda x: x[2])[:5]:
            print(f"    {t1} vs {t2}: {corr:.3f}")
    else:
        print(f"  No significant negative correlations found")

    oof_final = oof_clipped

    print(f"\n  Final post-processed predictions:")
    print(f"    Range: [{oof_final.min():.6f}, {oof_final.max():.6f}]")
    print(f"    Mean: {oof_final.mean():.6f}")
    print(f"    Std: {oof_final.std():.6f}")

    macro_final = macro_roc_auc(y_val, oof_final)
    print(f"  Final Macro ROC-AUC: {macro_final:.5f}")

    np.save(f"{data_dir}/oof_predictions_postprocessed.npy", oof_final)

    with open(f"{data_dir}/postprocessing_results.json", "w") as f:
        json.dump({
            "clip_threshold": clip_threshold,
            "macro_before": float(macro_before),
            "macro_after_clip": float(macro_after_clip),
            "macro_final": float(macro_final),
            "n_negative_correlations": len(negative_pairs),
            "negative_pairs_sample": negative_pairs[:5] if negative_pairs else [],
        }, f, indent=2)

    del oof_preds, oof_clipped, oof_final, y_val, corr_matrix
    gc.collect()

    return {
        "macro_before": float(macro_before),
        "macro_after_clip": float(macro_after_clip),
        "macro_final": float(macro_final),
        "n_negative_correlations": len(negative_pairs),
    }
