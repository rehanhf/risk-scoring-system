import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import sys
import os
from typing import Tuple

sys.path.append(os.path.abspath("."))
from src.features.preprocess import apply_pipeline


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def pr_auc_eval(preds: np.ndarray, train_data: lgb.Dataset) -> Tuple[str, float, bool]:
    labels = train_data.get_label()
    if labels is None:
        raise ValueError("Labels not found in dataset.")
    # explicitly konversi ke numpy array untuk memenuhi strict checking type
    labels_arr = np.asarray(labels)
    pr_auc_val = compute_pr_auc(labels_arr, preds)
    return "pr_auc", pr_auc_val, True


def train_lgbm(
    train_path: str, val_path: str, pipeline_path: str, model_dir: str
) -> None:
    X_train, y_train = apply_pipeline(train_path, pipeline_path)
    X_val, y_val = apply_pipeline(val_path, pipeline_path)

    if y_train is None or y_val is None:
        raise ValueError("Target column missing from training or validation data.")

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    neg_count = (y_train_arr == 0).sum()
    pos_count = (y_train_arr == 1).sum()
    scale_pos_weight = float(neg_count / pos_count)

    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "objective": "binary",
        "metric": "custom",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train_arr)
    val_data = lgb.Dataset(X_val, label=y_val_arr, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        feval=pr_auc_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)],
    )

    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/lgbm_baseline.txt")

    val_preds = np.array(model.predict(X_val))
    val_roc = float(roc_auc_score(y_val_arr, val_preds))
    val_pr = compute_pr_auc(y_val_arr, val_preds)

    print(f"Validation ROC-AUC: {val_roc:.4f}")
    print(f"Validation PR-AUC: {val_pr:.4f}")


if __name__ == "__main__":
    train_lgbm(
        train_path="data/processed/train.csv",
        val_path="data/processed/val.csv",
        pipeline_path="data/processed/preprocessor.joblib",
        model_dir="src/models",
    )
