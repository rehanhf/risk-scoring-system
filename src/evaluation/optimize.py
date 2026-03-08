import numpy as np
import lightgbm as lgb
import sys
import os

sys.path.append(os.path.abspath("."))
from src.features.preprocess import apply_pipeline


def optimize_threshold(
    val_path: str, pipeline_path: str, model_path: str, c_fp: float, c_fn: float
) -> None:
    # 1. load Data and model
    X_val, y_val = apply_pipeline(val_path, pipeline_path)

    if y_val is None:
        raise ValueError("Validation target missing.")
    y_val_arr = y_val.to_numpy()

    model = lgb.Booster(model_file=model_path)

    # 2. prediksi probabilitas
    y_prob = model.predict(X_val)

    # 3. simulasi economic cost across thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        fp = np.sum((y_pred == 1) & (y_val_arr == 0))
        fn = np.sum((y_pred == 0) & (y_val_arr == 1))

        total_cost = (fp * c_fp) + (fn * c_fn)
        costs.append(total_cost)

    # 4. identifikasi threshold optimal
    min_cost_idx = np.argmin(costs)
    opt_threshold = thresholds[min_cost_idx]
    min_cost = costs[min_cost_idx]

    print(f"Optimal Threshold: {opt_threshold:.2f}")
    print(f"Minimum Expected Validation Cost: ${min_cost:,.2f}")

    # baseline cost (approve everyone -> threshold = 1.0 -> All defaults jadi FN)
    baseline_fn = np.sum(y_val_arr == 1)
    baseline_cost = baseline_fn * c_fn
    print(f"Baseline Cost (Approve All): ${baseline_cost:,.2f}")
    print(f"Cost Savings: ${(baseline_cost - min_cost):,.2f}")


if __name__ == "__main__":
    optimize_threshold(
        val_path="data/processed/val.csv",
        pipeline_path="data/processed/preprocessor.joblib",
        model_path="src/models/lgbm_baseline.txt",
        c_fp=500.0,
        c_fn=5000.0,
    )
