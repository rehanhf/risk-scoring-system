import numpy as np
import lightgbm as lgb
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath("."))
from src.features.preprocess import apply_pipeline


def generate_explanations(
    test_path: str, pipeline_path: str, model_path: str, output_dir: str
) -> None:
    # 1. load Artifacts
    X_test, y_test = apply_pipeline(test_path, pipeline_path)
    preprocessor = joblib.load(pipeline_path)
    model = lgb.Booster(model_file=model_path)

    # 2.extract feature names
    # preprocessor harus menyediakan nama kolom yang telah dikembangkan (misalnya, OHE categories).
    feature_names = preprocessor.get_feature_names_out()

    # 3. Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)

    # sample 1000 instances untuk mencegah bottleneck komputasi saat analisis globally
    X_sample = X_test[:1000]
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary klassifikasi returns list dari shap_values untuk beberapa versi, atau array.
    # extract array yang berkaitan dengan positive class (default)
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    os.makedirs(output_dir, exist_ok=True)

    # 4. global explanation: summary plot
    plt.figure()
    shap.summary_plot(
        shap_values_pos, X_sample, feature_names=feature_names, show=False
    )
    plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches="tight")
    plt.close()

    # 5. local explanation: single high-risk application
    # cari instance yang ter prediksi as high risk
    preds = np.array(model.predict(X_sample))
    high_risk_idx = int(np.argmax(preds))
    high_risk_prob = preds[high_risk_idx]

    print(
        f"Analyzing High-Risk Applicant Index {high_risk_idx} (Probability: {high_risk_prob:.4f})"
    )

    # force plot untuk yang specific applicant
    plt.figure()
    shap.force_plot(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list)
        else explainer.expected_value,
        shap_values_pos[high_risk_idx, :],
        X_sample[high_risk_idx, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.savefig(f"{output_dir}/shap_local_idx_{high_risk_idx}.png", bbox_inches="tight")
    plt.close()

    print(f"Explanations saved to {output_dir}")


if __name__ == "__main__":
    generate_explanations(
        test_path="data/processed/test.csv",
        pipeline_path="data/processed/preprocessor.joblib",
        model_path="src/models/lgbm_baseline.txt",
        output_dir="reports/figures",
    )
