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
    # 1. Load Artifacts
    X_test, y_test = apply_pipeline(test_path, pipeline_path)
    preprocessor = joblib.load(pipeline_path)
    model = lgb.Booster(model_file=model_path)

    # 2. Extract Feature Names
    # Preprocessor must provide the exact expanded column names (e.g., OHE categories)
    feature_names = preprocessor.get_feature_names_out()

    # 3. Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)

    # Sample 1000 instances to prevent computational bottleneck during global analysis
    X_sample = X_test[:1000]
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary classification returns a list of shap_values for some versions, or an array.
    # Extract the array corresponding to the positive class (Default)
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    os.makedirs(output_dir, exist_ok=True)

    # 4. Global Explanation: Summary Plot
    plt.figure()
    shap.summary_plot(
        shap_values_pos, X_sample, feature_names=feature_names, show=False
    )
    plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches="tight")
    plt.close()

    # 5. Local Explanation: Single High-Risk Application
    # Find an instance predicted as high risk
    preds = np.array(model.predict(X_sample))
    high_risk_idx = int(np.argmax(preds))
    high_risk_prob = preds[high_risk_idx]

    print(
        f"Analyzing High-Risk Applicant Index {high_risk_idx} (Probability: {high_risk_prob:.4f})"
    )

    # Force plot for the specific applicant
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
