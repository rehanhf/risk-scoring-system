import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import numpy as np


def build_and_save_pipeline(train_path: str, output_dir: str):
    # 1.load STRICTLY train data
    train = pd.read_csv(train_path)

    # 2.define feature sets
    target = "default payment next month"
    drop_cols = ["ID", target]

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    # PAY_0 to PAY_6 = ordinal/status codes. Perlakukan sebagai kategori atau biarkan, gunakan OHE untuk strictness.
    pay_features = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    categorical_features.extend(pay_features)

    numerical_features = [
        col for col in train.columns if col not in categorical_features + drop_cols
    ]

    # 3. buat sub-pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" Mencegah crashes in production jika kategori baru muncul.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # 4. gabung jaid ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop anything not explicitly defined
    )

    # 5. fit strictly on train
    X_train = train.drop(columns=drop_cols)
    preprocessor.fit(X_train)

    # 6. serialisasi fitted pipeline
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(preprocessor, f"{output_dir}/preprocessor.joblib")

    print(
        f"Pipeline fitted dan saved. Output dimensi feature : {preprocessor.transform(X_train).shape[1]}"
    )


def apply_pipeline(
    data_path: str, pipeline_path: str
) -> tuple[np.ndarray, pd.Series | None]:
    df = pd.read_csv(data_path)
    preprocessor = joblib.load(pipeline_path)

    target = "default payment next month"
    X = df.drop(columns=["ID", target], errors="ignore")
    y = df[target] if target in df.columns else None

    X_processed = preprocessor.transform(X)
    return X_processed, y


if __name__ == "__main__":
    build_and_save_pipeline("data/processed/train.csv", "data/processed")
