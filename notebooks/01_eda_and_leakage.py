import pandas as pd

# 1. load train data
train = pd.read_csv("./data/processed/train.csv")

# 2. target definition
target_col = "default payment next month"
print(f"Target prevalence (Base Rate): {train[target_col].mean():.4f}")

# 3. audit leakage
# drop ID. Kolom ini adalah integer monotonic yang mengikuti urutan chronological.
# model tree-based akan tersplit berdasarkan ID dan overfit terhadap sumbu temporal, hal ini dapat membuat catastrophic failure saat OOT test.
train = train.drop(columns=["ID"])

# define ruang lingkup operasional: Behavioral Scoring.
# memprediksi default bulan Oktober menggunakan data hingga bulan September.
# PAY_0 (September status) hingga PAY_6 (April status)adalah historical predictors yang valid.
# jika ruang lingkup operasional adalah Origination Scoring (memprediksi default saat kartu diterbitkan),
# maka semua kolom PAY_*, BILL_AMT_*, dan PAY_AMT_* dapat menjadi severe data leakage dan harus dihapus.

print("Features retained for Behavioral Scoring:")
print(train.columns.tolist())
