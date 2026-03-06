import pandas as pd
import os


def temporal_split(input_path: str, output_dir: str) -> None:
    # read_excel = xls. line 0 adalah deskripsi variabel/line 1 adalah header.
    df = pd.read_excel(input_path, header=1)

    # sort by ID untuk pengambilan data berdasarkan waktu melalui proxy.
    df = df.sort_values(by="ID").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"Train: {train.shape[0]}, Val: {val.shape[0]}, Test: {test.shape[0]}")


if __name__ == "__main__":
    temporal_split("data/raw/credit_data.xls", "data/processed")
