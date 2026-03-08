import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.append(os.path.abspath("."))
from src.features.preprocess import apply_pipeline
from src.models.mlp import CreditRiskMLP


def compute_pr_auc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def train_neural_network(
    train_path: str, val_path: str, pipeline_path: str, model_dir: str
):
    # 1. Load Data
    X_train, y_train = apply_pipeline(train_path, pipeline_path)
    X_val, y_val = apply_pipeline(val_path, pipeline_path)

    # insert strict type guard:
    if y_train is None or y_val is None:
        raise ValueError("Targets missing from training or validation data.")

    # 2.convert ke tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train.to_numpy()).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)

    # 3. buat DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True
    )

    # 4.imbalance handling
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    print(f"Neural Network pos_weight: {pos_weight.item():.2f}")

    # 5. inisialisasi
    input_dim = X_train.shape[1]
    model = CreditRiskMLP(input_dim)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 6. loop training
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 7. validasi evaluasi
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).numpy()

    val_roc = roc_auc_score(y_val, val_probs)
    val_pr = compute_pr_auc(y_val, val_probs)

    print(f"PyTorch Validation ROC-AUC: {val_roc:.4f}")
    print(f"PyTorch Validation PR-AUC: {val_pr:.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/mlp_baseline.pth")


if __name__ == "__main__":
    train_neural_network(
        train_path="data/processed/train.csv",
        val_path="data/processed/val.csv",
        pipeline_path="data/processed/preprocessor.joblib",
        model_dir="src/models",
    )
