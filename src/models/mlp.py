import torch
import torch.nn as nn


class CreditRiskMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(CreditRiskMLP, self).__init__()
        # Architecture: Input -> 64 -> 32 -> 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            # No Sigmoid here. BCEWithLogitsLoss applies it internally for numerical stability.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
