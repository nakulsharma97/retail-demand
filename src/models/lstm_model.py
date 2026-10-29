"""
PyTorch LSTM Demand Forecasting Model
Sequence-to-one architecture for time series prediction.
Demonstrates deep learning capability with PyTorch.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import mlflow
from loguru import logger


class RetailDemandDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def prepare_sequences(series: np.ndarray, seq_len: int = 12):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)


def train_lstm(train_series: np.ndarray, epochs: int = 30,
               seq_len: int = 12, batch_size: int = 32,
               lr: float = 1e-3) -> tuple:

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.reshape(-1,1)).flatten()
    X, y   = prepare_sequences(scaled, seq_len)
    X      = X.reshape(X.shape[0], seq_len, 1)

    dataset    = RetailDemandDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMForecaster(input_size=1).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="lstm_demand_forecast"):
        mlflow.log_params({"epochs": epochs, "seq_len": seq_len, "lr": lr})
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optim.zero_grad(); loss.backward(); optim.step()
                epoch_loss += loss.item()
            avg = epoch_loss / len(dataloader)
            if (epoch+1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg:.4f}")
                mlflow.log_metric("train_loss", avg, step=epoch)

    return model, scaler, seq_len, device
