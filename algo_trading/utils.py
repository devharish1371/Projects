import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import linalg

import torch
from torch.utils.data import DataLoader, TensorDataset


# ------------------------- Logging -------------------------
def get_logger(log_dir: str, name: str, filename: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicated handlers in interactive runs
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, filename))
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ------------------------- Graph Utils -------------------------
def calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: Optional[float] = None) -> sp.csr_matrix:
    """
    Compute scaled Laplacian: L_tilde = 2L / lambda_max - I
    where L = D - A (combinatorial Laplacian).
    """
    assert adj_mx.shape[0] == adj_mx.shape[1], "Adjacency must be square"
    A = sp.csr_matrix(adj_mx)
    d = np.array(A.sum(1)).flatten()
    D = sp.diags(d)
    L = D - A
    if lambda_max is None:
        lambda_max = linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
    identity_matrix = sp.identity(A.shape[0], format='csr')
    L_tilde = (2.0 / lambda_max) * L - identity_matrix
    return L_tilde.tocsr()


def calculate_random_walk_matrix(adj_mx: np.ndarray) -> sp.csr_matrix:
    A = sp.csr_matrix(adj_mx)
    d = np.array(A.sum(1)).flatten()
    d[d == 0] = 1.0
    D_inv = sp.diags(1.0 / d)
    return D_inv @ A


def calculate_wavelet(*args, **kwargs):
    # Placeholder to satisfy imports; wavelet filters not used in this project
    raise NotImplementedError("Wavelet filter not implemented for this project.")


# ------------------------- Scaling -------------------------
@dataclass
class StandardScaler:
    mean: float
    std: float

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * (self.std + 1e-8) + self.mean


# ------------------------- Data Loading -------------------------
class SimpleIterator:
    """
    Wrapper to provide .get_iterator() and .num_batch similar to the original codebase.
    """

    def __init__(self, dataloader: DataLoader):
        self._dl = dataloader
        self.num_batch = len(self._dl)

    def get_iterator(self):
        for batch in self._dl:
            yield batch


def _create_windows(series_tensor: torch.Tensor, seq_len: int, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create overlapping windows from time-series tensor.
    series_tensor: shape (T, N) where N=num_nodes (features per node = 1)
    Returns:
      X: (num_samples, seq_len, N)
      Y: (num_samples, horizon, N)
    """
    T, N = series_tensor.shape
    num_samples = T - seq_len - horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough timesteps to create windows with given seq_len and horizon.")
    xs = []
    ys = []
    for t in range(num_samples):
        x = series_tensor[t : t + seq_len]  # (seq_len, N)
        y = series_tensor[t + seq_len : t + seq_len + horizon]  # (horizon, N)
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)  # (B, seq_len, N)
    Y = torch.stack(ys)  # (B, horizon, N)
    return X, Y


def _to_model_shapes(X: torch.Tensor, Y: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert to shapes expected by DCRNN:
      inputs: (seq_len, B, num_nodes * input_dim) with input_dim=1
      labels: (horizon, B, num_nodes * output_dim) with output_dim=1
    """
    B, seq_len, N = X.shape
    assert N == num_nodes
    X_ = X.permute(1, 0, 2).contiguous().view(seq_len, B, num_nodes * 1)
    H = Y.shape[1]
    Y_ = Y.permute(1, 0, 2).contiguous().view(H, B, num_nodes * 1)
    return X_, Y_


def _build_adjacency_from_correlation(train_returns: np.ndarray, min_corr: float = 0.2) -> np.ndarray:
    """
    Build undirected adjacency from Pearson correlation of node returns.
    train_returns: (T, N)
    """
    corr = np.corrcoef(train_returns.T)
    np.fill_diagonal(corr, 0.0)
    A = (np.abs(corr) >= min_corr).astype(float)
    # ensure symmetry
    A = np.maximum(A, A.T)
    return A


def load_dataset(
    data_path: str,
    seq_len: int,
    horizon: int,
    batch_size: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    price_column: str = "Close",
    use_returns: bool = True,
) -> Dict:
    """
    Load combined BSE500 CSV from data_fetch.get_bse500_data, pivot to date x ticker matrix,
    compute returns if requested, scale based on train split, and create DataLoaders.

    Returns dict with keys: train_loader, val_loader, test_loader, scaler, adjacency
    """
    df = pd.read_csv(data_path, parse_dates=["Date"])  # expects columns: Date, Ticker, price_column
    pivot = df.pivot_table(index="Date", columns="Ticker", values=price_column).sort_index()
    # forward/backward fill occasional missing values
    pivot = pivot.ffill().bfill()
    if use_returns:
        pivot = pivot.pct_change().dropna()

    values = pivot.values.astype(np.float32)  # (T, N)
    num_timesteps, num_nodes = values.shape
    # splits on time
    n_test = int(num_timesteps * test_ratio)
    n_val = int(num_timesteps * val_ratio)
    n_train = num_timesteps - n_val - n_test
    if n_train <= (seq_len + horizon):
        raise ValueError("Not enough data for the specified seq_len and horizon.")

    train_vals = values[:n_train]
    val_vals = values[: n_train + n_val]

    # scaler based on train
    scaler = StandardScaler(mean=float(train_vals.mean()), std=float(train_vals.std() + 1e-8))

    def build_loader(section_vals: np.ndarray) -> SimpleIterator:
        tensor = torch.from_numpy(section_vals)  # (T, N)
        X_win, Y_win = _create_windows(tensor, seq_len, horizon)  # (B, seq_len, N), (B, H, N)
        # scale both X and Y
        X_scaled = scaler.transform(X_win)
        Y_scaled = scaler.transform(Y_win)
        # convert to model input shapes
        X_model, Y_model = _to_model_shapes(X_scaled, Y_scaled, num_nodes)
        dataset = TensorDataset(X_model, Y_model)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return SimpleIterator(dl)

    train_loader = build_loader(train_vals)
    val_loader = build_loader(val_vals)
    test_loader = build_loader(values)

    # adjacency from train returns
    adjacency = _build_adjacency_from_correlation(train_vals)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "adjacency": adjacency,
        "num_nodes": num_nodes,
        "tickers": list(pivot.columns),
    }


