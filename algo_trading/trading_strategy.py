import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from utils import StandardScaler
from dcrnn_model import DCRNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    state = torch.load(path, map_location=device)
    scaler = StandardScaler(mean=state['scaler_mean'], std=state['scaler_std'])
    model_kwargs = state['model_kwargs']
    adj = state['adjacency']
    model = DCRNNModel(adj, logger=None, **model_kwargs)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()
    return model, scaler, adj, model_kwargs


def prepare_sequence(prices_df: pd.DataFrame, seq_len: int, horizon: int, scaler: StandardScaler):
    prices_df = prices_df.sort_index().ffill().bfill()
    returns = prices_df.pct_change().dropna()
    values = returns.values.astype(np.float32)  # (T, N)
    if values.shape[0] < seq_len:
        raise ValueError("Not enough data for inference window.")
    x = torch.from_numpy(values[-seq_len:])  # (seq_len, N)
    x = scaler.transform(x)
    x = x.unsqueeze(1)  # (seq_len, 1, N)
    x = x.view(seq_len, 1, -1)  # (seq_len, 1, num_nodes*1)
    return x


def infer_next_returns(checkpoint_path: str, recent_data_csv: str) -> pd.DataFrame:
    model, scaler, adj, model_kwargs = load_checkpoint(checkpoint_path)
    seq_len = int(model_kwargs['seq_len'])
    horizon = int(model_kwargs['horizon'])

    df = pd.read_csv(recent_data_csv, parse_dates=['Date'])  # columns: Date, Ticker, Close
    pivot = df.pivot_table(index='Date', columns='Ticker', values='Close').sort_index()
    x = prepare_sequence(pivot, seq_len, horizon, scaler)

    with torch.no_grad():
        out = model(x)  # (horizon, 1, num_nodes)
        out_inv = scaler.inverse_transform(out)
    preds = out_inv.squeeze(1).cpu().numpy()  # (horizon, num_nodes)

    tickers = list(pivot.columns)
    dates = [pivot.index[-1] + timedelta(days=i+1) for i in range(horizon)]
    out_df = pd.DataFrame(preds, columns=tickers, index=dates)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--recent_csv', required=True)
    parser.add_argument('--out_csv', default='predictions.csv')
    args = parser.parse_args()

    pred_df = infer_next_returns(args.checkpoint, args.recent_csv)
    pred_df.to_csv(args.out_csv)
    print(f"Saved predictions to {args.out_csv}")
