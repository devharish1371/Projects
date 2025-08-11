import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from data_fetch import get_bse500_data
from dcrnn_supervisor import DCRNNSupervisor
from trading_strategy import infer_next_returns


def parse_ticker_list(tickers_arg: str) -> List[str]:
    return [t.strip() for t in tickers_arg.split(',') if t.strip()]


def latest_checkpoint(model_dir: str) -> str:
    files = [f for f in os.listdir(model_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    files.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
    return os.path.join(model_dir, files[-1])


def recommend_long_portfolio(pred_df: pd.DataFrame, capital: float, threshold: float, top_n: int) -> pd.DataFrame:
    """
    Use first horizon row predictions to select long positions.
    Allocate equal capital across top_n tickers with prediction > threshold.
    """
    next_row = pred_df.iloc[0]
    ranked = next_row.sort_values(ascending=False)
    selected = ranked[ranked > threshold]
    if top_n > 0:
        selected = selected.head(top_n)
    num = max(len(selected), 1)
    alloc_per = capital / num
    rec = pd.DataFrame({
        'Ticker': selected.index,
        'Predicted_Return': selected.values,
        'Capital_Allocated': alloc_per,
        'Weight': 1.0 / num
    })
    rec = rec.reset_index(drop=True)
    return rec


def main():
    parser = argparse.ArgumentParser(description='DCRNN demo: fetch, train, predict, and generate a long-only strategy.')
    parser.add_argument('--tickers', type=str, required=True, help='Comma-separated tickers, e.g., RELIANCE.BO,TCS.BO,INFY.BO')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (e.g., 1d, 1h, 15m)')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--capital', type=float, default=500000.0)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()

    tickers = parse_ticker_list(args.tickers)
    os.makedirs(args.out_dir, exist_ok=True)
    data_csv = os.path.join(args.out_dir, 'data.csv')
    logs_dir = os.path.join(args.out_dir, 'logs')
    ckpt_dir = os.path.join(logs_dir, 'checkpoints')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) Fetch data
    print('Fetching data...')
    df = get_bse500_data(start=args.start, end=args.end, tickers=tickers, save_path=data_csv, interval=args.interval)
    if df.empty:
        raise RuntimeError('No data fetched. Aborting.')

    # 2) Train model (quick demo)
    print('Training model...')
    sup = DCRNNSupervisor(
        data=dict(
            data_path=data_csv,
            seq_len=args.seq_len,
            horizon=args.horizon,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            price_column='Close',
            use_returns=True,
        ),
        model=dict(
            rnn_units=64,
            num_rnn_layers=2,
            filter_type='random_walk',
            seq_len=args.seq_len,
            horizon=args.horizon,
        ),
        train=dict(
            epochs=args.epochs,
            base_lr=1e-3,
            log_dir=logs_dir,
            model_dir=ckpt_dir,
            lr_steps=[int(max(args.epochs * 0.6, 1)), int(max(args.epochs * 0.8, 2))],
            lr_decay_ratio=0.5,
            patience=max(int(args.epochs * 0.2), 3),
        ),
        log_level='INFO',
    )
    sup.train()

    # 3) Load latest checkpoint and run inference
    ckpt_path = latest_checkpoint(ckpt_dir)
    print(f'Using checkpoint: {ckpt_path}')
    preds = infer_next_returns(ckpt_path, data_csv)
    preds_csv = os.path.join(args.out_dir, 'predictions.csv')
    preds.to_csv(preds_csv)
    print(f'Saved predictions to {preds_csv}')

    # 4) Simple long-only strategy for next period
    rec = recommend_long_portfolio(preds, capital=args.capital, threshold=args.threshold, top_n=args.top_n)
    rec_csv = os.path.join(args.out_dir, 'strategy.csv')
    rec.to_csv(rec_csv, index=False)
    print('Suggested long-only allocation for next period:')
    print(rec)
    print(f'Saved strategy to {rec_csv}')


if __name__ == '__main__':
    main()


