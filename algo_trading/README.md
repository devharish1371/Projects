Quick start

1) Install dependencies

    pip install -r requirements.txt

2) Fetch data (replace tickers with your BSE500 list)

    python data_fetch.py

This produces bse500_data.csv with columns Date,Ticker,Close,...

3) Train DCRNN

    python train.py --data_csv bse500_data.csv --seq_len 24 --horizon 4 --epochs 30

4) Inference

Prepare a recent CSV in the same format (Date,Ticker,Close). Then:

    python trading_strategy.py --checkpoint logs/checkpoints/model_epoch_XX.pt --recent_csv recent.csv --out_csv predictions.csv

5) Backtest (single ticker demo)

    python backtesting.py

Notes
- Long-only; thresholding on predicted next-period returns can be integrated with execution layer.
- The adjacency is built from train correlations; you can replace with a sector/industry graph.

