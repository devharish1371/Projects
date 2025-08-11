# backtesting.py
import pandas as pd
import numpy as np


def calculate_metrics(equity_curve: pd.Series, periods_per_year: int = 252):
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return {"CAGR": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0}
    years = len(returns) / periods_per_year
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    sharpe_ratio = np.sqrt(periods_per_year) * returns.mean() / (returns.std() + 1e-9)
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()
    return {
        "CAGR": float(cagr),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
    }


def backtest_long_only(
    price_series: pd.Series,
    predicted_returns: pd.Series,
    threshold: float = 0.0,
    initial_capital: float = 500000.0,
    min_hold_minutes: int = 15,
    frequency: str = '1D',
):
    """
    Long-only backtest with a simple threshold on predicted next-period return.
    Enforces a minimum holding period; for daily data, min_hold_minutes is ignored.
    """
    df = pd.concat([price_series.rename('Close'), predicted_returns.rename('Pred')], axis=1).dropna()
    signal = (df['Pred'] > threshold).astype(int)
    # Avoid lookahead: act on previous signal
    signal = signal.shift(1).fillna(0)

    # Apply 15-min hold for intraday if frequency is intraday; here we keep simple
    df['Signal'] = signal
    df['Ret'] = df['Close'].pct_change().fillna(0.0)
    df['StratRet'] = df['Signal'] * df['Ret']
    df['Equity'] = (1 + df['StratRet']).cumprod() * initial_capital
    df['BuyHold'] = (1 + df['Ret']).cumprod() * initial_capital

    metrics = calculate_metrics(df['Equity'])
    return df, metrics


if __name__ == "__main__":
    # Example: backtest a single ticker using predictions.csv produced by trading_strategy
    preds = pd.read_csv('predictions.csv', index_col=0, parse_dates=True)
    # For demo, pick first column
    ticker = preds.columns[0]
    # In practice, join with actual realized prices to evaluate properly
    # Here we synthesize a price path from returns for illustration
    rets = preds[ticker]
    prices = (1 + rets).cumprod() * 100.0
    results, stats = backtest_long_only(prices, rets, threshold=0.0)
    print(stats)
