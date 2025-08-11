import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List


def get_bse500_data(
    start: str,
    end: str,
    tickers: List[str],
    save_path: str = "bse500_data.csv",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for provided BSE500 tickers from Yahoo Finance.
    Returns a concatenated DataFrame with columns [Date, Open, High, Low, Close, Adj Close, Volume, Ticker].
    """
    all_data = []
    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if df is None or df.empty:
                continue
            df = df.dropna()
            df["Ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")
    if not all_data:
        print("No data fetched.")
        return pd.DataFrame()
    combined_df = pd.concat(all_data)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"index": "Date"}, inplace=True)
    combined_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")
    return combined_df


if __name__ == "__main__":
    # Minimal subset for quick run; replace with full BSE500 universe list
    subset = ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO"]
    start_date = "2020-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    get_bse500_data(start=start_date, end=end_date, tickers=subset, interval="1d")
