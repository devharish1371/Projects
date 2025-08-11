import pandas as pd
import numpy as np


def compute_features_per_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Given a single ticker's OHLCV DataFrame (with 'Date' index),
    compute simple technical features. Not strictly needed by DCRNN pipeline, but useful for EDA.
    """
    df = df_ticker.copy().sort_index()
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df['Return_15'] = df['Close'].pct_change(15)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_15'] = df['Close'].rolling(window=15).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Volatility_15'] = df['Return_1'].rolling(window=15).std()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(window=14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df


def build_feature_df(all_data_csv: str, feature_save_path: str = "features_bse500.parquet") -> pd.DataFrame:
    df_all = pd.read_csv(all_data_csv, parse_dates=['Date'])
    feature_frames = []
    for ticker, df_t in df_all.groupby('Ticker'):
        df_t = df_t.sort_values('Date').set_index('Date')
        df_feat = compute_features_per_ticker(df_t)
        df_feat['Ticker'] = ticker
        feature_frames.append(df_feat.reset_index())
    if not feature_frames:
        raise RuntimeError("No ticker data processed for feature building.")
    df_features = pd.concat(feature_frames, ignore_index=True)
    df_features.to_parquet(feature_save_path, index=False)
    print(f"[feature_engineering] Features saved to {feature_save_path}")
    return df_features


if __name__ == "__main__":
    features = build_feature_df("bse500_data.csv")
    print(features.head())
    print(features.shape)
