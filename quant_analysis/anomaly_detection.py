import numpy as np
import pandas as pd

def detect_anomalies(data, z_threshold, vol_window, vol_multiplier, ticker):
    mean_ret = data['Relative_Return'].mean()
    std_ret = data['Relative_Return'].std()
    data['Z_score'] = (data['Relative_Return'] - mean_ret) / std_ret
    data['Z_Event'] = np.where(abs(data['Z_score']) > z_threshold,
                               np.where(data['Z_score'] > 0, 'Positive Outlier', 'Negative Outlier'),
                               None)
    data['Rolling_STD'] = data['Relative_Return'].rolling(window=vol_window).std()
    data['Vol_Spike'] = np.where(abs(data['Relative_Return']) > vol_multiplier * data['Rolling_STD'],
                                  'Volatility Spike', None)

    events = data[(data['Z_Event'].notna()) | (data['Vol_Spike'].notna())].copy()
    events['Event_Type'] = events[['Z_Event', 'Vol_Spike']].fillna('').agg(', '.join, axis=1).str.strip(', ')
    events['Ticker'] = ticker
    return events.reset_index()[['Date', 'Ticker', 'Return', 'Z_score', 'Rolling_STD', 'Event_Type']]

