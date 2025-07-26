
import yfinance as yf
import pandas as pd

def fetch_market_data(ticker, benchmark, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    benchmark_data = yf.download(benchmark, start=start_date, end=end_date)

    data['Return'] = data['Close'].pct_change()
    benchmark_data['Benchmark_Return'] = benchmark_data['Close'].pct_change()
    data = data.join(benchmark_data[['Benchmark_Return']], how='left')
    data['Relative_Return'] = data['Return'] - data['Benchmark_Return']

    return data

