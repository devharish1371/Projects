import requests
from yahoo_fin import news
from datetime import timedelta, datetime
import pandas as pd
from config import FINNHUB_API_KEY

def fetch_yahoo_news(ticker):
    all_news = news.get_yf_rss(ticker)
    news_df = pd.DataFrame(all_news)
    news_df['date'] = news_df['published_parsed'].apply(lambda x: datetime(*x[:6]))
    news_df['date_only'] = news_df['date'].dt.date
    return news_df

def get_historical_news(ticker, date, window_days=1):
    start = (date - timedelta(days=window_days)).strftime('%Y-%m-%d')
    end = (date + timedelta(days=window_days)).strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={end}&token={FINNHUB_API_KEY}"
    r = requests.get(url)
    articles = r.json()
    if isinstance(articles, list):
        return '; '.join([a['headline'] for a in articles[:5]])
    return ''

def enrich_with_news(events, ticker):
    events['News_Headlines'] = events['Date'].apply(lambda d: get_historical_news(ticker, d))
    return events[events['News_Headlines'].notna() & (events['News_Headlines'].str.strip() != '')].copy()

