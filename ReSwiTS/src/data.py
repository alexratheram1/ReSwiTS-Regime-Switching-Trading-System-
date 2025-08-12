import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def load_ohlcv(ticker: str, period_years: int = 8, interval: str = "id") -> pd.DataFrame:
    period = f"{period_years}y"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    df = df.dropna().rename(columns={
        "Open":"open","High": "high","Low":"low","Close":"close","Volume":"volume"
    })
    return df

def load_universe(tickers, years=8, interval="id"):
    return{t: load_ohlcv(t, years, interval) for t in tickers}