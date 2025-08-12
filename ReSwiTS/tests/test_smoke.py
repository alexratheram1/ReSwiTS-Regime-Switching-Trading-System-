from src.data import load_ohlcv

def test_load():
    df = load_ohlcv("SPY", period_years=1, interval="1d")
    assert len(df) > 50
    assert {"open","high","low","close","volume"}.issubset(df.columns)