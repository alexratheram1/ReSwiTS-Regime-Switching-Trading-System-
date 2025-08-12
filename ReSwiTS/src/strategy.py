import pandas as pd
import numpy as np

def trend_playbook(df: pd.DataFrame) -> pd.Series:
    fast = df["close"].rolling(20).mean()
    slow = df["close"].rolling(50).mean()
    # Force Series math, then flatten to 1D
    signal = ((fast > slow).astype(int) - (fast < slow).astype(int)).values.ravel()
    return pd.Series(signal, index=df.index).shift(1).fillna(0.0).astype(float)

def mean_revert_playbook(df: pd.DataFrame) -> pd.Series:
    ma = df["close"].rolling(20).mean()
    z = (df["close"] - ma) / df["close"].rolling(20).std()
    signal = (-z).clip(-1, 1).values.ravel()
    return pd.Series(signal, index=df.index).shift(1).fillna(0.0).astype(float)

def regime_router(df: pd.DataFrame, regime: pd.Series) -> pd.Series:
    regime = regime.reindex(df.index).fillna("chop")
    trend_sig = trend_playbook(df)
    mr_sig = mean_revert_playbook(df)

    pos = pd.Series(0.0, index=df.index, dtype=float)
    pos.loc[regime == "trend"] = trend_sig.loc[regime == "trend"].values
    pos.loc[regime != "trend"] = mr_sig.loc[regime != "trend"].values
    return pos
