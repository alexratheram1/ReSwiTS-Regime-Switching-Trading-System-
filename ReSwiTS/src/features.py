import numpy as np
import pandas as pd

def _safe_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # Manual ATR (no ta dependency), robust to NaNs/short histories
    close_prev = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - close_prev).abs(),
        (df["low"]  - close_prev).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Guard: if not enough rows, fail early with a clear message
    if out.shape[0] < 60:
        raise ValueError(f"Not enough data ({out.shape[0]} rows). Try a longer period or use interval='1d' first.")

    out["ret"] = out["close"].pct_change()
    out["log_ret"] = np.log(out["close"]).diff()
    out["atr"] = _safe_atr(out, window=14)
    out["rv"] = out["log_ret"].rolling(20).std() * np.sqrt(252)  # realized vol proxy

    # Entropy proxy on sign of returns
    signs = np.sign(out["ret"]).replace({-1: 0})
    def shannon_entropy(x):
        counts = np.bincount(x.astype(int), minlength=2)
        p = counts / max(1, counts.sum())
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())
    out["entropy"] = signs.rolling(50).apply(shannon_entropy, raw=False)

    out = out.dropna()
    return out

def feature_matrix(df: pd.DataFrame, cols=("ret","rv","atr","entropy")) -> np.ndarray:
    return df[list(cols)].values