import numpy as np
import pandas as pd


def backtest(df: pd.DataFrame, position: pd.Series, fee_bps=1.0, slippage_bps=2.0, rf=0.0):
    # Align
    df = df.copy()
    position = position.reindex(df.index).fillna(0)

    # Returns + costs
    gross = position.shift(1) * df["ret"]
    turnover = position.diff().abs().fillna(0)
    cost = (fee_bps + slippage_bps) / 1e4 * turnover
    net = gross - cost

    equity = (1 + net).cumprod()
    dd = equity / equity.cummax() - 1

    stats = {
        "CAGR": (equity.iloc[-1]) ** (252/len(equity)) - 1 if len(equity)>0 else np.nan,
        "Sharpe": np.sqrt(252) * (net.mean() - rf/252) / (net.std() + 1e-12),
        "Sortino": np.sqrt(252) * (net.mean() - rf/252) / (net[net<0].std() + 1e-12),
        "MaxDD": dd.min(),
        "HitRate": (net>0).mean(),
    }
    out = pd.DataFrame({
        "gross": gross,
        "net": net,
        "equity": equity,
        "drawdown": dd,
        "position": position,
    })
    return out, stats