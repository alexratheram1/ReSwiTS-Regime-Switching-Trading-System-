import numpy as np
import pandas as pd


def var_cvar(returns: pd.Series, alpha=0.95):
    q = returns.quantile(1-alpha)
    cvar = returns[returns <= q].mean()
    return float(-q), float(-cvar)


def attribution_by_regime(bt_df: pd.DataFrame, regime: pd.Series):
    tmp = bt_df.join(regime.rename("regime"))
    grp = tmp.groupby("regime")["net"].agg(["mean","sum","count"])
    return grp