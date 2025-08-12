import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def fit_hmm(X: np.ndarray, n_states: int = 3, covariance_type: str = "full", random_state: int = 42) -> GaussianHMM:
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, random_state=random_state, n_iter=200)
    model.fit(X)
    return model

def infer_states(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    return model.predict(X)

def label_states_by_stats(df: pd.DataFrame, states: np.ndarray) -> pd.Series:
    """Map raw HMM state IDs to human labels using mean return & vol.
    Robust if 'ret' missing; will compute from 'close' if needed.
    """
    tmp = df.copy()

    # Ensure ret exists
    if "ret" not in tmp.columns:
        if "close" in tmp.columns:
            tmp["ret"] = tmp["close"].pct_change()
        else:
            raise KeyError("Expected column 'ret' or 'close' to compute returns for state labelling.")

    # Align lengths
    if len(tmp) != len(states):
        # Trim to the last len(states) rows (common after dropna)
        tmp = tmp.iloc[-len(states):].copy()

    tmp["state"] = states

    # If dataset is tiny, fall back to neutral labels
    if tmp["state"].nunique() == 0 or tmp.shape[0] < 10:
        return pd.Series(["chop"] * len(df), index=df.index[-len(states):])

    # Aggregate by state
    stats = (
        tmp.groupby("state")["ret"]
           .agg(mean_ret="mean", vol="std")
           .sort_values("mean_ret")
    )

    # Build mapping: worst mean -> "risk_off", best -> "trend", middles -> "chop"
    order = list(stats.index)
    mapping = {}
    if len(order) == 1:
        mapping[order[0]] = "chop"
    elif len(order) == 2:
        mapping[order[0]] = "risk_off"
        mapping[order[1]] = "trend"
    else:
        mapping[order[0]] = "risk_off"
        mapping[order[-1]] = "trend"
        for s in order[1:-1]:
            mapping[s] = "chop"

    labeled = pd.Series(states, index=df.index[-len(states):]).map(mapping)
    return labeled