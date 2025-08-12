import streamlit as st
import pandas as pd
import yaml
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.data import load_ohlcv
from src.features import add_features, feature_matrix
from src.regimes import fit_hmm, infer_states, label_states_by_stats
from src.strategy import regime_router
from src.backtest import backtest
from src.risk import var_cvar, attribution_by_regime

st.set_page_config(page_title="Regime Switcher", layout="wide")

st.title("Regime-Switching Trading System (MVP)")

with open("config.yaml","r") as f:
    cfg = yaml.safe_load(f)

ticker = st.sidebar.selectbox("Ticker", cfg["data"]["tickers"])
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
states = st.sidebar.slider("HMM states", 2, 5, cfg["hmm"]["n_states"])
fee = st.sidebar.number_input("Fee (bps)", 0.0, 50.0, cfg["backtest"]["fee_bps"])
slip = st.sidebar.number_input("Slippage (bps)", 0.0, 100.0, cfg["backtest"]["slippage_bps"])

# Load + features
raw = load_ohlcv(ticker, cfg["data"]["lookback_years"], interval)
feat = add_features(raw)
X = feature_matrix(feat)

# HMM
model = fit_hmm(X, n_states=states, covariance_type=cfg["hmm"]["covariance_type"], random_state=cfg["hmm"]["random_state"])
state_ids = infer_states(model, X)
regime = label_states_by_stats(feat, state_ids)

# Strategy & backtest
position = regime_router(feat, regime)
bt, stats = backtest(feat, position, fee_bps=fee, slippage_bps=slip)

# Layout
import plotly.graph_objects as go
from src.plots import price_with_regimes  # we'll update this too

# --- Price chart with regimes ---
st.subheader("Price & Regimes")
price_fig = price_with_regimes(feat, regime)
st.plotly_chart(price_fig, use_container_width=True)

# --- Equity + drawdown chart ---
eq_fig = go.Figure()
eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name="Equity", line=dict(color="green")))
eq_fig.add_trace(go.Scatter(x=bt.index, y=bt["drawdown"], name="Drawdown", line=dict(color="red"), yaxis="y2"))

# Second y-axis for drawdown
eq_fig.update_layout(
    yaxis=dict(title="Equity"),
    yaxis2=dict(title="Drawdown", overlaying="y", side="right"),
    legend=dict(orientation="h", y=-0.2)
)
st.subheader("Equity & Drawdown")
st.plotly_chart(eq_fig, use_container_width=True)

# --- Stats table ---
stats_fmt = {k: f"{v:.2%}" if "Rate" not in k and abs(v) < 1 else f"{v:.2f}" for k,v in stats.items()}
VaR, CVaR = var_cvar(bt["net"])
stats_fmt["VaR(95%)"] = f"{VaR:.2%}"
stats_fmt["CVaR(95%)"] = f"{CVaR:.2%}"
st.subheader("Performance Metrics")
st.table(pd.DataFrame(stats_fmt.items(), columns=["Metric", "Value"]))

# --- Attribution table ---
attrib = attribution_by_regime(bt, regime)
attrib["mean"] = attrib["mean"].apply(lambda x: f"{x:.4f}")
attrib["sum"] = attrib["sum"].apply(lambda x: f"{x:.2f}")
st.subheader("PnL Attribution by Regime")
st.dataframe(attrib)