import plotly.graph_objects as go

def price_with_regimes(df, regime):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="black")))

    colors = {"trend": "rgba(33,150,243,0.15)",
              "chop": "rgba(158,158,158,0.15)",
              "risk_off": "rgba(244,67,54,0.15)"}

    current_regime = regime.iloc[0]
    start_date = df.index[0]

    for date, reg in zip(df.index[1:], regime.iloc[1:]):
        if reg != current_regime:
            fig.add_vrect(x0=start_date, x1=date, fillcolor=colors.get(current_regime, "rgba(0,0,0,0)"), line_width=0)
            current_regime = reg
            start_date = date

    # Close the last segment
    fig.add_vrect(x0=start_date, x1=df.index[-1], fillcolor=colors.get(current_regime, "rgba(0,0,0,0)"), line_width=0)

    fig.update_layout(title="Price with Regimes", xaxis_title="Date", yaxis_title="Price")
    return fig