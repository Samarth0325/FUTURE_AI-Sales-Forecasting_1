
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Sales Forecast Dashboard")
st.caption("Upload historical sales and predict future trends with classic time-series models (naive, moving average, Holt-Winters).")

# ---- Sidebar: Data input ----
with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("No file uploaded. Using a small demo dataset.", icon="ℹ️")
        # Create demo data (2 years of daily sales with weekly seasonality + trend + noise)
        rng = pd.date_range("2023-01-01", periods=730, freq="D")
        base = 200 + 0.1 * np.arange(len(rng))                           # slow trend
        seasonal = 20 * np.sin(2 * np.pi * rng.dayofweek / 7)            # weekly seasonality
        noise = np.random.normal(0, 8, size=len(rng))
        demo = pd.DataFrame({"date": rng, "sales": base + seasonal + noise}).round(2)
        df_raw = demo
    else:
        df_raw = pd.read_csv(uploaded)

    st.header("2) Columns")
    # Try to guess
    guessed_date = None
    for c in df_raw.columns:
        if "date" in c.lower() or "time" in c.lower():
            guessed_date = c
            break
    date_col = st.selectbox("Date column", options=df_raw.columns.tolist(), index=(df_raw.columns.get_loc(guessed_date) if guessed_date in df_raw.columns else 0))
    target_col = st.selectbox("Target (sales) column", options=[c for c in df_raw.columns if c != date_col])

    st.header("3) Resampling & prep")
    freq = st.selectbox("Aggregate to frequency", options=["D (Daily)", "W (Weekly)", "M (Monthly)"], index=0)
    agg = st.selectbox("Aggregate method", options=["sum", "mean"], index=0)
    holdout = st.slider("Validation share (last % of data)", min_value=10, max_value=40, value=20, step=5)
    fill_missing = st.selectbox("Handle missing values", ["Forward-fill", "Zero-fill", "Interpolate"], index=0)

    st.header("4) Model & horizon")
    model_choice = st.selectbox(
        "Model",
        [
            "Naive (last value)",
            "Moving Average",
            "Simple Exponential Smoothing",
            "Holt-Winters (additive trend & seasonality)",
            "Holt-Winters (multiplicative trend & seasonality)",
        ],
        index=3
    )
    horizon = st.number_input("Forecast horizon (# periods)", min_value=1, max_value=365, value=30, step=1)
    ma_window = st.number_input("Moving Average window (if used)", min_value=2, max_value=365, value=7, step=1)
    season_length = st.number_input("Season length (periods)", min_value=1, max_value=365, value=7, step=1)

# ---- Data preparation ----
def to_freq_code(label: str) -> str:
    return {"D (Daily)": "D", "W (Weekly)": "W", "M (Monthly)": "M"}[label]

def aggregate_time_series(df: pd.DataFrame, date_col: str, target_col: str, freq_label: str, agg: str) -> pd.Series:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    # keep only needed cols
    df = df[[date_col, target_col]].dropna()
    freq = to_freq_code(freq_label)
    if agg == "sum":
        y = df.set_index(date_col)[target_col].resample(freq).sum()
    else:
        y = df.set_index(date_col)[target_col].resample(freq).mean()
    return y

def fill_series(y: pd.Series, method: str) -> pd.Series:
    if method == "Forward-fill":
        return y.ffill()
    if method == "Zero-fill":
        return y.fillna(0)
    if method == "Interpolate":
        return y.interpolate()
    return y

def split_series(y: pd.Series, holdout_pct: int):
    n = len(y)
    h = max(1, int(n * holdout_pct / 100))
    train = y.iloc[:-h] if h < n else y.iloc[:0]
    valid = y.iloc[-h:]
    return train, valid

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100

def fit_and_forecast(train: pd.Series, valid: pd.Series, model_choice: str, season_length: int, ma_window: int, horizon: int):
    # Forecast on validation range
    if model_choice == "Naive (last value)":
        last = train.iloc[-1]
        valid_pred = pd.Series([last] * len(valid), index=valid.index)
        future_index = pd.date_range(start=valid.index[-1] + (valid.index[-1] - valid.index[-2]) if len(valid) > 1 else train.index[-1] + (train.index[-1] - train.index[-2]), periods=horizon, freq=pd.infer_freq(pd.Index(train.index.append(valid.index))) or "D")
        future_pred = pd.Series([last] * horizon, index=future_index)
        return valid_pred, future_pred

    if model_choice == "Moving Average":
        rolling = train.rolling(window=ma_window, min_periods=1).mean()
        last = rolling.iloc[-1]
        valid_pred = pd.Series([last] * len(valid), index=valid.index)
        future_index = pd.date_range(start=valid.index[-1] + (valid.index[-1] - valid.index[-2]) if len(valid) > 1 else train.index[-1] + (train.index[-1] - train.index[-2]), periods=horizon, freq=pd.infer_freq(pd.Index(train.index.append(valid.index))) or "D")
        future_pred = pd.Series([last] * horizon, index=future_index)
        return valid_pred, future_pred

    if model_choice == "Simple Exponential Smoothing":
        model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        valid_pred = model.forecast(len(valid))
        future_pred = model.forecast(len(valid) + horizon).iloc[-horizon:]
        future_pred.index = pd.date_range(start=valid.index[-1] + (valid.index[-1] - valid.index[-2]) if len(valid) > 1 else train.index[-1] + (train.index[-1] - train.index[-2]), periods=horizon, freq=pd.infer_freq(pd.Index(train.index.append(valid.index))) or "D")
        return valid_pred, future_pred

    # Holt-Winters
    seasonal = "add" if "additive" in model_choice else "mul"
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=season_length,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=True)
    valid_pred = model.forecast(len(valid))
    future_pred = model.forecast(len(valid) + horizon).iloc[-horizon:]
    # set continuous future index
    step = (train.index[-1] - train.index[-2]) if len(train) > 1 else pd.Timedelta(days=1)
    start_next = valid.index[-1] + (valid.index[-1] - valid.index[-2]) if len(valid) > 1 else train.index[-1] + step
    future_pred.index = pd.date_range(start=start_next, periods=horizon, freq=pd.infer_freq(pd.Index(train.index.append(valid.index))) or step)
    return valid_pred, future_pred

# Aggregate
y = aggregate_time_series(df_raw, date_col, target_col, freq, agg)
y = fill_series(y, fill_missing)

if len(y) < 10:
    st.warning("Not enough data after aggregation. Provide more rows or lower the validation share.", icon="⚠️")

train, valid = split_series(y, holdout)
valid_pred, future_pred = fit_and_forecast(train, valid, model_choice, int(season_length), int(ma_window), int(horizon))

# ---- Metrics ----
val_rmse = math.sqrt(mean_squared_error(valid, valid_pred)) if len(valid) > 0 else np.nan
val_mape = mape(valid, valid_pred) if len(valid) > 0 else np.nan

# ---- Plot ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name="Actual"))
if len(valid) > 0:
    fig.add_trace(go.Scatter(x=valid.index, y=valid_pred.values, mode="lines", name="Validation Forecast"))
if len(future_pred) > 0:
    fig.add_trace(go.Scatter(x=future_pred.index, y=future_pred.values, mode="lines", name="Future Forecast"))
fig.update_layout(
    title="Sales History & Forecast",
    xaxis_title="Date",
    yaxis_title=target_col,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Cards ----
c1, c2, c3 = st.columns(3)
c1.metric("Data points", f"{len(y)}")
c2.metric("Validation RMSE", f"{val_rmse:.2f}" if not np.isnan(val_rmse) else "—")
c3.metric("Validation MAPE (%)", f"{val_mape:.2f}" if not np.isnan(val_mape) else "—")

# ---- Table & Download ----
st.subheader("Forecast Table")
hist_df = y.reset_index()
hist_df.columns = ["date", "actual"]
val_df = pd.DataFrame({"date": valid.index, "forecast": valid_pred.values})
fut_df = pd.DataFrame({"date": future_pred.index, "forecast": future_pred.values})
out = pd.concat([hist_df, val_df], ignore_index=True, sort=False)
out = out.sort_values("date")

st.dataframe(out.tail(100), use_container_width=True)

csv = fut_df.to_csv(index=False).encode("utf-8")
st.download_button("Download future forecast (CSV)", data=csv, file_name="future_forecast.csv", mime="text/csv")

st.divider()
with st.expander("How to use"):
    st.markdown(
        """
1. Upload a CSV with at least a **date** column and a **sales** column (any name).
2. Choose aggregation frequency (daily/weekly/monthly) and how to fill gaps.
3. Pick a model:  
   - *Naive*: repeats the last observed value.  
   - *Moving Average*: smooth baseline.  
   - *Simple Exp Smoothing*: level adaptation.  
   - *Holt‑Winters*: trend + seasonality (additive or multiplicative).
4. Set **season length** (e.g., 7 for weekly seasonality on daily data; 12 for monthly data with yearly seasonality).
5. Tune validation share and check RMSE/MAPE.
6. Download the forecast CSV.
        """
    )
