## ANN Time-Series Price Forecasting (MLPRegressor)
_Configurable time-series forecasting with Artificial Neural Networks (ANN) + Direct Forecast & Holdout Backtest_

This module implements a neural-network–based time series forecaster on financial price data (e.g. gold futures `GC=F`) using an **MLP (Multi-Layer Perceptron)** via `sklearn.neural_network.MLPRegressor`.

It supports **two workflows**:

1. **Direct Forecast (Future)**  
   Train on recent history and forecast the next `horizon` points (future-only).

2. **Holdout Backtest (Offset Split)**  
   Pick a split point in the past (`offset_back`), train on the window before it (`back_points`), then forecast the next `horizon` points and compute **out-of-sample RMSE**.

---

### What the pipeline does (end-to-end)

- Fetches **daily closing prices** from Yahoo Finance via `yfinance`.
- Builds a **lag-based supervised dataset** using the last `N` data points.
- Trains an `MLPRegressor` (ANN) with configurable:
  - `window` size (number of lags)
  - number of hidden layers
  - neurons per hidden layer
- Splits the data into **Train / Validation / Test** sets in **time order** (no shuffle) using fixed ratios (default `0.70 / 0.15 / 0.15`).
- Computes **R² / RMSE** on each split (and **holdout RMSE** in backtest mode).
- Forecasts iteratively (**recursive forecasting**) and plots:
  - historical prices
  - predicted prices
  - split point between history and forecast
  - error segments (actual vs predicted) in predicted zone

---

## Installation

Create venv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# .\venv\Scripts\activate     # Windows

pip install numpy pandas matplotlib scikit-learn yfinance
```

---

## How to Run

Save this module as `ann.py` (or similar) and run:

```bash
python ann.py
```

You will see:

```text
welcome to the MENU ====
1) Direct forecast (future)
2) Holdout backtest (offset_back + back_points + horizon)
```

Then answer the prompts:
- `Symbol` (press Enter for `GC=F`)
- `window`, `hidden_layers`, `neurons`
- scaling `y/n`
- plus mode-specific inputs (below)

> If you want to integrate into a bigger application, **do not auto-run on import**.  
> Remove the last line `ann()` and call `ann()` explicitly from your main entrypoint (e.g. `app.py`).

---

## Mode 1 — Direct Forecast (Future)

**Goal:** Train on the latest `n_points` and forecast the next `horizon` points into the future.

Prompts:
- `n_points` (default `3000`)
- `horizon` (default `30`)
- train/val/test ratios (default `0.7 / 0.15 / 0.15`)

Steps:
1. Select last `n_points` from the close series.
2. Build supervised dataset (sliding window).
3. Fit on train, report R²/RMSE for train/val/test.
4. Refit on the full selected history.
5. Forecast `horizon` points recursively.
6. Plot:
   - history + future forecast
   - a future-only close-up plot

---

## Mode 2 — Holdout Backtest (Offset Split)

**Goal:** Simulate “real” out-of-sample forecasting on a **past** holdout window.

Prompts (your “3 knobs”):
- `offset_back`: go back from the last point by how many points (default `500`)
- `back_points`: how many points before split to train on (default `3000`)
- `horizon`: how many points after split to forecast (default `90`)
- train/val/test ratios (default `0.7 / 0.15 / 0.15`)

Slices:
- Train slice: `[split - back_points : split)`
- Holdout slice: `[split : split + horizon)`

Metrics:
- Train/Val/Test metrics computed **only** inside the train slice (no leakage)
- Holdout (out-of-sample) RMSE computed on the holdout slice

Plots:
- Full view: train + holdout actual + holdout forecast (with error segments)
- Zoom view: predicted zone only (with error segments)
- Extra close-up predicted-only chart

---

## 1. Data Source & Preprocessing

Data is loaded from Yahoo Finance:

```python
import yfinance as yf

t = yf.Ticker(symbol)
hist = t.history(period="max", interval="1d", auto_adjust=True)

close = hist["Close"].dropna()
```

Notes:
- `auto_adjust=True` uses adjusted prices.
- Timezone is stripped if present to keep plotting/index handling stable.

---

## 2. Lag-Based Supervised Dataset (Sliding Window)

For a given `window` (lag length), the model uses the last `window` prices to predict the next one.

Input vector at time \(t\):

$$
X_t = [p_{t-w}, \dots, p_{t-1}]
$$

Target:

$$
y_t = p_t
$$

Implementation:

```python
X, y = [], []
for i in range(window, len(series_1d)):
    X.append(series_1d[i - window:i])
    y.append(series_1d[i])
```

This is standard **sliding window supervised learning** on a univariate time series.

---

## 3. Model — MLPRegressor (ANN) + Optional Scaling

The model is a feed-forward neural network:

- Implementation: `sklearn.neural_network.MLPRegressor`
- Activation: `relu`
- Optimizer: `adam`
- Early stopping: enabled
- Max iterations: `max_iter=1200` (in your current code)
- Hidden layers: configurable (`hidden_layers`, `neurons`)

Architecture creation:

```python
from sklearn.neural_network import MLPRegressor

hl_sizes = tuple([neurons] * hidden_layers)

base = MLPRegressor(
    hidden_layer_sizes=hl_sizes,
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=1200,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
)
```

### Scaling option (recommended)
If `use_scaling = y`, the pipeline:
- scales inputs \(X\) via `StandardScaler`
- scales targets \(y\) via `TransformedTargetRegressor(..., transformer=StandardScaler())`

Why this matters:
- raw price levels can be large and unstable
- scaling usually improves training stability and convergence

---

## 4. Train / Validation / Test Split (No Leakage)

The dataset is split **chronologically** (no shuffling):

```python
train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15

train_end = int(n_samples * train_ratio)
val_end   = train_end + int(n_samples * val_ratio)

X_train, y_train = X[:train_end],        y[:train_end]         # oldest 70%
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]  # next 15%
X_test,  y_test  = X[val_end:],          y[val_end:]           # most recent 15%
```

This preserves temporal ordering (future never leaks into past), which is critical for time-series modeling.

---

## 5. Evaluation Metrics (What they mean)

After training, the model evaluates performance on each split using:

### • MSE (Mean Squared Error)
Average squared difference between actual and predicted values.

Plain form:

```text
MSE = (1/n) * Σ (y_i - ŷ_i)²
```

LaTeX:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

### • RMSE (Root Mean Squared Error)
Square root of MSE — same unit as the target variable.

Plain form:

```text
RMSE = sqrt(MSE)
```

LaTeX:

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

### • R² (Coefficient of Determination)
Indicates how much variance in the target variable is explained by the model (1.0 = perfect).

Plain form:

```text
R² = 1 - [ Σ (y_i - ŷ_i)² ] / [ Σ (y_i - ȳ)² ]
```

LaTeX:

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Higher R² means the model fits historical data better (more variance explained).

Example console output:

```text
==== TIME SERIES NN PERFORMANCE ====
[TRAIN] R²: 0.9821 | RMSE: 30.825123
[VAL]   R²: 0.7420 | RMSE: 47.963325
[TEST]  R²: 0.7013 | RMSE: 50.993694
====================================
```

Interpretation example:
- If gold is around 4,000 USD:
  - `RMSE ≈ 30` → typical prediction error is about 30 USD

Overfitting rule:
- If \(R^2_{\text{train}}\) is very high (~0.99) but \(R^2_{\text{test}}\) is much lower/unstable → you’re overfitting.

---

## 6. Forecasting the Future (Recursive / Iterative Forecast)

You choose:
- `horizon`: how many future points to forecast

Forecasting is **recursive**:
1. take the last `window` values
2. predict next
3. append prediction into window
4. repeat

Pseudocode:

```python
import numpy as np

last_window = series[-window:].copy()
future_preds = []

for _ in range(horizon):
    next_val = model.predict(last_window.reshape(1, -1))[0]
    future_preds.append(next_val)

    last_window = np.roll(last_window, -1)
    last_window[-1] = next_val
```

Note: recursive forecasts compound error as horizon increases.

---

## 7. Plotting (Full + Zoom + Error Segments)

The plotting layer produces:
- **Full view**: train + predicted zone (actual) + forecast
- **Zoom view**: predicted zone only
- **Error segments**: vertical red dashed lines showing actual vs predicted error at each date

Minimal example of plotting structure:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 6))
plt.plot(full_dates, full_actual, label="Actual", linewidth=2)
plt.plot(pred_dates, pred_forecast, label="Forecast", linewidth=2)

plt.axvline(split_date, linestyle=":", label="Forecast start")

for d, a, p in zip(pred_dates, pred_actual, pred_forecast):
    plt.plot([d, d], [a, p], "r--", alpha=0.35, linewidth=1)

plt.title(f"{symbol} - Full View | HOLDOUT RMSE={rmse:.3f}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 8. Window Size & Overfitting Warning (Don’t ignore this)

Oversized window sizes can easily cause overfitting.

Example:

```text
window        = 1000
hidden_layers = 2
neurons       = 64
```
(BTW I offer 3 layers and 32 neurons. It works way better for 14days forcasts.)

Approximate parameter count:

- Layer 1: \(1000 \times 64 \approx 64{,}000\) weights
- Layer 2: \(64 \times 64 \approx 4{,}096\) weights
- Output: \(64 \times 1 \approx 64\) weights

Total:

$$
\text{params} \approx 68{,}000 \ (\text{+ biases}) \Rightarrow 70{,}000+ \text{ parameters}
$$

If you have ~1,400 training samples and ~70k parameters, you’re not “modeling” — you’re begging the network to memorize.

Practical guidance:
- Start modest:
  - `window = 20` or `40`
  - `hidden_layers = 1–2`
  - `neurons = 32` or `64`
  - scaling = `y`
- Watch the generalization gap:
  - small gap → OK
  - huge gap → overfitting

---

## 9. Possible Extensions (Next steps)

If you want to push this beyond baseline:

- log-returns instead of price levels
- feature engineering (MA/RSI/VIX/etc.)
- rolling-window backtests (walk-forward validation)
- save/export model + metrics to disk
- compare against naive baselines (last value, moving average, ARIMA)

---

## Disclaimer

Educational/research use only. Not financial advice.
