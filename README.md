## Price Forecasting using ANN  
_Configurable Time-Series Forecasting with Artificial Neural Networks_

This module implements a neural-network–based time series forecaster on financial price data (e.g. gold futures `GC=F`) using an **MLP (Multi-Layer Perceptron)**.

The core function **`do_ts_forecast()`** performs the entire pipeline end-to-end:

- Fetches **daily closing prices** from Yahoo Finance via `yfinance`.
- Builds a **lag-based supervised dataset** using the last `N` data points.
- Trains an `MLPRegressor` (ANN) with configurable:
  - `window` size (number of lags),
  - number of hidden layers,
  - neurons per hidden layer.
- Splits the data into **Train / Validation / Test** sets in **time order** with fixed ratios `0.7 / 0.15 / 0.15`.
- Computes **R² / MSE / RMSE** for each split.
- Forecasts the next `horizon` days iteratively and plots:
  - historical prices,
  - future predictions,
  - the split point between history and forecast.

---

### 1. Data Source & Preprocessing

Data is loaded from Yahoo Finance:

```python
import yfinance as yf

t = yf.Ticker(symbol)
hist = t.history(period="max", interval="1d")
close = hist["Close"].dropna()
```

You choose:

- `symbol` (default: `"GC=F"` for gold futures),
- `n_points`: how many of the most recent daily closes to use (default: `3000`).

Only the last `n_points` observations are used:

```python
series = close.values[-n_points:]
```

#### Lag-based supervised dataset

For a given `window` (lag length), the model uses the last `window` prices to predict the next one.

Input vector at time \(t\):

$$
X_t = [p_{t-w}, \dots, p_{t-1}]
$$

Target:

$$
y_t = p_t
$$

In code:

```python
X, y = [], []
for i in range(window, len(series)):
    X.append(series[i - window:i])
    y.append(series[i])
```

This is standard **“sliding window” supervised learning** on a univariate time series.

---

### 2. Model: `MLPRegressor` (Artificial Neural Network)

The model is a feed-forward neural network:

- Implementation: `sklearn.neural_network.MLPRegressor`
- Activation: `relu`
- Optimizer: `adam`
- Max iterations: `max_iter=500`
- Hidden layers: configurable

You configure:

- `hidden_layers` → number of hidden layers (e.g. `1`, `2`, `3`, …),
- `neurons` → neurons per hidden layer (e.g. `32`, `64`, `128`, …).

Architecture construction:

```python
from sklearn.neural_network import MLPRegressor

hl_sizes = tuple([neurons] * hidden_layers)

model = MLPRegressor(
    hidden_layer_sizes=hl_sizes,
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=500,
)
```

Example – for `hidden_layers = 2` and `neurons = 64`:

- **Input**: `window` features  
- **Hidden Layer 1**: 64 neurons (ReLU)  
- **Hidden Layer 2**: 64 neurons (ReLU)  
- **Output**: 1 neuron (next price)

---

### 3. Train / Validation / Test Split

The dataset is split **chronologically** (no shuffling):

```python
train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15
```

For `n_samples` examples:

```python
train_end = int(n_samples * train_ratio)
val_end   = train_end + int(n_samples * val_ratio)

X_train, y_train = X[:train_end],        y[:train_end]         # oldest 70%
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]  # next 15%
X_test,  y_test  = X[val_end:],          y[val_end:]           # most recent 15%
```

This preserves **temporal ordering** (future never leaks into past), which is critical for time-series modeling.

---

## 4. Evaluation Metrics

After training, the model evaluates performance on each split using:

### **• MSE (Mean Squared Error)**
Measures average squared difference between actual and predicted values.
```
MSE = (1/n) * Σ (y_i - ŷ_i)²
```

### **• RMSE (Root Mean Squared Error)**
Square root of MSE — same unit as the target variable.
```
RMSE = sqrt(MSE)
```

### **• R² (Coefficient of Determination)**
Indicates how much variance in the target variable is explained by the model (1.0 = perfect).
```
R² = 1 - [ Σ (y_i - ŷ_i)² ] / [ Σ (y_i - ȳ)² ]
```

Higher R² means the model fits the historical data better.
Which means more variance in the target variable is explained by the model (1.0 = perfect fit).

Example console output:

```text
==== TIME SERIES NN PERFORMANCE ====
[TRAIN] R²: 0.9821 | MSE:  950.123456 | RMSE: 30.825123
[VAL]   R²: 0.7420 | MSE: 2300.789012 | RMSE: 47.963325
[TEST]  R²: 0.7013 | MSE: 2600.456789 | RMSE: 50.993694
====================================
```

If you are forecasting gold futures and the price is around **4,000 USD**, then:

- `RMSE ≈ 30` → typical prediction error is about **30 USD**.
- `MSE ≈ 900` → square of that; units are **USD²**.

Always compare **train vs validation vs test**:

- If \(R^2_{\text{train}}\) is very high (~0.99) but \(R^2_{\text{test}}\) is much lower or unstable → you are **overfitting**.

---

### 5. Forecasting the Future (`horizon`)

You choose:

- `horizon`: how many **future days** to forecast (default: `30`).

The model generates forecasts **iteratively**:

1. Take the last `window` **real prices**.
2. Predict the next price → append to `future_preds`.
3. Slide the window by dropping the oldest value and adding the new prediction.
4. Repeat until `horizon` steps are produced.

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

---

### 6. Plotting

The function plots:

- The last `n_points` days of **historical prices**.
- The next `horizon` days of **predicted prices** (dashed line).
- A vertical line separating **history** and **forecast**.

Approximate plotting code:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(hist_tail.index, hist_tail.values, label="Actual (history)")
plt.plot(
    future_idx,
    future_preds,
    label=f"Forecast (next {horizon} days)",
    linestyle="--",
)

plt.axvline(last_date, color="gray", linestyle=":", label="Forecast start")

plt.title(
    f"{symbol} - ANN Time Series Forecast\n"
    f"TRAIN R²={r2_train:.3f}, TEST R²={r2_test:.3f}, "
    f"window={window}, layers={hidden_layers}, neurons={neurons}"
)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
```

---

### 7. Window Size & Overfitting Warning

Oversized window sizes can easily lead to overfitting.

Example configuration:

```text
window        = 1000
hidden_layers = 2
neurons       = 64
```

Approximate parameter count:

- Layer 1: \(1000 \times 64 \approx 64{,}000\) weights  
- Layer 2: \(64 \times 64 \approx 4{,}096\) weights  
- Output: \(64 \times 1 \approx 64\) weights  

Total:

$$
\text{params} \approx 68{,}000 \text{ (plus biases)} \Rightarrow 70{,}000+ \text{ parameters}
$$

If you only have **≈ 1,400 training examples**, you are trying to estimate ~70k parameters with 1.4k samples:

- The model is **far more complex** than the available data justifies.
- It becomes an extremely flexible function that can **memorize** the training series.

You will typically see:

- \(R^2_{\text{train}} \approx 0.99+\),
- but validation / test performance is **unstable** and often disappointing.

**Practical guidance:**

- Keep `window` **modest** (e.g. `20–60`) unless you really know what you’re doing.
- Recommended starting point:
  - `window = 20` or `40`,
  - `hidden_layers = 1–2`,
  - `neurons = 32` or `64`.

Watch the gap between **train** and **test** metrics:

- small gap → model generalizes reasonably,
- huge gap → you’re overfitting.

---

### 8. How to Run

Assuming you’re inside your project folder and the environment is set up:

```bash
# Activate virtualenv
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\activate       # Windows

# Run the script
python ann.py
```

Then follow the prompts:

- Enter ticker symbol (or press **Enter** for `GC=F`).
- Enter `n_points` (or press **Enter** for `3000`).
- Enter `horizon` (future days to predict).
- Enter `window`, `hidden_layers`, `neurons`.

The script will:

1. Train the ANN.
2. Print **R² / MSE / RMSE** for train / val / test.
3. Open a **matplotlib** window with history + forecast plotted.

Possible extensions:

- Input normalization / scaling,
- Using log-returns instead of levels,
- Multiple feature inputs (e.g. technical indicators),
- Exporting the trained model / metrics to disk.
