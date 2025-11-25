Price Forecasting using ANN
(Configurable Time-Series Forecasting using Artificial Neural Networks)
This module implements a neural-network-based time series forecaster on financial price data (e.g. gold futures GC=F) using an MLP (Multi-Layer Perceptron).
The core function, do_ts_forecast(), performs the entire pipeline end-to-end:


Fetches daily closing prices from Yahoo Finance via yfinance.


Builds a lag-based supervised dataset using the last N data points.


Trains an MLPRegressor (ANN) with configurable:


window size (number of lags),


number of hidden layers,


neurons per hidden layer.




Splits the data into Train / Validation / Test sets in time order
using fixed ratios: 0.7 / 0.15 / 0.15.


Computes R² / MSE / RMSE for each split.


Forecasts the next horizon days iteratively and plots:


historical prices,


future predictions,


split point between history and forecast.





1. Data Source and Preprocessing


Data is loaded using:
import yfinance as yf
t = yf.Ticker(symbol)
hist = t.history(period="max", interval="1d")
close = hist["Close"].dropna()



You choose:


symbol (default: "GC=F" for gold futures),


n_points: how many of the most recent daily closes to use (default: 3000).




Only the last n_points observations are used:
series = close.values[-n_points:]

Lag-based supervised dataset
For a given window (lag length), the model uses the last window prices to predict the next one:


Input vector at time t:
Xt=[pt−window,…,pt−1]X_t = [p_{t-window}, \dots, p_{t-1}]Xt​=[pt−window​,…,pt−1​]


Target:
yt=pty_t = p_tyt​=pt​


In code:
X, y = [], []
for i in range(window, len(series)):
    X.append(series[i - window:i])
    y.append(series[i])

This is standard “sliding window” supervised learning on a univariate time series.

2. Model: MLPRegressor (Artificial Neural Network)
The model is a feed-forward neural network:


Implementation: sklearn.neural_network.MLPRegressor


Activation: relu


Optimizer: adam


Iterations: max_iter=500


Hidden layers: configurable


You configure:


hidden_layers → number of hidden layers (e.g. 1, 2, 3…)


neurons → neurons per hidden layer (e.g. 32, 64, 128…)


The code then builds:
hl_sizes = tuple([neurons] * hidden_layers)
model = MLPRegressor(
    hidden_layer_sizes=hl_sizes,
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=500,
)

So for hidden_layers = 2 and neurons = 64 you get an architecture:


Input: window features


Hidden Layer 1: 64 neurons (ReLU)


Hidden Layer 2: 64 neurons (ReLU)


Output: 1 neuron (next price)



3. Train / Validation / Test Split
The dataset is split chronologically (no shuffling):


train_ratio = 0.7


val_ratio   = 0.15


test_ratio  = 0.15


On n_samples examples:
train_end = int(n_samples * train_ratio)
val_end   = train_end + int(n_samples * val_ratio)

X_train, y_train = X[:train_end], y[:train_end]       # oldest 70%
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]   # next 15%
X_test,  y_test  = X[val_end:], y[val_end:]           # most recent 15%

This preserves temporal ordering (future never leaks into past), which is critical for time series modeling.

4. Evaluation Metrics
After training, the model predicts on each split and computes:


MSE (Mean Squared Error):
MSE=1n∑(y−y^)2\text{MSE} = \frac{1}{n} \sum (y - \hat{y})^2MSE=n1​∑(y−y^​)2


RMSE (Root Mean Squared Error):
RMSE=MSE\text{RMSE} = \sqrt{\text{MSE}}RMSE=MSE​


R² (Coefficient of Determination):
measures how much variance in y is explained by the model (1.0 = perfect).


Output looks like:
==== TIME SERIES NN PERFORMANCE ====
[TRAIN] R²: 0.9821 | MSE:  950.123456 | RMSE: 30.825123
[VAL]   R²: 0.7420 | MSE: 2300.789012 | RMSE: 47.963325
[TEST]  R²: 0.7013 | MSE: 2600.456789 | RMSE: 50.993694
====================================

If you are forecasting gold futures and the price is around 4,000 USD, then:


RMSE ≈ 30 → typical prediction error is about 30 USD.


MSE ≈ 900 → square of that; units are USD².


Always compare train vs validation vs test:


If R²_train is very high (~0.99) but R²_test is much lower or unstable, you are overfitting.



5. Forecasting the Future (horizon)
You choose:


horizon: how many future days to forecast (default: 30).


The model generates forecasts iteratively:


Take the last window real prices.


Predict the next price → append to predictions.


Slide the window forward by dropping the oldest value and adding the new prediction.


Repeat until horizon steps are produced.


Pseudocode:
last_window = series[-window:].copy()
future_preds = []

for _ in range(horizon):
    next_val = model.predict(last_window.reshape(1, -1))[0]
    future_preds.append(next_val)
    last_window = np.roll(last_window, -1)
    last_window[-1] = next_val


6. Plotting
The function plots:


The last n_points days of historical prices.


The next horizon days of predicted prices (dashed line).


A vertical line separating history and forecast.


Roughly:
plt.figure(figsize=(12, 6))
plt.plot(hist_tail.index, hist_tail.values, label="Actual (history)")
plt.plot(future_idx, future_preds, label=f"Forecast (next {horizon} days)", linestyle="--")
plt.axvline(last_date, color="gray", linestyle=":", label="Forecast start")
plt.title(f"{symbol} - ANN Time Series Forecast\nTRAIN R²=..., TEST R²=..., window=..., layers=..., neurons=...")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()


7. Window Size & Overfitting Warning
The guide text in the docstring explicitly warns about oversized window sizes.
Example:


window = 1000, hidden_layers = 2, neurons = 64


Parameter count:


Layer 1: 1000 × 64 ≈ 64,000 weights


Layer 2: 64 × 64 ≈ 4,096 weights


Output: 64 × 1 ≈ 64 weights
→ Total ≈ 68,000 parameters, plus biases → easily 70,000+ parameters.


If you only have ~1,400 training examples, you’re trying to estimate 70k parameters with 1.4k samples. That means:


Model is far more complex than the available data justifies.


It becomes an extremely flexible function that can memorize the training series.


You’ll see:


R²_train ≈ 0.99+,


but validation / test performance will be unstable and often disappointing.




Practical guidance:


Keep window modest (e.g. 20–60) unless you know what you’re doing.


Start with:


window = 20 or 40,


hidden_layers = 1–2,


neurons = 32 or 64.




Watch the gap between train and test metrics:


small gap → model generalizes reasonably,


huge gap → overfitting.





8. How to Run
Assuming you’re inside your project folder and the environment is set up:
# Activate virtualenv
source venv/bin/activate  # on macOS / Linux
# .\venv\Scripts\activate  # on Windows

# Run the script
python ann.py

Then follow the prompts:


Enter ticker symbol (or press Enter for GC=F).


Enter n_points (or press Enter for 3000).


Enter horizon (future days to predict).


Enter window, hidden_layers, neurons.


The script will:


train the ANN,


print R² / MSE / RMSE for train/val/test,


open a matplotlib window with history + forecast plotted.



If you want, next step we can bolt on:


input normalization / scaling,


log-returns instead of levels,


multiple feature inputs (e.g., technical indicators),


or export the trained model / metrics to disk.

