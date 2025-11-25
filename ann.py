import yfinance as yf

def do_ts_forecast() -> None:
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error, r2_score
    except ImportError as e:
        print("Required libraries for time series forecasting are missing:")
        print("Required: numpy, pandas, matplotlib, scikit-learn")
        print("Example install: pip install numpy pandas matplotlib scikit-learn")
        print(f"Details: {e}")
        return

    try:
        symbol = input("Symbol (leave empty for GC=F): ").strip() or "GC=F"

        n_points_str = input(
            "Number of past data points to use for training (leave empty for 3000): "
        ).strip()
        n_points = int(n_points_str) if n_points_str else 3000

        horizon_str = input(
            "How many days ahead do you want to forecast? (leave empty for 30): "
        ).strip()
        horizon = int(horizon_str) if horizon_str else 30

        window_str = input(
            "Lag window size (e.g. 20, leave empty for 20): "
        ).strip()
        window = int(window_str) if window_str else 20

        hl_str = input(
            "Number of hidden layers (leave empty for 2): "
        ).strip()
        hidden_layers = int(hl_str) if hl_str else 2

        neurons_str = input(
            "Number of neurons per layer (leave empty for 64): "
        ).strip()
        neurons = int(neurons_str) if neurons_str else 64

        # you should set your rates here
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            print("Train/val/test ratios do not sum to 1, check the code.")
            return

        if n_points <= window + 5:
            print("Number of past data points is too small compared to the window size. Increase them.")
            return
        if hidden_layers <= 0 or neurons <= 0:
            print("Hidden layer count and neuron count must be positive.")
            return

        t = yf.Ticker(symbol)
        hist = t.history(period="max", interval="1d")
        if hist.empty or "Close" not in hist:
            print("Could not fetch price data.")
            return

        close = hist["Close"].dropna()
        if len(close) < window + 20:
            print("Not enough price data.")
            return

        if n_points > len(close):
            n_points = len(close)

        # Work with the last n_points
        series = close.values[-n_points:]

        if len(series) <= window + 5:
            print("Selected n_points and window combination is too small.")
            return

        # to understand the data behaves eachother at the scale of observations
        X, y = [], []
        for i in range(window, len(series)):
            X.append(series[i - window:i])
            y.append(series[i])

        X = np.array(X)
        y = np.array(y)

        n_samples = len(X)
        if n_samples < 10:
            print("Not enough observations for the model.")
            return

        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        if train_end < 5:
            print("Train set is too small. Fetch more data or reduce the window.")
            return
        if val_end - train_end < 0:
            print("Validation set size is negative, check the ratios in the code.")
            return
        if (n_samples - val_end) < 5:
            print("Test set is too small. Fetch more data or reduce the window.")
            return

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        hl_sizes = tuple([neurons] * hidden_layers)

        print(
            f"\nBuilding model: MLPRegressor(hidden_layer_sizes={hl_sizes}, "
            f"window={window}, n_samples={n_samples}, "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)})"
        )

        model = MLPRegressor(
            hidden_layer_sizes=hl_sizes,
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=500,
        )
        model.fit(X_train, y_train)

        # PREDICTIONS
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val) if len(X_val) > 0 else None
        y_pred_test = model.predict(X_test) if len(X_test) > 0 else None

        # TRAINING SIDE
        mse_train = float(mean_squared_error(y_train, y_pred_train))
        rmse_train = float(np.sqrt(mse_train))
        r2_train = float(r2_score(y_train, y_pred_train))

        # VALIDATION SIDE
        if y_pred_val is not None and len(y_val) > 0:
            mse_val = float(mean_squared_error(y_val, y_pred_val))
            rmse_val = float(np.sqrt(mse_val))
            r2_val = float(r2_score(y_val, y_pred_val))
        else:
            mse_val = rmse_val = r2_val = float("nan")

        # TEST SIDE
        if y_pred_test is not None and len(y_test) > 0:
            mse_test = float(mean_squared_error(y_test, y_pred_test))
            rmse_test = float(np.sqrt(mse_test))
            r2_test = float(r2_score(y_test, y_pred_test))
        else:
            mse_test = rmse_test = r2_test = float("nan")

        print("\n==== TIME SERIES NN PERFORMANCE ===========")
        print(f"[TRAIN] R2: {r2_train:.4f} | MSE: {mse_train:.6f} | RMSE: {rmse_train:.6f}")
        print(f"[VAL]   R2: {r2_val:.4f} | MSE: {mse_val:.6f} | RMSE: {rmse_val:.6f}")
        print(f"[TEST]  R2: {r2_test:.4f} | MSE: {mse_test:.6f} | RMSE: {rmse_test:.6f}")
        print("============(PS: price error is rmse)===================================")

        # FUTURE HORIZON PREDICTIONS
        last_window = series[-window:].copy()
        future_preds = []

        for _ in range(horizon):
            next_val = float(model.predict(last_window.reshape(1, -1))[0])
            future_preds.append(next_val)

            last_window = np.roll(last_window, -1)
            last_window[-1] = next_val

        # PAST AND FUTURE
        hist_tail = close.iloc[-n_points:]
        last_date = hist_tail.index[-1]
        future_idx = pd.date_range(
            last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

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
            f"{symbol} - NN Time Series Forecast\n"
            f"TRAIN R2={r2_train:.3f}, TEST R2={r2_test:.3f}, "
            f"window={window}, layers={hidden_layers}, neurons={neurons}"
        )
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Time series forecast error: {e}")


do_ts_forecast()
