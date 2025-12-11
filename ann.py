def ann():    
    try:
        import yfinance as yf
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import TransformedTargetRegressor
    except ImportError as e:
        print("Missing libs. Install:")
        print("pip install numpy pandas matplotlib scikit-learn yfinance")
        print(f"Details: {e}")
        return

    # Helpers

    def fetch_close(symbol: str):
        t = yf.Ticker(symbol)
        hist = t.history(period="max", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            return None
        close = hist["Close"].dropna()
        if hasattr(close.index, "tz") and close.index.tz is not None:
            close.index = close.index.tz_convert(None)
        return close
    def build_supervised(series_1d: "np.ndarray", window: int):
        X, y = [], []
        for i in range(window, len(series_1d)):
            X.append(series_1d[i - window:i])
            y.append(series_1d[i])
        return np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    def make_model(hidden_layers: int, neurons: int, use_scaling: bool):
        base = MLPRegressor(
            hidden_layer_sizes=tuple([neurons] * hidden_layers),
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=1200,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.1,
        )
        if not use_scaling:
            return base
        x_pipe = Pipeline([("scaler", StandardScaler()), ("mlp", base)])
        return TransformedTargetRegressor(regressor=x_pipe, transformer=StandardScaler())
    def split_time(X, y, train_ratio, val_ratio, test_ratio):
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:val_end], y[train_end:val_end]),
            (X[val_end:], y[val_end:]),
        )
    def safe_r2_rmse(model, Xp, yp):
        if len(Xp) == 0:
            return float("nan"), float("nan")
        pred = model.predict(Xp)
        r2 = float(r2_score(yp, pred))
        rmse = float(np.sqrt(mean_squared_error(yp, pred)))
        return r2, rmse
    def forecast_recursive(model, last_window: "np.ndarray", steps: int):
        w = last_window.astype(float).copy()
        preds = []
        for _ in range(steps):
            nxt = float(model.predict(w.reshape(1, -1))[0])
            preds.append(nxt)
            w = np.roll(w, -1)
            w[-1] = nxt
        return np.asarray(preds, dtype=float)
    def plot_full_and_zoom_holdout(
        train_dates, train_actual,
        pred_dates, pred_actual, pred_forecast,
        split_date, symbol, rmse
    ):
        import matplotlib.pyplot as plt
        # FULL VIEW
        plt.figure(figsize=(13, 6))
        full_dates = list(train_dates) + list(pred_dates)
        full_actual = list(train_actual) + list(pred_actual)
        plt.plot(full_dates, full_actual, label="Actual", color="blue", linewidth=2)
        plt.plot(pred_dates, pred_forecast, label="Forecast", color="darkgreen", linewidth=2)
        plt.axvline(split_date, color="gray", linestyle=":", label="Forecast start")
        for d, a, p in zip(pred_dates, pred_actual, pred_forecast):
            plt.plot([d, d], [a, p], "r--", alpha=0.35, linewidth=1)
        plt.title(f"{symbol} - Full View | HOLDOUT RMSE={rmse:.3f}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # ZOOM VIEW (pred zone only this is quite important)
        plt.figure(figsize=(13, 6))
        plt.plot(pred_dates, pred_actual, label="Actual (pred zone)", color="blue", linewidth=2)
        plt.plot(pred_dates, pred_forecast, label="Forecast (pred zone)", color="darkgreen", linewidth=2)
        for d, a, p in zip(pred_dates, pred_actual, pred_forecast):
            plt.plot([d, d], [a, p], "r--", alpha=0.35, linewidth=1)
        plt.title(f"{symbol} - Zoom (Predicted Zone Only) | RMSE={rmse:.3f}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_direct_forecast(hist_dates, hist_actual, future_dates, future_pred, symbol, meta=""):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 6))
        plt.plot(hist_dates, hist_actual, label="Actual (history)", color="blue", linewidth=2)
        plt.plot(future_dates, future_pred, label="Forecast (future)", color="darkgreen", linewidth=2)
        plt.axvline(hist_dates[-1], color="gray", linestyle=":", label="Forecast start")
        plt.title(f"{symbol} - Direct Forecast {meta}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Menu + Base Inputs

    print("\n welcome to the MENU ====")
    print("1) Direct forecast (future)")
    print("2) Holdout backtest (offset_back + back_points + horizon)")
    mode = input("Select (1/2) [default 1]: ").strip() or "1"
    symbol = input("Symbol (empty=GC=F): ").strip() or "GC=F"
    close = fetch_close(symbol)
    if close is None or close.empty:
        print("Could not fetch price data.")
        return
    window = int(input("Lag window size (default 20): ").strip() or "20")
    hidden_layers = int(input("Hidden layers (default 2): ").strip() or "2")
    neurons = int(input("Neurons per layer (default 64): ").strip() or "64")
    use_scaling = (input("Use scaling? (y/n) [default y]: ").strip().lower() or "y") == "y"
    if window <= 1 or hidden_layers <= 0 or neurons <= 0:
        print("window>1, hidden_layers>0, neurons>0 olmalı.")
        return
    model = make_model(hidden_layers, neurons, use_scaling)

    # Mode 1: Direct Forecast (future)

    if mode == "1":
        n_points = int(input("\nHow many past points for training (default 3000): ").strip() or "3000")
        horizon = int(input("Forecast horizon (future points) (default 30): ").strip() or "30")
        train_ratio = float(input("Train ratio (default 0.7): ").strip() or "0.7")
        val_ratio = float(input("Val ratio (default 0.15): ").strip() or "0.15")
        test_ratio = float(input("Test ratio (default 0.15): ").strip() or "0.15")
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            print("Train/Val/Test ratios must sum to 1.0")
            return
        if n_points > len(close):
            n_points = len(close)
        hist_tail = close.iloc[-n_points:]
        series = hist_tail.values.astype(float)
        if len(series) <= window + 50:
            print("Not enough data for this window. Increase n_points or reduce window.")
            return
        # Build supervised on the whole selected history
        X_all, y_all = build_supervised(series, window)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_time(X_all, y_all, train_ratio, val_ratio, test_ratio)
        model.fit(X_tr, y_tr)
        r2_tr, rmse_tr = safe_r2_rmse(model, X_tr, y_tr)
        r2_va, rmse_va = safe_r2_rmse(model, X_va, y_va)
        r2_te, rmse_te = safe_r2_rmse(model, X_te, y_te)
        print("\n HI, YOU CAN THE CHECK PERFORMANCE METRICS (on selected history) ====")
        print(f"[TRAIN] R2={r2_tr:.4f} | RMSE={rmse_tr:.4f}")
        print(f"[VAL]   R2={r2_va:.4f} | RMSE={rmse_va:.4f}")
        print(f"[TEST]  R2={r2_te:.4f} | RMSE={rmse_te:.4f}")
        # Refit on full history slice for final future forecast
        model.fit(X_all, y_all)
        last_window = series[-window:].copy()
        future_pred = forecast_recursive(model, last_window, horizon)
        import pandas as pd
        last_date = hist_tail.index[-1]
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        meta = f"| window={window}, layers={hidden_layers}, neurons={neurons}"
        plot_direct_forecast(hist_tail.index, hist_tail.values.astype(float), future_idx, future_pred, symbol, meta)
        # Additional close-up view: only forecasted future period
        plt.figure(figsize=(13, 6))
        plt.plot(future_idx, future_pred, label="Forecast (future)", color="darkgreen", linewidth=2, marker='s', markersize=5)
        plt.axvline(future_idx[0], color="gray", linestyle=":", label="Forecast start", alpha=0.7)
        plt.title(f"{symbol} - Close-Up: Future Forecast Only {meta}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # Mode 2: Holdout backtest (your 3 knobs)

    if mode == "2":
        offset_back = int(input("\n1) Go back from LAST by how many points? (e.g. 500): ").strip() or "500")
        back_points = int(input("2) How many points BEFORE that to train on? (e.g. 3000): ").strip() or "3000")
        horizon = int(input("3) Forecast horizon points AFTER split? (e.g. 90): ").strip() or "90")

        train_ratio = float(input("Train ratio (default 0.7): ").strip() or "0.7")
        val_ratio = float(input("Val ratio (default 0.15): ").strip() or "0.15")
        test_ratio = float(input("Test ratio (default 0.15): ").strip() or "0.15")

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            print("Train/Val/Test ratios must sum to 1.0 otherwise it cant fit itself u know")
            return

        N = len(close)
        split_idx = N - offset_back
        if split_idx <= 0 or split_idx >= N:
            print("offset_back is invalid for this dataset length.")
            return

        train_start_idx = split_idx - back_points
        if train_start_idx < 0:
            print(f"Not enough history. Need back_points={back_points} before split.")
            return

        max_h = N - split_idx
        if horizon > max_h:
            print(f"Warning: horizon={horizon} > available post-split points={max_h}. Using {max_h}.")
            horizon = max_h

        train_close = close.iloc[train_start_idx:split_idx]
        holdout_close = close.iloc[split_idx: split_idx + horizon]

        if len(train_close) <= window + 50:
            print("Train slice too small for this window. Reduce window or increase back_points.")
            return
        if len(holdout_close) < 5:
            print("Holdout slice too small. Increase horizon or adjust offset_back.")
            return

        train_series = train_close.values.astype(float)
        holdout_actual = holdout_close.values.astype(float)

        X_all, y_all = build_supervised(train_series, window)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_time(X_all, y_all, train_ratio, val_ratio, test_ratio)

        model.fit(X_tr, y_tr)
        r2_tr, rmse_tr = safe_r2_rmse(model, X_tr, y_tr)
        r2_va, rmse_va = safe_r2_rmse(model, X_va, y_va)
        r2_te, rmse_te = safe_r2_rmse(model, X_te, y_te)

        print("\n HI AGAIN, CHECK THE PERFORMANCE METRICS (ONLY on train slice) ====")
        print(f"[TRAIN] R2={r2_tr:.4f} | RMSE={rmse_tr:.4f}")
        print(f"[VAL]   R2={r2_va:.4f} | RMSE={rmse_va:.4f}")
        print(f"[TEST]  R2={r2_te:.4f} | RMSE={rmse_te:.4f}")

        # refit full train slice then forecast holdout horizon
        model.fit(X_all, y_all)
        last_window = train_series[-window:].copy()
        holdout_pred = forecast_recursive(model, last_window, horizon)

        rmse_oos = float(np.sqrt(mean_squared_error(holdout_actual, holdout_pred)))
        print("\n==== HOLDOUT (OUT-OF-SAMPLE) ====")
        print(f"Split date      : {train_close.index[-1].date()}")
        print(f"Train slice     : {train_close.index[0].date()} .. {train_close.index[-1].date()}  (n={len(train_close)})")
        print(f"Forecast slice  : {holdout_close.index[0].date()} .. {holdout_close.index[-1].date()}  (n={len(holdout_close)})")
        print(f"HOLDOUT RMSE    : {rmse_oos:.4f}")

        plot_full_and_zoom_holdout(
            train_dates=train_close.index,
            train_actual=train_close.values.astype(float),
            pred_dates=holdout_close.index,
            pred_actual=holdout_actual,
            pred_forecast=holdout_pred,
            split_date=train_close.index[-1],
            symbol=symbol,
            rmse=rmse_oos,
        )
        
        # Additional close-up view: only predicted period
        plt.figure(figsize=(13, 6))
        plt.plot(holdout_close.index, holdout_actual, label="Actual", color="blue", linewidth=2, marker='o', markersize=4)
        plt.plot(holdout_close.index, holdout_pred, label="Forecast", color="darkgreen", linewidth=2, marker='s', markersize=4)
        
        for d, a, p in zip(holdout_close.index, holdout_actual, holdout_pred):
            plt.plot([d, d], [a, p], "r--", alpha=0.35, linewidth=1)
        
        plt.title(f"{symbol} - Close-Up: Predicted Period Only | RMSE={rmse_oos:.3f}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return

    print("Invalid selection. Choose 1 or 2.")
    
ann()
 