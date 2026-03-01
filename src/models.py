"""
models.py - ARIMA/SARIMA, LSTM, GRU models for HR forecasting + comparison
"""

import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = '') -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        'model': model_name,
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'mape': round(mape, 4),
        'n_test': len(y_true),
    }


def train_test_split_timeseries(series: np.ndarray, test_ratio: float = 0.2):
    split = int(len(series) * (1 - test_ratio))
    return series[:split], series[split:]


# ─────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────

def prepare_hr_series(rr_ms: np.ndarray, window_s: int = 10, fs_rr: float = 1.0) -> np.ndarray:
    """Convert RR intervals to instantaneous HR, then compute windowed mean HR."""
    hr = 60000.0 / rr_ms
    # Rolling mean to smooth
    kernel = np.ones(max(1, window_s)) / max(1, window_s)
    return np.convolve(hr, kernel, mode='valid')


def create_sequences(series: np.ndarray, look_back: int = 20) -> tuple:
    """Create (X, y) sequences for supervised learning."""
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i + look_back])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# ARIMA / SARIMA
# ─────────────────────────────────────────────

def fit_arima(train: np.ndarray, test: np.ndarray,
              order: tuple = (2, 1, 2), seasonal_order: tuple = None,
              verbose: bool = False) -> dict:
    """
    Fit ARIMA or SARIMA model and evaluate on test set.
    Falls back gracefully if statsmodels unavailable.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        history = list(train)
        predictions = []

        # Use SARIMA if seasonal order provided
        ModelClass = SARIMAX if seasonal_order else ARIMA

        for i in range(len(test)):
            try:
                if seasonal_order:
                    model = ModelClass(history, order=order, seasonal_order=seasonal_order)
                else:
                    model = ModelClass(history, order=order)
                fit = model.fit(disp=False)
                yhat = fit.forecast(steps=1)[0]
            except Exception:
                yhat = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)

            predictions.append(yhat)
            history.append(test[i])

            if verbose and i % 50 == 0:
                print(f"  ARIMA step {i}/{len(test)}")

        predictions = np.array(predictions)
        metrics = compute_metrics(test, predictions, 'ARIMA')

        return {
            'model_name': 'ARIMA',
            'predictions': predictions,
            'metrics': metrics,
            'order': order,
        }

    except ImportError:
        print("  statsmodels not available, using naive forecast for ARIMA")
        # Naive: use last known value
        predictions = np.array([train[-1]] + list(test[:-1]))
        metrics = compute_metrics(test, predictions, 'ARIMA_naive')
        return {'model_name': 'ARIMA_naive', 'predictions': predictions, 'metrics': metrics}


def auto_arima(train: np.ndarray, test: np.ndarray, max_p: int = 3, max_q: int = 3) -> dict:
    """Try a few ARIMA orders and pick lowest AIC."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        best_aic = np.inf
        best_order = (1, 1, 1)

        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(train, order=(p, 1, q))
                    fit = model.fit(disp=False)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, 1, q)
                except Exception:
                    pass

        print(f"  Best ARIMA order: {best_order} (AIC={best_aic:.1f})")
        return fit_arima(train, test, order=best_order)

    except ImportError:
        return fit_arima(train, test)


# ─────────────────────────────────────────────
# LSTM / GRU (TensorFlow/Keras)
# ─────────────────────────────────────────────

def build_lstm(look_back: int = 20, units: int = 64, dropout: float = 0.2,
               learning_rate: float = 0.001):
    """Build LSTM model for univariate time series forecasting."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(units, input_shape=(look_back, 1), return_sequences=True),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    except ImportError:
        return None


def build_gru(look_back: int = 20, units: int = 64, dropout: float = 0.2,
              learning_rate: float = 0.001):
    """Build GRU model for univariate time series forecasting."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            GRU(units, input_shape=(look_back, 1), return_sequences=True),
            Dropout(dropout),
            GRU(units // 2),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model
    except ImportError:
        return None


def fit_deep_model(series: np.ndarray, model_type: str = 'LSTM',
                   look_back: int = 20, epochs: int = 20,
                   batch_size: int = 32, test_ratio: float = 0.2,
                   verbose: bool = False) -> dict:
    """
    Train LSTM or GRU model on HR time series.
    Falls back to simple regression if TensorFlow unavailable.
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

        train, test = train_test_split_timeseries(series_scaled, test_ratio)
        X_train, y_train = create_sequences(train, look_back)
        X_test, y_test = create_sequences(test, look_back)

        if len(X_train) < 10:
            raise ValueError("Insufficient data for deep learning")

        # Reshape for LSTM/GRU: (samples, timesteps, features)
        X_train = X_train.reshape(-1, look_back, 1)
        X_test = X_test.reshape(-1, look_back, 1)

        # Build model
        builder = build_lstm if model_type == 'LSTM' else build_gru
        model = builder(look_back=look_back)

        if model is None:
            raise ImportError("TensorFlow not available")

        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[es],
            verbose=1 if verbose else 0
        )

        # Predictions
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()

        # Inverse transform
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        metrics = compute_metrics(y_true_actual, y_pred, model_type)

        return {
            'model_name': model_type,
            'model': model,
            'predictions': y_pred,
            'actuals': y_true_actual,
            'metrics': metrics,
            'training_loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', []),
            'scaler': scaler,
        }

    except (ImportError, Exception) as e:
        print(f"  {model_type} fallback (sklearn): {e}")
        # Simple linear regression fallback
        from sklearn.linear_model import LinearRegression
        train, test = train_test_split_timeseries(series, test_ratio)
        X_train, y_train = create_sequences(train, look_back)
        X_test, y_test = create_sequences(test, look_back)

        if len(X_train) < 2:
            predictions = np.full(len(test), np.mean(train))
            actuals = test
        else:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            predictions = lr.predict(X_test)
            actuals = y_test

        metrics = compute_metrics(actuals, predictions, f'{model_type}_LinearFallback')
        return {
            'model_name': f'{model_type}_LinearFallback',
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics,
        }


# ─────────────────────────────────────────────
# Model Comparison Framework
# ─────────────────────────────────────────────

def compare_models(rr_ms: np.ndarray, verbose: bool = True) -> dict:
    """
    Run all models and compare performance.
    Returns dict with results and summary DataFrame.
    """
    if len(rr_ms) < 100:
        print("  Insufficient data for model comparison (need ≥100 RR intervals)")
        return {}

    print("\n  Preparing HR time series...")
    hr_series = prepare_hr_series(rr_ms, window_s=5)

    if len(hr_series) < 60:
        print("  HR series too short after windowing")
        return {}

    train, test = train_test_split_timeseries(hr_series, test_ratio=0.2)
    results = {}

    # ── ARIMA ──
    print("  Fitting ARIMA...")
    arima_len = min(500, len(train))  # Limit for speed
    arima_test_len = min(100, len(test))
    arima_result = fit_arima(
        train[-arima_len:], test[:arima_test_len],
        order=(2, 1, 2), verbose=False
    )
    results['ARIMA'] = arima_result
    if verbose:
        m = arima_result['metrics']
        print(f"  ARIMA  → RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}")

    # ── LSTM ──
    print("  Fitting LSTM...")
    look_back = min(20, len(train) // 4)
    lstm_result = fit_deep_model(hr_series, model_type='LSTM',
                                  look_back=look_back, epochs=30, verbose=False)
    results['LSTM'] = lstm_result
    if verbose:
        m = lstm_result['metrics']
        print(f"  LSTM   → RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}")

    # ── GRU ──
    print("  Fitting GRU...")
    gru_result = fit_deep_model(hr_series, model_type='GRU',
                                 look_back=look_back, epochs=30, verbose=False)
    results['GRU'] = gru_result
    if verbose:
        m = gru_result['metrics']
        print(f"  GRU    → RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}")

    # ── Summary ──
    summary_rows = [r['metrics'] for r in results.values()]
    summary_df = pd.DataFrame(summary_rows).set_index('model')
    results['summary'] = summary_df
    results['hr_series'] = hr_series
    results['train'] = train
    results['test'] = test

    if verbose:
        print("\n  === Model Comparison Summary ===")
        print(summary_df.to_string())

    return results


def save_model_results(results: dict, output_dir: str, record_id: int):
    """Save metrics and predictions to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    if 'summary' in results:
        results['summary'].to_csv(
            os.path.join(output_dir, f"{record_id}_model_metrics.csv")
        )
    for name, res in results.items():
        if isinstance(res, dict) and 'predictions' in res:
            pred_df = pd.DataFrame({
                'predicted': res['predictions'],
                'actual': res.get('actuals', res['predictions'])
            })
            pred_df.to_csv(
                os.path.join(output_dir, f"{record_id}_{name}_predictions.csv"),
                index=False
            )


if __name__ == "__main__":
    from load_data import generate_synthetic_record
    from preprocess import preprocess_record

    rec = generate_synthetic_record(100, duration_s=600, mean_hr=72, hrv_std=40)
    rec = preprocess_record(rec, verbose=False)

    print(f"RR intervals: {len(rec['rr_clean'])}")
    results = compare_models(rec['rr_clean'], verbose=True)
