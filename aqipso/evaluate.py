"""Regression, AQI-category and statistical evaluation."""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             median_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, root_mean_squared_error)

from . import config as cfg


def to_band(aqi) -> np.ndarray:
    """Convert AQI values to CPCB category labels."""
    uppers = [upper for upper, _ in cfg.AQI_BANDS]
    labels = np.array([label for _, label in cfg.AQI_BANDS])
    return labels[np.searchsorted(uppers, np.asarray(aqi, dtype=float), side="left")]


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": r2_score(y_true, y_pred),
    }


def category_metrics(y_true, y_pred) -> dict:
    """Score predictions after mapping both sides to CPCB AQI categories."""
    t, p = to_band(y_true), to_band(y_pred)
    high = (np.asarray(y_true) > cfg.HIGH_POLLUTION_AQI).astype(int)
    return {
        "Accuracy": accuracy_score(t, p),
        "Precision_macro": precision_score(t, p, average="macro", zero_division=0),
        "Recall_macro": recall_score(t, p, average="macro", zero_division=0),
        "F1_macro": f1_score(t, p, average="macro", zero_division=0),
        "F1_weighted": f1_score(t, p, average="weighted", zero_division=0),
        "AUC_high_pollution": roc_auc_score(high, np.asarray(y_pred)),
    }


def all_metrics(y_true, y_pred) -> dict:
    return {**regression_metrics(y_true, y_pred), **category_metrics(y_true, y_pred)}


def block_bootstrap_rmse_gain(y_true, pred_a, pred_b, block: int = 7,
                              n_boot: int = 5000, seed: int = 0) -> dict:
    """Moving-block bootstrap of RMSE(a) - RMSE(b) on the test period.

    Daily errors are autocorrelated, so whole weeks are resampled instead of single
    days. A positive gain means model b has the lower error.
    """
    rng = np.random.default_rng(seed)
    err_a = (np.asarray(y_true) - np.asarray(pred_a)) ** 2
    err_b = (np.asarray(y_true) - np.asarray(pred_b)) ** 2
    n = len(err_a)
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    gains = np.sqrt(err_a[idx].mean(axis=1)) - np.sqrt(err_b[idx].mean(axis=1))
    return {
        "observed_gain": float(np.sqrt(err_a.mean()) - np.sqrt(err_b.mean())),
        "ci95": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "prob_gain_positive": float((gains > 0).mean()),
    }
