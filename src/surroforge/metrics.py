"""Regression metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute common regression metrics over all outputs."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = pred - true
    absolute = np.abs(error)
    squared = error**2
    denominator = np.where(np.abs(true) < 1e-12, np.nan, np.abs(true))
    mape = np.nanmean(absolute / denominator) * 100.0
    ss_res = float(np.sum(squared))
    ss_tot = float(np.sum((true - np.mean(true, axis=0)) ** 2))
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "mape": float(mape) if np.isfinite(mape) else 0.0,
        "max_error": float(np.max(absolute)) if absolute.size else 0.0,
    }
