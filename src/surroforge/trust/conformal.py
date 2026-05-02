"""Simple split-conformal intervals."""

from __future__ import annotations

import numpy as np


class ConformalIntervals:
    """Calibrate symmetric prediction intervals from residuals."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.radius_: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> ConformalIntervals:
        """Fit interval radii from absolute residuals."""
        residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        if residuals.ndim == 1:
            residuals = residuals[:, None]
        quantile = min(1.0, np.ceil((len(residuals) + 1) * (1 - self.alpha)) / len(residuals))
        self.radius_ = np.quantile(residuals, quantile, axis=0, method="higher")
        return self

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper interval bounds."""
        if self.radius_ is None:
            raise RuntimeError("ConformalIntervals has not been fitted")
        predictions = np.asarray(y_pred, dtype=float)
        return predictions - self.radius_, predictions + self.radius_
