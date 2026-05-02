"""scikit-learn model wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class SklearnRegressor:
    """Wrap any scikit-learn style estimator with fit/predict methods."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.output_names: list[str] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        output_names: list[str] | None = None,
    ) -> SklearnRegressor:
        self.output_names = output_names or [f"y{i}" for i in range(np.asarray(y).shape[1])]
        self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.asarray(self.estimator.predict(x), dtype=float)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        return predictions

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump({"estimator": self.estimator, "output_names": self.output_names}, handle)
        return output
