"""Ensemble regressor for uncertainty estimation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from surroforge.models.mlp import MLPRegressor


class EnsembleRegressor:
    """Fit multiple regressors and estimate uncertainty from disagreement."""

    def __init__(
        self,
        members: int = 5,
        *,
        model_factory: Callable[..., Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if members <= 0:
            raise ValueError("members must be positive")
        self.members = members
        self.model_factory = model_factory or MLPRegressor
        self.model_kwargs = dict(model_kwargs or {})
        self.models_: list[Any] = []
        self.output_names: list[str] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        output_names: list[str] | None = None,
    ) -> EnsembleRegressor:
        """Fit every ensemble member."""
        self.models_ = []
        self.output_names = output_names or [f"y{i}" for i in range(np.asarray(y).shape[1])]
        for index in range(self.members):
            kwargs = dict(self.model_kwargs)
            kwargs.setdefault("seed", 10_000 + index)
            model = self.model_factory(**kwargs)
            model.fit(x, y, output_names=self.output_names)
            self.models_.append(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict ensemble mean."""
        mean, _std = self.predict_with_uncertainty(x)
        return mean

    def predict_with_uncertainty(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and standard deviation across ensemble members."""
        if not self.models_:
            raise RuntimeError("EnsembleRegressor has not been fitted")
        predictions = np.stack([model.predict(x) for model in self.models_], axis=0)
        return predictions.mean(axis=0), predictions.std(axis=0)

    def save(self, path: str | Path) -> Path:
        """Save each model under a directory."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for index, model in enumerate(self.models_):
            model.save(directory / f"member-{index:02d}.pt")
        return directory
