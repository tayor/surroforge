"""Model protocols and shared errors."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ModelNotFittedError(RuntimeError):
    """Raised when prediction is requested before fitting a model."""


class SurrogateModel(Protocol):
    """Small protocol implemented by SurroForge regressors."""

    output_names: list[str]

    def fit(
        self, x: np.ndarray, y: np.ndarray, *, output_names: list[str] | None = None
    ) -> SurrogateModel: ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def save(self, path: str | Path) -> Path: ...
