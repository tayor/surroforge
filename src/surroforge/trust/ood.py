"""Out-of-distribution and bounds checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from surroforge.design import DesignSpace, DesignSpaceError


def bounds_warnings(
    space: DesignSpace,
    params: Mapping[str, Any],
    *,
    boundary_margin: float = 0.05,
) -> list[str]:
    """Return warnings for out-of-bounds or near-boundary inputs."""
    warnings: list[str] = []
    try:
        validated = space.validate(params)
    except DesignSpaceError as exc:
        return [str(exc)]
    for name, parameter in space.parameters.items():
        if parameter.type == "categorical":
            continue
        low, high = parameter.bounds()
        span = max(high - low, 1.0)
        value = float(validated[name])
        normalized = (value - low) / span
        if normalized <= boundary_margin or normalized >= 1.0 - boundary_margin:
            warnings.append(f"{name} is near boundary of training domain")
    return warnings


def distance_to_training_score(x: np.ndarray, training_x: np.ndarray | None) -> float | None:
    """Return nearest-neighbor distance in normalized feature space."""
    if training_x is None or len(training_x) == 0:
        return None
    query = np.asarray(x, dtype=float)
    if query.ndim == 1:
        query = query[None, :]
    train = np.asarray(training_x, dtype=float)
    scale = np.where(train.std(axis=0) < 1e-12, 1.0, train.std(axis=0))
    distances = np.linalg.norm((train[None, :, :] - query[:, None, :]) / scale, axis=2)
    return float(np.min(distances))


def trust_level(
    *,
    warnings: list[str],
    distance_score: float | None = None,
    uncertainty: float | None = None,
) -> str:
    """Map warnings, distance, and uncertainty into a coarse trust level."""
    if warnings and any(
        "outside" in warning or "constraint failed" in warning for warning in warnings
    ):
        return "low"
    if uncertainty is not None and uncertainty > 1.0:
        return "medium"
    if distance_score is not None and distance_score > 3.0:
        return "medium"
    if warnings:
        return "medium"
    return "high"
