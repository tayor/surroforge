"""Candidate scoring for active learning."""

from __future__ import annotations

import numpy as np


def score_candidates(
    candidate_x: np.ndarray,
    *,
    model=None,
    training_x: np.ndarray | None = None,
    acquisition: str = "max_uncertainty",
) -> np.ndarray:
    """Score candidate feature vectors for acquisition."""
    candidates = np.asarray(candidate_x, dtype=float)
    if (
        acquisition == "max_uncertainty"
        and model is not None
        and hasattr(model, "predict_with_uncertainty")
    ):
        _mean, std = model.predict_with_uncertainty(candidates)
        return np.asarray(std).mean(axis=1)
    if acquisition == "boundary_exploration":
        minimum = candidates.min(axis=0)
        maximum = candidates.max(axis=0)
        span = np.where(maximum - minimum < 1e-12, 1.0, maximum - minimum)
        normalized = (candidates - minimum) / span
        return np.min(np.abs(normalized - 0.5), axis=1)
    if acquisition in {"diversity", "max_uncertainty"}:
        return _diversity_scores(candidates, training_x)
    raise ValueError(f"unknown acquisition strategy: {acquisition}")


def select_batch(
    candidate_x: np.ndarray,
    *,
    batch_size: int,
    model=None,
    training_x: np.ndarray | None = None,
    acquisition: str = "max_uncertainty",
) -> list[int]:
    """Select candidate row indices for a simulation batch."""
    if batch_size <= 0:
        return []
    scores = score_candidates(
        candidate_x,
        model=model,
        training_x=training_x,
        acquisition=acquisition,
    )
    order = np.argsort(scores)[::-1]
    return [int(index) for index in order[:batch_size]]


def _diversity_scores(candidate_x: np.ndarray, training_x: np.ndarray | None) -> np.ndarray:
    if training_x is None or len(training_x) == 0:
        center = candidate_x.mean(axis=0, keepdims=True)
        return np.linalg.norm(candidate_x - center, axis=1)
    train = np.asarray(training_x, dtype=float)
    scale = np.where(train.std(axis=0) < 1e-12, 1.0, train.std(axis=0))
    distances = np.linalg.norm((candidate_x[:, None, :] - train[None, :, :]) / scale, axis=2)
    return distances.min(axis=1)
