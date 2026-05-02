"""Design-space sampling implementations."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from surroforge.design import DesignSpace, DesignSpaceError

Sampler = Callable[..., list[dict[str, Any]]]


def _filter_valid(
    space: DesignSpace,
    unit_samples: np.ndarray,
    n: int,
    *,
    max_attempts: int,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    attempts = 0
    for row in unit_samples:
        attempts += 1
        candidate = space.decode_unit(row)
        if space.contains(candidate):
            accepted.append(candidate)
            if len(accepted) == n:
                return accepted
        if attempts >= max_attempts:
            break
    if len(accepted) < n:
        raise DesignSpaceError(
            f"only generated {len(accepted)} valid samples after {attempts} attempts"
        )
    return accepted


def random(
    n: int,
    space: DesignSpace,
    *,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> list[dict[str, Any]]:
    """Generate random design-space samples."""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    attempts = max_attempts or max(n * 50, 100)
    unit = rng.random((attempts, len(space)))
    return _filter_valid(space, unit, n, max_attempts=attempts)


def sobol(
    n: int,
    space: DesignSpace,
    *,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> list[dict[str, Any]]:
    """Generate Sobol low-discrepancy samples."""
    if n <= 0:
        return []
    attempts = max_attempts or max(n * 8, 16)
    sampler = qmc.Sobol(d=len(space), scramble=True, seed=seed)
    unit = sampler.random(attempts)
    return _filter_valid(space, unit, n, max_attempts=attempts)


def lhs(
    n: int,
    space: DesignSpace,
    *,
    seed: int | None = None,
    max_attempts: int | None = None,
) -> list[dict[str, Any]]:
    """Generate Latin-hypercube samples."""
    if n <= 0:
        return []
    attempts = max_attempts or max(n * 8, 16)
    sampler = qmc.LatinHypercube(d=len(space), seed=seed)
    unit = sampler.random(attempts)
    return _filter_valid(space, unit, n, max_attempts=attempts)


def grid(levels: int, space: DesignSpace) -> list[dict[str, Any]]:
    """Generate a small Cartesian grid over the unit cube."""
    if levels <= 0:
        return []
    points = np.linspace(0.0, np.nextafter(1.0, 0.0), levels)
    samples = []
    for row in itertools.product(points, repeat=len(space)):
        candidate = space.decode_unit(row)
        if space.contains(candidate):
            samples.append(candidate)
    return samples


def from_csv(path: str | Path, space: DesignSpace) -> list[dict[str, Any]]:
    """Load user-supplied samples from a CSV file."""
    frame = pd.read_csv(path)
    samples = []
    for row in frame.to_dict(orient="records"):
        samples.append(space.validate({name: row[name] for name in space.names}))
    return samples


def resolve_sampler(method: str | Sampler) -> Sampler:
    """Resolve a sampler name or callable."""
    if callable(method):
        return method
    registry: dict[str, Sampler] = {
        "random": random,
        "sobol": sobol,
        "lhs": lhs,
        "latin_hypercube": lhs,
        "latin-hypercube": lhs,
    }
    try:
        return registry[method]
    except KeyError as exc:
        raise ValueError(f"unknown sampling method: {method}") from exc
