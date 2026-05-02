"""High-level SurroForge orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from surroforge.active import select_batch
from surroforge.design import DesignSpace
from surroforge.metrics import regression_metrics
from surroforge.models import EnsembleRegressor, MLPRegressor
from surroforge.report import write_report
from surroforge.sampling import random as random_samples
from surroforge.sampling import resolve_sampler
from surroforge.schemas import Prediction, RunRecord
from surroforge.simulators import Simulator
from surroforge.store import RunStore
from surroforge.trust import bounds_warnings, distance_to_training_score, trust_level


class Forge:
    """End-to-end simulator-to-surrogate workflow manager."""

    def __init__(self, design_space: DesignSpace, simulator: Simulator, store: str | Path) -> None:
        self.design_space = design_space
        self.simulator = simulator
        self.store = RunStore(
            store,
            metadata={
                "package": "surroforge",
                "design_space": self.design_space.to_dict(),
            },
        )
        self.model: Any | None = None
        self.training_inputs_: np.ndarray | None = None
        self.output_names_: list[str] = []
        self._store_lock = Lock()

    def sample(
        self,
        n: int,
        method: str | Callable[..., list[dict[str, Any]]] = "sobol",
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate and register pending simulation samples."""
        sampler = resolve_sampler(method)
        try:
            samples = sampler(n, self.design_space, seed=seed, **kwargs)
        except TypeError:
            samples = sampler(n, self.design_space)
        self.store.add_samples(samples, seed=seed)
        return samples

    def run_pending(self, *, workers: int = 1, stop_on_error: bool = False) -> list[RunRecord]:
        """Run all pending simulations."""
        pending = self.store.pending()
        if workers <= 1:
            for record in pending:
                self._run_one(record, stop_on_error=stop_on_error)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(self._run_one, record, stop_on_error) for record in pending
                ]
                for future in as_completed(futures):
                    future.result()
        return self.store.records()

    def train(
        self,
        model: str | Any = "mlp",
        *,
        output_names: list[str] | None = None,
        ensemble: int | None = None,
        **model_kwargs: Any,
    ) -> Any:
        """Train a surrogate model from completed scalar outputs."""
        x, y, names = self.dataset(output_names=output_names)
        model_obj: Any
        if ensemble is not None and ensemble > 1:
            model_obj = EnsembleRegressor(members=ensemble, model_kwargs=model_kwargs)
        elif model == "mlp":
            model_obj = MLPRegressor(**model_kwargs)
        elif isinstance(model, str):
            raise ValueError(f"unknown model: {model}")
        else:
            model_obj = model
        model_obj.fit(x, y, output_names=names)
        self.model = model_obj
        self.training_inputs_ = x
        self.output_names_ = names
        model_path = (
            self.store.root / "models" / ("last" if ensemble and ensemble > 1 else "last.pt")
        )
        model_obj.save(model_path)
        return model_obj

    def dataset(
        self, *, output_names: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return numeric training arrays from completed runs."""
        records = self.store.completed()
        if not records:
            raise RuntimeError("no completed simulation runs are available")
        names = output_names or [
            name
            for name, value in records[0].outputs.items()
            if isinstance(value, int | float | np.number) and not isinstance(value, bool)
        ]
        if not names:
            raise RuntimeError("no scalar outputs are available for training")
        x = self.design_space.encode_many(record.params for record in records)
        y = np.asarray(
            [[record.outputs[name] for name in names] for record in records], dtype=float
        )
        return x, y, names

    def validate(self) -> dict[str, float]:
        """Evaluate the fitted model on completed runs and write reports."""
        if self.model is None:
            raise RuntimeError("train or assign a model before validate()")
        x, y, _names = self.dataset(output_names=self.output_names_ or None)
        predictions = self.model.predict(x)
        metrics = regression_metrics(y, predictions)
        write_report(self.store.root / "reports", records=self.store.records(), metrics=metrics)
        return metrics

    def predict(self, params: Mapping[str, Any], *, safe: bool = True) -> Prediction:
        """Predict with optional trust warnings."""
        if self.model is None:
            raise RuntimeError("train or assign a model before predict()")
        warnings = bounds_warnings(self.design_space, params) if safe else []
        validated = self.design_space.validate(params)
        x = self.design_space.encode(validated)[None, :]
        if hasattr(self.model, "predict_with_uncertainty"):
            mean, std = self.model.predict_with_uncertainty(x)
            prediction = mean[0]
            uncertainty = std[0]
        else:
            prediction = self.model.predict(x)[0]
            uncertainty = np.zeros_like(prediction)
        distance_score = distance_to_training_score(x, self.training_inputs_) if safe else None
        values = {name: float(prediction[index]) for index, name in enumerate(self.output_names_)}
        uncertainty_values = {
            name: float(uncertainty[index]) for index, name in enumerate(self.output_names_)
        }
        return Prediction(
            values=values,
            uncertainty=uncertainty_values,
            trust_level=trust_level(
                warnings=warnings,
                distance_score=distance_score,
                uncertainty=float(np.mean(uncertainty)) if uncertainty.size else None,
            ),
            warnings=warnings,
            metadata={"distance_to_training": distance_score},
        )

    def active_learn(
        self,
        *,
        rounds: int = 1,
        candidates: int = 10_000,
        batch_size: int = 16,
        acquisition: str = "max_uncertainty",
        run: bool = True,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Suggest and optionally run active-learning samples."""
        selected: list[dict[str, Any]] = []
        for round_index in range(rounds):
            candidate_samples = random_samples(
                candidates,
                self.design_space,
                seed=None if seed is None else seed + round_index,
            )
            candidate_x = self.design_space.encode_many(candidate_samples)
            training_x = self.training_inputs_
            if training_x is None:
                completed = self.store.completed()
                training_x = (
                    self.design_space.encode_many(record.params for record in completed)
                    if completed
                    else None
                )
            indices = select_batch(
                candidate_x,
                batch_size=batch_size,
                model=self.model,
                training_x=training_x,
                acquisition=acquisition,
            )
            batch = [candidate_samples[index] for index in indices]
            self.store.add_samples(batch, seed=seed)
            selected.extend(batch)
            if run:
                self.run_pending()
        return selected

    def export(self, path: str | Path) -> Path:
        """Export the trained model."""
        if self.model is None:
            raise RuntimeError("train or assign a model before export()")
        return self.model.save(path)

    def report(self) -> dict[str, Path]:
        """Write a dataset-only report."""
        return write_report(self.store.root / "reports", records=self.store.records())

    def _run_one(self, record: RunRecord, stop_on_error: bool = False) -> None:
        start = time.perf_counter()
        with self._store_lock:
            self.store.mark_running(record.run_id)
        try:
            outputs = self.simulator.execute(record.params, self.store.workdir(record.run_id))
        except Exception as exc:
            elapsed = time.perf_counter() - start
            with self._store_lock:
                self.store.mark_failed(record.run_id, str(exc), runtime_seconds=elapsed)
            if stop_on_error:
                raise
            return
        elapsed = time.perf_counter() - start
        with self._store_lock:
            self.store.mark_completed(record.run_id, outputs, runtime_seconds=elapsed)
