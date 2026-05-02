"""Local on-disk run store."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

from surroforge.design import DesignSpace
from surroforge.schemas import RunRecord, RunStatus


class RunStore:
    """Append-friendly local store for samples, run metadata, and datasets."""

    records_filename = "records.jsonl"

    def __init__(self, root: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for child in ["artifacts", "logs", "models", "reports"]:
            (self.root / child).mkdir(exist_ok=True)
        forge_yaml = self.root / "forge.yaml"
        if metadata is not None or not forge_yaml.exists():
            forge_yaml.write_text(
                yaml.safe_dump(dict(metadata or {}), sort_keys=False),
                encoding="utf-8",
            )
        self._records_path.touch(exist_ok=True)

    @property
    def _records_path(self) -> Path:
        return self.root / self.records_filename

    def add_samples(
        self,
        samples: Iterable[Mapping[str, Any]],
        *,
        seed: int | None = None,
    ) -> list[str]:
        """Add pending samples and return new run IDs."""
        records = self.records()
        existing = {_hash_params(record.params) for record in records}
        new_records: list[RunRecord] = []
        now = _utc_now()
        for params in samples:
            params_dict = dict(params)
            params_hash = _hash_params(params_dict)
            if params_hash in existing:
                continue
            run_id = f"run-{len(records) + len(new_records) + 1:05d}"
            new_records.append(
                RunRecord(
                    run_id=run_id,
                    params=params_dict,
                    status=RunStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                    seed=seed,
                    metadata={"params_hash": params_hash},
                )
            )
            existing.add(params_hash)
        if new_records:
            self._write_records([*records, *new_records])
            self.write_samples_csv()
        return [record.run_id for record in new_records]

    def records(self) -> list[RunRecord]:
        """Return all records in insertion order."""
        records = []
        text = self._records_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                records.append(RunRecord.model_validate_json(line))
        return records

    def pending(self) -> list[RunRecord]:
        """Return pending records."""
        return [record for record in self.records() if record.status == RunStatus.PENDING]

    def completed(self) -> list[RunRecord]:
        """Return completed records."""
        return [record for record in self.records() if record.status == RunStatus.COMPLETED]

    def mark_running(self, run_id: str) -> None:
        """Mark a run as running."""
        self.update(run_id, status=RunStatus.RUNNING, failure_reason=None)

    def mark_completed(
        self,
        run_id: str,
        outputs: Mapping[str, Any],
        runtime_seconds: float,
    ) -> None:
        """Mark a run as completed."""
        self.update(
            run_id,
            status=RunStatus.COMPLETED,
            outputs=dict(outputs),
            failure_reason=None,
            runtime_seconds=runtime_seconds,
        )
        self.write_samples_csv()
        self.write_dataset_h5()

    def mark_failed(self, run_id: str, reason: str, runtime_seconds: float | None = None) -> None:
        """Mark a run as failed."""
        self.update(
            run_id,
            status=RunStatus.FAILED,
            failure_reason=reason,
            runtime_seconds=runtime_seconds,
        )
        self.write_samples_csv()

    def update(self, run_id: str, **changes: Any) -> None:
        """Update one record by run ID."""
        records = self.records()
        for index, record in enumerate(records):
            if record.run_id == run_id:
                data = record.model_dump()
                data.update(changes)
                data["updated_at"] = _utc_now()
                records[index] = RunRecord(**data)
                self._write_records(records)
                return
        raise KeyError(f"unknown run_id: {run_id}")

    def workdir(self, run_id: str) -> Path:
        """Return the artifact directory for one run."""
        path = self.root / "artifacts" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def as_frame(self) -> pd.DataFrame:
        """Return records as a flat DataFrame."""
        rows = []
        for record in self.records():
            row: dict[str, Any] = {
                "run_id": record.run_id,
                "status": record.status,
                "runtime_seconds": record.runtime_seconds,
                "failure_reason": record.failure_reason,
            }
            row.update({f"param.{key}": value for key, value in record.params.items()})
            row.update({f"output.{key}": value for key, value in record.outputs.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def write_samples_csv(self) -> Path:
        """Write a human-readable samples CSV snapshot."""
        path = self.root / "samples.csv"
        self.as_frame().to_csv(path, index=False)
        return path

    def write_dataset_h5(self) -> Path | None:
        """Write completed scalar outputs to HDF5 for downstream training."""
        completed = self.completed()
        if not completed:
            return None
        input_names, inputs = self._encoded_inputs(completed)
        output_names = [name for name, value in completed[0].outputs.items() if _is_scalar(value)]
        if not output_names:
            return None
        outputs = np.asarray(
            [[record.outputs[name] for name in output_names] for record in completed], dtype=float
        )
        path = self.root / "dataset.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("inputs", data=inputs)
            handle.create_dataset("outputs", data=outputs)
            handle.attrs["input_names"] = json.dumps(input_names)
            handle.attrs["output_names"] = json.dumps(output_names)
        return path

    def _write_records(self, records: Iterable[RunRecord]) -> None:
        payload = "".join(record.model_dump_json() + "\n" for record in records)
        self._records_path.write_text(payload, encoding="utf-8")

    def _encoded_inputs(self, records: list[RunRecord]) -> tuple[list[str], np.ndarray]:
        metadata = yaml.safe_load((self.root / "forge.yaml").read_text(encoding="utf-8")) or {}
        if "design_space" in metadata:
            space = DesignSpace.from_dict(metadata["design_space"])
            return space.names, space.encode_many(record.params for record in records)
        input_names = list(records[0].params)
        return input_names, np.asarray(
            [[record.params[name] for name in input_names] for record in records], dtype=float
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_params(params: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(params), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_scalar(value: Any) -> bool:
    return isinstance(value, int | float | np.number) and not isinstance(value, bool)
