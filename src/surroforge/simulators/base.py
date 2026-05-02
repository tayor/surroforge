"""Base simulator interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SimulatorError(RuntimeError):
    """Raised when a simulator fails to prepare, run, or collect results."""


class Simulator(ABC):
    """Minimal adapter contract for external or in-process simulators."""

    params_filename = "params.json"

    def prepare(self, params: Mapping[str, Any], workdir: str | Path) -> None:
        """Prepare a work directory for a simulation run."""
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.params_filename).write_text(
            json.dumps(dict(params), indent=2) + "\n",
            encoding="utf-8",
        )

    def run(self, workdir: str | Path) -> None:
        """Execute the simulation."""
        return None

    @abstractmethod
    def collect(self, workdir: str | Path) -> dict[str, Any]:
        """Collect output values from a completed run."""

    def execute(self, params: Mapping[str, Any], workdir: str | Path) -> dict[str, Any]:
        """Prepare, run, and collect one simulation."""
        self.prepare(params, workdir)
        self.run(workdir)
        outputs = self.collect(workdir)
        if not isinstance(outputs, dict):
            raise SimulatorError("simulator collect() must return a dictionary")
        return outputs

    @classmethod
    def read_params(cls, workdir: str | Path) -> dict[str, Any]:
        """Read the standard params file from a work directory."""
        return json.loads((Path(workdir) / cls.params_filename).read_text(encoding="utf-8"))


def ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """Validate simulator output as a dictionary."""
    if not isinstance(value, Mapping):
        raise SimulatorError(f"{context} must return a mapping, got {type(value).__name__}")
    return dict(value)
