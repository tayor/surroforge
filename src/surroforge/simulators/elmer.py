"""Elmer FEM adapter built around repeatable case directories."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from surroforge.simulators.base import SimulatorError
from surroforge.simulators.subprocess import SubprocessSimulator


class PyElmerSimulator(SubprocessSimulator):
    """Prepare an Elmer case with a factory, run ElmerSolver, and collect files."""

    def __init__(
        self,
        *,
        case_factory: str | Callable[[Mapping[str, Any], Path], Any],
        outputs: Mapping[str, str],
        command: str = "ElmerSolver case.sif",
        timeout: float | None = None,
    ) -> None:
        super().__init__(command=command, timeout=timeout)
        self.case_factory = case_factory
        self.outputs = dict(outputs)

    def prepare(self, params: Mapping[str, Any], workdir: str | Path) -> None:
        super().prepare(params, workdir)
        factory = self._resolve_factory()
        factory(dict(params), Path(workdir))

    def collect(self, workdir: str | Path) -> dict[str, Any]:
        directory = Path(workdir)
        collected: dict[str, Any] = {}
        for name, relative_path in self.outputs.items():
            path = directory / relative_path
            if not path.exists():
                raise SimulatorError(f"expected Elmer output '{name}' at {path}")
            collected[name] = _read_output_value(path)
        return collected

    def _resolve_factory(self) -> Callable[[Mapping[str, Any], Path], Any]:
        if callable(self.case_factory):
            return self.case_factory
        module_name, _, attribute = self.case_factory.partition(":")
        if not module_name or not attribute:
            raise SimulatorError("case_factory must be a callable or 'module:callable' string")
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
        if not callable(factory):
            raise SimulatorError(f"case factory {self.case_factory!r} is not callable")
        return factory


def _read_output_value(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if frame.shape == (1, 1):
            value = frame.iloc[0, 0]
            return value.item() if hasattr(value, "item") else value
        return frame.to_dict(orient="records")
    return str(path)
