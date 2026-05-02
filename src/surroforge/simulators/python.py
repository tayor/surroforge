"""In-process simulator adapters."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from surroforge.simulators.base import Simulator, SimulatorError, ensure_mapping


class PythonCallableSimulator(Simulator):
    """Wrap a Python callable as a simulator."""

    def __init__(
        self,
        function: Callable[..., Mapping[str, Any]],
        *,
        output: str = "output.json",
    ) -> None:
        self.function = function
        self.output = output

    def run(self, workdir: str | Path) -> None:
        directory = Path(workdir)
        params = self.read_params(directory)
        result = self._call(params, directory)
        output_path = directory / self.output
        output_path.write_text(
            json.dumps(ensure_mapping(result, context="PythonCallableSimulator"), indent=2) + "\n",
            encoding="utf-8",
        )

    def collect(self, workdir: str | Path) -> dict[str, Any]:
        output_path = Path(workdir) / self.output
        if not output_path.exists():
            raise SimulatorError(f"expected simulator output at {output_path}")
        return ensure_mapping(
            json.loads(output_path.read_text(encoding="utf-8")),
            context="output JSON",
        )

    def _call(self, params: dict[str, Any], workdir: Path) -> Mapping[str, Any]:
        signature = inspect.signature(self.function)
        if len(signature.parameters) >= 2:
            return self.function(params, workdir)
        return self.function(params)


class CSVReplaySimulator(Simulator):
    """Replay deterministic simulator outputs from a CSV fixture or dataset."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        input_columns: Sequence[str] | None = None,
        output_columns: Sequence[str] | None = None,
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.input_columns = list(input_columns) if input_columns is not None else None
        self.output_columns = list(output_columns) if output_columns is not None else None
        self.rtol = rtol
        self.atol = atol

    def collect(self, workdir: str | Path) -> dict[str, Any]:
        params = self.read_params(workdir)
        frame = pd.read_csv(self.csv_path)
        input_columns = self.input_columns or [
            column for column in params if column in frame.columns
        ]
        output_columns = self.output_columns or [
            column for column in frame.columns if column not in set(input_columns)
        ]
        for row in frame.to_dict(orient="records"):
            if self._matches(row, params, input_columns):
                return {column: row[column] for column in output_columns}
        raise SimulatorError(f"no CSV replay row matched params {params!r}")

    def _matches(
        self,
        row: Mapping[str, Any],
        params: Mapping[str, Any],
        columns: Sequence[str],
    ) -> bool:
        for column in columns:
            if column not in params:
                return False
            left = row[column]
            right = params[column]
            if _is_number(left) and _is_number(right):
                if not math.isclose(
                    float(left),
                    float(right),
                    rel_tol=self.rtol,
                    abs_tol=self.atol,
                ):
                    return False
            elif left != right:
                return False
        return True


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
