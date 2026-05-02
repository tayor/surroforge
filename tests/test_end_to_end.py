from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from surroforge import DesignSpace, Forge
from surroforge.simulators import PythonCallableSimulator
from surroforge.store import RunStore

pytestmark = pytest.mark.e2e


class LeastSquaresRegressor:
    def __init__(self) -> None:
        self.output_names: list[str] = []
        self.coef_: np.ndarray | None = None

    def fit(self, x, y, *, output_names=None):
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim == 1:
            y_array = y_array[:, None]
        design = np.column_stack([np.ones(len(x_array)), x_array])
        self.coef_ = np.linalg.lstsq(design, y_array, rcond=None)[0]
        self.output_names = output_names or [f"y{index}" for index in range(y_array.shape[1])]
        return self

    def predict(self, x):
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            x_array = x_array[None, :]
        design = np.column_stack([np.ones(len(x_array)), x_array])
        return design @ self.coef_

    def save(self, path):
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"output_names": self.output_names, "coef": self.coef_.tolist()}),
            encoding="utf-8",
        )
        return output


def test_public_api_surrogate_workflow_end_to_end(tmp_path):
    space = DesignSpace()
    space.float("thickness", 0.5, 4.0)
    space.float("power", 10.0, 120.0)
    space.categorical("material", ["aluminum", "copper"])

    def simulate(params, workdir):
        material_bonus = 3.0 if params["material"] == "copper" else 0.0
        max_temperature = 20.0 + 0.8 * params["power"] / params["thickness"] - material_bonus
        mass = 2.5 * params["thickness"] + material_bonus
        (workdir / "solver.log").write_text("completed\n", encoding="utf-8")
        return {"max_temperature": max_temperature, "mass": mass}

    forge = Forge(space, PythonCallableSimulator(simulate), store=tmp_path / "runs")
    samples = forge.sample(16, method="lhs", seed=42)
    assert len(samples) == 16

    records = forge.run_pending(workers=2)
    assert len(records) == 16
    assert all(record.status == "completed" for record in records)
    assert (tmp_path / "runs" / "samples.csv").exists()
    assert (tmp_path / "runs" / "dataset.h5").exists()
    assert (tmp_path / "runs" / "artifacts" / "run-00001" / "solver.log").exists()

    model = forge.train(model=LeastSquaresRegressor())
    assert model.output_names == ["max_temperature", "mass"]
    assert (tmp_path / "runs" / "models" / "last.pt").exists()

    metrics = forge.validate()
    assert set(metrics) >= {"mae", "rmse", "r2", "mape", "max_error"}
    assert (tmp_path / "runs" / "reports" / "report.md").exists()
    assert (tmp_path / "runs" / "reports" / "report.html").exists()

    prediction = forge.predict({"thickness": 1.25, "power": 80.0, "material": "copper"}, safe=True)
    assert set(prediction.values) == {"max_temperature", "mass"}
    assert prediction.trust_level in {"high", "medium"}

    active_samples = forge.active_learn(rounds=1, candidates=32, batch_size=4, run=True, seed=100)
    assert len(active_samples) == 4
    assert not forge.store.pending()
    assert sum(record.status == "completed" for record in forge.store.records()) >= 16

    exported = forge.export(tmp_path / "exported-model.json")
    assert exported.exists()

    with h5py.File(tmp_path / "runs" / "dataset.h5", "r") as handle:
        assert handle["inputs"].shape[1] == 3
        assert handle["outputs"].shape[1] == 2
        assert json.loads(handle.attrs["output_names"]) == ["max_temperature", "mass"]


def test_cli_project_workflow_end_to_end(tmp_path):
    project = tmp_path / "thermal"
    run_cli(tmp_path, "init", "thermal", "--directory", str(tmp_path))

    run_cli(
        project,
        "sample",
        "--n",
        "8",
        "--method",
        "sobol",
        "--seed",
        "7",
    )
    run_result = run_cli(project, "run", "--callable", "simulator:simulate", "--workers", "2")
    assert "Completed 8 runs; failed 0" in run_result.stdout

    report_result = run_cli(project, "report")
    assert "report.md" in report_result.stdout

    store = project / "runs" / "default"
    records = RunStore(store).records()
    assert len(records) == 8
    assert all(record.status == "completed" for record in records)
    assert (store / "samples.csv").exists()
    assert (store / "dataset.h5").exists()
    assert (store / "reports" / "report.md").exists()
    assert (store / "reports" / "report.html").exists()


def run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    source_path = str(repo_root / "src")
    env["PYTHONPATH"] = (
        source_path if not env.get("PYTHONPATH") else source_path + os.pathsep + env["PYTHONPATH"]
    )
    result = subprocess.run(
        [sys.executable, "-m", "surroforge.cli", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result
