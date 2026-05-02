from __future__ import annotations

import json
import sys
import time

import pandas as pd
import pytest

from surroforge.simulators import CSVReplaySimulator, PythonCallableSimulator, SimulatorError
from surroforge.simulators.elmer import PyElmerSimulator
from surroforge.simulators.subprocess import SubprocessSimulator


def test_python_callable_simulator(tmp_path):
    def simulate(params, workdir):
        assert workdir.exists()
        return {"y": params["x"] + 1}

    simulator = PythonCallableSimulator(simulate)
    assert simulator.execute({"x": 2}, tmp_path / "run") == {"y": 3}


def test_subprocess_simulator_collects_json(tmp_path):
    script = tmp_path / "simulate.py"
    script.write_text(
        "import json; "
        "params=json.load(open('params.json')); "
        "json.dump({'y': params['x'] * 2}, open('output.json','w'))",
        encoding="utf-8",
    )
    simulator = SubprocessSimulator([sys.executable, str(script)], timeout=5)
    assert simulator.execute({"x": 4}, tmp_path / "run") == {"y": 8}


def test_subprocess_timeout_kills_process_group(tmp_path):
    simulator = SubprocessSimulator(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)",
        ],
        timeout=0.2,
        terminate_grace_seconds=0.1,
    )
    started = time.perf_counter()
    with pytest.raises(SimulatorError, match="timed out"):
        simulator.execute({"x": 1}, tmp_path / "timeout")
    assert time.perf_counter() - started < 3.0


def test_csv_replay_simulator(tmp_path):
    csv_path = tmp_path / "replay.csv"
    pd.DataFrame([{"x": 1.5, "material": "steel", "y": 9.0}]).to_csv(csv_path, index=False)
    simulator = CSVReplaySimulator(csv_path, input_columns=["x", "material"], output_columns=["y"])
    assert simulator.execute({"x": 1.5, "material": "steel"}, tmp_path / "run") == {"y": 9.0}


def test_pyelmer_collects_declared_outputs(tmp_path):
    def build_case(params, workdir):
        del params
        results = workdir / "results"
        results.mkdir()
        (results / "max_temperature.csv").write_text("value\n42.5\n", encoding="utf-8")

    simulator = PyElmerSimulator(
        case_factory=build_case,
        outputs={"max_temperature": "results/max_temperature.csv"},
        command=[sys.executable, "-c", "pass"],
    )
    assert simulator.execute({"power": 100}, tmp_path / "elmer") == {"max_temperature": 42.5}
    assert json.loads((tmp_path / "elmer" / "params.json").read_text())["power"] == 100
