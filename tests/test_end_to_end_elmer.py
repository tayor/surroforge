from __future__ import annotations

import importlib
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from surroforge import DesignSpace, Forge
from surroforge.models import MLPRegressor
from surroforge.simulators import PyElmerSimulator

gmsh = pytest.importorskip("gmsh")
pytest.importorskip("torch")
pytest.importorskip("pyelmer")
elmer = importlib.import_module("pyelmer.elmer")
execute = importlib.import_module("pyelmer.execute")

ELMER_SOLVER = shutil.which("ElmerSolver")
ELMER_GRID = shutil.which("ElmerGrid")
if ELMER_SOLVER is None or ELMER_GRID is None:
    pytest.skip("ElmerSolver and ElmerGrid are required for this test", allow_module_level=True)

pytestmark = [pytest.mark.e2e, pytest.mark.elmer, pytest.mark.torch]


def test_pyelmer_neural_workflow_end_to_end(tmp_path: Path) -> None:
    space = DesignSpace()
    space.float("left_temperature", 10.0, 30.0)
    space.float("right_temperature", -5.0, 5.0)

    forge = Forge(space, _make_pyelmer_simulator(), store=tmp_path / "runs")
    forge.store.add_samples(
        {
            "left_temperature": left_temperature,
            "right_temperature": right_temperature,
        }
        for left_temperature in (10.0, 20.0, 30.0)
        for right_temperature in (-5.0, 0.0, 5.0)
    )

    records = forge.run_pending(workers=1)
    assert len(records) == 9
    assert all(record.status == "completed" for record in records)
    assert (tmp_path / "runs" / "dataset.h5").exists()

    model = forge.train(
        model="mlp",
        hidden=[32, 32],
        lr=1e-2,
        epochs=400,
        patience=80,
        batch_size=4,
        seed=7,
        device="cpu",
    )
    assert isinstance(model, MLPRegressor)
    assert model.output_names == ["mean_temperature", "max_temperature"]
    assert model.history_
    checkpoint = tmp_path / "runs" / "models" / "last.pt"
    assert checkpoint.exists()

    metrics = forge.validate()
    assert metrics["mae"] < 0.75
    assert metrics["rmse"] < 1.0
    assert metrics["r2"] > 0.98
    assert (tmp_path / "runs" / "reports" / "report.md").exists()
    assert (tmp_path / "runs" / "reports" / "report.html").exists()

    held_out = {"left_temperature": 18.0, "right_temperature": 1.5}
    actual = _make_pyelmer_simulator().execute(held_out, tmp_path / "held-out")
    assert actual["max_temperature"] == pytest.approx(held_out["left_temperature"], abs=0.25)
    assert actual["mean_temperature"] == pytest.approx(
        0.5 * (held_out["left_temperature"] + held_out["right_temperature"]),
        abs=0.25,
    )

    prediction = forge.predict(held_out, safe=True)
    assert prediction.values["mean_temperature"] == pytest.approx(
        actual["mean_temperature"],
        abs=1.0,
    )
    assert prediction.values["max_temperature"] == pytest.approx(
        actual["max_temperature"],
        abs=1.0,
    )

    restored = MLPRegressor.load(checkpoint, device="cpu")
    restored_outputs = dict(
        zip(restored.output_names, restored.predict(space.encode(held_out))[0], strict=True)
    )
    assert restored_outputs["mean_temperature"] == pytest.approx(
        actual["mean_temperature"],
        abs=1.0,
    )
    assert restored_outputs["max_temperature"] == pytest.approx(
        actual["max_temperature"],
        abs=1.0,
    )


def _make_pyelmer_simulator() -> PyElmerSimulator:
    assert ELMER_SOLVER is not None
    command = (
        f"{shlex.quote(ELMER_SOLVER)} case.sif && "
        f"{shlex.quote(sys.executable)} extract_metrics.py"
    )
    return PyElmerSimulator(
        case_factory=_build_plate_case,
        outputs={
            "mean_temperature": "mean_temperature.csv",
            "max_temperature": "max_temperature.csv",
        },
        command=command,
        timeout=60,
    )


def _build_plate_case(params: dict[str, float], workdir: Path) -> None:
    left_temperature = float(params["left_temperature"])
    right_temperature = float(params["right_temperature"])

    gmsh.initialize()
    try:
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add("thermal_plate")
        factory = gmsh.model.occ
        surface = factory.addRectangle(0.0, 0.0, 0.0, 0.8, 0.2)
        factory.synchronize()

        left_edges = _boundary_tags_for_x(surface, 0.0)
        right_edges = _boundary_tags_for_x(surface, 0.8)
        if not left_edges or not right_edges:
            raise RuntimeError("failed to locate left/right plate boundaries in gmsh")

        body_id = gmsh.model.addPhysicalGroup(2, [surface], 1)
        left_id = gmsh.model.addPhysicalGroup(1, left_edges, 2)
        right_id = gmsh.model.addPhysicalGroup(1, right_edges, 3)
        gmsh.model.setPhysicalName(2, body_id, "body")
        gmsh.model.setPhysicalName(1, left_id, "left")
        gmsh.model.setPhysicalName(1, right_id, "right")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(workdir / "mesh.msh"))
    finally:
        gmsh.finalize()

    execute.run_elmer_grid(str(workdir), "mesh.msh", out_dir=str(workdir))

    simulation = elmer.Simulation()
    simulation.settings.update(
        {
            "Max Output Level": 1,
            "Coordinate System": '"Cartesian 2D"',
            "Coordinate Mapping(3)": "1 2 3",
            "Simulation Type": '"Steady State"',
            "Steady State Max Iterations": 1,
            "Output Intervals": 1,
        }
    )
    heat_solver = elmer.Solver(
        simulation,
        "heat_solver",
        {
            "Equation": '"Heat Equation"',
            "Procedure": '"HeatSolve" "HeatSolver"',
            "Variable": '"Temperature"',
            "Variable DOFs": 1,
            "Stabilize": True,
            "Bubbles": False,
            "Lumped Mass Matrix": False,
            "Optimize Bandwidth": True,
            "Steady State Convergence Tolerance": 1.0e-10,
            "Linear System Solver": '"Direct"',
            "Linear System Direct Method": '"UMFPACK"',
        },
    )
    output_solver = elmer.Solver(
        simulation,
        "result_output",
        {
            "Procedure": '"ResultOutputSolve" "ResultOutputSolver"',
            "Output Format": '"Vtu"',
            "Binary Output": False,
            "Output File Name": '"case"',
            "Scalar Field 1": '"Temperature"',
        },
    )
    equation = elmer.Equation(simulation, "main", [heat_solver, output_solver])
    material = elmer.Material(
        simulation,
        "solid",
        {
            "Heat Conductivity": 2.0,
            "Density": 1.0,
            "Heat Capacity": 1.0,
        },
    )
    body = elmer.Body(simulation, "body", [body_id])
    body.material = material
    body.equation = equation
    elmer.Boundary(simulation, "left", [left_id], {"Temperature": left_temperature})
    elmer.Boundary(simulation, "right", [right_id], {"Temperature": right_temperature})

    simulation.write_startinfo(str(workdir))
    simulation.write_sif(str(workdir))
    _write_metrics_extractor(workdir)


def _boundary_tags_for_x(surface_tag: int, x_target: float, tol: float = 1e-6) -> list[int]:
    tags: list[int] = []
    for dim, tag in gmsh.model.getBoundary([(2, surface_tag)], oriented=False, recursive=False):
        if dim != 1:
            continue
        xmin, _ymin, _zmin, xmax, _ymax, _zmax = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmin - x_target) < tol and abs(xmax - x_target) < tol:
            tags.append(tag)
    return tags


def _write_metrics_extractor(workdir: Path) -> None:
    (workdir / "extract_metrics.py").write_text(
        """
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

root = ET.parse("case_t0001.vtu").getroot()
temperatures = []
for data_array in root.iter("DataArray"):
    if data_array.attrib.get("Name", "").lower() != "temperature":
        continue
    temperatures.extend(float(value) for value in (data_array.text or "").split())

if not temperatures:
    raise RuntimeError("temperature field not found in case_t0001.vtu")

metrics = {
    "mean_temperature.csv": sum(temperatures) / len(temperatures),
    "max_temperature.csv": max(temperatures),
}

for filename, value in metrics.items():
    with Path(filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["value"])
        writer.writerow([f"{value:.10f}"])
""".strip()
        + "\n",
        encoding="utf-8",
    )