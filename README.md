# SurroForge

Forge neural surrogates from expensive physics simulators.

SurroForge is a simulator-first Python library for turning slow engineering and physics workflows into reusable, uncertainty-aware surrogate models. Define a design space, sample it, run a wrapped simulator, train a PyTorch surrogate, estimate trust, actively choose the next simulations, and export a fast predictor.


## Install

Base install:

```bash
pip install surroforge
```

Recommended local development setup with `uv`:

```bash
uv venv --python 3.11 .venv
uv sync --extra dev
```

Optional extras:

```bash
pip install "surroforge[torch]"   # PyTorch MLP and ensembles
pip install "surroforge[elmer]"   # pyelmer/gmsh adapter support
pip install "surroforge[vtk]"     # mesh and VTU tooling
pip install "surroforge[sklearn]" # scikit-learn wrapper
pip install "surroforge[all]"
```

Elmer itself is intentionally not bundled. If `ElmerSolver` is already installed on your machine, use `PyElmerSimulator` to standardize case setup, execution, logs, and output collection.

## Quick Start

```python
from surroforge import DesignSpace, Forge
from surroforge.models import MLPRegressor
from surroforge.simulators import PythonCallableSimulator


def expensive_sim(params):
    thickness = params["thickness"]
    power = params["power"]
    return {
        "max_temp": 40.0 + 2.5 * power / thickness,
        "stress": 100.0 * thickness**2,
    }


space = DesignSpace()
space.float("thickness", low=0.5, high=5.0, unit="mm")
space.float("power", low=10.0, high=200.0, unit="W")

forge = Forge(
    design_space=space,
    simulator=PythonCallableSimulator(expensive_sim),
    store="./runs/thermal_demo",
)

forge.sample(n=64, method="sobol", seed=7)
forge.run_pending()
forge.train(model=MLPRegressor(hidden=[128, 128], epochs=300))
metrics = forge.validate()
prediction = forge.predict({"thickness": 0.7, "power": 180.0}, safe=True)
forge.export("./surrogate.pt")

print(metrics)
print(prediction.values, prediction.uncertainty, prediction.trust_level)
```

The base package can sample, run, store, and report without PyTorch installed. Training with `MLPRegressor` or `EnsembleRegressor` requires the `torch` extra.

## Design Spaces

```python
space = DesignSpace()
space.float("thickness", 0.5, 5.0, unit="mm")
space.integer("fins", 0, 12)
space.categorical("material", ["copper", "aluminum", "steel"])
space.add_constraint("thickness * (fins + 1) <= 40")
space.to_yaml("design_space.yaml")
```

Supported parameter types:

- Continuous floats
- Integers
- Categoricals
- Units and descriptions
- Constraint expressions
- JSON and YAML serialization

## Sampling

```python
forge.sample(100, method="random", seed=1)
forge.sample(100, method="sobol", seed=2)
forge.sample(100, method="lhs", seed=3)
```

The sampling module also supports small Cartesian grids and user-supplied CSV files.

## Simulators

All simulators follow the same contract:

```python
class Simulator:
    def prepare(self, params, workdir): ...
    def run(self, workdir): ...
    def collect(self, workdir): ...
```

Included adapters:

- `PythonCallableSimulator` for pure Python functions
- `SubprocessSimulator` for external commands that write JSON outputs
- `CSVReplaySimulator` for tests, tutorials, and offline datasets
- `PyElmerSimulator` for pyelmer/Elmer case factories

`SubprocessSimulator` captures stdout/stderr, enforces timeouts, and escalates from `SIGTERM` to `SIGKILL` for process groups that do not exit cleanly.

## Elmer Adapter

```python
from surroforge.simulators.elmer import PyElmerSimulator

sim = PyElmerSimulator(
    case_factory="examples.elmer_thermal_plate.build_case:build_case",
    outputs={
        "max_temp": "results/max_temperature.csv",
        "field": "results/temperature.vtu",
    },
    command="ElmerSolver case.sif",
    timeout=120,
)
```

The case factory receives `(params, workdir)` and writes a valid Elmer case folder. SurroForge owns run IDs, metadata, logs, output parsing, retries at the workflow level, and datasets.

## CLI

```bash
surroforge init thermal-plate
cd thermal-plate
surroforge sample --n 100 --method sobol --seed 7
surroforge run --callable simulator:simulate --workers 4
surroforge train --epochs 300
surroforge validate
surroforge active --rounds 2 --batch-size 8 --candidates 1000
surroforge predict params.yaml
surroforge report
surroforge export model.pt
```

The CLI uses Typer and Rich. It is intentionally file-based, so it works cleanly with local scripts, CI, and solver work directories.

## Data Store

Each `Forge` store contains:

```text
runs/default/
  forge.yaml
  records.jsonl
  samples.csv
  dataset.h5
  artifacts/
  logs/
  models/
  reports/
```

The JSONL records are append-friendly and human-auditable. `samples.csv` is a convenient flat snapshot. `dataset.h5` stores completed scalar datasets for training.

## Trust Layer

SurroForge predictions include:

- Bounds and near-boundary warnings
- Distance-to-training-data score
- Ensemble disagreement when using `EnsembleRegressor`
- A coarse trust level: `high`, `medium`, or `low`

The surrogate should say when it is uncertain. That is part of the product, not an afterthought.

## Development

```bash
uv venv --python 3.11 .venv
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src/surroforge
uv build
uv run twine check dist/*
```

Torch tests are marked with `@pytest.mark.torch` and are skipped automatically when torch is not installed. They use tiny datasets and very small epoch counts so the default suite finishes in minutes.


## License

MIT. See [LICENSE](LICENSE).
