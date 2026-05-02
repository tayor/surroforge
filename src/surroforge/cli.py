"""Command-line interface for SurroForge."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console

from surroforge import __version__
from surroforge.design import DesignSpace
from surroforge.forge import Forge
from surroforge.models import MLPRegressor
from surroforge.sampling import resolve_sampler
from surroforge.simulators import PythonCallableSimulator
from surroforge.store import RunStore

app = typer.Typer(
    help="Forge neural surrogates from expensive physics simulations.",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    """SurroForge command group."""
    if version:
        console.print(f"surroforge {__version__}")
        raise typer.Exit()


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Project directory to create.")],
    directory: Annotated[Path, typer.Option("--directory", "-d")] = Path("."),
) -> None:
    """Create a small SurroForge project skeleton."""
    root = directory / name
    root.mkdir(parents=True, exist_ok=True)
    design_space = DesignSpace()
    design_space.float("thickness", 0.5, 5.0, unit="mm")
    design_space.float("power", 10.0, 200.0, unit="W")
    design_space.to_yaml(root / "design_space.yaml")
    (root / "simulator.py").write_text(_SIMULATOR_TEMPLATE, encoding="utf-8")
    (root / "surroforge.yaml").write_text(
        yaml.safe_dump(
            {
                "design_space": "design_space.yaml",
                "simulator": "simulator:simulate",
                "store": "runs/default",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    console.print(f"Created SurroForge project at {root}")


@app.command()
def sample(
    n: Annotated[int, typer.Option("--n", min=1)],
    method: Annotated[str, typer.Option("--method")] = "sobol",
    design_space: Annotated[Path, typer.Option("--design-space")] = Path("design_space.yaml"),
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Generate pending samples into a run store."""
    space = DesignSpace.from_yaml(design_space)
    sampler = resolve_sampler(method)
    samples = sampler(n, space, seed=seed)
    run_store = RunStore(store, metadata={"design_space": space.to_dict()})
    run_ids = run_store.add_samples(samples, seed=seed)
    console.print(f"Added {len(run_ids)} samples to {store}")


@app.command()
def run(
    callable_path: Annotated[str, typer.Option("--callable", help="module:function simulator")],
    design_space: Annotated[Path, typer.Option("--design-space")] = Path("design_space.yaml"),
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Run pending samples with a Python callable simulator."""
    forge = Forge(
        DesignSpace.from_yaml(design_space),
        PythonCallableSimulator(_load_callable(callable_path)),
        store,
    )
    records = forge.run_pending(workers=workers)
    completed = sum(record.status == "completed" for record in records)
    failed = sum(record.status == "failed" for record in records)
    console.print(f"Completed {completed} runs; failed {failed}")


@app.command()
def train(
    design_space: Annotated[Path, typer.Option("--design-space")] = Path("design_space.yaml"),
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 250,
    ensemble: Annotated[int | None, typer.Option("--ensemble", min=1)] = None,
) -> None:
    """Train a PyTorch MLP surrogate from completed runs."""
    forge = Forge(
        DesignSpace.from_yaml(design_space),
        PythonCallableSimulator(lambda params: {}),
        store,
    )
    forge.train(epochs=epochs, ensemble=ensemble)
    console.print(f"Saved model under {store / 'models'}")


@app.command()
def validate(
    design_space: Annotated[Path, typer.Option("--design-space")] = Path("design_space.yaml"),
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
) -> None:
    """Validate the latest MLP checkpoint against completed runs."""
    forge = _forge_with_latest_model(design_space, store)
    metrics = forge.validate()
    for key, value in metrics.items():
        console.print(f"{key}: {value}")


@app.command()
def active(
    design_space: Annotated[Path, typer.Option("--design-space")] = Path("design_space.yaml"),
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
    rounds: Annotated[int, typer.Option("--rounds", min=1)] = 1,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 16,
    candidates: Annotated[int, typer.Option("--candidates", min=1)] = 1_000,
) -> None:
    """Suggest active-learning samples without running them."""
    forge = _forge_with_latest_model(design_space, store, required=False)
    samples = forge.active_learn(
        rounds=rounds,
        batch_size=batch_size,
        candidates=candidates,
        run=False,
    )
    console.print(f"Added {len(samples)} active-learning samples")


@app.command()
def predict(
    params: Annotated[Path, typer.Argument(help="YAML file containing parameter values.")],
    design_space: Annotated[Path, typer.Option("--design-space")] = Path("design_space.yaml"),
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
) -> None:
    """Predict with the latest MLP checkpoint."""
    forge = _forge_with_latest_model(design_space, store)
    payload = yaml.safe_load(params.read_text(encoding="utf-8")) or {}
    prediction = forge.predict(payload)
    console.print(prediction)


@app.command()
def report(store: Annotated[Path, typer.Option("--store")] = Path("runs/default")) -> None:
    """Generate a dataset report."""
    run_store = RunStore(store)
    paths = __import__("surroforge.report", fromlist=["write_report"]).write_report(
        store / "reports", records=run_store.records()
    )
    console.print(f"Wrote {paths['markdown']} and {paths['html']}")


@app.command()
def export(
    output: Annotated[Path, typer.Argument(help="Destination model path.")],
    store: Annotated[Path, typer.Option("--store")] = Path("runs/default"),
) -> None:
    """Copy the latest MLP checkpoint to a destination."""
    source = store / "models" / "last.pt"
    if not source.exists():
        raise typer.BadParameter(f"no model checkpoint at {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    console.print(f"Exported {output}")


def _load_callable(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise typer.BadParameter("callable must use module:function syntax")
    module = importlib.import_module(module_name)
    function = getattr(module, attribute)
    if not callable(function):
        raise typer.BadParameter(f"{path!r} is not callable")
    return function


def _forge_with_latest_model(
    design_space: Path,
    store: Path,
    *,
    required: bool = True,
) -> Forge:
    forge = Forge(
        DesignSpace.from_yaml(design_space),
        PythonCallableSimulator(lambda params: {}),
        store,
    )
    checkpoint = store / "models" / "last.pt"
    if checkpoint.exists():
        model = MLPRegressor.load(checkpoint)
        forge.model = model
        forge.output_names_ = model.output_names
        completed = forge.store.completed()
        if completed:
            forge.training_inputs_ = forge.design_space.encode_many(
                record.params for record in completed
            )
        else:
            forge.training_inputs_ = None
    elif required:
        raise typer.BadParameter(f"no model checkpoint at {checkpoint}")
    return forge


_SIMULATOR_TEMPLATE = """def simulate(params):
    thickness = params["thickness"]
    power = params["power"]
    return {
        "max_temp": 40.0 + 2.5 * power / thickness,
        "stress": 100.0 * thickness**2,
    }
"""


if __name__ == "__main__":
    app()
