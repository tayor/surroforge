from __future__ import annotations

from pathlib import Path

import numpy as np

from surroforge import DesignSpace, Forge
from surroforge.simulators import PythonCallableSimulator


class MeanModel:
    def __init__(self) -> None:
        self.output_names: list[str] = []
        self.mean: np.ndarray | None = None

    def fit(self, x, y, *, output_names=None):
        assert x.shape[0] == y.shape[0]
        self.output_names = output_names or ["y"]
        self.mean = np.asarray(y, dtype=float).mean(axis=0)
        return self

    def predict(self, x):
        assert self.mean is not None
        return np.tile(self.mean, (len(x), 1))

    def save(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("mean-model\n", encoding="utf-8")
        return output


def make_forge(tmp_path) -> Forge:
    space = DesignSpace()
    space.float("x", 0.0, 1.0)
    space.float("z", 1.0, 2.0)

    def simulate(params):
        return {"y": params["x"] + 2 * params["z"]}

    return Forge(space, PythonCallableSimulator(simulate), store=tmp_path / "runs")


def test_forge_runs_stores_trains_validates_and_predicts(tmp_path):
    forge = make_forge(tmp_path)
    forge.sample(6, method="random", seed=5)
    records = forge.run_pending()
    assert sum(record.status == "completed" for record in records) == 6
    assert (tmp_path / "runs" / "samples.csv").exists()
    assert (tmp_path / "runs" / "dataset.h5").exists()

    model = forge.train(model=MeanModel())
    assert model.output_names == ["y"]
    metrics = forge.validate()
    assert set(metrics) >= {"mae", "rmse", "r2", "mape", "max_error"}
    assert (tmp_path / "runs" / "reports" / "report.md").exists()

    prediction = forge.predict({"x": 0.1, "z": 1.2})
    assert prediction.values.keys() == {"y"}
    assert prediction.trust_level in {"high", "medium"}

    exported = forge.export(tmp_path / "exported-model.txt")
    assert exported.exists()


def test_active_learning_adds_pending_samples(tmp_path):
    forge = make_forge(tmp_path)
    samples = forge.active_learn(rounds=1, candidates=20, batch_size=4, run=False, seed=10)
    assert len(samples) == 4
    assert len(forge.store.pending()) == 4
