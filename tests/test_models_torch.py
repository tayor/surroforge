from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.torch


def test_mlp_regressor_smoke(tmp_path):
    pytest.importorskip("torch")
    from surroforge.models import MLPRegressor

    rng = np.random.default_rng(123)
    x = rng.random((12, 2))
    y = x[:, :1] * 2.0 + x[:, 1:] * 0.5
    model = MLPRegressor(hidden=[8], epochs=6, patience=3, batch_size=4, seed=123)
    model.fit(x, y, output_names=["temperature"])
    predictions = model.predict(x[:2])
    assert predictions.shape == (2, 1)
    path = model.save(tmp_path / "model.pt")
    loaded = MLPRegressor.load(path)
    assert loaded.predict(x[:1]).shape == (1, 1)


def test_ensemble_regressor_smoke():
    pytest.importorskip("torch")
    from surroforge.models import EnsembleRegressor

    x = np.linspace(0.0, 1.0, 10)[:, None]
    y = 3.0 * x
    model = EnsembleRegressor(
        members=2,
        model_kwargs={"hidden": [4], "epochs": 4, "patience": 2, "batch_size": 5},
    )
    model.fit(x, y, output_names=["y"])
    mean, std = model.predict_with_uncertainty(x[:3])
    assert mean.shape == std.shape == (3, 1)
