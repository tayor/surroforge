from __future__ import annotations

import pandas as pd
import pytest

from surroforge import DesignSpace
from surroforge.design import DesignSpaceError
from surroforge.sampling import from_csv, grid, lhs, random, sobol


def make_space() -> DesignSpace:
    space = DesignSpace()
    space.float("x", 0.0, 1.0, description="continuous variable")
    space.integer("n", 1, 3)
    space.categorical("material", ["copper", "steel"])
    space.add_constraint("x * n <= 2.5")
    return space


def test_design_space_validation_and_roundtrip(tmp_path):
    space = make_space()
    params = space.validate({"x": 0.4, "n": 2, "material": "steel"})
    assert params == {"x": 0.4, "n": 2, "material": "steel"}
    assert space.encode(params).tolist() == [0.4, 2.0, 1.0]

    with pytest.raises(DesignSpaceError):
        space.validate({"x": 2.0, "n": 2, "material": "steel"})

    yaml_path = tmp_path / "space.yaml"
    json_path = tmp_path / "space.json"
    space.to_yaml(yaml_path)
    space.to_json(json_path)

    assert DesignSpace.from_yaml(yaml_path).names == ["x", "n", "material"]
    assert DesignSpace.from_json(json_path).to_dict() == space.to_dict()


def test_sampling_methods_generate_valid_samples(tmp_path):
    space = make_space()
    for sampler in [random, sobol, lhs]:
        samples = sampler(8, space, seed=123)
        assert len(samples) == 8
        assert all(space.contains(sample) for sample in samples)

    grid_samples = grid(2, space)
    assert grid_samples
    assert all(space.contains(sample) for sample in grid_samples)

    csv_path = tmp_path / "samples.csv"
    pd.DataFrame(
        [
            {"x": 0.1, "n": 1, "material": "copper"},
            {"x": 0.2, "n": 2, "material": "steel"},
        ]
    ).to_csv(csv_path, index=False)
    assert from_csv(csv_path, space) == [
        {"x": 0.1, "n": 1, "material": "copper"},
        {"x": 0.2, "n": 2, "material": "steel"},
    ]
