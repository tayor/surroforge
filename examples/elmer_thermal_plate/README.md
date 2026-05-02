# Elmer Thermal Plate Example

This folder sketches the first Elmer integration target. It is intentionally not part of the default fast test suite because real FEM runs can take longer and depend on local solver configuration.

```python
from surroforge import DesignSpace, Forge
from surroforge.simulators.elmer import PyElmerSimulator

space = DesignSpace()
space.float("thickness", 0.5, 5.0, unit="mm")
space.float("power", 25.0, 200.0, unit="W")

sim = PyElmerSimulator(
    case_factory="build_case:build_case",
    outputs={"max_temperature": "results/max_temperature.csv"},
    command="ElmerSolver case.sif",
    timeout=120,
)

forge = Forge(space, sim, store="runs/elmer_thermal_plate")
```

Use this as the place to add a real pyelmer case once geometry, mesh density, and result extraction are fixed for your machine.
