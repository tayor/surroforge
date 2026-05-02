"""Small pure-Python thermal plate demo."""

from __future__ import annotations

from pathlib import Path

from surroforge import DesignSpace, Forge
from surroforge.models import MLPRegressor, torch_available
from surroforge.simulators import PythonCallableSimulator


def thermal_plate(params):
    thickness = params["thickness"]
    power = params["power"]
    conductivity = params["conductivity"]
    convection = params["convection"]
    resistance = thickness / conductivity + 1.0 / convection
    max_temperature = 22.0 + power * resistance
    mean_temperature = 22.0 + 0.62 * power * resistance
    hotspot_x = 0.5 + 0.08 * (power / 200.0) - 0.03 * (conductivity / 250.0)
    hotspot_y = 0.5 + 0.05 * (thickness / 5.0) - 0.02 * (convection / 50.0)
    return {
        "max_temperature": max_temperature,
        "mean_temperature": mean_temperature,
        "hotspot_x": hotspot_x,
        "hotspot_y": hotspot_y,
    }


def build_forge(store: str | Path = "runs/thermal_plate_python") -> Forge:
    space = DesignSpace()
    space.float("thickness", 0.5, 5.0, unit="mm")
    space.float("power", 25.0, 200.0, unit="W")
    space.float("conductivity", 12.0, 250.0, unit="W/mK")
    space.float("convection", 5.0, 50.0, unit="W/m^2K")
    return Forge(space, PythonCallableSimulator(thermal_plate), store=store)


def main() -> None:
    forge = build_forge()
    forge.sample(32, method="sobol", seed=4)
    forge.run_pending()
    if torch_available():
        forge.train(model=MLPRegressor(hidden=[32, 32], epochs=25, patience=5), ensemble=None)
        print(forge.validate())
        print(
            forge.predict(
                {"thickness": 1.0, "power": 120.0, "conductivity": 80.0, "convection": 20.0}
            )
        )
    else:
        print("Install surroforge[torch] to train the MLP surrogate for this demo.")


if __name__ == "__main__":
    main()
