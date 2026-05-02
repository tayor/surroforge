# Bring Your Own Simulator

SurroForge is designed for black-box workflows. A simulator can be a Python function, a shell command, an Elmer case, a legacy executable, or a script that writes output files.

## Python Function

```python
from surroforge.simulators import PythonCallableSimulator


def simulate(params):
    return {"temperature": params["power"] / params["thickness"]}


simulator = PythonCallableSimulator(simulate)
```

If the function accepts two arguments, SurroForge passes `(params, workdir)` so the simulator can write artifacts.

## External Command

```python
from surroforge.simulators import SubprocessSimulator

simulator = SubprocessSimulator(
    command="python simulate.py --params params.json --out output.json",
    output="output.json",
    timeout=60,
)
```

The command runs inside a per-sample work directory. SurroForge writes `params.json` before execution and expects `output.json` to contain a JSON object of scalar or artifact outputs.

## Elmer Case Factory

```python
from surroforge.simulators.elmer import PyElmerSimulator

simulator = PyElmerSimulator(
    case_factory="my_project.elmer_cases:build_case",
    outputs={"max_temp": "results/max_temperature.csv"},
    command="ElmerSolver case.sif",
)
```

The factory should write mesh, SIF, and solver inputs into `workdir`. This keeps Elmer visible and debuggable while SurroForge handles campaign orchestration.

## Output Guidelines

- Use stable output names across all runs.
- Prefer JSON for scalar/vector outputs.
- Store large fields or meshes as files and return their relative paths.
- Keep solver logs in the run work directory for later debugging.
- Set timeouts for external commands so failed solver calls do not hang a campaign.
