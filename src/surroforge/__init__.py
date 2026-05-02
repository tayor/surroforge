"""SurroForge: simulator-first neural surrogate modeling."""

from surroforge.design import DesignSpace, Parameter
from surroforge.forge import Forge
from surroforge.schemas import Prediction, RunRecord, RunStatus
from surroforge.simulators import CSVReplaySimulator, PythonCallableSimulator, SubprocessSimulator

__all__ = [
    "CSVReplaySimulator",
    "DesignSpace",
    "Forge",
    "Parameter",
    "Prediction",
    "PythonCallableSimulator",
    "RunRecord",
    "RunStatus",
    "SubprocessSimulator",
]

__version__ = "0.1.0"
