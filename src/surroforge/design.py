"""Design-space definitions and validation."""

from __future__ import annotations

import builtins
import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ValidationError, field_validator


class DesignSpaceError(ValueError):
    """Raised when a design space or parameter value is invalid."""


class Parameter(BaseModel):
    """A single design parameter."""

    name: str
    type: Literal["float", "integer", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] | None = None
    unit: str | None = None
    description: str | None = None
    condition: str | None = None

    @field_validator("choices")
    @classmethod
    def _non_empty_choices(cls, choices: list[Any] | None) -> list[Any] | None:
        if choices is not None and not choices:
            raise ValueError("categorical parameters need at least one choice")
        return choices

    def model_post_init(self, __context: Any) -> None:  # noqa: D105
        if self.type in {"float", "integer"}:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.type} parameter '{self.name}' needs low and high")
            if float(self.low) >= float(self.high):
                raise ValueError(f"parameter '{self.name}' must have low < high")
        if self.type == "categorical" and not self.choices:
            raise ValueError(f"categorical parameter '{self.name}' needs choices")

    def validate_value(self, value: Any) -> Any:
        """Validate and coerce one value for this parameter."""
        if self.type == "float":
            assert self.low is not None and self.high is not None
            numeric = float(value)
            if numeric < float(self.low) or numeric > float(self.high):
                raise DesignSpaceError(
                    f"{self.name}={numeric} is outside [{self.low}, {self.high}]"
                )
            return numeric
        if self.type == "integer":
            assert self.low is not None and self.high is not None
            if isinstance(value, bool):
                raise DesignSpaceError(f"{self.name} must be an integer, not bool")
            numeric = int(value)
            if numeric < int(self.low) or numeric > int(self.high):
                raise DesignSpaceError(
                    f"{self.name}={numeric} is outside [{self.low}, {self.high}]"
                )
            return numeric
        assert self.choices is not None
        if value not in self.choices:
            raise DesignSpaceError(f"{self.name}={value!r} is not one of {self.choices!r}")
        return value

    def from_unit(self, value: float) -> Any:
        """Map a value in [0, 1] to the parameter domain."""
        clipped = min(max(float(value), 0.0), np.nextafter(1.0, 0.0))
        if self.type == "float":
            assert self.low is not None and self.high is not None
            return float(self.low) + clipped * (float(self.high) - float(self.low))
        if self.type == "integer":
            assert self.low is not None and self.high is not None
            low = int(self.low)
            high = int(self.high)
            return min(high, low + int(math.floor(clipped * (high - low + 1))))
        assert self.choices is not None
        index = min(len(self.choices) - 1, int(math.floor(clipped * len(self.choices))))
        return self.choices[index]

    def encode(self, value: Any) -> float:
        """Encode a parameter value as a numeric model feature."""
        value = self.validate_value(value)
        if self.type in {"float", "integer"}:
            return float(value)
        assert self.choices is not None
        return float(self.choices.index(value))

    def bounds(self) -> tuple[float, float]:
        """Return numeric feature bounds for this parameter."""
        if self.type == "categorical":
            assert self.choices is not None
            return 0.0, float(len(self.choices) - 1)
        assert self.low is not None and self.high is not None
        return float(self.low), float(self.high)


class DesignSpace:
    """Ordered set of parameters, constraints, and serializers."""

    def __init__(
        self,
        parameters: Iterable[Parameter | Mapping[str, Any]] | None = None,
        constraints: Iterable[str] | None = None,
    ) -> None:
        self.parameters: OrderedDict[str, Parameter] = OrderedDict()
        self.constraints = list(constraints or [])
        for parameter in parameters or []:
            parsed = parameter if isinstance(parameter, Parameter) else Parameter(**parameter)
            self._add(parsed)

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters.values())

    def __contains__(self, name: str) -> bool:
        return name in self.parameters

    @property
    def names(self) -> list[str]:
        """Parameter names in deterministic feature order."""
        return list(self.parameters)

    def _add(self, parameter: Parameter) -> None:
        if parameter.name in self.parameters:
            raise DesignSpaceError(f"duplicate parameter name: {parameter.name}")
        self.parameters[parameter.name] = parameter

    def float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        unit: str | None = None,
        description: str | None = None,
        condition: str | None = None,
    ) -> DesignSpace:
        """Add a continuous parameter."""
        self._add(
            Parameter(
                name=name,
                type="float",
                low=low,
                high=high,
                unit=unit,
                description=description,
                condition=condition,
            )
        )
        return self

    def integer(
        self,
        name: str,
        low: int,
        high: int,
        *,
        unit: str | None = None,
        description: str | None = None,
        condition: str | None = None,
    ) -> DesignSpace:
        """Add an integer parameter."""
        self._add(
            Parameter(
                name=name,
                type="integer",
                low=low,
                high=high,
                unit=unit,
                description=description,
                condition=condition,
            )
        )
        return self

    def categorical(
        self,
        name: str,
        choices: Iterable[Any],
        *,
        unit: str | None = None,
        description: str | None = None,
        condition: str | None = None,
    ) -> DesignSpace:
        """Add a categorical parameter."""
        self._add(
            Parameter(
                name=name,
                type="categorical",
                choices=list(choices),
                unit=unit,
                description=description,
                condition=condition,
            )
        )
        return self

    def add_constraint(self, expression: str) -> DesignSpace:
        """Add a boolean Python expression evaluated against parameter names."""
        self.constraints.append(expression)
        return self

    def validate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a complete parameter dictionary."""
        missing = [name for name in self.parameters if name not in params]
        if missing:
            raise DesignSpaceError(f"missing parameters: {', '.join(missing)}")
        extra = [name for name in params if name not in self.parameters]
        if extra:
            raise DesignSpaceError(f"unknown parameters: {', '.join(extra)}")
        validated = {
            name: parameter.validate_value(params[name])
            for name, parameter in self.parameters.items()
        }
        for expression in self.constraints:
            if not self._constraint_holds(expression, validated):
                raise DesignSpaceError(f"constraint failed: {expression}")
        return validated

    def contains(self, params: Mapping[str, Any]) -> bool:
        """Return True if parameters are valid and constraints are satisfied."""
        try:
            self.validate(params)
        except (DesignSpaceError, TypeError, ValueError):
            return False
        return True

    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        """Encode parameters as a numeric feature vector."""
        validated = self.validate(params)
        return np.asarray(
            [self.parameters[name].encode(validated[name]) for name in self.parameters],
            dtype=float,
        )

    def encode_many(self, samples: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """Encode many parameter dictionaries as a 2D array."""
        rows = [self.encode(sample) for sample in samples]
        if not rows:
            return np.empty((0, len(self.parameters)), dtype=float)
        return np.vstack(rows)

    def decode_unit(self, unit_vector: Iterable[builtins.float]) -> dict[str, Any]:
        """Decode a unit-cube vector into a parameter dictionary."""
        values = list(unit_vector)
        if len(values) != len(self.parameters):
            raise DesignSpaceError("unit vector dimension does not match design space")
        return {
            name: parameter.from_unit(values[index])
            for index, (name, parameter) in enumerate(self.parameters.items())
        }

    def numeric_bounds(self) -> np.ndarray:
        """Return numeric feature bounds as an array of shape (n_parameters, 2)."""
        return np.asarray(
            [parameter.bounds() for parameter in self.parameters.values()], dtype=float
        )

    def distance_to_bounds(self, params: Mapping[str, Any]) -> builtins.float:
        """Return normalized distance to the nearest numeric boundary."""
        vector = self.encode(params)
        bounds = self.numeric_bounds()
        spans = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0)
        normalized = (vector - bounds[:, 0]) / spans
        return float(np.min(np.minimum(normalized, 1.0 - normalized)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the design space to plain Python data."""
        return {
            "parameters": {
                name: parameter.model_dump(exclude_none=True, exclude={"name"})
                for name, parameter in self.parameters.items()
            },
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DesignSpace:
        """Create a design space from a dictionary."""
        parameters = []
        for name, raw in payload.get("parameters", {}).items():
            data = dict(raw)
            data["name"] = name
            try:
                parameters.append(Parameter(**data))
            except ValidationError as exc:
                raise DesignSpaceError(str(exc)) from exc
        return cls(parameters=parameters, constraints=payload.get("constraints", []))

    def to_json(self, path: str | Path) -> Path:
        """Write the design space to JSON."""
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_json(cls, path: str | Path) -> DesignSpace:
        """Read a design space from JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_yaml(self, path: str | Path) -> Path:
        """Write the design space to YAML."""
        output = Path(path)
        output.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return output

    @classmethod
    def from_yaml(cls, path: str | Path) -> DesignSpace:
        """Read a design space from YAML."""
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @staticmethod
    def _constraint_holds(expression: str, params: Mapping[str, Any]) -> bool:
        names = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        names.update(params)
        return bool(eval(expression, {"__builtins__": {}}, names))
