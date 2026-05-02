"""Shared schemas used across SurroForge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(str, Enum):
    """Lifecycle states for a simulation sample."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunRecord(BaseModel):
    """Persistent record for one simulation run."""

    model_config = ConfigDict(use_enum_values=True)

    run_id: str
    params: dict[str, Any]
    status: RunStatus = RunStatus.PENDING
    outputs: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    runtime_seconds: float | None = None
    created_at: str
    updated_at: str
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    """Prediction payload returned by :meth:`surroforge.Forge.predict`."""

    values: dict[str, float]
    uncertainty: dict[str, float] = field(default_factory=dict)
    trust_level: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
