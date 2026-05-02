"""Example Elmer case factory placeholder."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_case(params: dict[str, Any], workdir: Path) -> None:
    """Write a minimal case placeholder.

    Replace this with real pyelmer geometry, mesh, material, solver, and output
    definitions. The function is intentionally lightweight so importing the
    example never launches Elmer.
    """
    (workdir / "case.sif").write_text(
        f"! Generated placeholder for params: {params}\n",
        encoding="utf-8",
    )
    (workdir / "results").mkdir(exist_ok=True)
