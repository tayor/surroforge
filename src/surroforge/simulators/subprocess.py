"""Subprocess-based simulator adapter."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from surroforge.simulators.base import Simulator, SimulatorError, ensure_mapping


class SubprocessSimulator(Simulator):
    """Run an external command and collect a JSON output file."""

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        output: str = "output.json",
        timeout: float | None = None,
        terminate_grace_seconds: float = 2.0,
        env: Mapping[str, str] | None = None,
        shell: bool | None = None,
    ) -> None:
        self.command = command
        self.output = output
        self.timeout = timeout
        self.terminate_grace_seconds = terminate_grace_seconds
        self.env = dict(env or {})
        self.shell = isinstance(command, str) if shell is None else shell

    def run(self, workdir: str | Path) -> None:
        directory = Path(workdir)
        logs = directory / "logs"
        logs.mkdir(exist_ok=True)
        stdout_path = logs / "stdout.txt"
        stderr_path = logs / "stderr.txt"
        env = os.environ.copy()
        env.update(self.env)
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                self.command,
                cwd=directory,
                env=env,
                shell=self.shell,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                text=True,
            )
            try:
                return_code = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process_group(process)
                raise SimulatorError(f"simulation timed out after {self.timeout} seconds") from exc
        if return_code != 0:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            detail = f": {stderr_text}" if stderr_text else ""
            raise SimulatorError(f"simulation command failed with exit code {return_code}{detail}")

    def collect(self, workdir: str | Path) -> dict[str, Any]:
        output_path = Path(workdir) / self.output
        if not output_path.exists():
            raise SimulatorError(f"expected simulator output at {output_path}")
        return ensure_mapping(
            json.loads(output_path.read_text(encoding="utf-8")),
            context="output JSON",
        )

    def _terminate_process_group(self, process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self.terminate_grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=max(self.terminate_grace_seconds, 0.1))
