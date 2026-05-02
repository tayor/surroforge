from __future__ import annotations

from typer.testing import CliRunner

from surroforge.cli import app


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "surroforge" in result.output


def test_cli_init_and_sample(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["init", "thermal", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    project = tmp_path / "thermal"
    assert (project / "design_space.yaml").exists()
    assert (project / "simulator.py").exists()

    result = runner.invoke(
        app,
        [
            "sample",
            "--n",
            "4",
            "--method",
            "sobol",
            "--design-space",
            str(project / "design_space.yaml"),
            "--store",
            str(project / "runs" / "default"),
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (project / "runs" / "default" / "records.jsonl").exists()
