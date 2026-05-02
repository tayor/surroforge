"""Static Markdown and HTML reports."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

from surroforge.schemas import RunRecord


def render_markdown_report(
    *,
    records: Sequence[RunRecord],
    metrics: Mapping[str, float] | None = None,
    title: str = "SurroForge Report",
) -> str:
    """Render a compact Markdown report."""
    completed = [record for record in records if record.status == "completed"]
    failed = [record for record in records if record.status == "failed"]
    lines = [f"# {title}", "", "## Dataset", ""]
    lines.append(f"- Total samples: {len(records)}")
    lines.append(f"- Completed: {len(completed)}")
    lines.append(f"- Failed: {len(failed)}")
    if metrics:
        lines.extend(["", "## Metrics", ""])
        for name, value in metrics.items():
            lines.append(f"- {name}: {value:.6g}")
    if failed:
        lines.extend(["", "## Failed Simulations", ""])
        for record in failed:
            lines.append(f"- {record.run_id}: {record.failure_reason}")
    lines.extend(["", "## Model Card", ""])
    lines.append("SurroForge models are data-driven surrogates and should be validated before use.")
    return "\n".join(lines) + "\n"


def render_html_report(
    *,
    records: Sequence[RunRecord],
    metrics: Mapping[str, float] | None = None,
    title: str = "SurroForge Report",
) -> str:
    """Render a dependency-free HTML report."""
    markdown = render_markdown_report(records=records, metrics=metrics, title=title)
    body = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- ") or line:
            body.append(f"<p>{html.escape(line)}</p>")
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <style>
    body {{
      font-family: system-ui, sans-serif;
      max-width: 860px;
      margin: 3rem auto;
      padding: 0 1rem;
      line-height: 1.5;
    }}
    h1, h2 {{ color: #1f2937; }}
    p {{ color: #374151; }}
  </style>
</head>
<body>
{body}
</body>
</html>
""".format(title=html.escape(title), body="\n".join(body))


def write_report(
    directory: str | Path,
    *,
    records: Sequence[RunRecord],
    metrics: Mapping[str, float] | None = None,
    title: str = "SurroForge Report",
) -> dict[str, Path]:
    """Write Markdown and HTML reports to a directory."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    markdown_path.write_text(
        render_markdown_report(records=records, metrics=metrics, title=title),
        encoding="utf-8",
    )
    html_path.write_text(
        render_html_report(records=records, metrics=metrics, title=title),
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "html": html_path}
