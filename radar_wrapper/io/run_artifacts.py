"""Run directory creation and resolved config saving."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(cfg) -> Path:
    output_root = Path(cfg.get("run", "output_root", fallback="outputs/runs"))
    run_name = cfg.get("run", "name", fallback=None)

    if not run_name or run_name.strip() == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    run_dir = output_root / run_name

    # Guard against silently overwriting an existing run.  Append a counter
    # until we find a fresh name.
    suffix = 1
    candidate = run_dir
    while candidate.exists():
        candidate = output_root / f"{run_name}_{suffix}"
        suffix += 1

    run_dir = candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(cfg, run_dir: Path) -> Path:
    output_path = Path(run_dir) / "resolved_config.ini"
    with open(str(output_path), "w", encoding="utf-8") as fh:
        cfg.write(fh)
    return output_path
    