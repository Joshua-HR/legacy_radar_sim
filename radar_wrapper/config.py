"""Canonical config loading for the radar replay wrapper.

Config stacking order (later overrides earlier):
    1. configs/default.ini          (compiled-in defaults)
    2. user-supplied *.ini          (scenario / preset)
    3. -- CLI overrides
"""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Radar replay pipeline",
    )
    parser.add_argument("--config", required=True, help="Path to scenario config (.ini)")
    parser.add_argument("--run-name", default=None, help="Override run.name")
    parser.add_argument("--backend", default=None, help="Override postprocess.backend (mx|rftests)")
    parser.add_argument("--source", default=None, help="Override input.source (synthetic|measured)")
    return parser.parse_args(argv)


def load_config(
    config_path: str | None = None,
    cli_args: argparse.Namespace | None = None,
) -> configparser.ConfigParser:
    """Load default.ini, layer in the user config, then apply CLI overrides."""
    cfg = configparser.ConfigParser()

    # 1. compiled-in defaults (relative to project root)
    _root = Path(__file__).resolve().parents[1]
    default_ini = _root / "configs" / "default.ini"
    cfg.read(str(default_ini))

    # 2. user config
    if config_path is not None:
        cfg.read(config_path)

    # 3. CLI overrides
    if cli_args is not None:
        apply_cli_overrides(cfg, cli_args)

    validate_config(cfg)
    return cfg


def apply_cli_overrides(
    cfg: configparser.ConfigParser,
    args: argparse.Namespace,
) -> None:
    if args.run_name is not None:
        if "run" not in cfg:
            cfg["run"] = {}
        cfg["run"]["name"] = args.run_name
    
    if args.backend is not None:
        if "postprocess" not in cfg:
            cfg["postprocess"] = {}
        cfg["postprocess"]["backend"] = args.backend

    if args.source is not None:
        if "input" not in cfg:
            cfg["input"] = {}
        cfg["input"]["source"] = args.source


def validate_config(cfg: configparser.ConfigParser) -> None:
    source = cfg.get("input", "source", fallback="").strip().lower()
    backend = cfg.get("postprocess", "backend", fallback="").strip().lower()

    if source not in ("synthetic", "measured"):
        raise ValueError(f"input.source must be 'synthetic' or 'measured', got: {source!r}")

    # "none" is the CIR-only mode: run the source, write cir_data.npz, and stop
    # before any postprocessing. Used by physics/geometry validation, where the
    # observable is the CIR itself and a detection pass would only add cost and
    # threshold sensitivity.
    if backend not in ("mx", "rftests", "none"):
        raise ValueError(
            f"postprocess.backend must be 'mx', 'rftests' or 'none', got: {backend!r}"
        )
