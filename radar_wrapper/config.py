"""Canonical config loading for the radar replay wrapper.

Config stacking order (later overrides earlier):
    1. configs/default.ini          (compiled-in defaults)
    2. user-supplied *.ini          (scenario / preset)
    3. -- CLI overrides
"""

from __future__ import annotations

import argparse
import configparser
import warnings
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Recognised configuration surface
# ---------------------------------------------------------------------------

#: Every section the wrapper reads. An recognised section is hard error
#: because a section-name typo cannot be recovered from downstream: configparser
#: treats "[mx ]" (trailing space) as a section distinct from "[mx]", so the
#: whole block silently becomes dead config and the run proceeds with defaults.
#: That exact typo sat in configs/pipelines/synthetic_mx_validation.ini from
#: 2026-08-13 until it was found by this audit, and it was inherited by a
#: teammate who copied that file. See docs/standard_validation_geometry.md.
KNOWN_SENCTIONS: Tuple[str, ...] = (
    "run",
    "input",
    "synthetic",
    "measured",
    "radar",
    "postprocess",
    "mx",
    "rftests",
    "visualization",
)

#: Wrapper [radar] keys that are IGNORED for synthetic runs. The MX backend is
#: configured from CIRMetadata (radar_wrapper/porting/config_mapper.py), and for
#: a synthetic run SyntheticCIRSource fills those metadata fields from the
#: CIRgenerator platform INI's actual generated shape / [radar] period / [cir]
#: bin_time_s -- not from this sections. Setting them here looks effective and
#: is not.
SYNTHETIC_IGNORED_RADAR_KEYS: Tuple[str, ...] = (
    "num_taps",
    "pri_s",
    "fast_time_resolution_ns",
)

#: [mx] keys the hex ingestion path cannot honour. RadarPostprocessing (hex) is
#: a different class from RadarDopplerProcessor (complex): its constructor takes
#: no detection thresholds at all and it never sees
#: ConfigMapper.build_mx_params(), so these keys reach nothing under
#: ingest_mode = hex. Meausred: with hex, cfar_threshold 0 vs 30 dB and
#: min_dynamic_excess_db 0 vs 30 both give the same detections; with complex the
#: same sweep gives 9 vs 0. Use ingest_mode = complex to tune thresholds.
HEX_UNSUPPORTED_MX_KEYS: Tuple[str, ...] = (
    "segment_size",
    "hop_size",
    "skip_bins",
    "min_range_cm",
    "max_range_cm",
    "min_abs_velocity_mps",
    "calib_segments",
    "min_dynamic_excess_db",
    "min_cells",
    "min_unique_taps",
    "min_range_span_cm",
    "persistence_window",
    "persistence_hits",
    "persistence_range_cm",
    "reject_edge_doppler_bins",
    "adaptive_baseline_update",
    "adaptive_baseline_alpha",
    "adaptive_baseline_guard_db",
    "require_baseline_for_human",
    "enable_static_detection",
)


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

    _reject_unknown_sections(cfg)
    _warn_ineffective_keys(cfg, source=source, backend=backend)


def _reject_unknown_sections(cfg: configparser.ConfigParser) -> None:
    """Fail on any section the wrapper does not read.

    Deliberately a hard error rather than a warning: the failure this catches is
    a *silent* one. "[mx ]" parses fine, shadows nothing, and leaves the run
    using configs/default.ini's thresholds while appearing to override them.
    """
    unknown = [section for section in cfg.sections() if section not in KNOWN_SENCTIONS]
    if not unknown:
        return

    hints = []
    for section in unknown:
        stripped = section.strip()
        if stripped in KNOWN_SENCTIONS and stripped != section:
            hints.append(
                f"  [{section}] -> did you mean [{stripped}]? "
                "(surrounding whitespace is part of the section name)"
            )
        else:
            hints.append(f"  [{section}]")

    raise ValueError(
        "unrecognised config section(s):\n"
        + "\n".join(hints)
        + "\nKnown sections: "
        + ", ".join(KNOWN_SENCTIONS)
    )


def _warn_ineffective_keys(
    cfg: configparser.ConfigParser,
    source: str,
    backend: str,
) -> None:
    """Warn about keys that are present, look effective, and are not.

    A warning rather than an error: these keys are shipped in
    configs/default.ini, so erroring would rejecte every existing config. The
    point is that a reader of an INI cannot otherwise tell that a value is dead.
    """
    if source == "synthetic" and cfg.has_section("radar"):
        present = [
            key for key in SYNTHETIC_IGNORED_RADAR_KEYS if cfg.has_option("radar", key)
        ]
        if present:
            warnings.warn(
                "[radar] "
                + ", ".join(present)
                + " are ignored for input.source = synthetic: the MX backend reads "
                "them from CIRMetadata, which SyntheticCIRSource fills from the "
                "CIRgenerator platform INI ([radar] period / [cir] bin_time_s and "
                "the generated tap count). Change the platform INI selected by "
                "[synthetic] default_syn_config_path instead.",
                stacklevel=3,
            )

    if backend == "mx" and cfg.has_section("mx"):
        ingest_mode = cfg.get("mx", "ingest_mode", fallback="hex").strip().lower()
        if ingest_mode == "hex":
            present = [
                key for key in HEX_UNSUPPORTED_MX_KEYS if cfg.has_option("mx", key)
            ]
            if present:
                warnings.warn(
                    "[mx] ingest_mode = hex cannot honour these detection "
                    "thresholds: "
                    + ", ".join(present)
                    + ". The hex path uses RadarPostprocessing, whose constructor "
                    "accepts no thesholds and which never sees "
                    "ConfigMapper.build_mx_params(). Set ingest_mode = complex to "
                    "make them effective.",
                    stacklevel=3,
                )
        elif ingest_mode == "complex" and not cfg.getboolean(
            "mx", "debug_enable", fallback=True
        ):
            # debug_enable = false silently breaks ingest_mode = complex
            # entirely, not just its printing: MX_CodeV1_Mustafa's
            # _process_segment() has its final `return result` indented INSIDE
            # `if cfg.debug_enable:`, so once debug_enable is false every
            # segment that reaches the normal (non-CALIBRATING,
            # non-near-zone-blocked) code path falls off the end of the
            # function and returns None implicitly; push_cir()'s
            # `except Exception: pass` around result.rx_health then swallows
            # the resulting AttributeError, so nothing raises. Meausred: 7
            # detections / 12 non-None segments with debug_enable = true vs 0 /
            # 2 with it false, on the identical CIR. MX_CodeV1_Mustafa is
            # legacy and untouched (CLAUDE.md), so this can only be caught
            # here. debug_plot_enable is the correct key to disable for speed.
            warnings.warn(
                "[mx] ingest_mode = complex with debug_enable = false will "
                "silently return zero detections: MX_CodeV1_Mustafa's "
                "_process_segment() only returns its result inside the "
                "debug_enable branch (a legacy indentation defect, left "
                "unmodified per CLAUDE.md). Set debug_enable = true and use "
                "debug_plot_enable = false to disable only the per-segment "
                "plots.",
                stacklevel=3,
            )
            