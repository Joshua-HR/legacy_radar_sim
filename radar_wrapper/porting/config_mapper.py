"""Maps wrapper canonical config + CIR metadata to backend-specific parameters."""

from __future__ import annotations

from typing import Dict


class ConfigMapper:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # MX parameters
    # ------------------------------------------------------------------

    def build_mx_params(self, cir_metadata) -> Dict[str, object]:
        """Build the keyword arguments needed to configure the MX processor.

        This maps wrapper canonical config onto ``ProcessingConfig`` from
        ``MX_CodeV1_Mustafa/radar_processing.py``.
        """
        params: Dict[str, object] = {
            "cir_taps": cir_metadata.num_taps,
            "segment_size": self.cfg.getint("mx", "segment_size", fallback=32),
            "hop_size": self.cfg.getint("mx", "hop_size", fallback=0),
            "period_ms": cir_metadata.pri_s * 1000.0,
            "channel": int(cir_metadata.channel),
            "cfar_threshold_db": self.cfg.getfloat("mx", "cfar_threshold", fallback=7.0),
            "skip_bins": self.cfg.getint("mx", "skip_bins", fallback=5),
            "min_range_cm": self.cfg.getfloat("mx", "min_range_cm", fallback=50.0),
            "max_range_cm": self.cfg.getfloat("mx", "max_range_cm", fallback=300.0),
            "min_abs_velocity_mps": self.cfg.getfloat("mx", "min_abs_velocity_mps", fallback=0.0),
            "empty_calibration_segments": self.cfg.getint("mx", "calib_segments", fallback=3),
            "min_dynamic_excess_db": self.cfg.getfloat("mx", "min_dynamic_excess_db", fallback=6.0),
            "min_cells_for_human": self.cfg.getint("mx", "min_cells", fallback=2),
            "min_unique_taps_for_human": self.cfg.getint("mx", "min_unique_taps", fallback=2),
            "min_range_span_cm_for_human": self.cfg.getfloat("mx", "min_range_span_cm", fallback=15.0),
            "adaptive_baseline_update": self.cfg.getboolean("mx", "adaptive_baseline_update", fallback=True),
            "adaptive_baseline_alpha": self.cfg.getfloat("mx", "adaptive_baseline_alpha", fallback=0.03),
            "adaptive_baseline_guard_db": self.cfg.getfloat("mx", "adaptive_baseline_guard_db", fallback=12.0),
            "require_baseline_for_human": self.cfg.getboolean("mx", "require_baseline_for_human", fallback=True),
            "enable_static_detection": self.cfg.getboolean("mx", "enable_static_detection", fallback=False),
            "persistence_window": self.cfg.getint("mx", "persistence_window", fallback=4),
            "persistence_min_hits": self.cfg.getint("mx", "persistence_hits", fallback=2),
            "persistence_range_gate_cm": self.cfg.getfloat("mx", "persistence_range_cm", fallback=45.0),
            "reject_edge_doppler_bins": self.cfg.getint("mx", "reject_edge_doppler_bins", fallback=1),
            # Near-zone ghost-motion block. Exposed because its false-alarm rate
            # scales with segment_size and it does NOT respect skip_bins or
            # min_range_cm: _near_zone_block_status() builds its own mask over
            # every tap below near_zone_block_max_cm (50 cm -> taps 0..3) and
            # suppresses the WHOLE segment if >= near_zone_block_min_cells of
            # those (segment_size x 4) cells exceed near_zone_block_excess_db.
            # At segment_size = 32 that is 128 candidate cells; at 256 it is
            # 1024, and against a 3-segment 95th-percentile baseline a couple of
            # them clear 6 dB on noise alone. A long coherent window is exactly
            # what micro-Doppler (breathing) needs, so the two interact -- see
            # docs/standard_validation_geometry.md. Defaults match
            # ProcessingConfig, so no existing run changes.
            "near_zone_block_enabled": self.cfg.getboolean("mx", "near_zone_block_enabled", fallback=True),
            "near_zone_block_max_cm": self.cfg.getfloat("mx", "near_zone_block_max_cm", fallback=50.0),
            "near_zone_block_excess_db": self.cfg.getfloat("mx", "near_zone_block_excess_db", fallback=6.0),
            "near_zone_block_min_cells": self.cfg.getint("mx", "near_zone_block_min_cells", fallback=2),
            # Debug printing / plotting. ProcessingConfig defaults both to True,
            # which floods stdout with per-segment [DBG] lines and writes a PNG
            # per segment into outputs/debugs/ -- and raises FileNotFoundError if
            # that directory does not already exist. Harmless for a single
            # interactive run, prohibitive for a parameter sweep, so both are
            # exposed and default to the legacy True so no existing run changes.
            "debug_enable": self.cfg.getboolean("mx", "debug_enable", fallback=True),
            "debug_plot_enable": self.cfg.getboolean("mx", "debug_plot_enable", fallback=True),
        }
        return params

    # ------------------------------------------------------------------
    # RFtests parameters (stub - not part of MVP)
    # ------------------------------------------------------------------

    def generate_rftests_config(self, cir_metadata, output_path) -> None:
        raise NotImplementedError("RFtests config mapping is not implemented yet")
            