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

        This maps wrapper canonical config onto ``PreprocessingConfig`` from
        ```MX_CodeV1_Mustafa/radar_processing.py``.
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
        }
        return params

    # ------------------------------------------------------------------
    # RFtests parameters (stub - not part of MVP)
    # ------------------------------------------------------------------

    def generate_rftests_config(self, cir_metadata, output_path) -> None:
        raise NotImplementedError("RFtests config mapping is not implemented yet")
            