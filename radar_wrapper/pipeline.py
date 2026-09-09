"""Top-level replay pipeline: source -> adapter -> results -> plots."""

from __future__ import annotations

from radar_wrapper.adapters.mx_adapter import MXPostprocessingAdapter
from radar_wrapper.io.npz_writer import save_cir_npz
from radar_wrapper.io.result_writer import save_results
from radar_wrapper.io.run_artifacts import create_run_dir, save_run_config
from radar_wrapper.schemas.cir import CIRData
from radar_wrapper.sources.measured_cir import MeasuredCIRSource
from radar_wrapper.sources.synthetic_cir import SyntheticCIRSource
from radar_wrapper.visualization.range_angle import plot_range_angle
from radar_wrapper.visualization.range_time import plot_range_time


class RadarReplayPipeline:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.run_dir = create_run_dir(cfg)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> tuple[CIRData, list]:
        """Execute the full replay pipeline.

        Returns:
            (cir, detections) - canonical CIR data and detection list.
        """
        save_run_config(self.cfg, self.run_dir)

        # --- source ---
        source = self._create_source()
        cir = source.load()

        save_cir_npz(cir, self.run_dir)

        # --- postprocessing ---
        # backend = none is the CIR-only mode: the run stops after the CIR has
        # been written. Intended for physics/geometry validation, where the
        # observable is cir_data.npz and a detection pass would only add cost
        # and threshold sensitivity.
        adapter = self._create_adapter()
        detections = [] if adapter is None else adapter.run_replay(cir)

        # --- artifacts ---
        # Written even for backend = none (as empty lists) so every run
        # directory has the same file layout.
        save_results(detections, cir, self.cfg, self.run_dir)

        # --- plots ---
        # Both plots are detection scatter plots, so they are skipped rather
        # than rendered empty when there is no backend.
        if adapter is not None:
            self._visualize(cir, detections)

        return cir, detections

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def _create_source(self):
        source_name = self.cfg.get("input", "source", fallback="").strip().lower()

        if source_name == "synthetic":
            return SyntheticCIRSource(self.cfg)

        if source_name == "measured":
            return MeasuredCIRSource(self.cfg)

        raise ValueError(f"Unsupported input.source: {source_name!r}")

    def _create_adapter(self):
        """Build the postprocessing adapter, or None for the CIR-only mode.

        Returns:
            An Adapter implementing run_replay(cir), or None when
            postprocess.backend == "none".
        """
        backend = self.cfg.get("postprocess", "backend", fallback="").strip().lower()

        if backend == "mx":
            return MXPostprocessingAdapter(cfg=self.cfg, run_dir=self.run_dir)

        if backend == "none":
            return None

        if backend == "rftests":
            raise NotImplementedError(
                "RFtests postprocessing backend is not implemented in this MVP."
            )

        raise ValueError(f"Unsupported postprocess.backend: {backend!r}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _visualize(self, cir: CIRData, detections: list) -> None:
        if not self.cfg.getboolean("visualization", "enabled", fallback=True):
            return

        show = self.cfg.getboolean("visualization", "show", fallback=False)
        save = self.cfg.getboolean("visualization", "save", fallback=True)

        if self.cfg.getboolean("visualization", "range_angle", fallback=True):
            plot_range_angle(detections, self.run_dir, show=show, save=save)

        if self.cfg.getboolean("visualization", "range_time", fallback=True):
            plot_range_time(detections, cir, self.run_dir, show=show, save=save)
            