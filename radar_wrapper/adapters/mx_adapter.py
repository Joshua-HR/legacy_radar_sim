# radar_wrapper/adapters/mx_adapter.py

"""MX adapter - supports both complex push_cir and Q8.8 hex fill_data_by_hex replay."""

from __future__ import annotations

from queue import Queue
from typing import Any

import numpy as np

from radar_wrapper.paths import add_mx_paths
from radar_wrapper.adapters.base import PostprocessingAdapter
from radar_wrapper.porting.config_mapper import ConfigMapper
from radar_wrapper.porting.hex_codec import frame_to_hex_string
from radar_wrapper.schemas.cir import CIRData
from radar_wrapper.schemas.results import Detection

import inspect


def _construct_with_supported_kwargs(cls, **kwargs):
    """Instantiate cls using only kwargs supported by cls.__init__.

    This avoids breakage when Mustafa/RFtests/older RadarPostprocessing
    versions have slightly different constructor signatures.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # If constructor accepts **kwargs, pass everything.
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )
    if has_var_kw:
        return cls(**kwargs)

    supported = (
        name
        for name, p in params.items()
        if name != "self"
        and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )

    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in supported
    }

    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print(
            "[MX hex] RadarPostprocessing.__init__ does not support "
            f"these kwargs, dropped: {dropped}"
        )

    return cls(**filtered)


class MXPostprocessingAdapter(PostprocessingAdapter):
    """Replay adapter for MX postprocessing

    Supported ingestion modes:

    1. complex
        - Uses radar_processing.RadarDopplerProcessor.push_cir()
        - Feeds complex ndarray directly

    2. hex
        - Uses radar_utils_new.RadarPostprocessing.fill_data_by_hex()
        - Converts complex CIR to Q8.8 hex first
        - Closer to HW/log style ingestion path
    """

    def __init__(self, cfg, run_dir=None) -> None:
        self.cfg = cfg
        self.run_dir = run_dir

        self.ingest_mode = self.cfg.get(
            "mx",
            "ingest_mode",
            fallback="hex",
        ).strip().lower()

        self.processor = None
        self.postproc = None
        self.target_queue = None

        self.detections: list[Detection] = []
        self.segment_results: list[Any] = []
        self._segment_idx = 0

    # ------------------------------------------------------------------
    # Configure
    # ------------------------------------------------------------------

    def configure(self, cir: CIRData) -> None:
        if self.ingest_mode == "complex":
            self._configure_complex_processor(cir)
            return

        if self.ingest_mode == "hex":
            self._configure_hex_postprocessor(cir)
            return

        raise ValueError(
            f"Unsupported [mx] ingest_mode={self.ingest_mode!r}. "
            "Use 'hex' or 'complex'."
        )

    def _configure_complex_processor(self, cir: CIRData) -> None:
        add_mx_paths()
        # print(mx_radar_utils.__file__)
        from radar_processing import ProcessingConfig, RadarDopplerProcessor  # type: ignore

        mapper = ConfigMapper(self.cfg)
        params = mapper.build_mx_params(cir.metadata)

        proc_cfg = ProcessingConfig(**params)

        self.processor = RadarDopplerProcessor(proc_cfg)
        self.postproc = None
        self.target_queue = None

        self._segment_idx = 0
        self.detections = []
        self.segment_results = []

    def _configure_hex_postprocessor(self, cir: CIRData) -> None:
        from queue import Queue
        import numpy as np

        from radar_wrapper.paths import add_mx_paths

        add_mx_paths()

        from test_modules import radar_utils as mx_radar_utils
        RadarPostprocessing = mx_radar_utils.RadarPostprocessing

        # for module import debugging
        if self.cfg.getboolean("mx", "verbose_import", fallback=False):
            import inspect
            print(f"[MX hex] radar_utils imported from: {mx_radar_utils.__file__}")
            print(
                "[MX hex] RadarPostprocessing.__init__ signature: ",
                inspect.signature(RadarPostprocessing.__init__),
            )

        SEGMENT_LENGTH = getattr(
            mx_radar_utils,
            "SEGMENT_LENGTH",
            self.cfg.getint("mx", "slow_time_size", fallback=256),
        )

        THRESHOLD = getattr(
            mx_radar_utils,
            "THRESHOLD",
            self.cfg.getfloat("mx", "cfar_threshold", fallback=0.0),
        )

        BEAM_DIRECTIONS_DEG = getattr(
            mx_radar_utils,
            "BEAM_DIRECTIONS_DEG",
            [0.0],
        )

        self.target_queue = Queue()

        self.postproc = _construct_with_supported_kwargs(
            RadarPostprocessing,
            ant_num=cir.metadata.num_antennas,
            slow_time_size=SEGMENT_LENGTH,
            fast_time_size=cir.metadata.num_taps,
            PRI=cir.metadata.pri_s,
            fast_time_resolution_ns=cir.metadata.fast_time_resolution_ns,
            channel=cir.metadata.channel,
            beam_dirs_deg=np.array(BEAM_DIRECTIONS_DEG),
            CFAR_threshold=THRESHOLD,
            target_queue=self.target_queue,
            status_dict={},
            use_threads=self.cfg.getboolean("mx", "use_threads", fallback=False),
        )


        self.processor = None
        self._segment_idx = 0
        self.detections = []
        self.segment_results = []

    # ------------------------------------------------------------------
    # run_replay
    # ------------------------------------------------------------------

    def run_replay(self, cir: CIRData):
        if self.processor is None and self.postproc is None:
            self.configure(cir)

        if self.ingest_mode == "complex":
            return self._run_replay_complex(cir)

        if self.ingest_mode == "hex":
            return self._run_replay_hex(cir)

        raise ValueError(f"Unsupported MX ingest_mode: {self.ingest_mode!r}")

    # ------------------------------------------------------------------
    # complex path: RadarDopplerProcessor.push_cir
    # ------------------------------------------------------------------

    def _run_replay_complex(self, cir: CIRData):
        if self.processor is None:
            raise RuntimeError("Complex processor is not configured")

        num_frames, num_antennas, _num_taps = cir.data.shape
        input_gain = self.cfg.getfloat("mx", "input_gain", fallback=1.0)

        for frame_idx in range(num_frames):
            last_result = None

            for ant_idx in range(min(num_antennas, 2)):
                cir_tap = cir.data[frame_idx, ant_idx, :] * input_gain
                
                last_result = self.processor.push_cir(
                    rx_index=ant_idx,
                    sequence_id=frame_idx,
                    cir=cir_tap
                )

                if self.cfg.getboolean("mx", "verbose_push", fallback=False):
                    print(f"[mx complex] frame={frame_idx}, ant={ant_idx}, result={last_result}")

            if last_result is not None:
                self._collect_segment_result_from_complex_path(
                    result=last_result,
                    frame_idx=frame_idx,
                )

        return self.detections

    def _collect_segment_result_from_complex_path(self, result, frame_idx: int) -> None:
        self.segment_results.append(result)

        for target in getattr(result, "targets", []) or []:
            det = Detection(
                frame_idx=frame_idx,
                segment_idx=self._segment_idx,
                range_cm=getattr(target, "range_cm", None),
                angle_deg=getattr(target, "angle_deg", None),
                power_db=getattr(target, "power_db", None),
                doppler_hz=getattr(target, "doppler_hz", None),
                backend="mx",
                raw=target.to_dict() if hasattr(target, "to_dict") else None,
            )
            self.detections.append(det)

        self._segment_idx += 1

    # ------------------------------------------------------------------
    # hex path: RadarPostprocessing.fill_data_by_hex
    # ------------------------------------------------------------------

    def _run_replay_hex(self, cir: CIRData):
        if self.postproc is None:
            raise RuntimeError("Hex postprocessor is not configured")

        num_frames, num_antennas, _num_taps = cir.data.shape

        ant_bitmap_list = self._get_ant_bitmap_list(num_antennas)

        input_gain = self.cfg.getfloat("mx", "input_gain", fallback=1.0)
        q_scale = self.cfg.getint("mx", "q_format_scale", fallback=256)
        q_overflow = self.cfg.get("mx", "q_format_overflow", fallback="saturate")

        for frame_idx in range(num_frames):
            for ant_idx in range(num_antennas):
                frame = cir.data[frame_idx, ant_idx, :] * input_gain

                cir_hex = frame_to_hex_string(
                    frame=frame,
                    scale=q_scale,
                    overflow=q_overflow,
                )

                ret = self.postproc.fill_data_by_hex(
                    cir_hex=cir_hex,
                    ant_bitmap=ant_bitmap_list[ant_idx],
                    st_idx=frame_idx,
                )

                if ret is not None:
                    self._collect_hex_queue_item(
                        ret,
                        frame_idx=frame_idx,
                    )

                if self.cfg.getboolean("mx", "verbose_push", fallback=False):
                    print(
                        f"[mx hex] frame={frame_idx}, ant={ant_idx}, "
                        f"ant_bitmap={ant_bitmap_list[ant_idx]}"
                    )

            self._drain_hex_target_queue(frame_idx=frame_idx)

        self._flush_hex_tail_segment_if_needed(frame_idx=num_frames - 1)
        self._drain_hex_target_queue(frame_idx=num_frames - 1)

        return self.detections

    def _get_ant_bitmap_list(self, num_antennas: int) -> list[str]:
        configured = self.cfg.get("mx", "ant_bitmap_list", fallback="").strip()

        if configured:
            values = [x.strip() for x in configured.split(",") if x.strip()]
            if len(values) < num_antennas:
                raise ValueError(
                    f"[mx] ant_bitmap_list has {len(values)} entries, "
                    f"but num_antennas={num_antennas}"
                )
            return values[:num_antennas]

        default = ["0101", "0201"]
        return default[:num_antennas]

    def _drain_hex_target_queue(self, frame_idx: int) -> None:
        if self.target_queue is None:
            return
        
        while not self.target_queue.empty():
            raw_item = self.target_queue.get()
            self._collect_hex_queue_item(raw_item, frame_idx=frame_idx)

    def _collect_hex_queue_item(self, raw_item, frame_idx: int) -> None:
        """Convert RadarPostprocessing queue output to warpper Detection.

        The old MX path often pushes either:
        - list[dict]
        - list[target-like object]
        - a single dict
        - a backend-specific result object

        This function is intentionally tolerant.
        """
        self.segment_results.append(raw_item)

        if raw_item is None:
            self._segment_idx += 1
            return

        if isinstance(raw_item, dict):
            targets = raw_item.get("targets", raw_item.get("detections", [raw_item]))
        elif isinstance(raw_item, list):
            targets = raw_item
        else:
            targets = getattr(raw_item, "targets", None)
            if targets is None:
                targets = getattr(raw_item, "detections", None)
            if targets is None:
                targets = [raw_item]

        for target in targets or []:
            det = self._target_to_detection_from_hex_path(
                target=target,
                frame_idx=frame_idx,
                segment_idx=self._segment_idx,
            )

            if det is not None:
                self.detections.append(det)

        self._segment_idx += 1

    def _target_to_detection_from_hex_path(
        self,
        target,
        frame_idx: int,
        segment_idx: int,
    ) -> Detection | None:
        if target is None:
            return None

        if isinstance(target, dict):
            raw = dict(target)

            range_cm = (
                target.get("range_cm")
                if target.get("range_cm") is not None
                else target.get("range")
            )

            angle_deg = (
                target.get("angle_deg")
                if target.get("angle_deg") is not None
                else target.get("direction")
            )

            power_db = (
                target.get("power_db")
                if target.get("power_db") is not None
                else target.get("power")
            )

            doppler_hz = target.get("doppler_hz")

        else:
            raw = target.to_dict() if hasattr(target, "to_dict") else None

            range_cm = getattr(target, "range_cm", None)
            if range_cm is None:
                range_cm = getattr(target, "range", None)

            angle_deg = getattr(target, "angle_deg", None)
            if angle_deg is None:
                angle_deg = getattr(target, "direction", None)

            power_db = getattr(target, "power_db", None)
            if power_db is None:
                power_db = getattr(target, "power", None)

            doppler_hz = getattr(target, "doppler_hz", None)

        return Detection(
            frame_idx=frame_idx,
            segment_idx=segment_idx,
            range_cm=range_cm,
            angle_deg=angle_deg,
            power_db=power_db,
            doppler_hz=doppler_hz,
            backend="mx_hex",
            raw=raw,
        )

    def _flush_hex_tail_segment_if_needed(self, frame_idx: int) -> None:
        if self.postproc is None:
            return

        flush_tail = self.cfg.getboolean(
            "mx",
            "flush_tail_segment",
            fallback=True,
        )

        if not flush_tail:
            return

        segment_idx = getattr(self.postproc, "segment_idx", 0)

        if segment_idx > 0 and hasattr(self.postproc, "run_postprocessing"):
            ret = self.postproc.run_postprocessing(
                self.postproc.segment.copy()
            )

            if ret is not None:
                self._collect_hex_queue_item(
                    ret,
                    frame_idx=frame_idx,
                )