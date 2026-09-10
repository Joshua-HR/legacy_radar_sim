from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional
import time

import numpy as np

from radar_types import Detection, GroupedTarget, SegmentResult

# Try to import matplotlib for plotting (optional dependency)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

SPPED_OF_LIGHT = 299_792_458.0
CHANNEL_FREQ_HZ = {
    5: 6489.6e6,
    6: 6988.8e6,
    8: 7488.0e6,
    9: 7987.2e6,
    10: 8486.4e6,
    12: 8985.6e6,
}


@dataclass
class ProcessingConfig:
    # ---------- Basic Parameters ----------
    cir_taps: int = 31                          # Number of taps per CIR (Complex Impulse Response) frame. # 1 tap ≈ 15 cm range resolution → 31 taps = 0-465 cm total
    segment_size: int = 16                      # Number of frames (packets) to collect in one segment (time window). # Example: 16 x 20 ms = 320 ms window

    hop_size: int = 0                           # Overlap length between segments. # 0 means completely new window, >0 preserves (segment_size - hop) frames # as the starting point for the next segment.
    period_ms: float = 5.0                      # Interval between radar frames (PRI) [ms]. # PRI = 0.005 s → PRF = 200 Hz
    channel: int = 9                            # Frequency channel number to use. # Actual Hz is converted in the 'center_freq_hz' property.
    antenna_spacing_m: float = 19e-3            # Distance between 2RX antennas [m]. # Used in AoA calculation combined with wavelength and distance to compute phase difference → angle.
    aoa_calibration_deg: float = 0.0            # Overall AoA correction values (not used in current system)
    aoa_cal_rx0_deg: float = 0.0                # Rx0 phase correction (deg). Applied as complex rotation.
    aoa_cal_rx1_deg: float = -62.0              # Rx1 phase correction (deg). Applied during calibration step.
    beam_direction_deg: float = 0.0             # Default beam direction offset for future beamformaing (deg).

    # ---------- Range/Velocity Limits ----------
    skip_bins: int = 5                          # Number of front taps (close range) to ignore. # Typically the first 5 taps (≈ 75cm) are assumed to be noise/direct reflection.
    min_range_cm: float = 75.0                  # Minimum allowed detection range [cm].
    max_range_cm: float = 430.0                 # Maximum allowed detection range [cm].
    cfar_threshold_db: float = 9.0              # SNR threshold used for CFAR detection (dB).
    min_abs_velocity_mps: float = 0.06          # Minimum absolute velocity (m/s) - object too slow are ignored.
    max_abs_angle_deg: float = 75.0             # Maximum allowed absolute AoA value (deg).
                                                # Detections outside this range are discarded.

    # ---------- CFAR Auxiliary Parameters ----------
    guard_range: int = 1                        # Number of guard cells in range dimension (CFAR protection region).
    guard_doppler: int = 1                      # Number of guard cells in doppler dimension.
    ref_range: int = 4                          # Number of reference cells in range dimension (for statistics).
    ref_doppler: int = 3                        # Number of reference cells in doppler dimension.
    max_detections: int = 16                    # Maximum number of detections to return per segment.

    # ---------- Grouping Parameters ----------
    group_range_cm: float = 45.0                # Allowed range difference (cm) to combine two detections as the same target.
    group_velocity_mps: float = 0.25            # Allowed velocity difference (m/s) to group as the same target.
    group_angle_deg: float = 18.0               # Allowed AoA difference range (deg).

    # ---------- Persistence Parameters ----------
    persistence_window: int = 5                 # Length of recent N range history to store.
    persistence_min_hits: int = 3               # How many times the same range must appear # in the above window to be considered a "persistence presence".
    persistence_range_gate_cm: float = 45.0     # Allowed torleance to judge the same range within history.

    # ---------- Doppler Edge Blocking ----------
    reject_edge_doppler_bins: int = 1           # Automatically ignore edge doppler bins (both sides). # Prevents noise/aliasing in frequently affected regions.

    # ---------- Minimum Conditions for Human Detection ----------
    min_cells_for_human: int = 2                # Minimum number of detection cells for a target to be a human candidate.
    min_rx_balance_db: float = -18.0            # Lower bound of Rx0/Rx1 power ratio (dB).
    max_rx_balance_db: float = 18.0             # Upper bound of Rx0/Rx1 power ratio (dB).
    min_motion_power_db: float = -60.0          # Minimum absolute power for motion detection (dBFS).

    active_rx_std_threshold: float = 1e-3       # Minimum CIR standard deviation to judge an Rx as "active". (vs noise level)

    require_two_rx_for_motion: bool = True      # Both 2Rx must be active for motion detection.
    bad_rx_floor_db_threshold: float = -100.0   # Power floor (dBFS) for "bad-RX" judgement.

    # ---------- Debug Options ----------
    debug_enable: bool = True                   # Enable debug print statements
    debug_plot_enable: bool = True              # Enable matplotlib plotting (if available)
    debug_print_stage: bool = True              # Print timing for each processing stage
    debug_print_detections: bool = True         # Print detection details
    debug_print_targets: bool = True            # Print target details
    debug_plot_yaxis_mode: str = "velocity"     # Y-axis mode for debug plots: "doppler_bin", "velocity", or "doppler_freq"
    debug_plot_xaxis_mode: str = "range_cm"     # X-axis mode for debug plots: "range_tap" or "range_cm"

    # ---------- Baseline (Empty-room) Calibration ----------
    empty_calibration_segments: int = 30        # Minimum number of segments needed to build baseline.
    baseline_percentiles: float = 95.0          # Percentile to calculate baseline noise level
    min_dynamic_excess_db: float = 6.0          # How many dB above baseline current power must be # to be considered dynamic detection.
    require_baseline_for_human: bool = True     # Baseline must be ready before human detection.

    # ---------- Human Shape (Tap/Range Span) Conditions ----------
    min_unique_taps_for_human: int = 2          # How many different taps must be activated to be a human candidate.
    min_range_span_cm_for_human: float = 15.0   # Minimum range width a human candidate must occupy.

    # ---------- Baseline Adaptive Update ----------
    adaptive_baseline_update: bool = True
    adaptive_baseline_alpha: float = 0.03       # How much to blend new frame into baseline (0 → fixed, 1 → immediate replacement).
    adaptive_baseline_guard_db: float = 12.0    # If excess is too large, do not use for baseline update.

    # ---------- Static Object Detection Parameters ----------
    static_excess_threshold_db: float = 12.0
    static_min_delta_db: float = -45.0
    static_min_unique_taps: int = 1
    static_rx_consistency: bool = False         # Must detection be consistent in both 2Rx?
    static_persistence_window: int = 5
    static_persistence_min_hits: int = 2
    static_baseline_percentile: float = 95.0
    enable_static_detection: bool = False       # Enable/disable static detection feature.

    # ---------- Packet Quality Related ----------
    packet_bad_truncated_ratio: float = 0.25    # Maximum truncation ratio for a packet to be considered "bad".
    packet_warn_truncated_ratio: float = 0.10   # "Warn" level truncation ratio.
    suppress_motion_on_bad_packets: bool = True # Suppress motion detection if packet is damaged.

    # ---------- Synthetic Single-tap Artifact Filter ----------
    drop_symmetric_single_tap_artifacts: bool = True
    near_artifact_max_range_cm: float = 120.0
    symmetric_velocity_gate_mps: float = 0.10
    symmetric_snr_gate_db: float = 8.0
    single_tap_far_min_range_cm: float = 150.0
    promote_far_single_tap_to_motion: bool = False  # Promote far (single-tap) detection to motion.

    # ---------- Range-track-motion (Walking Verification) ----------
    enable_range_track_motion: bool = True
    track_window_segments: int = 10             # History length (segments) to use for range-track.
    track_min_points: int = 3                   # Minimum number of valid points.
    track_min_range_change_cm: float = 45.0     # Minimum total range change.
    track_min_slope_cm_per_segment: float = 5.0 # Minimum average slope.
    track_max_misses: int = 2                   # Allowed number of consecutive missed frames.
    track_min_abs_velocity_abs: float = 0.06    # Minimum velocity to use for tracking.
    track_min_score: float = 0.0                # Lower bound of tracking candidate score.
    track_min_step_cm: float = 10.0             # Minimum individual step (consecutive range difference).
    track_min_significant_steps: int = 2        # Minimum number of significant steps.
    track_min_direction_consistency: float = 0.75   # Step direction consistency ratio.
    track_max_internal_misses: int = 1          # Allowed miss gap within history.
    require_range_track_for_human: bool = True  # Require range-track result for human judgement.

    # ---------- CIR Energy Centroid Based Direction Estimation ----------
    enable_cir_centroid_direction: bool = True
    cir_centroid_window_segments: int = 8       # History length to use for centroid tracking.
    cir_centroid_min_points: int = 3            # Minimum number of history points.
    cir_centroid_min_excess_db: float = 5.0     # Minimum excess to calculate centroid.
    cir_centroid_min_delta_cm: float = 7.0      # Minimum range change to judge movement.
    cir_centroid_min_total_weight: float = 2.0  # Cumulative weight must be at least this value for centroid to be significant.
    cir_centroid_max_internal_misses: int = 2   # Allowed missed frames within history.
    cir_centroid_min_consistency: float = 0.60  # Lower bound of direction consistency.

    # ---------- Near-zone Block (Suppress False Motion at Close Range) ----------
    near_zone_block_enabled: bool = True
    near_zone_block_max_cm: float = 50.0        # Maximum range to check for blocking.
    near_zone_block_excess_db: float = 6.0      # Minimum required excess in near-zone.
    near_zone_block_min_cells: int = 2          # Minimum number of cells needed for blocking judgement.
    near_zone_block_min_power_db: float = -95.0 # Power floor condition.

    @property
    def pri_s(self) -> float:
        """Pulse-Repitition-Interval (seconds)"""
        return self.period_ms * 1e-3

    @property
    def center_freq_hz(self) -> float:
        """Center frequency (Hz) corresponding to channel number. Default is channel 9 (7.9872 GHz)."""
        return CHANNEL_FREQ_HZ.get(int(self.channel), 7987.2e6)

    @property
    def wavelength_m(self) -> float:
        """Radar wavelength (m) = c / fo"""
        return SPPED_OF_LIGHT / self.center_freq_hz


class RadarDopplerProcessor:
    """Range-gated Doppler detector for UA200 two-RX CIR radar.

    v5 changes:
    - Detects inactive/frozen RX channels per segment.
    - Combines only active RX channels for Doppler detection.
    - Disables AoA automatically when only one RX is active.
    - Uses linear-power CFAR internally to avoid false detections caused by dB floor values.
    """

    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg                                                                          # Store the passed configuration object

        self.segment = np.zeros((2, cfg.segment_size, cfg.cir_taps), dtype=np.complex64)          # Complex CIR buffer of size 2-channel x segment_size x cir_taps (0-Rx,1-Rx | frame index | distance tap)
        self.valid = np.zeros((2, cfg.segment_size), dtype=bool)                                # Bool flag array indicating whether actual CIR data is present for each (rx, frame)
        self.hop_size = int(cfg.hop_size) if int(cfg.hop_size) > 0 else int(cfg.segment_size)   # hop_size setting: use full segment_size if 0 or less
        self.hop_size = max(1, min(self.hop_size, int(cfg.segment_size)))                       # Clamp hop_size to minimum 1, maximum segment_size
        self.write_idx = 0                                                                      # Initialize position index for writing new frame to current segment buffer
        self.segment_id = 0                                                                     # Initialize consecutive number of processed segments (time windows)
        self.pending: dict[int, dict[int, np.ndarray]] = {}                                     # Initialize temporary storage by sequence (seq → {rx:cir})
        self.max_pending_sequences = max(8, cfg.segment_size * 4)                               # Maximum number of sequences to keep in pending (prevent memory explosion)
        self.last_committed_sequence: Optional[int] = None                                      # Last successfully completed sequence number (None if not yet)
        self.packets_seen = 0                                                                   # Initialize total received packet (frame) counter
        self.complete_pairs = 0                                                                 # Initialize cumulative counter for how many times complete (Rx0+Rx1) pairs occurred
        self.target_history: deque[float] = deque(maxlen=cfg.persistence_window)                # Initialize history for range persistence judgement (recent N ranges)
        self.raw_candidate_history: deque[dict[str, float]] = deque(maxlen=max(4, cfg.track_window_segments))   # Initialize Raw-candidate history
        self.cir_centroid_history: deque[dict[str, float]] = deque(maxlen=max(4, int(getattr(cfg, "cir_centroid_window_segments", 8)))) # Initialize CIR-centroid history (track energy center movement)

        self.baseline_frames: list[np.ndarray] = []                                             # Frame list for Baseline (Empty-room) Calibration
        self.baseline_db: Optional[np.ndarray] = None                                           # Percentile-based noise level (dB), not yet available
        self.baseline_ready: bool = False                                                       # Baseline readiness flag (initially False)

        self.static_baseline_frames: list[np.ndarray] = []                                      # Frame list used to create average CIR for static object detection
        self.static_reference: Optional[np.ndarray] = None                                      # Static CIR average reference (complex), not yet available
        self.static_noise_db: Optional[np.ndarray] = None                                       # Static noise level (dB), not yet available
        self.static_history: deque[bool] = deque(maxlen=cfg.static_persistence_window)          # Initialize static change persistence history (recent N detection results)

        self.last_result: Optional[SegmentResult] = None                                        # Initialize cache for most recently returned SegmentResult
        self.last_rx_health: dict[str, object] = {}                                             # Initialize dictionary to stor Rx health information measured in last frame

        self.range_cm = np.arange(cfg.cir_taps, dtype=float) * 15.0                             # Generate range table: tap index x 15 cm → distance (cm)
        self.doppler_hz = np.fft.fftshift(np.fft.fftfreq(cfg.segment_size, d=cfg.pri_s))        # Generate doppler (Hz) table: get frequency bins with np.fft.fftfreq and move DC to center with fftshift

        self.velocity_mps = self.doppler_hz * cfg.wavelength_m / 2.0                              # Velocity (m/s) table = v = f·λ/2 (λ = wavelength, f = doppler Hz)
        self.dc_index = int(np.argmin(np.abs(self.doppler_hz)))                                 # Store index of DC (stationary) doppler bin (position closest to 0 Hz)

        # Debug storage for plotting
        self._debug_data: dict = {}

    def _time_call_ms(self, timing_name: str, func, *args, **kwargs):
        """Time a selected processing function with very small overhead.

        This is debug instrumentation only. It does not modify the returned value
        or any detector threshold. Values are attached later to result.rx_health.
        """
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            try:
                self._last_function_timings[timing_name] = float((time.perf_counter() - t0) * 1000.0)
            except Exception:
                pass

    def push_cir(self, rx_index: int, sequence_id: int, cir: np.ndarray) -> Optional[SegmentResult]:
        push_t0 = time.perf_counter()                           # Start time of entire function execution (high resolution)
        push_timings: dict[str, float] = {}                    # Dictionary to stor execution (ms) for each step
        self.packets_seen += 1                                  # Increment total received packet (frame) count

        ##------------------------------------------------##
        t_stage = time.perf_counter()   # Preprocessing stage start time
        if rx_index not in (0, 1):
            return None
        if cir.size < self.cfg.cir_taps:
            return None

        cir = np.asarray(cir[:self.cfg.cir_taps], dtype=np.complex64)   # Force length and convert to complex64
        if self.last_committed_sequence is not None and sequence_id <= self.last_committed_sequence:    # Ignore if already processed (or old) sequence
            return None
        push_timings["processor_push_precheck_ms"] = float((time.perf_counter() - t_stage) * 1000.0)    # Record time taken for preprocessing (index/length/sequence) check

        ##------------------------------------------------##
        t_stage = time.perf_counter()                   # Pair assembly stage start time
        pair = self.pending.setdefault(sequence_id, {}) # Create empty dict if not yet present for this sequence
        pair[rx_index] = cir                            # Store current Rx CIR in sequence dict
        # If number of sequences stored in pending exceeds limit, delete oldest ones first
        if len(self.pending) > self.max_pending_sequences:
            for old_seq in sorted(self.pending.keys())[:-self.max_pending_sequences]:
                self.pending.pop(old_seq, None)

        if len(pair) < 2:       # Do nothing before receiving 2Rx
            return None
        push_timings["processor_push_pair_assembly_ms"] = float((time.perf_counter() - t_stage) * 1000.0)   # Record time taken until 2Rx received

        ##------------------------------------------------##
        t_stage = time.perf_counter()                   # Stage start time for writing to segment buffer
        self.segment[0, self.write_idx, :] = pair[0]    # Copy Rx0 data to current write_idx position
        self.segment[1, self.write_idx, :] = pair[1]    # Copy Rx1 data to current write_idx position
        self.valid[:, self.write_idx] = True            # Mark both channels as valid
        self.write_idx += 1                             # Move to index for receiving next frame
        self.complete_pairs += 1                        # Record that one complete Rx pair has been added
        self.last_committed_sequence = sequence_id      # Store the sequence number successfully recorded most recently
        self.pending.pop(sequence_id, None)             # Remove used sequence from pending
        push_timings["processor_push_segment_write_ms"] = float((time.perf_counter() - t_stage) * 1000.0)   # Record time taken for above buffer write operation

        if self.write_idx < self.cfg.segment_size:  # Wait for more frames until segment is fully filled
            return None

        ##------------------------------------------------##
        t_stage = time.perf_counter()
        process_segment = bool(np.all(self.valid))                       # True if all channels/frames are valid
        segment_copy = self.segment.copy() if process_segment else None # Make a copy (for buffer rearrangement next)
        last_seq = self.last_committed_sequence or sequence_id          # Last sequence number of current segment

        # hop (overlap) processing: if hop < segment_size, preserve remaining frames and clear the front.
        if process_segment and segment_copy is not None and self.hop_size < self.cfg.segment_size:
            keep = self.cfg.segment_size - self.hop_size
            self.segment[:, :keep, :] = self.segment[:, self.hop_size:, :]  # Copy hop amount from back to front
            self.valid[:, :keep] = self.valid[:, self.hop_size:]            # Move valid flags identically
            self.segment[:, keep:, :] = 0                                   # Initialize remaining back portion to 0
            self.valid[:, keep:] = False                                    # Set back portion valid to False
            self.write_idx = keep                                           # Reset next write position to keep
        else:                                                               # When hop==segment_size (i.e., no overlap)
            self.segment.fill(0)                                            # Initialize entire buffer to 0
            self.valid.fill(False)                                          # Set all valid flags to False
            self.write_idx = 0                                              # Reset write index to 0
        push_timings["processor_push window_shift_ms"] = float((time.perf_counter() - t_stage) * 1000.0)    # Record time taken for hiop/shift operation


        if not process_segment or segment_copy is None: # Terminate here if not a valid segment
            return None

        process_start_mono = time.monotonic()                               # Record start time of monotonic clock (actual wall-time)
        process_perf = time.perf_counter()                                  # Record start time of high-resolution counter
        self._last_function_timings = {}                                    # Initialize internal timing dictionary

        result = self._process_segment(segment_copy, last_seq)              # Call core algorithm → return SegmentResult

        process_done_mono = time.monotonic()                                # Processing end time (monotonic)
        process_ms = float((time.perf_counter() - process_perf) * 1000.0)   # Total processing time (ms)

        # ------------------- Add timing/metadata to rx_health -------------------
        try:
            h = result.rx_health or {}                                      # Get existing rx_health dict (or new dict if none)
            h.update(push_timings)                                          # Merge push stage timing (preprocessing/assembly/shift)
            h.update(getattr(self, "_last_function_timings", {}) or {})    # Merge timings recorded inside _process_segment
            h["processor_fn__process_segment_total_ms"] = process_ms
            h["processor_push_total_ms"] = float((time.perf_counter() - push_t0) * 1000.0)
            h["processor_segment_ready_mono_s"] = float(process_start_mono)
            h["processor_segment_done_mono_s"] = float(process_done_mono)
            h["processor_segment_compute_ms"] = float((process_done_mono - process_start_mono) * 1000.0)
            h["processor_segment_size"] = int(self.cfg.segment_size)
            h["processor_hop_size"] = int(self.hop_size)
            h["processor_period_ms"] = float(self.cfg.period_ms)
            h["processor_algorithm_window_ms"] = float(self.cfg.segment_size * self.cfg.period_ms)
            h["processor_algorithm_hop_ms"] = float(self.hop_size * self.cfg.period_ms)
            h["processor_complete_pairs"] = int(self.complete_pairs)
            h["processor_write_idx_after_shift"] = int(self.write_idx)
            result.rx_health = h    # Save modified health dict back to result
        except Exception:
            pass                    # Maintain processing flow even if health update fails
        self.last_result = result   # Cache the most recent result
        self.segment_id += 1        # Increment segment counter
        return result               # Return final SegmentResult


    def _rx_health(self, seg_hp: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        cfg = self.cfg
        usable = np.arange(cfg.cir_taps) >= cfg.skip_bins
        usable &= self.range_cm >= cfg.min_range_cm
        usable &= self.range_cm <= cfg.max_range_cm
        if not np.any(usable):
            usable[:] = True


        std_per_rx = np.median(np.std(seg_hp[:, :, usable], axis=1), axis=1)
        active = std_per_rx >= cfg.active_rx_std_threshold
        active_count = int(np.sum(active))
        bad_rx = bool((cfg.require_two_rx_for_motion and active_count < 2) or active_count == 0)

        health = {
            "rx_std": [float(x) for x in std_per_rx],
            "active_rx": [bool(x) for x in active],
            "active_rx_count": active_count,
            "bad_rx_segment": bad_rx,
            "bad_rx_reason": "need_RX1_RX2_active" if bad_rx else "OK",
        }
        self.last_rx_health = health
        return active, health

    def _make_no_motion_result(
        self,
        *,
        last_sequence_id: int,
        state: str,
        noise_floor_db: float,
        max_power_db: float,
        rx_health: dict[str, object],
        static_ready: bool = False,
    ) -> SegmentResult:

        self.target_history.append(float("nan"))
        self.raw_candidate_history.append({
            "segment_id": float(self.segment_id),
            "range_cm": float("nan"),
            "velocity_mps": float("nan"),
            "score": float("nan"),
            "snr_db": float("nan"),
            "unique_taps": 0.0,
            "detections_count": 0.0,
        })
        self.cir_centroid_history.append({
            "segment_id": float(self.segment_id),
            "centroid_range_cm": float("nan"),
            "total_weight": 0.0,
        })
        return SegmentResult(
            segment_id=self.segment_id,
            last_sequence_id=last_sequence_id,
            packets_seen=self.packets_seen,
            complete_pairs=self.complete_pairs,
            detections=[],
            targets=[],
            human_detected=False,
            noise_floor_db=float(noise_floor_db),
            max_power_db=float(max_power_db),
            rx_health=rx_health,
            object_change_detected=False,
            motion_detected=False,
            person_candidate=False,
            state=state,
            static_detection={"ready": bool(static_ready), "persistent": False, "active_taps": []},
        )

    def _process_segment(self, seg: np.ndarray, last_sequence_id: int) -> SegmentResult:
        cfg = self.cfg                          # Bind configuration object to local variable (for faster access)
        eps = 1e-12                             # Very small constant to prevent 0-divide in log/division

        # Debug: Clear previous debug data
        if cfg.debug_enable:
            self._debug_data = {"segment_id": self.segment_id}

        # ---------------------- 1 Preprocessing - DC Removal & Phase Correction ----------------------
        _stage_t0 = time.perf_counter()         # Record preprocessing stage start time (for timing measurement)
        dc_temp = np.mean(seg, axis=1, keepdims=True)
        seg_hp = seg - dc_temp                  # Subtract per-Rx mean (DC) to get zero-mean CIR
        
        seg_hp[0] *= np.exp(1j * np.deg2rad(cfg.aoa_cal_rx0_deg))   # Rx0 phase correction (complex rotation)
        seg_hp[1] *= np.exp(1j * np.deg2rad(cfg.aoa_cal_rx1_deg))   # Rx1 phase correction

        active_rx, rx_health = self._time_call_ms("processor_fn__rx_health_ms", self._rx_health, seg_hp)    # Get Rx active status and health info. Execution time is recorded in _last_function_timings

        # Save time (ms) for entire preprocessing stage to health dict (for debug)
        try:
            self._last_function_timings["processor_stage__preprocess_phase_ms"] = float((time.perf_counter() - _stage_t0) * 1000.0)
        except Exception:
            pass

        # ---------------------- 2 Window Function Application & FFT ----------------------
        _stage_t0 = time.perf_counter()                             # FFT pre-stage start time record
        win = np.hanning(cfg.segment_size).astype(np.float32)       # Generate Hanning window (prevent spectral leakage)
        if np.allclose(win, 0):                                     # special case like segment_size = 1 → use unit window
            win = np.ones(cfg.segment_size, dtype=np.float32)

        x = seg_hp * win[None, :, None]                             # Apply window only to frame dimension (Broadcast)
        fft_rx = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)     # Result shape = [rx, doppler, tap]  # Perform FFT on time (frame) axis → doppler spectrum
        power_rx = np.abs(fft_rx) ** 2                              # Calculate power (magnitude squared) of complex FFT result

        # === DEBUG PLOT: seg_hp per-frame, CIR (fast-time), fft_rx power ===
        if cfg.debug_enable and cfg.debug_plot_enable and MATPLOTLIB_AVAILABLE:
            self._plot_cir_debug(seg_hp, x, eps)
        # === END DEBUG PLOT ===


        # Sum power using only active Rx (both may be active, or only one)
        active_indices = np.where(active_rx)[0]                     # Array of active Rx indices
        power = np.sum(power_rx[active_indices, :, :], axis=0)      # [doppler, tap] power matrix
        power_db = 10.0 * np.log10(power + eps)                     # Convert power to dB scale

        # ---------------------- 3 Create Detection Mask ----------------------
        velocity_mask = np.abs(self.velocity_mps) >= cfg.min_abs_velocity_mps       # Doppler bins with minimum velocity or higher
        velocity_mask[self.dc_index] = False                                        # DC (stationary) bin must be excluded
        edge = max(0, int(cfg.reject_edge_doppler_bins))                            # Number of edge doppler bins to block
        if edge > 0 and velocity_mask.size > 2 * edge:
            velocity_mask[:edge] = False
            velocity_mask[-edge:] = False

        # Range mask: satisfy both specified min/max range and skip_bins
        range_mask = (
            (self.range_cm >= cfg.min_range_cm) &
            (self.range_cm <= cfg.max_range_cm) &
            (np.arange(cfg.cir_taps) >= cfg.skip_bins)
        )
        valid_map = velocity_mask[:, None] & range_mask[None, :]                    # 2-D (doppler x tap) valid cell mask

        # Record time for FFT-post mask generation (for debug)
        try:
            self._last_function_timings["processor_stage__fft_power_mask_ms"] = float((time.perf_counter() - _stage_t0) * 1000.0)
        except Exception:
            pass

        # Debug: Store intermediate data for plotting
        if cfg.debug_enable:
            self._debug_data.update({
                "power_db": power_db.copy(),
                "valid_map": valid_map.copy(),
                "active_rx": active_rx.copy(),
            })
            if cfg.debug_print_stage:
                print(f"[DBG] Seg {self.segment_id}: FFT done, shape={power_db.shape}, active_rx:{active_rx}")

        # ---------------------- 4 Bad-RX Processing (Insufficient Active Rx) ----------------------
        if bool(rx_health.get("bad_rx_segment", False)):
            # Extract only finite values among valid cells to estimate noise floor and max power
            finite = power_db[valid_map]
            finite = finite[np.isfinite(finite)]
            noise_floor_db = float(np.median(finite)) if finite.size else -120.0
            max_power_db = float(np.max(finite)) if finite.size else -120.0

            # Fill health dictionary with current baseline/tracking status
            rx_health["baseline_ready"] = bool(self.baseline_ready)
            rx_health["baseline_segments"] = len(self.baseline_frames)
            rx_health["baseline_required"] = int(cfg.empty_calibration_segments)
            rx_health["baseline_mode"] = ("bad_rx_skip" if self.baseline_ready else "bad_rx_skip_calibration")
            rx_health["max_dynamic_excess_db"] = -300.0
            rx_health["raw_targets_count"] = 0
            rx_health["credible_targets_count"] = 0
            rx_health["artifact_targets_count"] = 0
            rx_health["track_motion"] = False
            rx_health["track_direction"] = "BAD_RX"
            rx_health["track_valid_points"] = 0

            state = ("BAD_RX_SEGMENT" if self.baseline_ready else "BAD_RX_CALIBRATING")
            # Immediately return "no-motion" result without motion/human detection
            return self._make_no_motion_result(
                last_sequence_id=last_sequence_id,
                state=state,
                noise_floor_db=noise_floor_db,
                max_power_db=max_power_db,
                rx_health=rx_health,
                static_ready=bool(self.static_reference is not None),
            )

        #---------------------- 5 Baseline (Empty-room) Calibration ----------------------
        if not self.baseline_ready:
            _stage_t0 = time.perf_counter()                             # Record calibration stage start time
            if np.any(valid_map):
                self.baseline_frames.append(power_db.copy())            # If at least one valid cell exists, save current frame
                self.static_baseline_frames.append(np.mean(seg, axis=1).astype(np.complex64))

            calib_total = max(0, int(cfg.empty_calibration_segments))   # Required number of frames
            calib_count = len(self.baseline_frames)                     # Number of frames collected so far

            # Record current calibration progress in health
            rx_health["baseline_ready"] = False
            rx_health["baseline_segments"] = calib_count
            rx_health["baseline_required"] = calib_total
            rx_health["baseline_mode"] = "collecting_empty_room"


            if calib_total <= 0 or calib_count >= calib_total:          # If collected enough, create baseline
                # Create reference/noise for Baseline detection simultaneously
                if self.baseline_frames:
                    stack = np.stack(self.baseline_frames, axis=0)
                    self.baseline_db = np.percentile(stack, float(cfg.baseline_percentiles), axis=0)    # baseline_percentile: float = 95.0
                else:
                    self.baseline_db = np.full_like(power_db, -300.0)

                # Create reference/noise for static detection simultaneously
                if self.static_baseline_frames:
                    sstack = np.stack(self.static_baseline_frames, axis=0)  # [K,rx,tap]
                    self.static_reference = np.mean(sstack, axis=0).astype(np.complex64)
                    sdelta = sstack - self.static_reference[None, :, :]
                    sdelta_db = 20.0 * np.log10(np.abs(sdelta) + 1e-9)
                    self.static_noise_db = np.percentile(sdelta_db, float(cfg.static_baseline_percentile), axis=0)
                else:
                    self.static_reference = np.mean(seg, axis=1).astype(np.complex64)
                    self.static_noise_db = np.full((2, cfg.cir_taps), -120.0, dtype=float)

                self.baseline_ready = True
                rx_health["baseline_ready"] = True
                rx_health["baseline_mode"] = "ready"
                # Record time taken for calibration
                try:
                    self._last_function_timings["processor_stage__baseline_collect_or_build_ms"] = float((time.perf_counter() - _stage_t0) * 1000.0)
                except Exception:
                    pass

            else:       # When not yet collected enough, return "CALIBRATING" result with current noise floor / max power
                try:
                    self._last_function_timings["processor_stage__baseline_collect_or_build_ms"] = float((time.perf_counter() - _stage_t0) * 1000.0)
                except Exception:
                    pass
            finite_valid = power_db[valid_map]
            finite_valid = finite_valid[np.isfinite(finite_valid)]
            noise_floor_db = float(np.median(finite_valid)) if finite_valid.size else -300.0
            max_power_db = float(np.max(finite_valid)) if finite_valid.size else -300.0
            return self._make_no_motion_result(
                last_sequence_id=last_sequence_id,
                state="CALIBRATING",
                noise_floor_db=noise_floor_db,
                max_power_db=max_power_db,
                rx_health=rx_health,
                static_ready=False,
            )

        # ---------------------- 6 Assign Default Values if No Baseline (Protection Code) ----------------------
        if self.baseline_db is None:
            self.baseline_db = np.full_like(power_db, -300.0)   # Initialize to -300 dB (extremely low value)

        # ---------------------- 7 Dynamic Excess (Power - Baseline) Calculation & Final Valid Cell Mask ----------------------
        dynamic_excess_db = power_db - self.baseline_db     # Difference between current power and baseline (dB)

        # Valid only if both power and baseline are above floor (default -100 dB)
        valid_power_floor = power_db > float(getattr(cfg, "bad_rx_floor_db_threshold", -100.0))
        valid_baseline_floor = self.baseline_db > float(getattr(cfg, "bad_rx_floor_db_threshold", -100.0))

        # Final detection candidate cells: (velocity/range mask) ⋀ (power floor) ⋀ (baseline floor) ⋀ (excess >= min_dynamic_excess_db)
        dynamic_valid = valid_map & valid_power_floor & valid_baseline_floor & (dynamic_excess_db >= cfg.min_dynamic_excess_db)

        # ---------------------- 8 Auxiliary Module Calls ----------------------
        # 1) CIR-energy centroid (direction): Judge if movement is "TOWARD / BACK"
        centroid_info = self._time_call_ms("processor_fn__cir_energy_centroid_direction_ms", self._cir_energy_centroid_direction, dynamic_excess_db, power_db, valid_map)

        # 2) Near-zone block: Suppress false-motion if strong signal at very close range
        near_zone_info = self._time_call_ms("proceess_fn__near_zone_block_status_ms", self._near_zone_block_status, power_db, dynamic_excess_db, velocity_mask)

        # If near-zone blocking is activated, immediately return no-motion result
        if bool(near_zone_info.get("blocked", False)):
            finite_valid = power_db[valid_map]
            finite_valid = finite_valid[np.isfinite(finite_valid)]
            noise_floor_db = float(np.median(finite_valid)) if finite_valid.size else -300.0
            max_power_db = float(np.max(finite_valid)) if finite_valid.size else -300.0
            rx_health["baseline_ready"] = bool(self.baseline_ready)
            rx_health["baseline_segments"] = len(self.baseline_frames)
            rx_health["baseline_required"] = int(cfg.empty_calibration_segments)
            rx_health["baseline_mode"] = "ready"
            rx_health["near_zone_blocked"] = True
            rx_health.update(centroid_info)
            rx_health.update(near_zone_info)
            rx_health["max_dynamic_excess_db"] = float(np.max(dynamic_excess_db[valid_map])) if np.any(valid_map) else -300.0
            rx_health["raw_targets_count"] = 0
            rx_health["credible_targets_count"] = 0
            rx_health["artifact_targets_count"] = 0
            rx_health["track_motion"] = False
            rx_health["track_direction"] = "NEAR_ZONE"
            rx_health["track_valid_points"] = 0
            return self._make_no_motion_result(
                last_sequence_id=last_sequence_id,
                state="NEAR_ZONE_SUPPRESSED",
                noise_floor_db=noise_floor_db,
                max_power_db=max_power_db,
                rx_health=rx_health,
                static_ready=bool(self.static_reference is not None),
            )

        # ---------------------- 9 CFAR Detection & Grouping ----------------------
        detections = self._time_call_ms(
            "processor_fn__cfar_detect_ms", # Timing key
            self._cfar_detect,                  # Actual CFAR algorithm
            power,
            power_db,
            dynamic_excess_db,
            power_rx,
            fft_rx,
            dynamic_valid,
            active_rx,
        )
        # Sort by score in descending order and keep only top cfg.max_detections
        detections = sorted(detections, key=lambda d: d.score, reverse=True)[: cfg.max_detections]
        # Group Detections by range/velocity/AoA proximity to create `GroupedTarget` list
        raw_targets = self._time_call_ms("processor_fn__group_detections_ms", self._group_detections, detections)   # After grouping

        # Debug: Print detection info
        if cfg.debug_enable:
            self._debug_data["detections"] = detections
            self._debug_data["raw_targets"] = raw_targets
            if cfg.debug_print_detections and detections:
                print(f"[DBG] Seg {self.segment_id}: {len(detections)} detections")
                for i, d in enumerate(detections[:3]):
                    print(f"  [Detections {i}] range={d.range_cm:.1f}cm, vel={d.velocity_mps:.2f}m/s, snr={d.snr_db:.1f}dB, score={d.score:.1f}")

        # ---------------------- 10 Raw-candidagte History Update ----------------------
        if raw_targets:                 # If at least one exists, record the highset scoring (first) stage
            rt0 = raw_targets[0]        # Max scored detection info
            self.raw_candidate_history.append({
                "segment_id": float(self.segment_id),
                "range_cm": float(rt0.range_cm),
                "velocity_mps": float(rt0.velocity_mps),
                "score": float(rt0.score),
                "snr_db": float(rt0.snr_db),
                "unique_taps": float(rt0.unique_taps),
                "detections_count": float(rt0.detections_count),
            })
        else:                           # When no target, record NaN/0 for "no candidate"
            self.raw_candidate_history.append({
                "segment_id": float(self.segment_id),
                "range_cm": float("nan"),
                "velocity_mps": float("nan"),
                "score": float("nan"),
                "snr_db": float("nan"),
                "unique_taps": 0.0,
                "detections_count": 0.0,
            })
        
        # ---------------------- 11 Define Helpers for Artifact / Credible Target Classification ----------------------
        artifact_targets: list[GroupedTarget] = []  # List to hold targets later judged as artifact (false)
        credible_targets: list[GroupedTarget] = []  # List to hold actual "credible" targets

        # Helper to check minimum conditions for target (group) shape to be human candidate
        def _shape_ok(t: GroupedTarget) -> bool:
            return bool(
                t.detections_count >= cfg.min_cells_for_human           # Minimum cell count
                and t.unique_taps >= cfg.min_unique_taps_for_human      # Unique tap count
                and t.range_span_cm >= cfg.min_range_span_cm_for_human  # Range span
            )
        # ---------------------- 12 Helper to determine "symmetric single-tap artifact" ----------------------
        def _is_symmetric_near_artifact(t: GroupedTarget, all_targets: list[GroupedTarget]) -> bool:
            if not bool(getattr(cfg, "drop_symmetric_single_tap_artifacts", True)):     # 1) Return False immediately if artifact blocking is turned off in settings
                return False
            if t.unique_taps > 1:                                                       # 2) Only check single-tap (if unique_taps > 1, not an artifact)
                return False
            if t.range_cm > float(cfg.near_artifact_max_range_cm):                      # 3) If too far (near-artifact max range), reject as not artifact
                return False
            if abs(t.velocity_mps) < float(cfg.min_abs_velocity_mps):                   # 4) If velocity is nearly 0 (stationary), reject as not artifact
                return False

            for other in all_targets:
                if other is t:          # Skip self
                    continue
                if other.unique_taps:   # Other must also be single-tap
                    continue
                same_range = abs(other.range_cm - t.range_cm) <= 15.0               # Range difference <= 15 cm
                opposite_velocity = abs(other.velocity_mps + t.velocity_mps) <= float(cfg.symmetric_velocity_gate_mps)  # Velocity sign opposite, absolute value <= gate
                similar_power = abs(other.snr_db + t.snr_db) <= float(cfg.symmetric_snr_gate_db)                        # SNR difference <= gate
                if same_range and opposite_velocity and similar_power:
                    return True             # If all conditions satisfied, it's a symmetric artifact

            return False                    # In no pair found, not an artifact

        # ---------------------- 13 Iterate through Raw-target list to classify as "artifact / credible"
        for t in raw_targets:                           # Check each target created by CFAR-post grouping one by one
            shape_ok = _shape_ok(t)                     # 1) Does it satisfy minimum human shape conditions?
            # 2) Check "promote far single-tap to motion" option
            far_single_tap_ok = bool(
                bool(getattr(cfg, "promote_far_single_tap_to_motion", False))       # Is option turned on?
                and t.range_cm >= float(cfg.single_tap_far_min_range_cm)            # Is it far enough?
                and t.detections_count >= cfg.min_cells_for_human                   # Does it satisfy minimum cell count?
            )
            # 3) Check if it's a symmetric single-tap artifact
            symmetric_artifact = _is_symmetric_near_artifact(t, raw_targets)
            if symmetric_artifact:              # If symmetric artifact, immediately add to artifact list and continue to next target
                artifact_targets.append(t)
                continue
            if shape_ok or far_single_tap_ok:   # If shape is OK or option-promoted, add to credible list
                credible_targets.append(t)
            else:                               # if both conditions above are not met, consider as artifact
                artifact_targets.append(t)

        # ---------------------- 14 Sort credible targets by score in descending order → final target candidates ----------------------
        targets = sorted(credible_targets, key=lambda t: t.score, reverse=True)

        # Debug: print target info
        if cfg.debug_enable and cfg.debug_print_targets and targets:
            print(f"[DBG] Seg {self.segment_id}: {len(targets)} credible targets (artifact: {len(artifact_targets)})")
            for i, t in enumerate(targets[:3]):
                print(f"  [{i}] range={t.range_cm:.1f}, vel={t.velocity_mps:.2f}, taps={t.unique_taps}, span={t.range_span_cm:.1f}cm, score={t.score:.1f}")

        # ---------------------- 15 Range_track-motion verification (walking movement) - use timing measurement wrapper ----------------------
        track_motion, track_info = self._time_call_ms("processor_fn__range_track_motion_ms", self._range_track_motion, raw_targets) # (timing key, tracking function, input values)

        # ---------------------- 16 If tracking is true but no credible targets, forcibly use first-target as target
        if track_motion and not targets and raw_targets:
            targets = [raw_targets[0]]
        # ---------------------- 17 Static object change detection - record execution time with time-call wrapper ----------------------
        object_change_detected, static_info = self._time_call_ms("processor_fn__static_object_change_ms", self._static_object_change, seg)

        # ---------------------- 18 Judge "range persistence" based on current highest scoring target
        current_range = None
        best_shape_ok = False
        if targets:                             # If at least one credible target exists
            best = targets[0]                   # Select highest score target
            best_shape_ok = _shape_ok(best)     # Does this target satisfy human shape conditions?

            if best_shape_ok:                   # If shape is OK, save current range to history
                current_range = float(best.range_cm)
                self.target_history.append(current_range)
            else:                               # If shape doesn't match, record as NaN (ignore in persistence judgement)
                self.target_history.append(float("nan"))
        else:                                   # Even when no targets at all, insert NaN
            self.target_history.append(float("nan"))
        # ---------------------- 19 Persistence Check - How many times does current range appear in recent N range history? ----------------------
        persistent = False
        if current_range is not None:               # If current range is valid
            hits = 0
            for r in self.target_history:           # Check entire history
                if np.isfinite(r) and abs(float(r) - current_range) <= cfg.persistence_range_gate_cm:
                    hits += 1                       # Increment hit count if within tolerance
            persistent = hits >= cfg.persistence_min_hits   # If exceeds minimum hit count, persistence is True

        # ---------------------- 20 Human Detection Final Judgement - Must satisfy multiple conditions to be True ----------------------
        baseline_ok = self.baseline_ready or not cfg.require_baseline_for_human
        human_detected = bool(
            baseline_ok             # When baseline is ready or not required
            and targets             # At least one credible target exists
            and persistent          # Range persistence satisfied
            and best_shape_ok       # Best target satisfies shape criteria
            and ((not bool(getattr(cfg, "require_range_track_for_human", True))) or bool(track_info.get("track_motion", False))) # Depending on track-motion requirement option: if option off, always pass; otherwise track motion must be true
        )
        # If judged as human, assign flag to all targets with range within range
        for t in targets:
            t.human_like = bool(human_detected and abs(t.range_cm - targets[0].range_cm) <= cfg.persistence_range_gate_cm)

        # ---------------------- 21 Set Other Flags (motion, candidate, etc.) ----------------------
        motion_detected = bool(targets)             # If credible target exists, motion is True
        raw_doppler_candidate = bool(raw_targets)   # If raw-target exists at least one, candidate is True
        static_raw_change = bool(static_info.get("raw_change", False))
        person_candidate = bool(object_change_detected or motion_detected or static_raw_change)

        # ---------------------- 22 Determine Final `state` String - Status code for display in debug/UI ----------------------
        if human_detected:
            state = "HUMAN_CONFIRMED"
        elif object_change_detected and motion_detected:
            state = "OBJECT_CHANGE+MOTION"
        elif object_change_detected:
            state = "OBJECT_CHANGE"
        elif static_raw_change and motion_detected:
            state = "STATIC_CANDIDATE+MOTION"
        elif static_raw_change:
            state = "STATIC_CANDIDATE"
        elif bool(track_info.get("track_motion", False)) and motion_detected:
            state = "TRACK_MOTION"
        elif motion_detected:
            state = "MOTION"
        elif raw_doppler_candidate:
            state = "DOPPLER_CANDIDATE"
        else:
            state = "CLEAR"

        # ---------------------- 23 Determine Baseline Auto-update Status ----------------------
        baseline_updated = False

        baseline_freeze_reason = "none"
        freeze_baseline = bool(
            human_detected
            or person_candidate
            or raw_doppler_candidate
            or bool(targets)
            or bool(raw_targets)
            or static_raw_change
            or object_change_detected
        )
        if freeze_baseline:
            baseline_freeze_reason = "candidate_or_track_present"

        # Actually perform auto-update
        if (
            cfg.adaptive_baseline_update
            and self.baseline_ready
            and self.baseline_db is not None
            and not freeze_baseline
        ):
            alpha = float(np.clip(cfg.adaptive_baseline_alpha, 0.0, 1.0))
            if alpha > 0.0:
                guard = dynamic_excess_db >= float(cfg.adaptive_baseline_guard_db)      # Exclude cells with too large dynamic_excess as "guard", use only rest for update
                update_mask = valid_map & (~guard)
                self.baseline_db[update_mask] = (1.0 - alpha) * self.baseline_db[update_mask] + alpha * power_db[update_mask]

                baseline_updated = bool(np.any(update_mask))

        # ---------------------- 24 Calculate Final Noise-floor / Max Power (use only valid cells)
        finite_valid = power_db[valid_map]
        finite_valid = finite_valid[np.isfinite(finite_valid)]
        noise_floor_db = float(np.median(finite_valid)) if finite_valid.size else -300.0
        max_power_db = float(np.max(finite_valid)) if finite_valid.size else -300.0

        # ---------------------- 25 Fill All Final Metadata into `rx_health` Dictionary
        rx_health["baseline_ready"]             = bool(self.baseline_ready)
        rx_health["baseline_segments"]          = len(self.baseline_frames)
        rx_health["baseline_required"]          = int(cfg.empty_calibration_segments)
        rx_health["baseline_mode"]              = "ready"
        rx_health["max_dynamic_excess_db"]      = float(np.max(dynamic_excess_db[valid_map])) if np.any(valid_map) else -300.0
        rx_health["baseline_adaptive_update"]   = bool(locals().get("baseline_updated", False))
        rx_health["baseline_freeze"]            = bool(locals().get("freeze_baseline", False))
        rx_health["baseline_freeze_reason"]     = str(locals().get("baseline_freeze_reason", "non"))
        rx_health["near_zone_blocked"]          = bool(locals().get("near_zone_info", {}).get("blocked", False))
        if "near_zone_info" in locals():        # If near-zone detailed info exists, merge all
            rx_health.update(near_zone_info)

        rx_health["raw_targets_count"]          = int(len(raw_targets))
        rx_health["credible_targets_count"]     = int(len(targets))
        rx_health["artifact_targets_count"]     = int(len(artifact_targets))
        rx_health["motion_artifact_filter"]     = "v3_2_bad_rx_gate_plus_v3_2_1_range_track_validator"
        rx_health["promote_far_single_tap_to_motion"]   = bool(getattr(cfg, "promote_far_single_tap_to_motion", False))

        # Merge track-motion, CIR-centroid info into health dictionary as well
        rx_health.update(track_info)
        rx_health.update(centroid_info)

        # Store key parameters of top-raw (first among calculated raw-targets) in separate fields
        if raw_targets:
            rx_health["top_raw_range_cm"]       = float(raw_targets[0].range_cm)
            rx_health["top_raw_tap"]            = int(round(raw_targets[0].range_cm / 15.0))
            rx_health["top_raw_velocity_mps"]   = float(raw_targets[0].velocity_mps)
            rx_health["top-raw_snr_db"]         = float(raw_targets[0].snr_db)

        # Store range/velocity info of some artifact targets as samples (max 6)
        if artifact_targets:
            rx_health["artifact_range_cm"]          = [float(t.range_cm) for t in artifact_targets[:6]]
            rx_health["artifact_velocities_mps"]    = [float(t.velocity_mps) for t in artifact_targets[:6]]

        # Store unique tap/range-span info of highest score (first) among final credible targets
        if targets:
            rx_health["best_unique_taps"]           = int(targets[0].unique_taps)
            rx_health["best_range_span_cm"]         = float(targets[0].range_span_cm)

        # Store info obtained from static object detection (best excess, active taps, etc.)
        rx_health["static_best_excess_db"]          = float(static_info.get("best_excess_db", -300.0))
        rx_health["static_active_taps"]             = static_info.get("active_taps", [])

        # ---------------------- 26 Create and Return Final `SegmentResult` Object ----------------------
        result = SegmentResult(
            segment_id=self.segment_id,
            last_sequence_id=last_sequence_id,
            packets_seen=self.packets_seen,
            complete_pairs=self.complete_pairs,
            detections=detections,                  # Individual cell detection list from CFAR
            targets=targets,                        # Final credible target list (score descending)
            human_detected=human_detected,          # Human detection status
            noise_floor_db=noise_floor_db,          # Noise floor of current segment (dB)
            max_power_db=max_power_db,              # Maximum power within current segment (dB)
            rx_health=rx_health,                    # All metadata filled above
            object_change_detected = object_change_detected,
            motion_detected=motion_detected,
            person_candidate=person_candidate,
            state=state,                            # Status string (HUMAN_CONFIRMED, MOTION, ...)
            static_detection=static_info,           # Static object change detection result dictionary
        )

        # Debug: Print final state
        if cfg.debug_enable:
            self._debug_data["state"] = state
            self._debug_data["human_detected"] = human_detected
            self._debug_data["targets"] = targets
            if cfg.debug_print_stage:
                print(f"[DBG] Seg {self.segment_id}: State={state}, Human={human_detected}, Targets={len(targets)}")

                # Debug: Simple plot if enabled and matplotlib available
                if cfg.debug_plot_enable and MATPLOTLIB_AVAILABLE:
                    self._plot_debug_info()

                return result

    def _plot_debug_info(self):
        """Simple debug plot for radar processing visualization."""
        if not MATPLOTLIB_AVAILABLE or not self._debug_data:
            return

        try:
            # Get axis modes from config
            yaxis_mode = getattr(self.cfg, 'debug_plot_yaxis_mode', 'velocity')
            xaxis_mode = getattr(self.cfg, 'debug_plot_xaxis_mode', 'range_tap')    # 'range_tap' or 'range_cm'

            # Calculate extent for imshow based on axis mode
            # extent = [left, right, bottom, top] in data coordiates
            # Note: velocity_mps and doppler_hz are fftshit-ed, so they go from -max to +max
            n_doppler = len(self.doppler_hz)
            n_range = len(self.range_cm)

            # Debug: Print velocity/doppler info
            print(f"[DBG Plot] velocity_mps range: [{self.velocity_mps[0]:.3f}, {self.velocity_mps[-1]:.3f}] m/s")
            print(f"[DBG Plot] doppler_hz range: [{self.doppler_hz[0]:.1f}, {self.doppler_hz[-1]:1f}] Hz")
            print(f"[DBG Plot] segment_size={self.cfg.segment_size}, period_ms={self.cfg.period_ms}, PRF={1000/self.cfg.period_ms:.1f} Hz")
            print(f"[DBG Plot] wavelength={self.cfg.wavelength_m:.4f} m, channel={self.cfg.channel}")

            # Y-axis setup
            if yaxis_mode == "velocity":
                y_min = float(self.velocity_mps[0])     # Most negative velocity
                y_max = float(self.velocity_mps[-1])    # Most positive velocity
                y_label = "Velocity (m/s)"
            elif yaxis_mode == "doppler_freq":
                y_min = float(self.doppler_hz[0])       # Most negative frequency
                y_max = float(self.doppler_hz[-1])
                y_label = "Doppler Frequency (Hz)"
            else:   # doppler_bin
                y_min = -0.5
                y_max = n_doppler - 0.5
                y_label = "Doppler Bin"

            # X-axis setup
            if xaxis_mode == "range_cm":
                x_min = float(self.range_cm[0])
                x_max = float(self.range_cm[-1])
                x_label = "Range (cm)"
            else:   # range_tap
                x_min = -0.5
                x_max = n_range - 0.5
                x_label = "Range Tap"

            extent = [x_min, x_max, y_min, y_max]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Radar Debug - Segment {self._debug_data.get('segment_id', '?')} (X: {xaxis_mode}, Y: {yaxis_mode})")

            # Plot 1: power DB map
            if "power_db" in self._debug_data:
                power_db = self._debug_data["power_db"]
                im0 = axes[0, 0].imshow(power_db, aspect='auto', origin='lower', extent=extent, cmap='jet')
                axes[0, 0].set_title("Power (dB)")
                axes[0, 0].set_xlabel(x_label)
                axes[0, 0].set_ylabel(y_label)
                plt.colorbar(im0, ax=axes[0, 0])
                axes[0, 0].set_xlim(x_min, x_max/4)
                axes[0, 0].set_ylim(y_min, y_max)

            # Plot 2: Valid map
            if "valid_map" in self._debug_data:
                valid_map = self._debug_data["valid_map"]
                im1 = axes[0, 1].imshow(valid_map, aspect='auto', origin='lower', extent=extent, cmap='gray')
                axes[0, 1].set_title("Valid Map")
                axes[0, 1].set_xlabel(x_label)
                axes[0, 1].set_ylabel(y_label)
                axes[0, 1].set_xlim(x_min, x_max/4)
                axes[0, 1].set_ylim(y_min, y_max)

            # Plot 3: Detections on power map
            if "power_db" in self._debug_data and "detections" in self._debug_data:
                power_db = self._debug_data["power_db"]
                detections = self._debug_data["detections"]
                ax = axes[1, 0]
                im2 = ax.imshow(power_db, aspect='auto', origin='lower', extent=extent, cmap='jet')
                ax.set_title(f"Detections ({len(detections)})")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

                # Convert detection coordinates based on axis mode
                for det in detections[:10]:
                    if xaxis_mode == "range_cm":
                        x = float(self.range_cm[det.tap_index])
                    else:   # range_tap
                        x = det.tap_index

                    if yaxis_mode == "velocity":
                        y = float(self.velocity_mps[det.doppler_index])
                    elif yaxis_mode == "doppler_freq":
                        y = float(self.doppler_hz[det.doppler_index])
                    else:   # dopploer_bin
                        y = det.doppler_index
                    ax.plot(x, y, 'b+', markersize=15, markerdegewidth=2)
                plt.colorbar(im2, ax=ax)
                ax.set_xlim(x_min, x_max/4)
                ax.set_ylim(y_min, y_max)

            # Plot 4: Range-Velocity targets (axis modes apply)
            if "targets" in self._debug_data:
                targets = self._debug_data["targets"]
                ax = axes[1, 1]
                if targets:
                    # Convert target coordinates based on axis modes
                    if xaxis_mode == "range_cm":
                        x_vals = [t.range_cm for t in targets]
                    else:   # range_tap - convert cm to tap index using system variable
                        # 1 tap = range_cm[1] - range_cm[0] (typically ~15 cm, but use actual value)
                        tap_spacing_cm = float(self.range_cm[1] - self.range_cm[0]) if len(self.range_cm) > 1 else 15.0
                        x_vals = [t.range_cm / tap_spacing_cm for t in targets]

                    if yaxis_mode == "velocity":
                        y_vals = [t.velocity_mps for t in targets]
                    elif yaxis_mode == "doppler_freq":
                        # Convert velocity to doppler freq: f_d = 2*v/lambda
                        y_vals = [2 * t.velocity_mps / self.cfg.wavelength_m for t in targets]
                    else:   # doppler_bin - convert velocity to bin index
                        n_dop = len(self.doppler_hz)
                        v_max = max(abs(self.velocity_mps[0]), abs(self.velocity_mps[-1]))
                        y_vals = [int((t.velocity_mps / v_max) * (n_dop / 2) + n_dop / 2) for t in targets]

                    scores = [t.score for t in targets]
                    scatter = ax.scatter(x_vals, y_vals, c=scores, s=100, alpha=0.7, camp='viridis')
                    ax.set_title(f"Targets ({len(targets)})")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label="Score")
                else:
                    ax.text(0.5, 0.5, "No Targets", ha='center', va='center', fontsize=14)

                # Set axis limits to match Plot 1, 2, 3
                ax.set_xlim(x_min, x_max/4)
                ax.set_ylim(y_min, y_max)

            plt.tight_layout()
            plt.savefig(f"radar_debug_seg{self._debug_data.get('segment_id', 0)}_x{xaxis_mode}_y{yaxis_mode}.png", dpi=100)
            plt.close()
            print(f"[DBG] Plot saved: radar_debug_seg{self._debug_data.get('segment_id', 0)}_x{xaxis_mode}_y{yaxis_mode}.png")
        except Exception as e:
            print(f"[DBG] Plot error: {e}")

    def _plot_cir_debug(self, seg_hp: np.ndarray, x: np.ndarray, eps: float = 1e-12):
        """Plot CIR (Fast-time) debug visualization.

        Args:
            seg_hp: DC-removed and phase-corrected CIR data [rx, frame, tap]
            x: Windowed CIR data [rx, frame, tap]
            eps: Small constant for log calculation
        """
        cfg = self.cfg
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f"Segment {self.segment_id} - Debug Plot")

            # Row 1: seg_hp magnitude heatmap (all frames).
            for rx in range(2):
                im0 = axes[0, rx].imshow(20*np.log10(np.abs(seg_hp[rx]) + eps), aspect='auto',
                                        origin='lower', cmap='jet')
                axes[0, rx].set_title(f"seg_hp Rx{rx} - All Frames (dB)")
                axes[0, rx].set_xlabel("Tap (Fast-time)")
                axes[0, rx].set_ylabel("Frame (Slow-time)")
                #axes[0, rx].set_xlim(0, 64)
                plt.colorbar(im0, ax=axes[0, rx], label='dB')

            
            # Row 2: CIR per frame (Fast-time axis) - Selected single frames
            frame_sel = list(range(cfg.segment_size)) # [0, cfg.segment_size // 2, cfg.segment_size - 1] # First, Middle, Last
            colors = ['b', 'g', 'r']
            for rx in range(2):
                for i, f_idx in enumerate(frame_sel):
                    cir_mag = 20 * np.log10(np.abs(seg_hp[rx, f_idx, :]) + eps)
                    #axes[1, rx].plot(np.arange(cfg.cir_taps), cir_mag, color=colors[i], label=f'Frame {f_idx}', alpha=0.8, linewidth=1.5)
                    #axes[1, rx].plot(np.arange(cfg.cir_taps), cir_mag, label=f'Frame {f_idx}', alpha=0.8, linewidth=1.5)
                    axes[1, rx].plot(np.arange(cfg.cir_taps), cir_mag, alpha=0.8, linewidth=1.5)

                avg_cir_mag = 20 * np.log10(np.mean(np.abs(seg_hp[rx, :, :]) + eps, axis=0))
                axes[1, rx].plot(np.arange(cfg.cir_taps), avg_cir_mag, color='k', linestyle='--', linewidth=2.5, label='Avg (all frames)')

                axes[1, rx].set_title(f"CIR (Fast-time) Rx{rx} - Selected Frames")
                axes[1, rx].set_xlabel("Tap (Fast-time)")
                axes[1, rx].set_ylabel("Magnitude (dB)")
                axes[1, rx].legend(fontsize=8)
                axes[1, rx].grid(alpha=0.3)
                #axes[1, rx].set_xlim(0, min(64, cfg.cir_taps))
                axes[1, rx].set_ylim(-60, 10)


            # Row 3: x magnitude (after window)
            for rx in range(2):
                #im1 = axes[2, rx].imshow(20*np.log10(np.abs(x[rx]) + eps), aspect='auto', origin='lower', cmap='jet')
                im1 = axes[2, rx].imshow(20*np.log10(np.abs(x[rx]) + 1e-4), aspect='auto', origin='lower', cmap='jet')
                axes[2, rx].set_title(f"x (windowed Rx{rx} (dB)")
                axes[2, rx].set_xlabel("Tap")
                axes[2, rx].set_ylabel("Frame")
                #axes[2, rx].set_xlim(0, 64)
                plt.colorbar(im1, ax=axes[2, rx], label='dB')


            # # Row 4: fft_rx power (doppler-tap map)
            # for rx in range(2):
            #     im2 = axes[3, rx].imshow(10*np.log10(power_rx[rx] + eps), aspect='auto',
            #                            origin='lower', cmap='jet')
            #    axes[3, rx].set_title(f"fft_rx power Rx{rx} (dB)")
            #    axes[3, rx].set_xlabel("Tap")
            #    axes[3, rx].set_ylabel("Doppler")
            #    plt.colorbar(im2, ax=axes[3, rx], label='dB')

            plt.tight_layout()
            plt.savefig(f"radar_fft_debug_seg{self.segment_id}.png", dpi=100)
            plt.show()
            plt.close()
            print(f"[DBG] FFT plot saved: radar_fft_debug_seg{self.segment_id}.png")
        except Exception as e:
            print(f"[DBG] CIR plot error: {e}")


    def _cir_energy_centroid_direction(
        self,
        dynamic_excess_db: np.ndarray,
        power_db: np.ndarray,
        valid_map: np.ndarray,
    ) -> dict[str, object]:
        """Estimate TOWARD/BACK from CIR dynamic-energy centroid migration.
        """
        cfg = self.cfg
        info: dict[str, object] = {
            "cir_centroid_enabled": bool(getattr(cfg, "enable_cir_centroid_direction", True)),
            "cir_centroid_valid": False,
            "cir_centroid_range_cm": None,
            "cir_centroid_peak_tap": None,
            "cir_centroid_peak_range_cm": None,
            "cir_centroid_total_weight": 0.0,
            "cir_centroid_points": 0,
            "cir_centroid_delta_cm": 0.0,
            "cir_centroid_slope_cm_per_segment": 0.0,
            "cir_centroid_direction": "UNKNOWN",
            "cir_centroid_confidence": 0.0,
            "cir_centroid_consistency": 0.0,
        }
        if not bool(info["cir_centroid_enabled"]):
            return info

        try:
            min_ex = float(getattr(cfg, "cir_centroid_min_excess_db", 5.0))
            min_weight = float(getattr(cfg, "cir_centroid_min_total_weight", 2.0))
            usable = np.asarray(valid_map, dtype=bool) & np.isfinite(dynamic_excess_db) & np.isfinite(power_db)
            usable &= power_db >= float(getattr(cfg, "bad_rx_floor_db_threshold", -100.0))
            if not np.any(usable):
                self.cir_centroid_history.append({
                    "segment_id": float(self.segment_id),
                    "centroid_range_cm": float("nan"),
                    "total_weight": 0.0,
                })
                return info

            w_map = np.where(usable, np.maximum(dynamic_excess_db - min_ex, 0.0), 0.0)
            tap_w = np.sum(w_map, axis=0)
            total_w = float(np.sum(tap_w))
            info["cir_centroid_total_weight"] = total_w
            if total_w < min_weight:
                self.cir_centroid_history.append({
                    "segment_id": float(self.segment_id),
                    "centroid_range_cm": float("nan"),
                    "total_weight": total_w,
                })
                return info

            centroid_cm = float(np.sum(self.range_cm * tap_w) / total_w)
            peak_tap = int(np.argmax(tap_w))
            info["cir_centroid_valid"] = True
            info["cir_centroid_range_cm"] = centroid_cm
            info["cir_centroid_peak_tap"] = peak_tap
            info["cir_centroid_peak_range_cm"] = float(self.range_cm[peak_tap])

            self.cir_centroid_history.append({
                "segment_id": float(self.segment_id),
                "centroid_range_cm": centroid_cm,
                "total_weight": total_w,
            })

            pts = []
            for item in self.cir_centroid_history:
                sid = float(item.get("segment_id", float("nan")))
                r = float(item.get("centroid_range_cm", float("nan")))
                tw = float(item.get("total_weight", 0.0))
                if np.isfinite(sid) and np.isfinite(r) and tw >= min_weight:
                    pts.append((sid, r, tw))
            pts = sorted(pts, key=lambda p: p[0])
            info["cir_centroid_points"] = int(len(pts))
            if len(pts) < int(getattr(cfg, "cir_centroid_min_points", 3)):
                return info

            x = np.asarray([p[0] for p in pts], dtype=float)
            y = np.asarray([p[1] for p in pts], dtype=float)
            max_gap = int(np.max(np.diff(x)) - 1) if len(x) >= 2 else 999
            if max_gap > int(getattr(cfg, "cir_centroid_max_internal_misses", 2)):
                return info

            delta = float(y[-1] - y[0])
            info["cir_centroid_delta_cm"] = delta
            x0 = x - x[0]
            slope = float(np.polyfit(x0, y, 1)[0]) if np.max(x0) > 0 else 0.0
            info["cir_centroid_slope_cm_per_segment"] = slope

            min_delta = float(getattr(cfg, "cir_centroid_min_delta_cm", 7.0))
            if abs(delta) < min_delta and abs(slope) < (min_delta / max(1.0, len(pts) - 1.0)):
                info["cir_centroid_direction"] = "STABLE"
                return info

            direction = "BACK" if delta > 0 else "TOWARD"
            dy = np.diff(y)
            step_thr = max(3.0, min_delta * 0.5)
            significant = dy[np.abs(dy) >= step_thr]
            if significant.size == 0:
                consistency = 0.5
            else:
                sign = 1.0 if delta > 0 else -1.0
                consistency = float(np.sum(np.sign(significant) == sign) / max(1, significant.size))
            info["cir_centroid_consistency"] = consistency
            if consistency < float(getattr(cfg, "cir_centroid_min_consistency", 0.60)):
                info["cir_centroid_direction"] = "STABLE"
                info["cir_centroid_confidence"] = max(0.0, min(0.55, consistency))
                return info

            delta_score = min(1.0, abs(delta) / max(min_delta, 1.0))
            confidence = float(np.clip(0.35 + 0.45 * delta_score + 0.20 * consistency, 0.0, 0.95))
            info["cir_centroid_direction"] = direction
            info["cir_centroid_confidence"] = confidence
            return info
        except Exception as exc:
            info["cir_centroid_error"] = str(exc)
            return info


    def _near_zone_block_status(
        self,
        power_db: np.ndarray,
        dynamic_excess_db: np.ndarray,
        velocity_mask: np.ndarray,
    ) -> dict[str, object]:
        """Return whether early-tap activity should suppress ghost motion.
        """
        cfg = self.cfg
        info: dict[str, object] = {
            "near_zone_enabled": bool(getattr(cfg, "near_zone_block_enabled", True)),
            "near_zone_blocked": False,
            "near_zone_max_cm": float(getattr(cfg, "near_zone_block_max_cm", 50.0)),
            "near_zone_cells": 0,
            "near_zone_max_excess_db": -300.0,
            "near_zone_best_tap": None,
            "near_zone_best_range_cm": None,
        }
        if not bool(info["near_zone_enabled"]):
            return info

        max_cm = float(getattr(cfg, "near_zone_block_max_cm", 50.0))

        near_tap_mask = self.range_cm < max_cm
        if not np.any(near_tap_mask):
            return info

        min_power_db = float(getattr(cfg, "near_zone_block_min_power_db", -95.0))
        excess_thr = float(getattr(cfg, "near_zone_block_excess_db", 6.0))
        near_map = velocity_mask[:, None] & near_tap_mask[None, :]
        near_map &= np.isfinite(dynamic_excess_db)
        near_map &= np.isfinite(power_db)
        near_map &= power_db > min_power_db
        near_map &= dynamic_excess_db >= excess_thr

        cell_count = int(np.count_nonzero(near_map))
        info["near_zone_cells"] = cell_count
        if np.any(near_map):
            vals = np.where(near_map, dynamic_excess_db, -300.0)
            idx = np.unravel_index(int(np.argmax(vals)), vals.shape)
            info["near_zone_max_excess_db"] = float(vals[idx])
            info["near_zone_best_tap"] = int(idx[1])
            info["near_zone_best_range_cm"] = float(self.range_cm[idx[1]])

        if cell_count >= int(getattr(cfg, "near_zone_block_min_cells", 2)):
            info["blocked"] = True
            info["near_zone_blocked"] = True
        return info

    
    def _range_track_motion(self, raw_targets: list[GroupedTarget]) -> tuple[bool, dict[str, object]]:
        """Validate walking motion from the temporal movement of candidate range taps.
        """
        cfg = self.cfg
        info: dict[str, object] = {
            "track_motion": False,
            "track_direction": "NONE",
            "track_valid_points": 0,
            "track_range_span_cm": 0.0,
            "track_slope_cm_per_segment": 0.0,
            "track_recent_misses": 0,
        }
        if not bool(getattr(cfg, "enable_range_track_motion", True)):
            info["track_enabled"] = False
            return False, info
        info["track_enabled"] = True

        hist = list(self.raw_candidate_history)[-max(1, int(cfg.track_window_segments)):]
        if not hist:
            return False, info

        
        recent_misses = 0
        for item in reversed(hist):
            if np.isfinite(float(item.get("range_cm", float("nan")))):
                break
            recent_misses += 1
        info["track_recent_misses"] = int(recent_misses)
        if recent_misses > int(cfg.track_max_misses):
            return False, info

        pts = []
        for item in hist:
            r = float(item.get("range_cm", float("nan")))
            v = float(item.get("velocity_mps", float("nan")))
            sc = float(item.get("score", float("nan")))
            sid = float(item.get("segment_id", float("nan")))
            if not (np.isfinite(r) and np.isfinite(v) and np.isfinite(sc) and np.isfinite(sid)):
                continue
            if abs(v) < float(cfg.track_min_abs_velocity_abs):
                continue
            if sc < float(cfg.track_min_score):
                continue
            pts.append((sid, r, v, sc))


        pts = sorted(pts, key=lambda p: p[0])
        dedup = []
        seen = set()
        for p in pts:
            sid_key = int(round(p[0]))
            if sid_key in seen:
                continue
            seen.add(sid_key)
            dedup.append(p)
        pts = dedup

        info["track_valid_points"] = int(len(pts))
        if len(pts) < int(cfg.track_min_points):
            return False, info

        x = np.asarray([p[0] for p in pts], dtype=float)
        y = np.asarray([p[1] for p in pts], dtype=float)

        if len(x) >= 2:
            max_internal_gap = int(np.max(np.diff(x)) - 1)
        else:
            max_internal_gap = 999
        info["track_max_internal_gap"] = int(max_internal_gap)
        if max_internal_gap > int(getattr(cfg, "track_max_internal_misses", 1)):
            return False, info

        x0 = x - x[0]
        if np.max(x0) <= 0:
            return False, info
        slope = float(np.polyfit(x0, y, 1)[0])
        span = float(np.nanmax(y) - np.nanmin(y))
        info["track_range_span_cm"] = span
        info["track_slope_cm_per_segment"] = slope

        if span < float(cfg.track_min_range_change_cm):
            return False, info
        if abs(slope) < float(cfg.track_min_slope_cm_per_segment):
            return False, info

        dy = np.diff(y)
        step_thr = float(getattr(cfg, "track_min_step_cm", 10.0))
        significant = dy[np.abs(dy) >= step_thr]
        info["track_significant_steps"] = int(significant.size)
        if significant.size < int(getattr(cfg, "track_min_significant_steps", 2)):
            return False, info
        slope_sign = 1.0 if slope > 0 else -1.0
        same_dir = int(np.sum(np.sign(significant) == slope_sign))
        direction_ratio = float(same_dir / max(1, significant.size))
        info["track_direction_consistency"] = direction_ratio
        if direction_ratio < float(getattr(cfg, "track_min_direction_consistency", 0.75)):
            return False, info

        latest_seg = float(self.segment_id)
        if latest_seg - x[-1] > int(cfg.track_max_misses):
            return False, info

        direction = "AWAY" if slope > 0 else "TOWARD"
        info["track_motion"] = True
        info["track_direction"] = direction
        return True, info

    
    def _static_object_change(self, seg: np.ndarray) -> tuple[bool, dict[str, object]]:
        """Detect static object/person change using complex CIR reference.
        
        """
        cfg = self.cfg
        if not bool(getattr(cfg, "enable_static_detection", False)):
            self.static_history.append(False)
            return False, {
                "ready": False,
                "disabled": True,
                "raw_change": False,
                "persistent": False,
                "active_taps": [],
                "best_tap": None,
                "best_range_cm": None,
                "best_excess_db": -300.0,
                "unique_taps": 0,
            }
        if self.static_reference is None or self.static_noise_db is None:
            self.static_history.append(False)
            return False, {
                "ready": False,
                "raw_change": False,
                "persistenct": False,
                "active_taps": [],
                "best_tap": None,
                "best_range_cm": None,
                "best_excess_db": -300.0,
                "unique_taps": 0,
            }

        seg_mean = np.mean(seg, axis=1).astype(np.complex64)    # [rx, tap]
        delta = seg_mean - self.static_reference
        delta_db = 20.0 * np.log10(np.abs(delta) + 1e-9)
        excess_db = delta_db - self.static_noise_db

        roi = (
            (np.arange(cfg.cir_taps) >= cfg.skip_bins)
            & (self.range_cm >= cfg.min_range_cm)
            & (self.range_cm <= cfg.max_range_cm)
        )
        rx_hit = (excess_db >= cfg.static_excess_threshold_db) & (delta_db >= cfg.static_min_delta_db)
        rx_hit[:, ~roi] = False
        if cfg.static_rx_consistency:
            tap_hit = np.logical_and(rx_hit[0], rx_hit[1])
        else:
            tap_hit = np.logical_or(rx_hit[0], rx_hit[1])
        
        active_taps = np.where(tap_hit)[0]
        raw_change = bool(len(active_taps) >= cfg.static_min_unique_taps)
        self.static_history.append(raw_change)
        persistent_hits = sum(1 for x in self.static_history if x)
        persistent = persistent_hits >= cfg.static_persistence_min_hits

        best_tap = None
        best_range_cm = None
        best_excess = -300.0
        if len(active_taps):
            max_per_tap = np.max(excess_db[:, active_taps], axis=0)
            best_i = int(np.argmax(max_per_tap))
            best_tap = int(active_taps[best_i])
            best_range_cm = float(best_tap * 15.0)
            best_excess = float(max_per_tap[best_i])

        info = {
            "ready": True,
            "raw_change": raw_change,
            "persistent": bool(persistent),
            "history_hits": int(persistent_hits),
            "history_window": int(cfg.static_persistence_window),
            "active_taps": [int(t) for t in active_taps],
            "best_tap": best_tap,
            "best_range_cm": best_range_cm,
            "best_excess_db": float(best_excess),
            "unique_taps": int(len(active_taps)),
            "threshold_db": float(cfg.static_excess_threshold_db),
        }
        return bool(persistent), info

    def _cfar_detect(
        self,
        power_lin: np.ndarray,              # Linear (power) = |R|^2 (dopplerxrange cell)
        power_db: np.ndarray,               # 10-log10(power_lin) (dB)
        dynamic_excess_db: np.ndarray,     # dB values for dynamic/static ratio map
        power_rx: np.ndarray,               # Power per RX antenna (N_rx x N_doppler x N_range)
        fft_rx: np.ndarray,                 # FFT result per RX (N_rx x N_doppler x N_range, complex)
        valid_map: np.ndarray,              # Cell validity (True = usable)
        active_rx: np.ndarray,              # RX mask usable iin current frame
    ) -> list[Detection]:
        cfg = self.cfg
        eps = 1e-12
        detections: list[Detection] = []
        nd, nr = power_db.shape
        use_two_rx = bool(np.sum(active_rx) == 2)

        for di in range(nd):
            if not np.any(valid_map[di, :]):
                continue
            for ri in range(cfg.skip_bins, nr):
                if not valid_map[di, ri]:
                    continue

                cut_lin = float(power_lin[di, ri])
                cut_db = float(power_db[di, ri])
                excess_db = float(dynamic_excess_db[di, ri])
                if cut_db < cfg.min_motion_power_db:
                    continue
                if excess_db < cfg.min_dynamic_excess_db:
                    continue

                d0 = max(0, di - cfg.ref_doppler - cfg.guard_doppler)
                d1 = min(nd, di + cfg.ref_doppler + cfg.guard_doppler + 1)
                r0 = max(0, ri - cfg.ref_range - cfg.guard_range)
                r1 = min(nr, ri + cfg.ref_range + cfg.guard_range + 1)


                ref_excess_db = dynamic_excess_db[d0:d1, r0:r1].copy()
                ref_valid = valid_map[d0:d1, r0:r1].copy()
                gd0 = max(0, di - cfg.guard_doppler) - d0
                gd1 = min(nd, di + cfg.guard_doppler + 1) - d0
                gr0 = max(0, ri - cfg.guard_range) - r0
                gr1 = min(nr, ri + cfg.guard_range + 1) - r0
                ref_valid[gd0:gd1, gr0:gr1] = False

                ref_vals_db = ref_excess_db[ref_valid]
                ref_vals_db = ref_vals_db[np.isfinite(ref_vals_db)]
                if ref_vals_db.size < 6:
                    continue

                local_ref_db = float(np.median(ref_vals_db))
                snr_db = excess_db - local_ref_db
                if snr_db < cfg.cfar_threshold_db:
                    continue

                rx_balance_db = 0.0
                angle_deg = None
                if use_two_rx:
                    p1 = float(power_rx[0, di, ri])
                    p2 = float(power_rx[1, di, ri])
                    rx_balance_db = 10.0 * np.log10((p1 + eps) / (p2 + eps))
                    if not (cfg.min_rx_balance_db <= rx_balance_db <= cfg.max_rx_balance_db):
                        continue
                    angle_deg = self._estimate_aoa_deg(fft_rx[:, di, ri])
                    if angle_deg is not None and abs(angle_deg) > cfg.max_abs_angle_deg:
                        continue
                
                velocity = float(self.velocity_mps[di])
                range_cm = float(self.range_cm[ri])
                score = float(snr_db + 0.7 * excess_db + min(8.0, abs(velocity) * 12.0) - 0.04 * max(0.0, range_cm - 250.0))
                detections.append(
                    Detection(
                        segment_id=self.segment_id,
                        sequence_id=self.last_committed_sequence or 0,
                        range_cm=range_cm,
                        velocity_mps=velocity,
                        doppler_hz=float(self.doppler_hz[di]),
                        angle_deg=angle_deg,
                        snr_db=float(snr_db),
                        power_db=cut_db,
                        tap_index=int(ri),
                        doppler_index=int(di),
                        rx_balance_db=float(rx_balance_db),
                        score=score,
                    )
                )
        return detections

    def _estimate_aoa_deg(self, rx_fft_pair: np.ndarray) -> Optional[float]:
        """Estimate Angle of Arrival from two-RX FFT bin pair.

        Uses phase difference between RX0 and RX1 to compute AoA via arcsin.
        Calibration is applied via complex rotation in _process_segment() before FFT.
        Beam direction offset is added for future beamforming support.
        """
        cfg = self.cfg
        if rx_fft_pair.size != 2:
            return None
        if np.abs(rx_fft_pair[0]) < 1e-9 or np.abs(rx_fft_pair[1]) < 1e-9:
            return None

        phase = np.angle(rx_fft_pair[0] * np.conj(rx_fft_pair[1]))


        arg = cfg.wavelength_m * phase / (2.0 * np.pi * cfg.antenna_spacing_m)
        arg = float(np.clip(arg, -1.0, 1.0))
        angle_deg = float(np.rad2deg(np.arcsin(arg)))

        angle_deg += cfg.beam_direction_deg

        return angle_deg

    def _group_detections(self, detections: list[Detection]) -> list[GroupedTarget]:
        """
        Group the input single detections (Detection) list by range/velocity/angle criteria,
        calculate representative values (weighted average, max, etc.) per group,
        and return a list of `GroupedTarget` objects.
        """
        cfg = self.cfg                          # Store settings (range/velocity/angle tolerance, etc.) in local variable
        groups: list[list[Detection]] = []      # Initialize double list in form "group = Detection list"

        # 1) Iterate through detections and check if they can be added to existing groups
        for det in detections:                  # Check one by one
            assigned = False                    # Current detection not yet assigned to any group

            for group in groups:                # Search all existing groups
                ref = group[0]                  # Current detection not yet assigned to any group

                # ----- Angle condition judgment (ignore if angle info unavailable) -----
                angle_ok = True                 # Default value: angle condition passed
                if det.angle_deg is not None and ref.angle_deg is not None:
                    angle_ok = abs(det.angle_deg - ref.angle_deg) <= cfg.group_angle_deg    # If both have angles, check if difference is within allowed range

                # ----- If all three conditions (range/velocity/angle) are satisfied, include in same group -----
                if (abs(det.range_cm - ref.range_cm) <= cfg.group_range_cm                      # Range difference within tolerance
                    and abs(det.velocity_mps - ref.velocity_mps) <= cfg.group_velocity_mps      # Velocity difference within tolerance
                    and angle_ok):                                                              # Angle difference within tolerance (or None)
                    group.append(det)           # Add to current group
                    assigned = True             # Assignment successful
                    break                       # If included once, no need to search further

            # ----- If not added to any existing group, create new group -----
            if not assigned:
                groups.append([det])
        
        # 2) Calculate representative values/statistics for each group and create GroupedTarget objects
        targets: list[GroupedTarget] = []
        for group in groups:
            # ----- Prepare weights and basic arrays -----
            weights = np.array([max(1e-6, d.snr_db) for d in group], dtype=float)           # Weight in SNR (dB) value. Replace 0 or less with 1e-6 to maintain only positive values.
            ranges = np.array([d.range_cm for d in group], dtype=float)                     # Range/velocity/power value arrays
            vels = np.array([d.velocity_mps for d in group], dtype=float)                   # Range/velocity/power value arrays
            powers = np.array([d.power_db for d in group], dtype=float)                     # Range/velocity/power value arrays
            angles_valid = [d.angle_deg for d in group if d.angle_deg is not None]          # ----- Angle average (only if angle info exists) -----
            angle = float(np.average(angles_valid)) if angles_valid else None               # ----- Angle average (only if angle info exists) -----
            tap_indices = np.array([d.tap_index for d in group], dtype=int)                 # ----- Meta info (tap index/doppler values, etc.) -----
            doppler_vals = np.array([d.doppler_hz for d in group], dtype=float)             # ----- Meta info (tap index/doppler values, etc.) -----
            unique_taps = int(len(set(int(x) for x in tap_indices)))                        # Number of different tap (range) indices -> judge multo-tap usage
            range_span_cm = (float((np.max(tap_indices) - np.min(tap_indices)) * 15.0) if tap_indices.size else 0.0)    # Range (distance) span: multiply index difference by 1 tap ~= 15 cm to estimate actual distance difference
            doppler_span_hz = (float(np.max(doppler_vals) - np.min(doppler_vals)) if doppler_vals.size else 0.0)        # Doppler span (Hz): max-min dopper difference within group
            doppler_hz = (float(np.average(doppler_vals, weights=weights)) if doppler_vals.size else 0.0)               # Weighted average doppler (Hz) - weight = SNR

            # ----- Select representative detection (detection with highest score) -----
            best_det = max(group, key=lambda d: d.score)

            targets.append(
                GroupedTarget(
                    segment_id=self.segment_id,
                    range_cm=float(np.average(ranges, weights=weights)),                    # Range/velocity use SNR weighted average
                    velocity_mps=float(np.average(vels, weights=weights)),                  # Range/velocity use SNR weighted average
                    angle_deg=angle,                                                        # Average angle (or None)
                    snr_db=float(np.max([d.snr_db for d in group])),                        # Best SNR / best power within group
                    power_db=float(np.max(powers)),                                         # Best SNR / best power within group
                    detections_count=len(group),                                            # Number of detections included in group
                    score=float(np.sum([d.score for d in group]) / max(1, len(group))),     # Average score of group
                    human_like=False,
                    unique_taps=unique_taps,
                    range_span_cm=range_span_cm,
                    doppler_span_hz=doppler_span_hz,
                    doppler_hz=doppler_hz,
                    doppler_index=int(best_det.doppler_index),                              # Representative detection's doppler index
                )
            )
        return sorted(targets, key=lambda t: t.score, reverse=True)