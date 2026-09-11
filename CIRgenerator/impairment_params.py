"""Load and inject measurement-fitted impairment parameters.

Suppoerted schemas
------------------
* v2 canonical structued schema produced by
  ``notebooks/impairment_fitting.py``.  It uses the exact two-term static
  model plus ``noise.stochastic`` (complex-gain AR(1) + per-tap additive
  complex AR(1)).
* previous structured schema.  It is accepted with a white-noise fallback.
* legacy flat ``case*_params.json`` schema.  It remains a compatibility path.

The function mutates an already-built ``SimulationConfig`` and deliberately
imports no simulator class, avoiding an import cycle.  For the v2 stochastic
model, ``CIRDataGenerator`` must call
``fitted_impairment.build_realizer_from_config(cfg)`` and apply the resulting
realizer after deterministic target/room/leakage accumulation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np

try:
    from fitted_impairment import FittedImpairmentModel, SCHEMA_NAME, SCHEMA_VERSION
except ImportError:  # package-style import
    from CIRgenerator.fitted_impairment import (
        FittedImpairmentModel,
        SCHEMA_NAME,
        SCHEMA_VERSION,
    )


def load_case_params(path: Union[str, Path]) -> dict:
    """Read and minimally validate a fitted-impairment JSON file."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as input_file:
        params = json.load(input_file)
    if not isinstance(params, dict):
        raise ValueError(f"top-level fitted params must be a JSON object: {resolved}")
    return params


def _detect_schema(params: dict) -> str:
    """Return ``v2``, ``structued``, or ``legacy``."""

    schema = params.get("schema", {})
    if (
        isinstance(schema, Mapping)
        and schema.get("name") == SCHEMA_NAME
        and int(schema.get("version", -1)) == SCHEMA_VERSION
    ):
        return "v2"
    noise = params.get("noise", {})
    if isinstance(noise, Mapping) and isinstance(noise.get("stochastic"), Mapping):
        return "v2"
    if "static_complex_template" in params:
        return "structured"
    tf = params.get("tx_rx_feedthrough", {})
    if isinstance(tf, Mapping) and "ant0" in tf:
        return "structured"
    return "legacy"


def _num_antennas(params: Mapping[str, Any], default: int = 2) -> int:
    return int(params.get("meta", {}).get("num_antennas", default))


def _structured_white_fallback(params: Mapping[str, Any], num_ant: int) -> list[float]:
    """Return a variance-perserving scalar fallback for old readers."""

    noise = params.get("noise", {})
    explicit = noise.get("fallback_white_component_std_per_ant")
    if explicit is not None:
        values = [float(v) for v in explicit]
        if len(values) != num_ant:
            raise ValueError(
                "noise.fallback_white_component_std_per_ant length mismatch: "
                f"expected {num_ant}, got {len(values)}"
            )
        return values

    per_tap = noise.get("modeled_component_std_per_tap")
    if per_tap is None:
        per_tap = noise.get("modeled_total_component_std_per_tap")
    if per_tap is not None:
        array = np.asarray(per_tap, dtype=float)
        if array.ndim == 2 and array.shape[0] == num_ant:
            return np.sqrt(np.mean(array**2, axis=1)).tolist()

    coefficients = noise.get("coefficients", [])
    output = []
    for antenna_index in range(num_ant):
        coefficient = coefficients[antenna_index] if antenna_index < len(coefficients) else {}
        base = float(coefficient.get("base_std", 0.0))
        quant = float(coefficient.get("quant_floor_std", 0.0))
        output.append(float(np.hypot(base, quant)))
    return output


def _structured_to_injection(params: dict) -> dict:
    """Normalize either structured schema into the old flat assignmnet shape."""
    
    num_ant = _num_antennas(params)
    flat: dict = {}
    flat["noise"] = {
        "std_per_ant": _structured_white_fallback(params, num_ant),
    }

    if _detect_schema(params) == 'v2':
        # keep the complete canonical object for FittedImpairmentModel validation
        # and installation.  The scalar fallback above is never used when the
        # simulator implements the v2 runtime, but remains useful to old tools.
        flat["_v2_fitted_model_params"] = copy.deepcopy(params)

    flat["quantization"] = {}

    fd = params.get("frame_drift", {})
    phase_compat = fd.get(
        "phase_drift_sfd",
        [float(fd.get("common_phase_circular_std_rad", 0.0)) for _ in range(num_ant)],
    )
    flat["frame_drift"] = {
        "amplitude_drift_std": float(fd.get("common_amplitude_std", 0.0)),
        "timing_jitter_std_bins": float(fd.get("fractional_timing_jitter_std_bins", 0.0)),
        "phase_drift_std": [float(v) for v in phase_compat]
    }

    # Preserve gain_deembedding so apply_case_params_to_cfg can inject
    # rx1_gain/rx2_gain from the fit (structured JSONs carry this at top
    # level; legecy JSONs already have it, so the .get default is a no-op
    # for them).

    flat["gain_deembedding"] = params.get("gain_deembedding", {})
    tf = params["tx_rx_feedthrough"]
    flat_tf: dict = {
        "start_tap": int(tf.get("start_tap", 0)),
        "num_taps": int(tf["num_taps"]),
    }
    phase_slopes: list[float] = []
    for antenna_index in range(num_ant):
        ant = tf[f"ant{antenna_index}"]
        flat_tf[f"amp_ant{antenna_index}"] = ant["amp"]
        flat_tf[f"decay_ant{antenna_index}"] = ant["decay"]
        flat_tf[f"phase_ant{antenna_index}_rad"] = ant["phase_rad"]
        flat_tf[f"ripple_amp_ant{antenna_index}"] = 0.0
        flat_tf[f"ripple_freq_ant{antenna_index}"] = 0.0
        flat_tf[f"noise_std_used_ant{antenna_index}"] = 0.0
        phase_slopes.append(float(ant.get("phase_slope_rad_per_tap", 0.0)))
    flat["tx_rx_feedthrough"] = flat_tf
    flat["_tx_rx_feedthrough_phase_slope_per_ant"] = phase_slopes

    r = params.get("radiation_leakage", {})
    flat["radiation_leakage"] = {
        "delay_bins": int(r.get("delay_bins", 0)),
        "factor_12": float(r.get("factor_12", 0.0)),
        "factor_21": float(r.get("factor_21", 0.0)),
        "phase_offset_12_rad": float(r.get("phase_offset_12_rad", 0.0)),
        "phase_offset_21_rad": float(r.get("phase_offset_21_rad", 0.0)),
    }

    ringing = params.get("tx_rx_feedthrough_ringing")
    if ringing is not None:
        first_enabled = next (
            (
                ringing[f"ant{antenna_index}"]
                for antenna_index in range(num_ant)
                if ringing.get(f"ant{antenna_index}", {}).get("enabled", True)
            ),
            None,
        )
        if first_enabled is not None:
            flat_ring: dict = {
                "start_tap": int(first_enabled["start_tap"]),
                "num_taps": int(first_enabled["num_taps"]),
            }
        for antenna_index in range(num_ant):
            ant = ringing[f"ant{antenna_index}"]
            if not ant.get("enabled", True):
                flat_ring[f"offset_amp_ant{antenna_index}"] = 0.0
                flat_ring[f"offset_phase_ant{antenna_index}_rad"] = 0.0
                flat_ring[f"ring_amp_ant{antenna_index}"] = 0.0
                flat_ring[f"ring_decay_ant{antenna_index}"] = 0.0
                flat_ring[f"ring_phase_ant{antenna_index}_rad"] = 0.0
                flat_ring[f"ring_freq_ant{antenna_index}"] = 0.0
                continue
            if (
                int(ant["start_tap"]) != flat_ring["start_tap"]
                or int(ant["num_taps"]) != flat_ring["num_taps"]
            ):
                raise ValueError(
                    "all antennas must share the same ringing start_num_tap/num_taps"
                )
            flat_ring[f"offset_amp_ant{antenna_index}"] = float(ant["offset_amp"])
            flat_ring[f"offset_phase_ant{antenna_index}_rad"] = float(
                ant["offset_phase_rad"]
            )
            flat_ring[f"ring_amp_ant{antenna_index}"] = float(ant["ring_amp"])
            flat_ring[f"ring_decay_ant{antenna_index}"] = float(ant["ring_decay"])
            flat_ring[f"ring_phase_ant{antenna_index}_rad"] = float(
                ant["ring_phase_rad"]
            )
            flat_ring[f"ring_freq_ant{antenna_index}"] = float(
                ant["ring_freq_rad_per_tap"]
            )
        flat["tx_rx_feedthrough_ringing"] = flat_ring
    
    return flat


def _set_per_antenna_fields(obj: Any, prefix: str, values: list[float]) -> None:
    for antenna_index, value in enumerate(values):
        setattr(obj, f"{prefix}_ant{antenna_index}", float(value))


def apply_case_params_to_cfg(cfg: Any, case_params: dict) -> None:
    """Inject fitted parameters onto an already-built ``SimulationConfig``.

    Call this after the hardware profile's default values have been applied so
    no derived-parameter branch overwrites the fitted values.
    """

    schema = _detect_schema(case_params)
    original_params = case_params
    if schema in {"v2", "structured"}:
        case_params = _structured_to_injection(case_params)

    num_ant = _num_antennas(original_params, default=2)

    # ------------------------------------------------------------------
    # Legacy scalar noise fallback.  v2 later zeroes this and installs the
    # complete stochastic model, so there is never doubling counting.
    # ------------------------------------------------------------------
    noise_cfg = case_params.get("noise", {})
    noise_std_per_ant = noise_cfg.get("std_per_ant")
    if noise_std_per_ant:
        values = [float(v) for v in noise_std_per_ant]
        if len(values) != num_ant:
            raise ValueError(
                f"noise std length mismatch: expected {num_ant}, got {len(values)}"
            )
        cfg.cir.noise_std_per_ant = values
        cfg.cir.noise_std = float(np.mean(values))
    else:
        cfg.cir.noise_std = float(case_params.get("noise_std", 0.0))

    amp_factor = noise_cfg.get("amp_proportional_factor_per_ant")
    if amp_factor:
        cfg.cir.noise_amp_proportional_factor = [float(v) for v in amp_factor]

    # ------------------------------------------------------------------
    # Rx gain de-embedding: inject the measured per-antenna RX gains
    # from the fit so metadata and the legacy noise-scaling fallback
    # (_generate_one_frame line ~2219) see the real values.  When the fit
    # supplies per-antenna noise_std the noise path already skips the
    # rx_gain rescale (see the "No RX-gain rescaler here" comment there),
    # so this only affects metadata + the legacy noise_std-only fallback.
    # ------------------------------------------------------------------
    gain_deemb = case_params.get("gain_deembedding", {})
    rx_gain_db = gain_deemb.get("rx_gain_db")
    if rx_gain_db is not None:
        rx_gain_db = [float(v) for v in rx_gain_db]
        if len(rx_gain_db) < num_ant:
            print(
                f"[apply_case_params_to_cfg] WARNING: gain_deembedding.rx_gain_db"
                f" has {len(rx_gain_db)} values but num_antennas={num_ant};"
                f" injecting the first {len(rx_gain_db)} only."
            )
        if len(rx_gain_db) >= 1:
            cfg.radar.rx1_gain = int(round(rx_gain_db[0]))
        if len(rx_gain_db) >= 2:
            cfg.radar.rx2_gain = int(round(rx_gain_db[1]))
        # Keep the rx_gains list in sync (RadarTestConfig.__post_init__ sets it
        # once at construction; we must refresh it after changing rx1/rx2_gain).
        cfg.radar.rx_gains = [cfg.radar.rx1_gain, cfg.radar.rx2_gain]

    # Fontend stays off in the fitted reproduction path.  The v2 residual
    # already contains the measured quantization floor.
    cfg.frontend.enable_adc_clipping = False
    cfg.frontend.enable_pulse_spreading = False
    cfg.frontend.enable_quantization = False

    quantization = case_params.get("quantization", {})
    q = quantization.get("gain_deembedded", quantization)
    if q.get("step") and q["step"] > 0.0:
        step = float(q["step"])
        max_level = max(q["full_scale_est"] / step, 1.0)
        bits = int(np.clip(np.ceil(np.log2(max_level + 1)) + 1, 4, 16))
        cfg.frontend.quantization_bits = bits
        cfg.frontend.quantization_full_scale = float(q["full_scale_est"])
        cfg.frontend.quantization_dc_offset_i = float(q.get("dc_offset_i", 0.0))
        cfg.frontend.quantization_dc_offset_q = float(q.get("dc_offset_q", 0.0))

    drift = case_params.get("frame_drift", {})
    cfg.frontend.enable_frame_drift = False
    cfg.frontend.frame_amplitude_drift_std = float(
        drift.get("amplitude_drift_std", 0.0)
    )
    cfg.frontend.frame_timing_jitter_std_bins = float(
        drift.get("timing_jitter_std_bins", 0.0)
    )
    phase_values = drift.get("phase_drift_std", [0.0] * num_ant)
    cfg.frontend.frame_phase_drift_std = [float(v) for v in phase_values]

    cfg.optional.enable_gain_mismatch = False
    cfg.optional.enable_phase_mismatch = False

    # ------------------------------------------------------------------
    # Deterministic two-term static model
    # ------------------------------------------------------------------
    leakage = cfg.leakage
    feedthrough = case_params["tx_rx_feedthrough"]
    leakage.enable_tx_rx_feedthrough = True
    leakage.tx_antenna_index = 0
    leakage.tx_rx_feedthrough_start_tap = int(feedthrough["start_tap"])
    leakage.tx_rx_feedthrough_num_taps = int(feedthrough["num_taps"])

    _set_per_antenna_fields(
        leakage,
        "tx_rx_feedthrough_amp",
        [float(feedthrough[f"amp_ant{a}"]) for a in range(num_ant)],
    )
    _set_per_antenna_fields(
        leakage,
        "tx_rx_feedthrough_decay",
        [float(feedthrough[f"decay_ant{a}"]) for a in range(num_ant)],
    )
    for antenna_index in range(num_ant):
        setattr(
            leakage,
            f"tx_rx_feedthrough_phase_ant{antenna_index}_rad",
            float(feedthrough[f"phase_ant{antenna_index}_rad"]),
        )
        setattr(
            leakage,
            f"tx_rx_feedthrough_ripple_amp_ant{antenna_index}",
            float(feedthrough.get(f"ripple_amp_ant{antenna_index}", 0.0)),
        )
        setattr(
            leakage,
            f"tx_rx_feedthrough_ripple_freq_ant{antenna_index}",
            float(feedthrough.get(f"ripple_freq_ant{antenna_index}", 0.0)),
        )
        setattr(
            leakage,
            f"tx_rx_feedthrough_random_std_ant{antenna_index}",
            float(feedthrough.get(f"noise_std_used_ant{antenna_index}", 0.0)),
        )

    phase_slope = case_params.get("_tx_rx_feedthrough_phase_slope_per_ant")
    if phase_slope is not None:
        for antenna_index, slope in enumerate(phase_slope):
            setattr(
                leakage,
                f"tx_rx_feedthrough_phase_slope_ant{antenna_index}",
                float(slope),
            )

    radiation = case_params.get("radiation_leakage")
    leakage.enable_radiation_leakage = False
    leakage.radiation_delay_bins = int(radiation.get("delay_bins", 0))
    leakage.radiation_leakage_factor_12 = float(radiation.get("factor_12", 0.0))
    leakage.radiation_leakage_factor_21 = float(radiation.get("factor_21", 0.0))
    leakage.radiation_phase_offset_12_rad = float(
        radiation.get("phase_offset_12_rad", 0.0)
    )
    leakage.radiation_phase_offset_21_rad = float(
        radiation.get("phase_offset_21_rad", 0.0)
    )
    leakage.enable_pcb_leakage = False
    leakage.enable_early_leakage = False

    ringing = case_params.get("tx_rx_feedthrough_ringing")
    if ringing is not None:
        leakage.enable_tx_rx_feedthrough_ringing = True
        leakage.tx_rx_feedthrough_ringing_start_tap = int(ringing["start_tap"])
        leakage.tx_rx_feedthrough_ringing_num_taps = int(ringing["num_taps"])
        for antenna_index in range(num_ant):
            setattr(
                leakage,
                f"tx_rx_feedthrough_ringing_offset_amp_ant{antenna_index}",
                float(ringing[f"offset_amp_ant{antenna_index}"]),
            )
            setattr(
                leakage,
                f"tx_rx_feedthrough_ringing_offset_phase_ant{antenna_index}_rad",
                float(ringing[f"offset_phase_ant{antenna_index}_rad"]),
            )
            setattr(
                leakage,
                f"tx_rx_feedthrough_ringing_amp_ant{antenna_index}",
                float(ringing[f"ring_amp_ant{antenna_index}"]),
            )
            setattr(
                leakage,
                f"tx_rx_feedthrough_ringing_decay_ant{antenna_index}",
                float(ringing[f"ring_decay_ant{antenna_index}"]),
            )
            setattr(
                leakage,
                f"tx_rx_feedthrough_ringing_phase_ant{antenna_index}_rad",
                float(ringing[f"ring_phase_ant{antenna_index}_rad"]),
            )
            setattr(
                leakage,
                f"tx_rx_feedthrough_ringing_freq_ant{antenna_index}",
                float(ringing[f"ring_freq_ant{antenna_index}"]),
            )
    else:
        leakage.enable_tx_rx_feedthrough_ringing = False

    # ------------------------------------------------------------------
    # Canonical v2 stochastic model.  This is deliberately installed last so
    # it can zero every legacy white-noise field and avoid double counting.
    # ------------------------------------------------------------------
    v2_params = case_params.get("_v2_fitted_model_params")
    if v2_params is not None:
        model = FittedImpairmentModel.load(v2_params)
        model.install_on_config(cfg)


__all__ = [
    "_detect_schema",
    "_structured_to_injection",
    "apply_case_params_to_cfg",
    "load_case_params",
]
