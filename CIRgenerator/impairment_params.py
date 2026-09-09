"""Shared injection of measurement-fitted impairment params onto a config.

This is a low-level helper that imports nothing from CIRDataGenerator (to avoid
an import cycle) and only mutates fields on an already-built SimulationConfig.

It is shared by:
    * ``validation/cir_validation_adapter.build_injected_simulator`` (called
      right after ``CIRSimulator(cfg)`` construction), and
    * ``CIRDataGenerator.CIRSimulator._inject_params_from_json`` for the
      config-driven ``hardware_profile = "fitted"`` profile.

Schema support: two ``*_params.json`` schemas are accepted.
    * ``*_params_structured.json`` (CANONICAL) - per-antenna nested sub-dicts and
      a ``static_complex_template``; feedthrough carries a per-tap
      ``phase_slope_rad_per_tap``.
    * legacy ``case*_params.json`` (READ-ONLY BACK-COMPAT SHIM) - flat
      ``*_ant0``/``*_ant1`` scalars, no phase slope.
``apply_case_params_to_cfg`` auto-detects the schema; the structured schema is
normalized up-front into the same flat dict the legacy path already consumed, so
a single field-assignment body serves both.  Legacy files hit byte-identical
assignments/values as before; only the structured path sets the new
``tx_rx_feedthrough_phase_slope_ant{i}`` fields.

Reproduction design (see CLAUDE.md, "TX-RX feedthrough ringing tail vs. antenna
crosstalk"): the injected realization uses ``noise_std`` + ``tx_rx_feedthrough``
+ ``tx_rx_feedthrough_ringing`` ONLY.  ADC clipping, quantization and frame
drift are intentionally DISABLED.  The quantization / frame-drift numeric fields
are still recorded (inert while their enable flags are off) so a caller can flip
a flag on later for experimentation.  ``radiation_leakage`` factors/phase/delay
are injected but the leakage itself is left OFF by design (avoids double-counting
with the ringing tail)

IMPORTANT: this must be called AFTER the hardware profile has been applied, so
that ``_update_derived_hardware_params()``'s formula branches do not overwrite
these fitted values (the "overwrite trap").
"""

from __future__ import annotations

import numpy as np


def _detect_schema(params: dict) -> str:
    """Return ``"structured"`` or ``"legacy"`` for a ``*_params.json`` dict.

    Structured files carry a top-level ``static_complex_template`` and a
    ``tx_rx_feedthrough`` with per-antenna sub-dicts (``ant0``/``ant1``).
    """
    if "static_complex_template" in params:
        return "structured"
    tf = params.get("tx_rx_feedthrough", {})
    if isinstance(tf, dict) and "ant0" in tf:
        return "structured"
    return "legacy"


def _structured_to_injection(params: dict) -> dict:
    """Convert a ``*_params_structured.json`` dict to the flat legacy shape.

    The returned dict is what the (unchanged) legacy field-assignment body in
    ``apply_case_params_to_cfg`` consumes.  ``ripple_*`` and
    ``noise_std_used_*`` are set to ``0.0`` (the structured feedthrough model is
    ``c * decay**k * exp(j*(phase + phase_slope*k))`` with no ripple/random
    term); the per-tap phase slope is carried under the private key
    ``_tx_rx_feedthrough_phase_slope_per_ant`` so the caller can set the new
    ``tx_rx_feedthrough_phase_slope_ant{i}`` fields.  ``radiation_leakage``,
    ``quantization`` and ``frame_drift`` are force-DISABLED in the reproduction
    path, so their numeric values do not affect output - they are synthesized
    here only so the legacy assignment body does not ``KeyError``.
    """
    num_ant = int(params.get("meta", {}).get("num_antennas", 2))

    flat: dict = {}

    # -- Noise: per-antenna base_std (matches try_simulator_validation). --
    coeffs = params.get("noise", {}).get("coefficients", [])
    flat["noise"] = {"std_per_ant": [float(c["base_std"]) for c in coeffs]}

    # -- Quantization: disabled; empty dict skips the step-derivation branch. --
    flat["quantization"] = {}

    # -- Frame drift: disabled; map common (per-frame) stats onto the flat keys. --
    fd = params.get("frame_drift", {})
    flat["frame_drift"] = {
        "amplitude_drift_std": float(fd.get("common_amplitude_std", 0.0)),
        "timing_jitter_std_bins": float(fd.get("fractional_timing_jitter_std_bins", 0.0)),
        "phase_drift_std": [
            float(fd.get("common_phase_circular_std_rad", 0.0)) for _ in range(num_ant)
        ],
    }

    # -- TX-RX feedthrough: per-antenna sub-dicts -> flat *_ant{i} scalars. --
    tf = params["tx_rx_feedthrough"]
    flat_tf: dict = {"start_tap": tf["start_tap"], "num_taps": tf["num_taps"]}
    phase_slopes: list[float] = []
    for a in range(num_ant):
        ant = tf[f"ant{a}"]
        flat_tf[f"amp_ant{a}"] = ant["amp"]
        flat_tf[f"decay_ant{a}"] = ant["decay"]
        flat_tf[f"phase_ant{a}_rad"] = ant["phase_rad"]
        flat_tf[f"ripple_amp_ant{a}"] = 0.0
        flat_tf[f"ripple_freq_ant{a}"] = 0.0
        flat_tf[f"noise_std_used_ant{a}"] = 0.0
        phase_slopes.append(float(ant.get("phase_slope_rad_per_tap", 0.0)))
    flat["tx_rx_feedthrough"] = flat_tf
    flat["_tx_rx_feedthrough_phase_slope_per_ant"] = phase_slopes

    # -- Radiation leakage: absent in structured; disabled -> synthesize zeros. --
    flat["radiation_leakage"] = {
        "delay_bins": 0,
        "factor_12": 0.0,
        "factor_21": 0.0,
        "phase_offset_12_rad": 0.0,
        "phase_offset_21_rad": 0.0,
    }

    # -- TX-RX feedthrough ringing: per-antenna sub-dicts -> flat *_ant{i}. --
    rg = params.get("tx_rx_feedthrough_ringing")
    if rg is not None:
        ant0 = rg["ant0"]
        flat_rg: dict = {
            "start_tap": ant0["start_tap"],
            "num_taps": ant0["num_taps"],
        }
        for a in range(num_ant):
            ant = rg[f"ant{a}"]
            flat_rg[f"offset_amp_ant{a}"] = ant["offset_amp"]
            flat_rg[f"offset_phase_ant{a}_rad"] = ant["offset_phase_rad"]
            flat_rg[f"ring_amp_ant{a}"] = ant["ring_amp"]
            flat_rg[f"ring_decay_ant{a}"] = ant["ring_decay"]
            flat_rg[f"ring_phase_ant{a}_rad"] = ant["ring_phase_rad"]
            flat_rg[f"ring_freq_ant{a}"] = ant["ring_freq_rad_per_tap"]
        flat["tx_rx_feedthrough_ringing"] = flat_rg
    
    return flat


def apply_case_params_to_cfg(cfg, case_params: dict) -> None:
    """Inject a ``case*_params.json``-style dict onto ``cfg`` in place.

    Args:
        cfg: a constructed ``SimulationConfig`` (its ``.cir``/``.frontend``/
            ``.optional``/``.leakage`` sub-configs are mutated).
        case_params: dict loaded from a ``*_params.json`` file produced by
            ``impairments_fitting``. Either the canonical structured schema
            (``*_params_structured.json``) or the legacy flat schema
            (``case*_params.json``) is accepted; the structured schema is
            normalized to the legacy flat shape up-front. The legacy schema
            requires ``tx_rx_feedthrough``, ``radiation_leakage``, ``noise``,
            ``frame_drift``, ``quantization``; ``tx_rx_feedthrough_ringing`` is
            optional for backward compatibility with params files saved before
            that model was added.
    """
    # Normalized the canonical structured schema to the legacy flat shape so the
    # single field-assignment body below serves both.  Legacy dicts pass through
    # untouched (byte-identical behavior).
    if _detect_schema(case_params) == "structured":
        case_params = _structured_to_injection(case_params)

    # -- Noise (per-antenna, gain-deembedded domain) --
    noise_cfg = case_params.get("noise", {})
    noise_std_per_ant = noise_cfg.get("std_per_ant")
    if noise_std_per_ant:
        noise_std_per_ant = [float(v) for v in noise_std_per_ant]
        cfg.cir.noise_std_per_ant = noise_std_per_ant
        cfg.cir.noise_std = float(np.mean(noise_std_per_ant))  # kept for scalar-only readers
    else:
        cfg.cir.noise_std = float(case_params.get("noise_std", 0.0))

    # Optional amplitude-proportional noise term (backward-compatible with
    # older case*_params.json files that predate this key).
    amp_proportional_factor = noise_cfg.get("amp_proportional_factor_per_ant")
    if amp_proportional_factor:
        cfg.cir.noise_amp_proportional_factor = [float(v) for v in amp_proportional_factor]

    # -- Frontend impairments: DISABLED for the reproduction (see module docstring).
    # Numerics are still recorded (inert while disabled) for later experimentation.
    cfg.frontend.enable_adc_clipping = False
    cfg.frontend.enable_pulse_spreading = False
    cfg.frontend.enable_quantization = False

    q = case_params["quantization"].get("gain_deembedded", case_params["quantization"])
    if q.get("step") and q["step"] > 0:
        step = float(q["step"])
        max_level = max(q["full_scale_set"] / step, 1.0)
        bits = int(np.clip(np.ceil(np.log2(max_level + 1)) + 1, 4, 16))
        cfg.frontend.quantization_bits = bits
        cfg.frontend.quantization_full_scale = float(q["full_scale_set"])
        cfg.frontend.quantization_dc_offset_i = float(q["dc_offset_i"])
        cfg.frontend.quantization_dc_offset_q = float(q["dc_offset_q"])

    d = case_params["frame_drift"]
    cfg.frontend.enable_frame_drift = False
    cfg.frontend.frame_amplitude_drift_std = float(d["amplitude_drift_std"])
    cfg.frontend.frame_timing_jitter_std_bins = float(d["timing_jitter_std_bins"])
    cfg.frontend.frame_phase_drift_std = [float(v) for v in d["phase_drift_std"]]

    cfg.optional.enable_gain_mismatch = False
    cfg.optional.enable_phase_mismatch = False

    # -- Leakage: tx_rx_feedthrough + ringing ON; radiation injected but OFF --
    L = cfg.leakage

    tf = case_params["tx_rx_feedthrough"]
    L.enable_tx_rx_feedthrough = True
    L.tx_antenna_index = 0
    L.tx_rx_feedthrough_start_tap = int(tf["start_tap"])
    L.tx_rx_feedthrough_num_taps = int(tf["num_taps"])
    L.tx_rx_feedthrough_amp_ant0 = float(tf["amp_ant0"])
    L.tx_rx_feedthrough_amp_ant1 = float(tf["amp_ant1"])
    L.tx_rx_feedthrough_decay_ant0 = float(tf["decay_ant0"])
    L.tx_rx_feedthrough_decay_ant1 = float(tf["decay_ant1"])
    L.tx_rx_feedthrough_phase_ant0_rad = float(tf["phase_ant0_rad"])
    L.tx_rx_feedthrough_phase_ant1_rad = float(tf["phase_ant1_rad"])
    L.tx_rx_feedthrough_ripple_amp_ant0 = float(tf["ripple_amp_ant0"])
    L.tx_rx_feedthrough_ripple_amp_ant1 = float(tf["ripple_amp_ant1"])
    L.tx_rx_feedthrough_ripple_freq_ant0 = float(tf["ripple_freq_ant0"])
    L.tx_rx_feedthrough_ripple_freq_ant1 = float(tf["ripple_freq_ant1"])
    L.tx_rx_feedthrough_random_std_ant0 = float(tf["noise_std_used_ant0"])
    L.tx_rx_feedthrough_random_std_ant1 = float(tf["noise_std_used_ant1"])

    # Per-tap feedthrough phase slope (structured schema only). Absent for
    # legacy files, so these fields stay unset -> the simulator's
    # getattr(..., 0.0) default keeps legacy callers byte-identical.
    phase_slope_per_ant = case_params.get("_tx_rx_feedthrough_phase_slope_per_ant")
    if phase_slope_per_ant is not None:
        for a, slope in enumerate(phase_slope_per_ant):
            setattr(L, f"tx_rx_feedthrough_phase_slope_ant{a}", float(slope))

    r = case_params["radiation_leakage"]
    # Same design decision as the measured fit: radiation_leakage stays OFF;
    # factor/phase/delay are injected so a caller could enable it later.
    L.enable_radiation_leakage = False
    L.radiation_delay_bins = int(r["delay_bins"])
    L.radiation_leakage_factor_12 = float(r["factor_12"])
    L.radiation_leakage_factor_21 = float(r["factor_21"])
    L.radiation_phase_offset_12_rad = float(r["phase_offset_12_rad"])
    L.radiation_phase_offset_21_rad = float(r["phase_offset_21_rad"])
    L.enable_pcb_leakage = False
    L.enable_early_leakage = False

    rg = case_params.get("tx_rx_feedthrough_ringing")
    if rg is not None:
        L.enable_tx_rx_feedthrough_ringing = True
        L.tx_rx_feedthrough_ringing_start_tap = int(rg["start_tap"])
        L.tx_rx_feedthrough_ringing_num_taps = int(rg["num_taps"])
        L.tx_rx_feedthrough_ringing_offset_amp_ant0 = float(rg["offset_amp_ant0"])
        L.tx_rx_feedthrough_ringing_offset_amp_ant1 = float(rg["offset_amp_ant1"])
        L.tx_rx_feedthrough_ringing_offset_phase_ant0_rad = float(rg["offset_phase_ant0_rad"])
        L.tx_rx_feedthrough_ringing_offset_phase_ant1_rad = float(rg["offset_phase_ant1_rad"])
        L.tx_rx_feedthrough_ringing_amp_ant0 = float(rg["ring_amp_ant0"])
        L.tx_rx_feedthrough_ringing_amp_ant1 = float(rg["ring_amp_ant1"])
        L.tx_rx_feedthrough_ringing_decay_ant0 = float(rg["ring_decay_ant0"])
        L.tx_rx_feedthrough_ringing_decay_ant1 = float(rg["ring_decay_ant1"])
        L.tx_rx_feedthrough_ringing_phase_ant0_rad = float(rg["ring_phase_ant0_rad"])
        L.tx_rx_feedthrough_ringing_phase_ant1_rad = float(rg["ring_phase_ant1_rad"])
        L.tx_rx_feedthrough_ringing_freq_ant0 = float(rg["ring_freq_ant0"])
        L.tx_rx_feedthrough_ringing_freq_ant1 = float(rg["ring_freq_ant1"])