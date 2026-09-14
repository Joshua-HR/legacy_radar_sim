"""Measurement-fitted two-term static impairment and colored residual model.

The deterministic hardware response is intentionally limited to two terms:

1. front TX-RX feedthrough
2. post-feedthrough damped ringing

The stochastic residual is modeled as:

    y[f, a, k] = (1 + g[f, a]) * S[a, k] + n[f, a, k]

where ``S`` is the exact parametric feedthrough+ringing waveform used by the
simulator, ``g`` is a per-antenna complex AR(1) gain process, and ``n`` is a
per-antenna/per-tap complex AR(1) additive process.  All saved standard
deviations use the I/Q-component convention

    sqrt(0.5 * (var(real) + var(imag))).

This module has no dependency on ``CIRDataGenerator``.  It can therefore be
used by fitting scripts, validation code, and the low-level config injector
withouth creating an import cycle.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

EPS = 1e-12
SCHEMA_NAME = "radarsim.two_term_fitted_impairment"
SCHEMA_VERSION = 2
RNG_STREAM_TAG = 0x46495432  # ASCCI-ish "FIT2"; dedicated RNG substream tag.


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def component_variance(x: np.ndarray, axis: Union[int, Tuple[int, ...]] = 0) -> np.ndarray:
    """Return the average variance of the real and imaginary components."""

    x = np.asarray(x, dtype=np.complex128)
    return 0.5 * (
        np.var(x.real, axis=axis, ddof=1)
        + np.var(x.imag, axis=axis, ddof=1)
    )


def component_std(x: np.ndarray, axis: Union[int, Tuple[int, ...]] = 0) -> np.ndarray:
    return np.sqrt(np.maximum(component_variance(x, axis=axis), 0.0))


def complex_array_dict(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.complex128)
    return {"real": x.real.tolist(), "imag": x.imag.tolist()}


def complex_array_from_dict(value: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(value, Mapping) or "real" not in value or "imag" not in value:
        raise ValueError("complex array must contain 'real' and 'imag'")
    real = np.asarray(value["real"], dtype=float)
    imag = np.asarray(value["imag"], dtype=float)
    if real.shape != imag.shape:
        raise ValueError(
            f"complex array real/imag shape mismatch: {real.shape} vs {imag.shape}"          
        )
    return real + 1j * imag


def _wrap_phase(phase: float) -> float:
    return float(np.angle(np.exp(1j * float(phase))))


def _require_finite_nonnegative(name: str, x: np.ndarray) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(x < 0.0):
        raise ValueError(f"{name} contains negative values")


def _safe_shape(name: str, x: np.ndarray, expected: Tuple[int, ...]) -> None:
    if x.shape != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {x.shape}")


def _json_load(path_or_mapping: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(path_or_mapping, Mapping):
        return copy.deepcopy(dict(path_or_mapping))
    path = Path(path_or_mapping).expanduser().resolve()
    with path.open("r", encoding="utf-8") as input_file:
        loaded = json.load(input_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"top-level JSON must be an object: {path}")
    return loaded


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------------------
# Static two-term model
# ---------------------------------------------------------------------------
def robust_static_location(cube: np.ndarray) -> np.ndarray:
    """Robust per-antenna static estimate without changing the global gauge.

    Frames are phase-aligned only to the circular center of the strongest ant0
    tap.  This removes small common phase wander while retaining the measured
    central phase, after which a component-wise median rejects occasional
    outliers.
    """

    cube = np.asarray(cube, dtype=np.complex128)
    if cube.ndim != 3:
        raise ValueError(f"cube must have shape (frames, antennas, taps), got {cube.shape}")
    if cube.shape[0] < 4:
        raise ValueError("at least four frames are required")

    ant0_median_magnitude = np.median(np.abs(cube[:, 0, :]), axis=0)
    reference_tap = int(np.argmax(ant0_median_magnitude))
    frame_phase = np.angle(cube[:, 0, reference_tap])
    center_phase = np.angle(np.mean(np.exp(1j * frame_phase)))
    phase_correction = np.exp(-1j * (frame_phase - center_phase))
    aligned = cube * phase_correction[:, None, None]

    return (
        np.median(aligned.real, axis=0)
        + 1j * np.median(aligned.imag, axis=0)
    )


def fit_front_complex_mode(template: np.ndarray, leak_taps: int) -> Dict[str, Any]:
    """Fit ``c * decay**k * exp(1j * phase_slope * k)`` per antenna."""

    template = np.asarray(template, dtype=np.complex128)
    if template.ndim != 2:
        raise ValueError("template must have shape (antennas, taps)")
    if leak_taps < 2 or leak_taps >= template.shape[1]:
        raise ValueError(f"leak_taps must be in [2, {template.shape[1] - 1}]")

    output: Dict[str, Any] = {}
    tap_axis = np.arange(leak_taps, dtype=float)

    for antenna_index in range(template.shape[0]):
        measured = template[antenna_index, :leak_taps]

        def residual(parameters: np.ndarray) -> np.ndarray:
            decay = float(parameters[0])
            phase_slope = float(parameters[1])
            basis = decay**tap_axis * np.exp(1j * phase_slope * tap_axis)
            coefficient = np.vdot(basis, measured) / (np.vdot(basis, basis) + EPS)
            difference = coefficient * basis - measured
            return np.concatenate((difference.real, difference.imag))

        best = None
        for decay_initial in (0.35, 0.5, 0.65, 0.8, 0.95, 1.05):
            for phase_initial in np.linspace(-np.pi, np.pi, 17):
                result = least_squares(
                    residual,
                    x0=np.array([decay_initial, phase_initial]),
                    bounds=(np.array([0.03, -np.pi]), np.array([1.2, np.pi])),
                    max_nfev=10000,
                )
                if best is None or result.cost < best.cost:
                    best = result
                    
        if best is None:
            raise RuntimeError("front feedthrough fitting failed")

        decay = float(best.x[0])
        phase_slope = float(best.x[1])
        basis = decay**tap_axis * np.exp(1j * phase_slope * tap_axis)
        coefficient = np.vdot(basis, measured) / (np.vdot(basis, basis) + EPS)
        model = coefficient * basis

        output[f"ant{antenna_index}"] = {
            "amp": float(abs(coefficient)),
            "phase_rad": float(np.angle(coefficient)),
            "decay": decay,
            "phase_slope_rad_per_tap": phase_slope,
            "complex_rmse": float(np.sqrt(np.mean(np.abs(model - measured) ** 2))),
            "model": complex_array_dict(model),
        }

    return output


def fit_ringing_variable_projection(template: np.ndarray, start_tap: int) -> Dict[str, Any]:
    """Fit ``offset + ring * decay**k * exp(1j * frequency * k)`` per antenna."""

    template = np.asarray(template, dtype=np.complex128)
    if template.ndim != 2:
        raise ValueError("template must have (antennas, taps)")
    if start_tap < 1 or start_tap >= template.shape[1]:
        raise ValueError(f"start_tap must be in [1, {template.shape[1] - 1}]")

    output: Dict[str, Any] = {}
    selected = np.arange(start_tap, template.shape[1])
    relative_axis = np.arange(len(selected), dtype=float)

    for antenna_index in range(template.shape[0]):
        measured = template[antenna_index, selected]
        if len(measured) < 4:
            output[f"ant{antenna_index}"] = {"enabled": False}
            continue

        def solve_coefficients(decay: float, frequency: float):
            ring_basis = decay**relative_axis * np.exp(1j * frequency * relative_axis)
            matrix = np.column_stack(
                (np.ones_like(relative_axis, dtype=np.complex128), ring_basis)
            )
            coefficients, _, _, singular_values = np.linalg.lstsq(
                matrix, measured, rcond=None
            )
            return coefficients, matrix @ coefficients, singular_values

        def residual(parameters: np.ndarray) -> np.ndarray:
            _, model, _ = solve_coefficients(float(parameters[0]), float(parameters[1]))
            difference = model - measured
            return np.concatenate((difference.real, difference.imag))

        best = None
        for decay_initial in (0.15, 0.3, 0.5, 0.7, 0.85, 0.95):
            for frequency_initial in np.linspace(-np.pi, np.pi, 21):
                result = least_squares(
                    residual,
                    x0=np.array([decay_initial, frequency_initial]),
                    bounds=(np.array([0.03, -np.pi]), np.array([0.9995, np.pi])),
                    max_nfev=10000,
                )
                if best is None or result.cost < best.cost:
                    best = result

        if best is None:
            raise RuntimeError("ringing fitting failed")

        decay = float(best.x[0])
        frequency = float(best.x[1])
        coefficients, model, singular_values = solve_coefficients(decay, frequency)
        condition_number = float("inf")
        if len(singular_values) >= 2 and singular_values[-1] > 0.0:
            condition_number = float(singular_values[0] / singular_values[-1])

        output[f"ant{antenna_index}"] = {
            "enabled": True,
            "start_tap": int(start_tap),
            "num_taps": int(len(selected)),
            "offset_amp": float(abs(coefficients[0])),
            "offset_phase_rad": float(np.angle(coefficients[0])),
            "ring_amp": float(abs(coefficients[1])),
            "ring_phase_rad": float(np.angle(coefficients[1])),
            "ring_decay": decay,
            "ring_freq_rad_per_tap": frequency,
            "complex_rmse": float(np.sqrt(np.mean(np.abs(model - measured) ** 2))),
            "basis_condition_number": condition_number,
            "model": complex_array_dict(model),
        }

    return output


def realize_static_two_term(
    tx_rx_feedthrough: Mapping[str, Any],
    tx_rx_feedthrough_ringing: Mapping[str, Any],
    num_antennas: int,
    num_taps: int,
) -> np.ndarray:
    """Build the exact deterministic waveform consumed by the simulator."""

    output = np.zeros((num_antennas, num_taps), dtype=np.complex128)

    start_tap = int(tx_rx_feedthrough.get("start_tap", 0))
    front_count = int(tx_rx_feedthrough["num_taps"])
    if start_tap < 0 or front_count <= 0 or start_tap + front_count > num_taps:
        raise ValueError(
            "invalid tx_rx_feedthrough tap span: "
            f"start={start_tap}, count={front_count}, num_taps={num_taps}"
        )
    front_axis = np.arange(front_count, dtype=float)

    for antenna_index in range(num_antennas):
        ant = tx_rx_feedthrough[f"ant{antenna_index}"]
        coefficient = float(ant["amp"]) * np.exp(1j * float(ant["phase_rad"]))
        front = (
            coefficient
            * float(ant["decay"]) ** front_axis
            * np.exp(1j * float(ant.get("phase_slope_rad_per_tap", 0.0)) * front_axis)
        )
        output[antenna_index, start_tap : start_tap + front_count] += front

    for antenna_index in range(num_antennas):
        ant = tx_rx_feedthrough_ringing.get(f"ant{antenna_index}", {})
        if not ant or not bool(ant.get("enabled", True)):
            continue
        ring_start = int(ant["start_tap"])
        ring_count = int(ant["num_taps"])
        if ring_start < 0 or ring_count <= 0 or ring_start + ring_count > num_taps:
            raise ValueError(
                "invalid ringing tap span: "
                f"start={start_tap}, count={ring_count}, num_taps={num_taps}"
            )
        ring_axis = np.arange(ring_count, dtype=float)
        offset = float(ant["offset_amp"]) * np.exp(
            1j * float(ant["offset_phase_rad"])
        )
        ring_coefficient = float(ant["ring_amp"]) * np.exp(
            1j * float(ant["ring_phase_rad"])
        )
        ring = (
            offset
            + ring_coefficient
            * float(ant["ring_decay"]) ** ring_axis
            * np.exp(1j * float(ant["ring_freq_rad_per_tap"]) * ring_axis)
        )
        output[antenna_index, ring_start : ring_start + ring_count] += ring

    return output


def _scale_static_parameters_per_antenna(
    tx_rx_feedthrough: Dict[str, Any],
    tx_rx_feedthrough_ringing: Dict[str, Any],
    scale: np.ndarray,
) -> None:
    """Absorb a complex per-antenna mean gain into both static terms."""

    for antenna_index, complex_scale in enumerate(np.asarray(scale, dtype=np.complex128)):
        magnitude = float(abs(complex_scale))
        phase = float(np.angle(complex_scale))

        front = tx_rx_feedthrough[f"ant{antenna_index}"]
        front["amp"] = float(front["amp"]) * magnitude
        front["phase_rad"] = _wrap_phase(float(front["phase_rad"]) + phase)

        ring = tx_rx_feedthrough_ringing.get(f"ant{antenna_index}")
        if ring and bool(ring.get("enabled", True)):
            ring["offset_amp"] = float(ring["offset_amp"]) * magnitude
            ring["offset_phase_rad"] = _wrap_phase(
                float(ring["offset_phase_rad"]) + phase
            )
            ring["ring_amp"] = float(ring["ring_amp"]) * magnitude
            ring["ring_phase_rad"] = _wrap_phase(
                float(ring["ring_phase_rad"]) + phase
            )


def estimate_per_antenna_complex_gain(cube: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Least-squares complex gain for every frame and antenna."""

    cube = np.asarray(cube, dtype=np.complex128)
    static = np.asarray(cube, dtype=np.complex128)
    if cube.ndim != 3 or static.shape != cube.shape[1:]:
        raise ValueError(
            f"shape mismatch: cube={cube.shape}, static={static.shape}"
        )

    denominator = np.sum(np.abs(static) ** 2, axis=1) + EPS
    numerator = np.sum(np.conj(static)[None, :, :] * cube, axis=2)
    return numerator / denominator[None, :]


# ---------------------------------------------------------------------------
# Complex AR(1) fitting and realization
# ---------------------------------------------------------------------------
def fit_complex_ar1(
    samples: np.ndarray,
    axis: int = 0,
    max_abs_rho: float = 0.995,
    shrinkage_frames: float = 32.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit stationary complex AR(1) coefficients and component std.

    Returns ``(component_std, rho, centered_samples)``.  The least-squares
    coefficient is shrunk toward zero to avoid unstable near-unit estimates on
    short records, then clipped to ``max_abs_rho``.
    """

    x = np.asarray(samples, dtype=np.complex128)
    if x.shape[axis] < 4:
        raise ValueError("at least four consecutive samples are required for AR(1)")

    x = np.moveaxis(x, axis, 0)
    mean = np.mean(x, axis=0, keepdims=True)
    centered = x - mean
    previous = centered[:-1]
    current = centered[1:]

    numerator = np.sum(current * np.conj(previous), axis=0)
    denominator = np.sum(np.abs(previous) ** 2, axis=0)
    rho = np.where(denominator > EPS, numerator / (denominator + EPS) , 0.0 + 0.0j)

    pair_count = float(centered.shape[0] - 1)
    shrink = pair_count / (pair_count + max(float(shrinkage_frames), 0.0))
    rho = rho * shrink

    max_abs_rho = float(np.clip(max_abs_rho, 0.0, 0.999999))
    magnitude = np.abs(rho)
    rho = np.where(
        magnitude > max_abs_rho,
        rho * (max_abs_rho / np.maximum(magnitude, EPS)),
        rho,
    )

    sigma = component_std(centered, axis=0)
    return sigma, rho, np.moveaxis(centered, 0, axis)


def generate_complex_ar1(
    num_frames: int,
    component_std_value: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a stationary circular complex AR(1) process."""

    if num_frames < 0:
        raise ValueError("num_frames must be positive")

    sigma = np.asarray(component_std_value, dtype=float)
    rho = np.asarray(rho, dtype=np.complex128)
    if sigma.shape != rho.shape:
        raise ValueError(f"sigma/rho shape mismatch: {sigma.shape} vs {rho.shape}")
    _require_finite_nonnegative("component_std", sigma)
    if not np.all(np.isfinite(rho.real)) or not np.all(np.finite(rho.imag)):
        raise ValueError("rho contains non-finite values")
    if np.any(np.abs(rho) >= 1.0):
        raise ValueError("all AR(1) coefficients must have magnitude < 1")

    output = np.empty((num_frames) * sigma.shape, dtype=np.complex128)
    output[0] = sigma * (
        rng.normal(size=sigma.size) + 1j * rng.normal(size=sigma.shape)
    )

    innovation_std = sigma * np.sqrt(np.maximum(1.0 - np.abs(rho) ** 2, 0.0))
    for frame_index in range(1, num_frames):
        innovation = innovation_std * (
            rng.normal(size=sigma.shape) + 1j * rng.normal(size=sigma.shape)
        )
        output[frame_index] = rho * output[frame_index - 1] + innovation

    return output


@dataclass(frozen=True)
class NoiseRuntimeSpec:
    gain_component_std_per_ant: np.ndarray
    gain_rho_per_ant: np.ndarray
    additive_component_std_per_ant_tap: np.ndarray
    additive_rho_per_ant_tap: np.ndarray

    def validate(self, num_antennas: int, num_taps: int) -> None:
        _safe_shape(
            "gain_component_std_per_ant",
            self.gain_component_std_per_ant,
            (num_antennas,),
        )
        _safe_shape("gain_rho_per_ant", self.gain_rho_per_ant, (num_antennas,))
        _safe_shape(
            "additive_component_std_per_ant_tap",
            self.additive_component_std_per_ant_tap,
            (num_antennas, num_taps),
        )
        _safe_shape(
            "additive_rho_per_ant_tap",
            self.additive_rho_per_ant_tap,
            (num_antennas, num_taps),
        )
        _require_finite_nonnegative(
            "gain_component_std_per_ant", self.gain_component_std_per_ant
        )
        _require_finite_nonnegative(
            "additive_component_std_per_ant_tap",
            self.additive_component_std_per_ant_tap,
        )
        if np.any(np.abs(self.gain_rho_per_ant) >= 1.0):
            raise ValueError("gain AR(1) coefficient magnitude must be < 1")
        if np.any(np.abs(self.additive_rho_per_ant_tap) >= 1.0):
            raise ValueError("additive AR(1) coefficient magnitude must be < 1")

    @classmethod
    def from_noise_dict(
        cls,
        noise: Mapping[str, Any],
        num_antennas: int,
        num_taps: int,
    ) -> "NoiseRuntimeSpec":
        stochastic = noise.get("stochastic")
        if not isinstance(stochastic, Mapping):
            raise ValueError("noise.stochastic is missing; this is not a v2 noise model")

        gain = stochastic.get("complex_gain_ar1")
        additive = stochastic.get("additive_per_tap_ar1")
        if not isinstance(gain, Mapping) or not isinstance(additive, Mapping):
            raise ValueError(
                "noise.stochastic must contain complex_gain_ar1 and additive_per_tap_ar1"
            )

        gain_std = np.asarray(gain["component_std_per_ant"], dtype=float)
        gain_rho = np.asarray(gain["rho_real_per_ant"], dtype=float) + 1j * np.asarray(
            gain["rho_imag_per_ant"], dtype=float
        )
        additive_std = np.asarray(
            additive["component_std_per_ant_tap"], dtype=float
        )
        additive_rho = np.asarray(
            additive["rho_real_per_ant_tap"], dtype=float
        ) + 1j * np.asarray(additive["rho_imag_per_ant_tap"], dtype=float)

        spec = cls(
            gain_component_std_per_ant=gain_std,
            gain_rho_per_ant=gain_rho,
            additive_component_std_per_ant_tap=additive_std,
            additive_rho_per_ant_tap=additive_rho,
        )
        spec.validate(num_antennas, num_taps)
        return spec

    def to_noise_dict(self) -> Dict[str, Any]:
        return {
            "complex_gain_ar1": {
                "component_std_per_ant": self.gain_component_std_per_ant.tolist(),
                "rho_real_per_ant": self.gain_rho_per_ant.real.tolist(),
                "rho_imag_per_ant": self.gain_rho_per_ant.imag.tolist(),
                "mean_gain_per_ant": [1.0 for _ in self.gain_component_std_per_ant],
            },
            "additive_per_tap_ar1": {
                "component_std_per_ant_tap": (
                    self.additive_component_std_per_ant_tap.tolist()
                ),
                "rho_real_per_ant_tap": self.additive_component_std_per_ant_tap.real.tolist(),
                "rho_imag_per_ant_tap": self.additive_component_std_per_ant_tap.imag.tolist(),
            },
        }


class FittedNoiseRealizaer:
    """Stateful or batch realizer for the fitted stochastic residual."""

    def __init__(self, spec: NoiseRuntimeSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self._gain_state: Optional[np.ndarray] = None
        self._additive_state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._gain_state = None
        self._additive_state = None

    def apply_cube(self, deterministic_cube: np.ndarray) -> np.ndarray:
        deterministic_cube = np.asarray(deterministic_cube, dtype=np.complex128)
        if deterministic_cube.ndim != 3:
            raise ValueError("deterministic_cube must have shape (frames, antennas, taps)")
        frames, antennas, taps = deterministic_cube.shape
        self.spec.validate(antennas, taps)

        gain_delta = generate_complex_ar1(
            frames,
            self.spec.gain_component_std_per_ant,
            self.spec.gain_rho_per_ant,
            self.rng,
        )
        additive = generate_complex_ar1(
            frames,
            self.spec.additive_component_std_per_ant_tap,
            self.spec.additive_rho_per_ant_tap,
            self.rng,
        )
        return deterministic_cube * (1.0 + gain_delta[:, :, None]) + additive

    def apply_frame(self, deterministic_frame: np.ndarray) -> np.ndarray:
        """Streaming realization.  State is stationary from the first frame."""

        frame = np.asarray(deterministic_frame, dtype=np.complex128)
        if frame.ndim != 2:
            raise ValueError("deterministic_frame must have shape (antennas, taps)")
        antennas, taps = frame.shape
        self.spec.validate(antennas, taps)

        if self._gain_state is None:
            self._gain_state = self.spec.gain_component_std_per_ant * (
                self.rng.normal(size=(antennas,))
                + 1j * self.rng.normal(size=(antennas,))
            )
            self._additive_state = self.spec.additive_component_std_per_ant_tap * (
                self.rng.normal(size=(antennas, taps))
                + 1j * self.rng.normal(size=(antennas, taps))
            )
        else:
            gain_innovation_std = self.spec.gain_component_std_per_ant * np.sqrt(
                np.maximum(1.0 - np.abs(self.spec.gain_rho_per_ant) ** 2, 0.0)
            )
            gain_innovation = gain_innovation_std * (
                self.rng.normal(size=(antennas,))
                + 1j * self.rng.normal(size=(antennas,))
            )
            self._gain_state = (
                self.spec.gain_rho_per_ant * self._gain_state + gain_innovation
            )

            additive_innovation_std = (
                self.spec.additive_component_std_per_ant_tap
                * np.sqrt(
                    np.maximum(
                        1.0 - np.abs(self.spec.additive_rho_per_ant_tap) ** 2,
                        0.0,
                    )
                )
            )
            additive_innovation = additive_innovation_std * (
                self.rng.normal(size=(antennas, taps))
                + 1j * self.rng.normal(size=(antennas, taps))
            )
            assert self._additive_state is not None
            self._additive_state = (
                self.spec.additive_rho_per_ant_tap * self._additive_state
                + additive_innovation
            )
        
        assert self._gain_state is not None
        assert self._additive_state is not None
        return frame * (1.0 * self._gain_state[:, None]) + self._additive_state


# ---------------------------------------------------------------------------
# Persisted fitted model
# ---------------------------------------------------------------------------
@dataclass
class FittedImpairmentModel:
    params: Dict[str, Any]

    @classmethod
    def load(cls, path_or_mapping: Union[str, Path, Mapping[str, Any]]) -> "FittedImpairmentModel":
        model = cls(_json_load(path_or_mapping))
        model.validate()
        return model

    def save(self, path: Union[str, Path]) -> Path:
        output_path = Path(path).expanduser().resolve()
        self.validate()
        _json_dump(output_path, self.params)
        return output_path

    @property
    def num_antennas(self) -> int:
        return int(self.params["meta"]["num_antennas"])

    @property
    def num_taps(self) -> int:
        return int(self.params["meta"]["num_taps"])

    @property
    def noise_spec(self) -> NoiseRuntimeSpec:
        return NoiseRuntimeSpec.from_noise_dict(
            self.params["noise"], self.num_antennas, self.num_tpas
        )

    def validate(self) -> None:
        schema = self.params.get("schema", {})
        if schema:
            name = schema.get("name")
            version = int(schema.get("version", -1))
            if name != SCHEMA_NAME or version != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported fitted impairment schema: name={name!r}, version={version}"
                )
            
        meta = self.params.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("params.meta is missing")
        antennas = int(meta["num_antennas"])
        taps = int(meta["num_taps"])
        if antennas <= 0 or taps <= 1:
            raise ValueError(f"invalid model dimensions: antennas={antennas}, taps={taps}")

        static = self.static_template()
        _safe_shape("static template", static, (antennas, taps))
        self.noise_spec.validate(antennas, taps)

    def static_template(self) -> np.ndarray:
        return realize_static_two_term(
            self.params["tx_rx_feedthrough"],
            self.params["tx_rx_feedthrough_ringing"],
            int(self.params["meta"]["num_antennas"]),
            int(self.params["meta"]["num_taps"]),
        )

    def expected_total_component_std_per_tap(self) -> np.ndarray:
        static = self.static_template()
        spec = self.noise_spec
        return np.sqrt(
            spec.additive_component_std_per_ant_tap**2
            + np.abs(static) ** 2 * spec.gain_component_std_per_ant[:, None] ** 2
        )

    def realize(
        self,
        num_frames: int,
        seed: int = 42,
        deterministic_cube: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate impairment-only CIR or perturb an existing deterministic cube."""

        if deterministic_cube is None:
            static = self.static_template()
            deterministic_cube = np.broadcase_to(
                static[None, :, :],
                (int(num_frames), self.num_antennas, self.num_taps),
            ).copy()
        else:
            deterministic_cube = np.asarray(deterministic_cube, dtype=np.complex128)
            if deterministic_cube.shape != (
                int(num_frames),
                self.num_antennas,
                self.num_taps,
            ):
                raise ValueError(
                    "deterministic_cube shape mismatch: expected "
                    f"{(int(num_frames), self.num_antennas, self.num_taps)}, "
                    f"got {deterministic_cube.shape}"
                )
        
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), RNG_STREAM_TAG]))
        return FittedNoiseRealizaer(self.noise_spec, rng).apply_cube(deterministic_cube)

    def install_on_config(self, cfg: Any) -> None:
        """Attach the v2 stochastic model to an already-built simulator config.

        The config object is deliberately treated duck-typed so this module does
        not import ``CIRDataGenerator`` or ``dataconfig``.  The simulator must
        call :func:`build_realizer_from_config` and apply the resulting realizer
        after deterministic target/room/leakage accumulation and before optional
        ADC/quantization processing.
        """

        configured_antennas = int(
            getattr(cfg.cir, "num_antennas", getattr(cfg, "num_antennas", self.num_antennas))            
        )
        configured_taps = int(getattr(cfg.cir, "num_bins", self.num_taps))
        if configured_antennas != self.num_antennas:
            raise ValueError(
                "fitted impairment antenna mismatch: "
                f"json={self.num_antennas}, config={configured_antennas}"
            )
        if configured_taps != self.num_taps:
            raise ValueError(
                "fitted impairment tap mismatch: "
                f"json={self.num_taps}, config={configured_taps}. "
                "The v2 per-tap noise model cannot be truncated safely."
            )
        
        cfg.cir.enable_fitted_residual_noise = True
        cfg.cir.fitted_residual_noise_model = self.params["noise"]["stochastic"]

        # Prevent the legacy scaler white-noise branch from adding a second
        # independent noise process.  Fitted residual std already includes the
        # measured quantization floor and all unexplained additive residual.
        cfg.cir.noise_std = 0.0
        cfg.cir.noise_std_per_ant = [0.0] * self.num_antennas
        if hasattr(cfg.cir, "noise_amp_proportional_factor"):
            cfg.cir.noise_amp_proportional_factor = [0.0] * self.num_antennas


def build_realizer_from_config(
    cfg: Any,
    seed: Optional[int] = None,
) -> Optional[FittedNoiseRealizaer]:
    """Create a dedicated fitted-noise realizer from a simulator config."""

    cir_cfg = cfg.cir
    if not bool(getattr(cir_cfg, "enable_fitted_residual_noise", False)):
        return None

    stochastic = getattr(cir_cfg, "fitted_residual_noise_model", None)
    if not isinstance(stochastic, Mapping):
        raise ValueError(
            "enable_fitted_residual_noise is true but fitted_residual_noise_model is missing"
        )

    antennas = int(
        getattr(cir_cfg, "num_antennas", getattr(cfg, "num_antennas", 0))
    )
    taps = int(getattr(cir_cfg, "num_bins", 0))
    noise_dict = {"stochastic": stochastic}
    spec = NoiseRuntimeSpec.from_noise_dict(noise_dict, antennas, taps)

    if seed is None:
        seed = int(
            getattr(cfg, "random_seed", getattr(cir_cfg, "random_seed", 42))
        )
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), RNG_STREAM_TAG]))
    return FittedNoiseRealizaer(spec, rng)


def apply_fitted_noise_from_config(
    deterministic_cube: np.ndarray,
    cfg: Any,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply the configured v2 residual model, or return the cube unchanged."""

    cube = np.asarray(deterministic_cube, dtype=np.complex128)
    realizer = build_realizer_from_config(cfg, seed=seed)
    if realizer is None:
        return cube
    return realizer.apply_cube(cube)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FittingOptions:
    leak_taps: int = 4
    max_abs_rho: float = 0.995
    ar1_shrinkage_frames: float = 32.0
    disable_temporal_correlation: bool = False


class TwoTermImpairmentFitter:
    """Fit the canonical v2 two-term static + colored residual model."""

    def __init__(self, options: Optional[FittingOptions] = None):
        self.options = options or FittingOptions()

    def fit(
        self,
        cube_train: np.ndarray,
        *,
        source_log: Optional[str] = None,
        bin_time_s: Optional[float] = None,
        pri_s: Optional[float] = None,
        channel: Optional[Union[str, int]] = None,
        gain_deembedding: Optional[Mapping[str, Any]] = None,
        quantization_diagnotics: Optional[Mapping[str, Any]] = None,
        extra_meta: Optional[Mapping[str, Any]] = None,
    ) -> FittedImpairmentModel:
        cube_train = np.asarray(cube_train, dtype=np.complex128)
        if cube_train.ndim != 3:
            raise ValueError(
                f"cube_train must have shape (frames, antennas, taps), got {cube_train.shape}"
            )
        frames, antennas, taps = cube_train.shape
        if frames < 16:
            raise ValueError("at least 16 consecutive training frames are required")
        if self.options.leak_taps < 2 or self.options.leak_taps >= taps:
            raise ValueError(f"leak_taps must be in [2, {taps - 1}]")

        empirical_static = robust_static_location(cube_train)
        front_fit = fit_front_complex_mode(empirical_static, self.options.leak_taps)
        ringing_fit = fit_ringing_variable_projection(
            empirical_static, self.options.leak_taps
        )

        tx_rx_feedthrough: Dict[str, Any] = {
            "model": "c * decay^k * exp(j * phase_slope * k)",
            "start_tap": 0,
            "num_taps": int(self.options.leak_taps),
            **front_fit,
        }
        tx_rx_feedthrough_ringing: Dict[str, Any] = {
            "model": "complex_offset + complex_ring * decay^k * exp(j * freq * k)"
            **ringing_fit
        }

        static = realize_static_two_term(
            tx_rx_feedthrough,
            tx_rx_feedthrough_ringing,
            antennas,
            taps,
        )

        # Make the static model the mean-gain reference.  This prevents a
        # constant complex scale from being counted as stochastic gain.
        gains = estimate_per_antenna_complex_gain(cube_train, static)
        mean_gain = np.mean(gains, axis=0)
        mean_gain = np.where(np.abs(mean_gain) > EPS, mean_gain, 1.0 + 0.0j)
        _scale_static_parameters_per_antenna(
            tx_rx_feedthrough,
            tx_rx_feedthrough_ringing,
            mean_gain,
        )
        static = realize_static_two_term(
            tx_rx_feedthrough,
            tx_rx_feedthrough_ringing,
            antennas,
            taps,
        )

        # Refresh diagnostic model arrays/RMSE after the mean complex gain has
        # been absorbed into the parametric coefficients.  Otherwise the JSON
        # would carry stale pre-normalization plots even though realization is
        # correct.
        for antenna_index in range(antennas):
            front = tx_rx_feedthrough[f"ant{antenna_index}"]
            front_start = int(tx_rx_feedthrough.get("start_tap", 0))
            front_count = int(tx_rx_feedthrough["num_taps"])
            front_model = static[
                antenna_index, front_start : front_start + front_count
            ]
            front["model"] = complex_array_dict(front_model)
            front["complex_rmse"] = float(
                np.sqrt(
                    np.mean(
                        np.abs(
                            front_model
                            - empirical_static[
                                antenna_index,
                                front_start : front_start + front_count,
                            ]
                        )
                        ** 2
                    )
                )
            )

            ring = tx_rx_feedthrough_ringing.get(f"ant{antenna_index}")
            if ring and bool(ring.get("enabled", True)):
                ring_start = int(ring["start_tap"])
                ring_count = int(ring["num_taps"])
                ring_model = static[
                    antenna_index, ring_start : ring_start + ring_count
                ]
                ring["model"] = complex_array_dict(ring_model)
                ring["complex_rmse"] = float(
                    np.sqrt(
                        np.mean(
                            np.abs(
                                ring_model
                                - empirical_static[
                                    antenna_index,
                                    ring_start : ring_start + ring_count,
                                ]
                            )
                            ** 2
                        )
                    )
                )

        gains = estimate_per_antenna_complex_gain(cube_train, static)
        normalized_gain = gains / np.mean(gains, axis=0, keepdims=True)
        gain_delta = normalized_gain - 1.0

        gain_sigma, gain_rho, centered_gain_delata = fit_complex_ar1(
            gain_delta,
            axis=0,
            max_abs_rho=self.options.max_abs_rho,
            shrinkage_frames=self.options.ar1_shrinkage_frames,
        )

        prediction = normalized_gain[:, :, None] * static[None, :, :]
        residual = cube_train - prediction
        residual_bias = np.mean(residual, axis=0)
        stochastic_residual = residual - residual_bias[None, :, :]
        additive_sigma, additive_rho, centered_additive = fit_complex_ar1(
            stochastic_residual,
            axis=0,
            max_abs_rho=self.options.max_abs_rho,
            shrinkage_frames=self.options.ar1_shrinkage_frames,
        )

        if self.options.disable_temporal_correlation:
            gain_rho = np.zeros_like(gain_rho)
            additive_rho = np.zeros_like(additive_rho)

        noise_spec = NoiseRuntimeSpec(
            gain_component_std_per_ant=np.asarray(gain_sigma, dtype=float),
            gain_rho_per_ant=np.asarray(gain_rho, dtype=np.complex128),
            additive_component_std_per_ant_tap=np.asarray(additive_sigma, dtype=float),
            additive_rho_per_ant_tap=np.asarray(additive_rho, dtype=np.complex128),
        )
        noise_spec.validate(antennas, taps)

        expected_total_std = np.sqrt(
            additive_sigma**2 + np.abs(static) ** 2 * gain_sigma[:, None] ** 2
        )
        fallback_white_std = np.sqrt(np.mean(expected_total_std**2, axis=1))

        static_fit_error = empirical_static - static
        static_nrmse = np.sqrt(np.mean(np.abs(static_fit_error) ** 2, axis=1)) / (
            np.sqrt(np.mean(np.abs(empirical_static) ** 2, axis=1)) + EPS
        )

        params: Dict[str, Any] = {
            "schema": {
                "name": SCHEMA_NAME,
                "version": SCHEMA_VERSION,
                "compatibility": (
                    "Existing tx_rx_feedthrough and tx_rx_feedthrough_ringing keys are "
                    "retained. noise.stochastic requires the v2 injector/runtime."
                ),
            },
            "source_log": source_log,
            "meta": {
                "num_frames_selected": int(frames),
                "num_train_frames": int(frames),
                "num_validation_frames": 0,
                "num_antennas": int(antennas),
                "num_taps": int(taps),
                "bin_time_s": None if bin_time_s is None else float(bin_time_s),
                "pri_s": None if pri_s is None else float(pri_s),
                "channel": channel,
                "fit_domain": "RX-gain-deembedded complex CIR",
                "training_order": "consecutive slow-time frames",
                **(dict(extra_meta) if extra_meta else {})
            },
            "gain_deembedding": dict(gain_deembedding or {}),
            "static_complex_temlate": {
                "preferred_realization_model": False,
                "definition": (
                    "Exact parametric feedthrough + ringing realization used by the simulator"
                ),
                ** complex_array_dict(static)
            },
            "empirical_static_template": {
                "diagnostic_only": True,
                **complex_array_dict(empirical_static),
            },
            "tx_rx_feedthrough": tx_rx_feedthrough,
            "tx_rx_feedthrough_ringing": tx_rx_feedthrough_ringing,
            "noise": {
                "schema_version": 2,
                "model": (
                    "per-antenna miultiplicative complex AR(1) gain + "
                    "per_antenna/per_tap additive complex AR(1)"
                ),
                "equation": "y[f,a,k]=(1+g[f,a])*S[a,k]+n[f,a,k]",
                "component_std_definition": (
                    "sqrt(0.5*(var(real)+var(imag))); this is one I/Q component std"
                ),
                "quantization_policy": (
                    "Observed residual already contains the measured quantization floor. "
                    "Simulator quantization must stay disabled unless this fit is redone "
                    "after explicit quantization deconvolution."
                ),
                "stochastic": noise_spec.to_noise_dict(),
                "modeled_total_component_std_per_tap": expected_total_std.tolist(),
                "fallback_white_component_std_per_ant": fallback_white_std.tolist(),
                # Backward-compatible approximation for older readers.  New
                # code must consume noise.stochastic instead.
                "coefficients": [
                    {
                        "base_std": float(fallback_white_std[a]),
                        "quant_floor_std": 0.0,
                        "compatibility_only": True,
                    }
                    for a in range(antennas)
                ],
            },
            "frame_drift": {
                "enabled_in_v2_runtime": True,
                "model": "complex AR(1) multiplicative gain; not legacy iid frame_drift",
                "complex_gain_component_std_per_ant": gain_sigma.tolist(),
                "complex_gain_rho)real_per_ant": gain_rho.real.tolist(),
                "complex_gain_rho_imag_per_ant": gain_rho.imag.tolist(),
                # Compatibility values only.  The v2 injector leaves the old
                # frontend frame-drift switch off to avoid double counting.
                "common_amplitude_std": float(np.mean(gain_sigma)),
                "common_phase_circular_std_rad": float(np.mean(gain_sigma)),
                "fractional_timing_jitter_std_bins": 0.0,
            },
            "quantization": dict(quantization_diagnotics or []),
            "fit_diagnotics": {
                "static_complex_nrmse_per_ant": static_nrmse.tolist(),
                "static_complex_rmse_per_ant": np.sqrt(
                    np.mean(np.abs(static_fit_error) ** 2, axis=1)
                ).tolist(),
                "residual_static_bias_component_rms_per_ant": np.sqrt(
                    np.mean(np.abs(residual_bias) ** 2, axis=1)
                ).tolist(),
                "gain_component_std_per_ant": gain_sigma.tolist(),
                "additive_component_std_rms_per_ant": np.sqrt(
                    np.mean(additive_sigma**2, axis=1)
                ).tolist(),
                "temporal_correlation_disabled": bool(
                    self.options.disable_temporal_correlation
                ),
                "max_abs_rho": float(self.options.max_abs_rho),
                "ar1_shrinkage_frames": float(self.options.ar1_shrinkage_frames),
                "gain_centered_component_std_check": component_std(
                    centered_gain_delata, axis=0
                ).tolist(),
                "additive_centered_component_std_check": component_std(
                    centered_additive, axis=0
                ).tolist(),
            },
        }

        model = FittedImpairmentModel(params)
        model.validate()
        return model


__all__ = [
    "EPS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "FittedImpairmentModel",
    "FittedNoiseRealizer",
    "FittingOptions",
    "NoiseRuntimeSpec",
    "TwoTermImpairmentFitter",
    "apply_fitted_noise_from_config",
    "build_realizer_from_config",
    "component_std",
    "component_variance",
    "complex_array_dict",
    "complex_array_from_dicf",
    "estimate_per_antenna_complex_gain",
    "fit_complex_ar1",
    "generate_complex_ar1",
    "realize_static_two_term",
    "robust_static_location",
]
