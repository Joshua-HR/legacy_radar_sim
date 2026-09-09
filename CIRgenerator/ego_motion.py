"""Radar ego motion: slow-time radar pose trajectories.

Purpose
-------
Reproduce the CIR changes caused by a user holding the radar-bearing device in
their hand (physiological hand tremor). The radar body acquires a 6-DoF pose
that varies from one slow-time sample (frame) to the next; the antenna local
offsets and phase centres are rigidly attached to that body.

Pipeline boundary (enforced, do not widen)
------------------------------------------
    slow_time + configured radar base pose
        -> pose trajectory (exactly one pose per slot-time sample)
        -> Tx/Rx antenna phase-centre world coordinates
        -> Path geometry -> delay -> carrier phase -> Doppler

The motion model NEVER writes a CIR phase or a tap. IT only updates geometry;
delay and phase are recomputed from that geometry at every slow-time sample by
CIRSimulator._generate_one_frame(). Doppler is therefore a *derived*
diagnostic of the time-varying total path length -- it is never multiplied in
as a separate phase term (no double application).

Convenctions
------------
* Right-handed coordinates, +Z up, SI units internally, unit suffixes on all
  field names (position_m, frequency_hz, rotation_rad, ...).
* Quaternions are scalar-first ``[w, x, y, z]```, active local-to-parent
  (body-to-world), composed with the Hamilton product.
* Angles reported to the antenna-pattern layer keep this project's existing
  HFSS-style convention (phi = azimuth, theta = polar angle from +Z); this
  module does not touch that layer. See docs/ego_motion_model.md for the
  documented consequence of that boundary.

Motion types
------------
* ``static``            -- default. Base pose at every slow-time sample. Takes a
                           legacy fast path so the generated CIR is bit-identical
                           to the pre-ego-motion engine.
* ``sampled_pose`       -- replay a recorded 6-DoF trajectory from CSV. File
                           contents are body-local deltas relative to the base
                           pose, NOT world pose.
* ``handled_jitter``    -- deterministic band-limited hand tremor synthesised
                           from K sinusoidal components per axis.

Out of scope by design (do not add here): constant-velocity radar translation
(express it as a sampled_pose CSV), rotation-only/sacn/circular/waypoint-spline
motion, acceleration or angular-rate dynamics, IMU noise, measured tremor
statistics, visibility/occlusion, NLOS multipath.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

#: Quaternion norm tolerance at CSV load time. Rows are rejected beyond this,
#: then normalised. Loose enough for 6-decimal CSV exports, tight enough to
#: catch a genuinely non-unit (e.g. unnormalised or garbage) quaternion.
QUAT_NORM_TOL = 1.0e-6

#: Above this |dot| the two quaternions are treated as parallel and SLERP
#: degenerates to normalised linear interpolation (sin(theta0) -> 0 guard).
#: Value fixed by the implementation contract.
SLERP_LINEAR_DOT_THRESHOLD = 0.9995

#: Relative tolerance for the "slow-time grid must be uniformly spaced" gate.
GRID_UNIFORMITY_RTOL = 1.0e-9

#: An axis whose configured RMS is > 0 but whose unnormalised synthesis has an
#: RMS below this cannot be renormalised onto the requested grid -> error.
MIN_ACHIEVABLE_RAW_RMS = 1.0e-30

#: Fixed 'EGO1' tag mixed into the ego-motion SeedSequence so this stream can
#: never coincide with the impairment/noise stream. See _ego_rng()
EGO_SEED_TAG = 0x45474F31

#: Exact CSV header required by sampled_pose, in this order.
SAMPLED_POSE_CSV_COLUMNS: Tuple[str, ...] = (
    "time_s",
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
)


# ---------------------------------------------------------------------------
# Quaternion algebra: scalar-first [w, x, y, z], active local-to-parent
# ---------------------------------------------------------------------------

IDENTITY_QUAT_WXYZ: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def quat_normalize(q) -> np.ndarray:
    """Normalize a scale_first quaternion to unit norm.

    Args:
        q: array-like [w, x, y, z].
    
    Returns:
        np.ndarray shape [4], unit form, float64.
    
    Raises:
        ValueError: if q is not length 4, is non-finite, or has zero sum
    """
    arr = np.asarray(q, dtype=float).reshape(-1)

    if arr.shape[0] != 4:
        raise ValueError(f"Quaternion must have 4 components [w,x,y,z], got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Quaternion has non-finite components: {arr!r}")
    
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("Quaternion has zero norm; cannot normalise")

    return arr / norm


def quat_multiply(a, b) ->  np.ndarray:
    """Hamilton product ``a (x) b`` of two scalar-first quaternions.

    With the active local-to-parent convention this composes rotations as
    "apply b in a's local frame, then a": R(a (x) b) == R(a) @ R(b). That is
    exactly the composition the base-pose/delta contract needs
    (``q_world = q_base (x) delta_q``).

    Args:
        a: array-like [w, x, y, z], the outer/parent rotation.
        b: array-like [w, x, y, z], the inner/local rotation.

    Returns:
        np.ndarray shape [4] (not renormalised; inputs are assumed unit).
    """
    aw, ax, ay, az = (float(v) for v in np.asarray(a, dtype=float).reshape(-1))
    bw, bx, by, bz = (float(v) for v in np.asarray(b, dtype=float).reshape(-1))

    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_to_rotation_matrix(q) -> np.ndarray:
    """Rotation matrix R such that ``v_parent = R @ v_local`` (active).

    Args:
        q: array-like unit quaternion [w, x, y, z]
    
    Returns:
    np.ndarray shape [3, 3], orthonormal with det == +1.
    """
    w, x, y, z = (float(v) for v in quat_normalize(q))

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_from_rotation_vector(r_rad) -> np.ndarray:
    """Exponential map ``q = exp(r)`` of a body-local rotation vector.

    The rotation vector's direction is the rotation axis and its magnitude is
    the rotation angle in radians, so
    ``q = [cos(|r|/2), sine(|r|/2) * r/|r|]``. This is the only rotation
    accumulation path used by this module -- Euler-angle accumulation is
    deliberately NOT supported, because summing Euler angles is not a group
    operation oon SO(3) and would drift for a multi-axis tremor.

    The ``sin(|r|/2)/|r| = 0.5 * sinc(|r|/(2*pi))`` form is exact at r == 0
    (numpy's sinc is the normalised sinc), so no small-angle branch is needed.

    Args:
        r_rad: array-like [3], body-local rotation vector in radians.

    Returns:
        np.ndarray shape [4], unit quaternion [w, x, y, z]
    """
    r = np.asarray(r_rad, dtype=float).reshape(-1)

    if r.shape[0] != 3:
        raise ValueError(f"Rotation vector must have 3 components, got {r.shape[0]}")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"Rotation vector has non-finite components: {r!r}")

    theta = float(np.linalg.norm(r))
    half_sinc = 0.5 * float(np.sinc(theta / (2.0 * np.pi)))

    return np.array(
        [np.cos(theta / 2.0), half_sinc * r[0], half_sinc * r[1], half_sinc * r[2]],
        dtype=float,
    )


def quat_slerp_shortest(q0, q1, u: float) -> np.ndarray:
    """Shortest-arc SLERP between two unit quaternions.

    The sign of q1 is flipped when the dot product is negative so the
    interpolation always takes the shorter of the two great-circle arcs
    (q and -q are the same rotation). When the two are near-parallel
    (|dot| > SLERP_LINEAR_DOT_THRESHOLD) the sin(theta0) denominator becomes
    ill-conditioned, so the limit is taken with normalised linear
    interpolation instead.

    Args:
        q0: array-like unit quaternion at u == 0.
        q1: array-like unit quaternion at u == 1.
        u (flot): interpolation parameter in [0, 1].

    Returns:
        np.ndarray shape [4], unit quaternion.
    """
    a = quat_normalize(q0)
    b = quat_normalize(q1)

    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot

    dot = min(dot, 1.0)

    if dot > SLERP_LINEAR_DOT_THRESHOLD:
        return quat_normalize(a + u * (b - a))

    theta0 = float(np.arccos(dot))
    sin_theta0 = float(np.sin(theta0))

    s0 = float(np.sin((1.0 - u) * theta0)) / sin_theta0
    s1 = float(np.sin(u * theta0)) / sin_theta0

    return quat_normalize(s0 * a + s1 * b)


def canonicalize_quaternion_sign_sequence(quats: np.ndarray) -> np.ndarray:
    """Make a quaternion time series sign-continuous, in place on a copy.

    A recorded trace may flip between q and -q between consecutive rows (both
    encode the same rotation). Left alone, that flip is invisible to a node
    lookup but makes interpolation take the long arc on that one segment. This
    pre-pass flips any row whose dot product with the previous *canonical* row
    is negative, so downstream SLERP never has to repair it.

    Args:
        quats (np.ndarray): shape [N, 4], scalar-first, already unit norm.

    Returns:
        np.ndarray shape [N, 4], sign-continuous copy.
    """
    out = np.array(quats, dtype=float, copy=True)

    for i in range(1, out.shape[0]):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    
    return out


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """An immutable rigid-body pose: world position + body-to-world rotation.

    Attributes:
        position_m (np.ndarray): shape [3], world position in metres.
        quaternion_wxyz (np.ndarray): shape [4], unit, scalar-first, active
            local-to-parent (body-to-world).
    """

    position_m: np.ndarray
    quaternion_wxyz: np.ndarray

    @staticmethod
    def from_parts(position_m, quaternion_wxyz=IDENTITY_QUAT_WXYZ) -> "Pose":
        """Build a validated Pose from array-likes."""
        p = np.asarray(position_m, dtype=float).reshape(-1)

        if p.shape[0] != 3:
            raise ValueError(f"Pose position must have 3 components, got {p.shape[0]}")
        if not np.all(np.isfinite(p)):
            raise ValueError(f"Pose position has non-finite components: {p!r}")

        return Pose(position_m, quaternion_wxyz=quat_normalize(quaternion_wxyz))
    
    def rotation_matrix(self) -> np.ndarray:
        """Body-to-world rotation matrix, shape [3, 3]."""
        return quat_to_rotation_matrix(self.quaternion_wxyz)

    def transform_body_points(self, points_body_m) -> np.ndarray:
        """Map rigidly-attached body-frame points into world coordinates.

        ``p_world = position_m + R @ p_body`` applied row-wise.

        Args:
            points_body_m: array-like shape [3] or [N, 3], body-frame metres.
        
        Returns:
            np.ndarray with the same shape as the input.
        """
        pts = np.asarray(points_body_m, dtype=float)
        single = pts.ndim == 1

        if single:
            pts = pts.reshape(1, 3)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Body points must be [3] or [N, 3], got {pts.shape}")

        world = self.position_m[None, :] + pts @ self.rotation_matrix().T

        return world[0] if single else world


def compose_base_relative(base_pose: Pose, delta_position_local_m, delta_quaternion_wxyz) -> Pose:
    """Compose a body-local delta onto a base pose.

        p_world(t) = p_base + R(q_base) * delta_p_local(t)
        q_world(t) = q_base (x) delta_q(t)

    This is the single place the ``base_pose_relative`` reference frame is
    realised; both sampled_pose and handheld_jitter route through it, so the
    two types cannot drift apart.

    Args:
        base_pose(Pose): the configured static radar pose.
        delta_position_local_m: array-like [3], body-local translation delta.
        delta_quaternion_wxyz: array-like [4], body-local rotation delta.

    Returns:
        Pose: the world pose at that slow-time sample.
    """
    delta_p = np.asarray(delta_position_local_m, dtype=float).reshape(-1)

    if delta_p.shape[0] != 3:
        raise ValueError(f"delta position must have 3 components, got {delta_p.shape[0]}")

    position_m = base_pose.position_m + base_pose.rotation_matrix() @ delta_p
    quaternion_wxyz = quat_multiply(base_pose.quaternion_wxyz, quat_normalize(delta_quaternion_wxyz))

    return Pose.from_parts(position_m, quaternion_wxyz)


# ---------------------------------------------------------------------------
# Slow-time grid validation
# ---------------------------------------------------------------------------


def validate_slow_time_grid(slow_time_s) -> Tuple[np.ndarray, float]:
    """Validate the simulation slow-time grid and return (grid, interval).

    Required by every non-static motion type: at least two samples, finite,
    strictly increasing, and uniformly spaced. Uniformly is what makes the
    Nyquist gate and the RMS renormalisation well defined.

    Args:
        slow_time_s: array-like [F], slow-time sample instants in seconds.

    Returns:
        (np.ndarray shape [F], float): the grid and its snapshot interval.

    Raises:
        ValueError: on any of the conditions above.
    """
    grid = np.asarray(slow_time_s, dtype=float).reshape(-1)

    if grid.shape[0] < 2:
        raise ValueError(
            f"Radar motion requires a slow-time grid with at least 2 samples, got {grid.shape[0]}. "
            f"Increase [cir] num_frames."
        )
    if not np.all(np.isfinite(grid)):
        raise ValueError("Slow-time grid has non-finite entries")

    intervals = np.diff(grid)

    if np.any(intervals <= 0.0):
        raise ValueError("Slow-time grid must be strictly increasing")

    snapshot_interval_s = float(intervals[0])

    if not np.allclose(intervals, snapshot_interval_s, rtol=GRID_UNIFORMITY_RTOL, atol=0.0):
        raise ValueError(
            "Radar motion requires a uniformly spaced slow-time grid; "
            f"intervals span [{intervals.min()!r}, {intervals.max()!r}] s"
        )

    return grid, snapshot_interval_s


# ---------------------------------------------------------------------------
# Motion models
# ---------------------------------------------------------------------------


class RadarMotion:
    """Base class. One method: evaluate() -> exactly one finite Pose.

    Subclasses must be side-effect free and deterministic: calling evaluate()
    twice with the same arguments returns the same pose, and evaluation order
    does not matter. Any randomness is drawn once at construction time from a
    dedicated stream (see _ego_rng)
    """

    #: Human-readable ``type`` value from the configuration.
    motion_type: str = "static"

    #: True only for the legacy fast path (see CIRSimulator).
    is_static: bool = False

    def evaluate(self, slow_time_s: float, base_pose: Pose) -> Pose:
        """Return the radar pose at one slow-time instant.

        Args:
            slow_time_s (float): slow-time instant in seconds.
            base_pose (Pose): the configured static radar pose.

        Returns:
            Pose: finite, immutable pose for that instant.
        """
        raise NotImplementedError

    def metadata(self) -> Dict[str, object]:
        """Reproducibility metadata for the run artifacts."""
        return {"motion_type": self.motion_type}


class StaticMotion(RadarMotion):
    """No ego motion: the base pose at every slow0time sample.

    The is the default (``[radar_motion]`` omitted) and is also what an
    explicit ``type = static`` selects. CIRSimulator short-circuits on
    ``is_static`` and reuses its pre-computed antenna array object, so the
    legacy path generators run in the legacy order and the resulting CIR is
    bit-identical to the pre-ego-motion engine.
    """

    motion_type = "static"
    is_static = True

    def evaluate(self, slow_time_s: float, base_pose: Pose) -> Pose:
        return base_pose


class SampledPoseMotion(RadarMotion):
    """Replay a recorded 6-DoF trajectory of body-local deltas from CSV.

    The CSV holds ``delta_p_local(t)`` / ``delta_q(t)`` relative to the
    configured base pose -- NOT world poses. Position is linearly
    interpolated; attitude uses shortest-arc SLERP. Extrapolation is refused:
    the whole simulation slow-time grid must lie inside the closed trace
    domain, and that coverage is checked at construction time so the failure
    happens before any path generation.
    """

    motion_type = "sampled_pose"

    def __init__(
        self,
        time_s: np.ndarray,
        delta_position_local_m: np.ndarray,
        delta_quaternion_wxyz: np.ndarray,
        source_path: Path,
    ):
        self._time_s = np.asarray(time_s, dtype=float)
        self._delta_position_local_m = np.asarray(delta_position_local_m, dtype=float)
        self._delta_quaternion_wxyz = np.asarray(delta_quaternion_wxyz, dtype=float)
        self._source_path = Path(source_path)

    # -- domain ------------------------------------------------------------

    @property
    def domain_s(self) -> Tuple[float, float]:
        """Closed trace domain [t_first, t_lat] in seconds."""
        return float(self._time_s[0]), float(self._time_s[-1])

    def assert_covers_grid(self, slow_time_s) -> None:
        """Fail if any slow-time sample falls outside the closed trace domain.

        Called at construction time (before path generation) so an
        out-of-domain scenario cannot produce a partially-valid CIR.
        """
        grid = np.asarray(slow_time_s, dtype=float).reshape(-1)
        t_first, t_last = self.domain_s

        if grid.size == 0:
            raise ValueError("Cannot validate trace coverage against an empty slow-time grid")

        if float(grid[0]) < t_first or float(grid[-1]) > t_last:
            raise ValueError(
                f"sampled_pose trace {self._source_path.name} does not cover the simulation "
                f"slow-time grid: trace domain is [{t_first!r}, {t_last!r}] s but the grid spans "
                f"[{float(grid[0])!r}, {float(grid[-1])!r}] s. Extrapolation is not permitted "
                f"(extrapolation = error); extend the trace or shorten the run."
            )
    
    # -- evaluation --------------------------------------------------------

    def evaluate(self, slow_time_s: float, base_pose: Pose) -> Pose:
        t = float(slow_time_s)
        t_first, t_last = self.domain_s

        if not np.isfinite(t):
            raise ValueError(f"sampled_pose evaluated at non-finite slow_time_s={t!r}")
        if t < t_first or t > t_last:
            raise ValueError(
                f"sampled_pose evaluated at slow_time_s={t!r} outside the closed trace domain "
                f"[{t_first!r}, {t_last!r}] s; extrapolation is not permitted"
            )

        idx = int(np.searchsorted(self._time_s, t, side="left"))

        # Exact node hit -> return the stored normalised node, no interpolation.
        if idx < self._time_s.shape[0] and self._time_s[idx] == t:
            return compose_base_relative(
                base_pose,
                self._delta_position_local_m[idx],
                self._delta_quaternion_wxyz[idx],
            )

        # Strictly inside segment (idx - 1, idx).
        i0 = idx - 1
        i1 = idx
        t0 = float(self._time_s[i0])
        t1 = float(self._time_s[i1])
        u = (t - t0) / (t1 - t0)

        delta_p = self._delta_position_local_m[i0] + u * (
            self._delta_position_local_m[i1] - self._delta_position_local_m[i0]
        )
        delta_q = quat_slerp_shortest(
            self._delta_quaternion_wxyz[i0], self._delta_quaternion_wxyz[i1], u
        )

        return compose_base_relative(base_pose, delta_p, delta_q)

    def metadata(self) -> Dict[str, object]:
        t_first, t_last = self.domain_s
        return {
            "motion_type": self.motion_type,
            "source_file": str(self._source_path),
            "reference": "base_pose_relative",
            "extrapolation": "error",
            "num_trace_nodes": int(self._time_s.shape[0]),
            "trace_domain_s": [t_first, t_last],
            "seed_derivation": "none (sampled_pose consumes no random numbers)",
        }


class HandheldJitterMotion(RadarMotion):
    """Deterministic band-limited hand tremor.

    Each of the 6 axes (3 body-local translation, 3 body-local rotation-vector
    components) is the sum of K sinusoids at band mid-point frequencies

        f_k = f_min + (k + 1/2) * (f_max - f_min) / K,   k = 0 .. K-1

        raw_a(t) = sum_k [ sin(2*pi*f_k*(t - t_0) + phi_{a,k}) - sin(phi_{a,k}) ]

    The per-component ``- sin(phi)`` anchoring makes ``raw_a(t_0) == 0``
    exactly, so the configured base pose is reproduced exactly at the first
    slow-time sample.

    Each axis is then renormalised on the ACTUAL slow-time grid:

        achieved = sqrt(mean_n(raw_a(t_n)**2))
        output_a = raw_a / achieved * configured_rms_a      (configured > 0 only)

    Consequences, all deliberate:
        * The reported achieved RMS equals the configured RMS by construction.
          It is a confirmation that renormalisation succeeded, NOT a measurment.
        * That RMS is taken about the INITIAL pose, not about the series mean:
          ``sqrt(mean_n((delta_a(t_n) - delta_a(t_0))**2))`` with
          ``delta_a(t_0) == 0``.
        * Anchoring introduces a DC component and the finite window introduces
          spectral leakage. The unnormalised harmonic components themselves lie
          inside the requested band.

    NOTE: the reference implementation's prose carries a ``sqrt(2/K)``
    normalisation factor that does not appear in its code. Renormalsation
    absorbs any such constant, so the output is identical either way; the code
    below follows the actual behaviour, not the prose. Do not reintroduce the
    literal.

    Rotation is accumulated as a body-local rotation-vector exponential map
    ``q_delta = exp(r_local)`` composed as ``q_base (x) q_delta``. Euler-angle
    accumulation is forbidden.

    Because renormalisation is grid-dependent, the trajectory is precomputed on
    the grid supplied at construction time and evaluation at any other
    slow-time instant is refused.
    """

    motion_type = "handheld_jitter"

    def __init__(
        self,
        slow_time_s: np.ndarray,
        snapshot_interval_s: float,
        component_frequencies_hz: np.ndarray,
        delta_axes: np.ndarray,
        achieved_translation_rms_local_m: np.ndarray,
        achieved_rotation_rms_local_rad: np.ndarray,
        requested_band_hz: Tuple[float, float],
        num_spectral_components: int,
        seed_derivation: str,
    ):
        self._slow_time_s = np.asarray(slow_time_s, dtype=float)
        self._snapshot_interval_s = float(snapshot_interval_s)
        self._component_frequencies_hz = np.asarray(component_frequencies_hz, dtype=float)
        # delta_axes shape [6, F]: rows 0..2 translation_m, rows 3..5 rotation_rad.
        self._delta_axes = np.asarray(delta_axes, dtype=float)
        self._achieved_translation_rms_local_m = np.asarray(
            achieved_translation_rms_local_m, dtype=float
        )
        self._achieved_rotation_rms_local_rad = np.asarray(
            achieved_rotation_rms_local_rad, dtype=float
        )
        self._requested_band_hz = (float(requested_band_hz[0]), float(requested_band_hz[1]))
        self._num_spectral_components = int(num_spectral_components)
        self._seed_derivation = str(seed_derivation)

    def _grid_index(self, slow_time_s: float) -> int:
        """Locate an exact slow-time sample, refusing anything off-grid."""
        t = float(slow_time_s)
        idx = int(np.searchsorted(self._slow_time_s, t, side="left"))

        if idx < self._slow_time_s.shape[0] and self._slow_time_s[idx] == t:
            return idx

        raise ValueError(
            f"handheld_jitter was precomputed on a fix slow-time grid of "
            f"{self._slow_time_s.shape[0]} samples at {self._snapshot_interval_s!r} s spacing; "
            f"it cannot be evaluated at slow_time_s={t!r}, which is not one of those samples. "
            f"The per-axis RMS renormalisation is grid-dependent, so re-evaluating off-grid "
            f"would silently change the achieved RMS."
        )

    def evaluate(self, slow_time_s: float, base_pose: Pose) -> Pose:
        idx = self._grid_index(slow_time_s)

        delta_p_local_m = self._delta_axes[0:3, idx]
        r_local_rad = self._delta_axes[3:6, idx]

        return compose_base_relative(
            base_pose, delta_p_local_m, quat_from_rotation_vector(r_local_rad)
        )

    def metadata(self) -> Dict[str, object]:
        f_min, f_max = self._requested_band_hz
        return {
            "motion_type": self.motion_type,
            "seed_derivation": self._seed_derivation,
            "requested_min_frequency_hz": f_min,
            "requested_max_frequency_hz": f_max,
            "num_spectral_components": self._num_spectral_components,
            "component_frequencies_hz": [float(v) for v in self._component_frequencies_hz],
            "snapshot_interval_s": self._snapshot_interval_s,
            "num_slow_time_samples": int(self._slow_time_s.shape[0]),
            "achieved_translation_rms_local_m": [
                float(v) for v in self._achieved_translation_rms_local_m
            ],
            "achieved_rotation_rms_local_rad": [
                float(v) for v in self._achieved_rotation_rms_local_rad
            ],
            "achieved_rms_definition": (
                "sqrt(mean_n((delta_a(t_n) - deltal_a(t_0))**2)) with delta_a(t_0) == 0; "
                "equals the configured RMS by construction (renormalisation confirmation, "
                " not a measurement)"
            ),
            "anchoring_note": (
                "per-component -sin(phi) anchoring yields delta(t_0)=0 and introduces a DC "
                "component; the finite window introduces spectral leakage. The unnormalised "
                "harmonic components lie inside the requested band."
            ),
        }


class SingleToneMotion(RadarMotion):
    """A single deterministic sinusoidal tone in translation and/or rotation.

    This is the analysis / derived-preset motion:

        s(t)                = sin(2*pi*f*(t - t_0) + phase_rad)
        delta_p_local(t)    = translation_amplitude_m * s(t) * translation_direction
        r_local(t)          = rotation_amplitude_rad * s(t) * rotation_axis
        delta_q(t)          = exp(r_local(t))

    then composed onto the base pose by compose_base_relative().

    Unlike HandheldJitterMotion this takes a **peak amplitude**, not an RMS, and
    draws no random numbers. That matters: for the anchored jitter waveform the
    achieved RMS is sqrt(0.5 + sin(phi)**2) of the raw tone, i.e. it depends on
    the randomly drawn phase, so an RMS-configured tone cannot be given an exact
    closed-form peak deviation. Every scenario that needs an analytically
    predictable modulation index therefore uses this class instead.

    ``translation_amplitude_m`` and ``rotation_amplitude_rad`` may be NEGATIVE,
    meaning anti-phase with respect to the reference tone. That is how the
    mirror/cancel presets express their sign (see build_radar_motion_from_profile).

    Closed form in t, so unlike HandheldJitterMotion it is not bound to the grid
    it was constructed with and my be evaluated at any instant. The slow-time
    grid is still validated at construction, to gate the frequency against
    Nyquist.

    With ``phase_rad == 0`` the anchoring is exact: s(t_0) == 0, so the
    configured base pose is reproduced bit-exactly at the first slow-time sample.
    """

    motion_type = "single_tone"

    def __init__(
        self,
        frequency_hz: float,
        phase_rad: float,
        t0_s: float,
        translation_amplitude_m: float = 0.0,
        translation_direction: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_amplitude_rad: float = 0.0,
        rotation_axis: Sequence[float] = (0.0, 0.0, 1.0),
        provenance: str = "",
    ):
        self._frequency_hz = float(frequency_hz)
        self._phase_rad = float(phase_rad)
        self._t0_s = float(t0_s)
        self._translation_amplitude_m = float(translation_amplitude_m)
        self._rotation_amplitude_rad = float(rotation_amplitude_rad)
        self._provenance = str(provenance)

        self._translation_direction = self._as_direction(
            translation_direction, "translation_direction", self._translation_amplitude_m
        )
        self._rotation_axis = self._as_direction(
            rotation_axis, "rotation_axis", self._rotation_amplitude_rad
        )

        for name, value in (
            ("frequency_hz", self._frequency_hz),
            ("phase_rad", self._phase_rad),
            ("translation_amplitude_m", self._translation_amplitude_m),
            ("rotation_amplitude_rad", self._rotation_amplitude_rad),
        ):
            if not np.isfinite(value):
                raise ValueError(f"SingleToneMotion {name} must be finite, got {value!r}")

        if self._frequency_hz <= 0.0:
            raise ValueError(
                f"SingleToneMotion frequency_hz must be > 0, got {self._frequency_hz!r}"
            )

    @staticmethod
    def _as_direction(vec, name: str, amplitude: float) -> np.ndarray:
        """Normalise a direction, tolerating a zero vector on a disabled axis."""
        arr = np.asarray(vec, dtype=float).reshape(-1)

        if arr.shape[0] != 3:
            raise ValueError(f"SingleToneMotion {name} must be a 3-vector, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"SingleToneMotion {name} has non-finite components: {arr!r}")

        norm = float(np.linalg.norm(arr))

        if norm <= 0.0:
            if amplitude != 0.0:
                raise ValueError(
                    f"SingleToneMotion {name} is the zero vector but its amplitude is "
                    f"{amplitude!r}; a non-zero amplitude need a direction."
                )
            return np.zeros(3, dtype=float)
        
        return arr / norm

    def evaluate(self, slow_time_s: float, base_pose: Pose) -> Pose:
        t = float(slow_time_s)

        if not np.isfinite(t):
            raise ValueError(f"SingleToneMotion evaluated at non-finite slow_time_s={t!r}")

        s = float(np.sin(2.0 * np.pi * self._frequency_hz * (t - self._t0_s) + self._phase_rad))

        delta_p_local_m = (self._translation_amplitude_m * s) * self._translation_direction
        r_local_rad = (self._rotation_amplitude_rad * s) * self._rotation_axis

        return compose_base_relative(
            base_pose, delta_p_local_m, quat_from_rotation_vector(r_local_rad)
        )

    def metadata(self) -> Dict[str, object]:
        return {
            "motion_type": self.motion_type,
            "seed_derivation": "none (single_tone is deterministic and consumes no random numbers)",
            "frequency_hz": self._frequency_hz,
            "phase_rad": self._phase_rad,
            "anchor_time_s": self._t0_s,
            "translation_amplitude_m": self._translation_amplitude_m,
            "translation_dirction": [float(v) for v in self._translation_direction],
            "rotation_amplitude_rad": self._rotation_amplitude_rad,
            "rotation_axis": [float(v) for v in self._rotation_axis],
            "amplitude_convection": "peak (NOT rms, unlike handheld_jitter)",
            "provenance": self._provenance,
        }

    
# --------------------------------------------------------------------------
# Random number ownership
# --------------------------------------------------------------------------

#: Documented derivation string stored in the run metadata.
EGO_SPEED_DERIVATION = (
    "numpy.random.Generator(PCG64(SeedSequence([random_seed, 0x45474F31]))); "
    "0x45474F31 is the fixed 'EGO1' tag. Consumes exactly one (K, 6) uniform "
    "phase draw in [0, 2*pi) and nothing else."
)


def _ego_rng(random_seed: int) -> np.random.Generator:
    """Dedicated ego-motion random stream, isolated from every other stream.

    The engine's impairment/thermal-noise/AWGN/quantization draws all come from
    ``CIRSimulator.self.rng``, a separate Generator object. Because this
    function returns a *different* Generator seeded from a distinct
    SeedSequence, enabling ego motion cannot shift the sample sequence any
    other stream sees -- the ego draws are not interleaved into it at all.

    Args:
        random_seed (int): the scenario'` ``random_seed``

    Returns:
        np.random.Generator: the ego-motion-only stream.
    """
    seed_sequence = np.random.SeedSequence([int(random_seed), EGO_SEED_TAG])
    return np.random.Generator(np.random.PCG64(seed_sequence))


# --------------------------------------------------------------------------
# sampled_pose: CSV loading
# --------------------------------------------------------------------------


def resolve_trace_path(file_value: str, scenario_dir: Optional[Path]) -> Path:
    """Resolve a sampled_pose ``file`` value relative to the scenario file.

    Absolute paths are refused so a scenario stays portable and reproducible
    on another machine.

    Args:
        file_value (str): the raw ``file`` value from the configuration.
        scenario_dir (Optional[Path]): directory of the scenario INI that
            declared the motion.
    
    Returns:
        Path: the resolved, existing trace path.
    """
    raw = (file_value or "").strip()

    if not raw:
        raise ValueError("[radar_motion] type = sampled_pose requires a non-empty 'file' key")

    candidate = Path(raw)

    if candidate.is_absolute():
        raise ValueError(
            f"[radar_motion] file must be relative to the scenario file, got absolute path "
            f"{raw!r}. Absolute paths are refused so scenarios stay portable."
        )

    if scenario_dir is None:
        raise ValueError(
            "[radar_motion] type = sampled_pose needs the scenario file's directory to resolve "
            f"a relative trace path ({raw!r}), but none was supplied."
        )
    
    resolved = (Path(scenario_dir) / candidate).resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"[radar_motion] e trace not found: {resolved} "
            f"(from file = {raw!r}, resolved against {scenario_dir})"
        )

    return resolved


def load_sampled_pose_csv(path: Path) -> SampledPoseMotion:
    """Strictly parse a sampled_pose CSV of body-local pose deltas.

    Enforced, all as ard errors:
        * strict UTF-8 (no BOM, no fallback encoding)
        * header exactly SAMPLED_POSE_CSV_COLUMNS, in that order
        * no comment lines, no blank lines, no duplicate columns, no extra
          columns, no short rows
        * every value finite and numeric
        * time_s strictly increasing
        * quaternions unit norm within QUAT_NORM_TOL, then normalised
        * at least 2 rows

    Quaternion signs are then made continuous (see
    canonicalize_quaternion_sign_sequence) so interpolation never takes the
    long arc across a recorded sign flip.

    Args:
        path (Path): resolved trace path.

    Returns:
        SampledPoseMotion
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        raw_lines = fh.read().split("\n")

    # Trailing newline at EOF is normal; anything else empty is a blank line.
    if raw_lines and raw_lines[-1] == "":
        raw_lines = raw_lines[:-1]

    if not raw_lines:
        raise ValueError(f"sampled_pose trace: {path.name} is empty")
    
    for lineno, line in enumerate(raw_lines, start=1):
        if line.strip() == "":
            raise ValueError(f"sampled_pose trace {path.name} line {lineno}: blank lines are not permitted")
        if line.lstrip().startswith("#"):
            raise ValueError(
                f"sampled_pose trace {path.name} line {lineno}: comment lines are not permitted"
            )

    rows = list(csv.reader(raw_lines))
    header = [cell.strip() for cell in rows[0]]

    if len(set(header)) != len(header):
        raise ValueError(f"sampled_pose trace {path.name}: duplicate column names in header {header!r}")

    if tuple(header) != SAMPLED_POSE_CSV_COLUMNS:
        raise ValueError(
            f"sampled_pose trace {path.name}: header must be exactly "
            f"{','.join(SAMPLED_POSE_CSV_COLUMNS)} (got {','.join(header)})"
        )
    
    data_rows = rows[1:]

    if len(data_rows) < 2:
        raise ValueError(
            f"sampled_pose trace {path.name}: needs at least 2 data rows, got {len(data_rows)}"
        )
    
    values = np.empty((len(data_rows), len(SAMPLED_POSE_CSV_COLUMNS)), dtype=float)

    for i, row in enumerate(data_rows):
        lineno = i + 2 # 1-based, header is line 1

        if len(row) != len(SAMPLED_POSE_CSV_COLUMNS):
            raise ValueError(
                f"sampled_pose trace {path.name} line {lineno}: expected "
                f"{len(SAMPLED_POSE_CSV_COLUMNS)} columns, got {len(row)}"
            )

        for j, cell in enumerate(row):
            text = cell.strip()
            try:
                parsed = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"sampled_pose trace {path.name} line {lineno}, column "
                    f"{SAMPLED_POSE_CSV_COLUMNS[j]}: {text!r} is not a number"
                ) from exc

            if not np.isfinite(parsed):
                raise ValueError(
                    f"sampled_pose trace {path.name} line {lineno}, column "
                    f"{SAMPLED_POSE_CSV_COLUMNS[j]}: non-finite value {text!r}"
                )

            values[i, j] = parsed

    time_s = values[:, 0]

    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"sampled_pose trace {path.name}: time_s must be strictly increasing")

    delta_position_local_m = values[:, 1:4]
    quats = values[:, 4:8]

    norms = np.linalg.norm(quats, axis=1)
    bad = np.nonzero(np.abs(norms - 1.0) > QUAT_NORM_TOL)[0]

    if bad.size > 0:
        first = int(bad[0])
        raise ValueError(
            f"sampled_pose trace {path.name} line {first + 2}: quaternion norm "
            f"{float(norms[first])!r} deviates from 1 by more than {QUAT_NORM_TOL!r}"
        )

    quats = quats / norms[:, None]
    quats = canonicalize_quaternion_sign_sequence(quats)

    return SampledPoseMotion(
        time_s=time_s,
        delta_position_local_m=delta_position_local_m,
        delta_quaternion_wxyz=quats,
        source_path=path,
    )


# --------------------------------------------------------------------------
# handheld_jitter: synthesis
# --------------------------------------------------------------------------


def build_handheld_jitter(
    translation_rms_local_m: Sequence[float],
    rotation_rms_local_rad: Sequence[float],
    min_frequency_hz: float,
    max_frequency_hz: float,
    num_spectral_components: int,
    slow_time_s,
    random_seed: int,
) -> HandheldJitterMotion:
    """Synthesis the deterministic band-limited tremor trajectory.
    
    See HandheldJitterMotion for the equations and for why renormalisation
    absorbs the reference prose's sqrt(2/K) factor.

    Axis order for the (K, 6) phase draw and for the returned delta rows is
    ``[translation_x, translation_y, translation_z, rotation_x, rotation_y,
    rotation_z]'', all in the initial radar body frame.

    Args:
        translation_rms_local_m: non-negative 3-vector, metres.
        rotation_rms_local_rad: non-negative 3-vector, radians.
        min_frequency_hz (float): band lower edge, > 0.
        max_frequency_hz (float): band upper edge, >= min.
        num_spectral_components (int): K > 0.
        slow_time_s: array-like [F], the simulation slow-time grid.
        random_seed (int): scenario random seed (ego stream is derived from it).

    Returns:
        HandheldJitterMotion
    """
    grid, snapshot_interval_s = validate_slow_time_grid(slow_time_s)

    f_min = float(min_frequency_hz)
    f_max = float(max_frequency_hz)
    k_count = int(num_spectral_components)

    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        raise ValueError(
            f"[radar_motion] handheld_jitter frequencies must be finite, got "
            f"min_frequency_hz={f_min!r}, max_frequency_hz={f_max!r}"
        )
    if not (0.0 < f_min <= f_max):
        raise ValueError(
            f"[radar_motion] handheld_jitter requires 0 < min_frequency_hz <= max_frequency_hz, "
            f"got min_frequency_hz={f_min!r}, max_frequency_hz={f_max!r}"
        )
    if k_count <= 0:
        raise ValueError(
            f"[radar_motion] handheld_jitter requires num_spectral_components > 0, got {k_count!r}"
        )

    nyquist_hz = 1.0 / (2.0 * snapshot_interval_s)

    # Strictly below Nyquist: equality is refused, because a component exactly
    # at Nyquist is not resolvable on this grid.
    if not (f_max < nyquist_hz):
        raise ValueError(
            f"[radar_motion] handheld_jitter max_frequency_hz={f_max!r} Hz must be strictly below "
            f"the slow-time Nyquist frequency {nyquist_hz!r} Hz "
            f"(snapshot interval) {snapshot_interval_s!r} s, i.e. [radar] period). "
            f"Lower max_frequency_hz or shorten the frame period."
        )

    translation_rms = np.asarray(translation_rms_local_m, dtype=float).reshape(-1)
    rotation_rms = np.asarray(rotation_rms_local_rad, dtype=float).reshape(-1)

    for name, vec in (
        ("translation_rms_local_m", translation_rms),
        ("rotation_rms_local_m", rotation_rms),
    ):
        if vec.shape[0] != 3:
            raise ValueError(
                f"[radar_motion] {name} must be 3-vector, got {vec.shape[0]} components"
            )
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"[radar_motion] {name} has non-finite components: {vec!r}")
        if np.any(vec < 0.0):
            raise ValueError(f"[radar_motion] {name} must be non-negative, got {vec!r}")

    configured_rms = np.concatenate([translation_rms, rotation_rms])

    # Band mid-point frequencies.
    k_indices = np.arange(k_count, dtype=float)
    component_frequencies_hz = f_min + (k_indices + 0.5) * (f_max - f_min) / float(k_count)

    # The one and only random draw: (K, 6) uniform phases.
    rng = _ego_rng(random_seed)
    phases_rad = rng.uniform(0.0, 2.0 * np.pi, size=(k_count, 6))

    t_rel = grid - grid[0] # t - t_0; anchroing reference is the first sample.

    # raw[axis, n] = sum_k [ sin(2*pi*f_k*t_rel[n] + phi[k, axis]) - sin(phi[k, axis]) ]
    angles = 2.0 * np.pi * component_frequencies_hz[:, None] * t_rel[None, :] # [K, F]
    raw = np.einsum(
        "kfa->af",
        np.sin(angles[:, :, None] + phases_rad[:, None, :]) - np.sin(phases_rad)[:, None, :],
    ) # [6, F]

    delta_axes = np.zeros_like(raw)
    achieved = np.zeros(6, dtype=float)

    for axis in range(6):
        target_rms = float(configured_rms[axis])

        if target_rms == 0.0:
            # Axis disabled: exactly zero, never renormalised.
            continue
        
        raw_rms = float(np.sqrt(np.mean(raw[axis] ** 2)))

        if not np.isfinite(raw_rms) or raw_rms < MIN_ACHIEVABLE_RAW_RMS:
            raise ValueError(
                f"[radar_motion] handheld_jitter axis {axis} has a positive configure RMS "
                f"({target_rms!r}) but its unnormalised synthesis RMS on this slow-time grid is "
                f"{raw_rms!r}, so the requested RMS cannot be realised. This happens when the "
                f"grid is too short relative to the requested band "
                f"({f_min!r}..{f_max!r} Hz over {float(t_rel[-1])!r} s); lengthen the run or "
                f"raise min_frequency_hz."
            )

        delta_axes[axis] = raw[axis] / raw_rms * target_rms
        # Renormalisation is exact, so this equals target_rms; recomputing it
        # from the output (rather than copying the target) keeps it an actual
        # confirmation that the normalisation held.
        achieved[axis] = float(np.sqrt(np.mean(delta_axes[axis] ** 2)))

    return HandheldJitterMotion(
        slow_time_s=grid,
        snapshot_interval_s=snapshot_interval_s,
        component_frequencies_hz=component_frequencies_hz,
        delta_axes=delta_axes,
        achieved_translation_rms_local_m=achieved[0:3],
        achieved_rotation_rms_local_rad=achieved[3:6],
        requested_band_hz=(f_min, f_max),
        num_spectral_components=k_count,
        seed_derivation=EGO_SPEED_DERIVATION,
    )


# --------------------------------------------------------------------------
# Strict configuration surface
# --------------------------------------------------------------------------

#: The configuration section name. Mirros the reference implementation's
#: ``radar.motion`` node in this project's INI world.
INI_SECTION = "radar_motion"

#: Keys accepted for each motion type. Every new field is optional and the
#: default (section omitted, or type = static) is inert, but within the section
#: an unknown or type-inapplicable key is hard error -- the configuration
#: surface stays strict so a typo can never silently disable the motion.
_ALLOWED_KEYS: Dict[str, Tuple[str, ...]] = {
    "static": ("type",),
    "sampled_pose": ("type", "file", "reference", "extrapolation"),
    "handheld_jitter": (
        "type",
        "translation_rms_local_m",
        "rotation_rms_local_rad",
        "min_frequency_hz",
        "max_frequency_hz",
        "num_spectral_components",
    ),
}

SUPPORTED_MOTION_TYPES: Tuple[str, ...] = tuple(sorted(_ALLOWED_KEYS))

#: sampled_pose accepts exactly one value for each of these.
_REQUIRED_LITERALS: Dict[str, str] = {
    "reference": "base_pose_relative",
    "extrapolation": "error",
}


def validate_ini_keys(motion_type: str, present_keys: Sequence[str]) -> None:
    """Reject unknown / type-inapplicable keys in the motion section.

    Args:
        motion_type (str): resolved ``type`` value.
        present_keys (Sequence[str]): keys actually present in the section
            (configparser lowercases them).

    Raises:
        ValueError: on an unknown type, or any key not allowed for that type.
    """
    if motion_type not in _ALLOWED_KEYS:
        raise ValueError(
            f"[{INI_SECTION}] unknown type = {motion_type!r}"
            f"supported types are {', '.join(SUPPORTED_MOTION_TYPES)}"
        )

    allowed = _ALLOWED_KEYS[motion_type]
    unknown = [key for key in present_keys if key not in allowed]

    if unknown:
        other_types = sorted(
            other
            for other in _ALLOWED_KEYS
            if other != motion_type and any(key in _ALLOWED_KEYS[other] for key in unknown)
        )
        hint = (
            f" Those keys belong to type = {', '.join(other_types)}."
            if other_types
            else ""
        )
        raise ValueError(
            f"[{INI_SECTION}] unknown key(s) for type = {motion_type}: "
            f"{', '.join(sorted(unknown))}. Allowed: {', '.join(allowed)}.{hint}"
        )


def validate_required_literal(key: str, value: str) -> str:
    """Enforce the single permitted value for ``reference`` / ``extrapolation``."""
    expected = _REQUIRED_LITERALS[key]
    got = (value or "").strip()

    if got != expected:
        raise ValueError(
            f"[{INI_SECTION}] {key} must be {expected!r} (the only supported value), got {got!r}"
        )
    
    return got


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_radar_motion(
    motion_cfg,
    slow_time_s,
    random_seed: int,
    scenario_dir: Optional[Path] = None,
) -> RadarMotion:
    """Build the radar motion model from a validated RadarMotionConfig.

    All heavy work (CSV parsing, tremor synthesis, grid coverage checks) happens
    HERE, at simulator construction time, so a bad configuration fails before
    any path generation rather than part-way through a run.

    Args:
        motion_cfg: a dataconfig.RadarMotionConfig instance.
        slow_time_s: array-like [F], the simulation slow-time grid in seconds.
        random_seed (int): scenario random seed; the ego stream is derived from
            it (see EGO_SPEED_DERIVATION) and consumes nothing from any other
            stream.
        scenario_dir (Optional[Path]): directory of the scenario INI, used to
            resolved a relative sampled_pose trace path.

    Returns:
        RadarMotion: StaticMotion, SampledPoseMotion or HandheldJitterMotion.
    """
    motion_type = (getattr(motion_cfg, "type", "static") or "static").strip().lower()

    if motion_type not in _ALLOWED_KEYS:
        raise ValueError(
            f"[{INI_SECTION}] unknown type = {motion_type!r}; "
            f"supported types are {', '.join(SUPPORTED_MOTION_TYPES)}"
        )

    if motion_type == "static":
        return StaticMotion()

    if motion_type == "sampled_pose":
        validate_required_literal("reference", getattr(motion_cfg, "reference", ""))
        validate_required_literal("extrapolation", getattr(motion_cfg, "extrapolation", ""))

        # A non-static motion still requires a well-formed grid: the coverage
        # check below is only meaningful on a validated grid.
        grid, _ = validate_slow_time_grid(slow_time_s)

        path = resolve_trace_path(getattr(motion_cfg, "file", ""), scenario_dir)
        motion = load_sampled_pose_csv(path)
        motion.assert_covers_grid(grid)

        return motion

    return build_handheld_jitter(
        translation_rms_local_m=getattr(motion_cfg, "translation_rms_local_m", (0.0, 0.0, 0.0)),
        rotation_rms_local_rad=getattr(motion_cfg, "rotation_rms_local_rad", (0.0, 0.0, 0.0)),
        min_frequency_hz=getattr(motion_cfg, "min_frequency_hz", 0.0),
        max_frequency_hz=getattr(motion_cfg, "max_frequency_hz", 0.0),
        num_spectral_components=getattr(motion_cfg, "num_spectral_components", 0),
        slow_time_s=slow_time_s,
        random_seed=random_seed,
    )


# --------------------------------------------------------------------------
# Named presets
#
# Presets parameters are hardcoded here, mirroring the established
# CIRSimulator._apply_hardware_profile() / _apply_room_profile() /
# _apply_target_profile() convention: a scenario selects physics by NAME, and the
# numbers live in Python rather than in a config surface.
#
# Two families:
#
#   * REALISM presets (handheld_*) -- plausible hand tremor, built on
#     handheld_jitter. Their amplitudes are RMS values.
#   * ANALYSIS presets (probe_*, mirror/cancel_target_micromotion) -- single
#     deterministic tones with PEAK amplitudes and a closed-form modulation
#     index, so a validation scenario can assert an exact predicted value. These
#     use SingleToneMotion and draw no random numbers at all.
#
# The `mirror_`/`cancel_` presets are DERIVED: they read the resolved target's
# own micro-motion parameters at build time, so they carry no duplicated
# literals and cannot drift out of sync with _apply_target_profile(). This works
# beacuse CIRSimulator._setup_radar_motion() runs last in __init__, after
# _apply_target_profile() and _sync_target_frame_period().
# --------------------------------------------------------------------------

#: Hand tremor RMS values shared by the bandheld_* presets. These are the same
#: numbers as configs/cirgen_configs/paltform_2d_legacy_handheld.ini's
#: [radar_motion] block, so the preset and that INI describe the same motion.
_HANDHELD_TRANSLATION_RMS_M = (0.0012, 0.0012, 0.0008)
_HANDHELD_ROTATION_RMS_RAD = (0.0040, 0.0040, 0.0020)
_HANDHELD_BAND_HZ = (4.0, 12.0)
_HANDHELD_COMPONENTS = 24

#: Analysis-probe constants. 1 mm of radial displacement is ~19.18 deg of
#: round-trip carrier phase at the Ch9 carrier but only ~0.0067 fast-time bins,
#: so a probe is a phase-domain stimulus. 8 Hz sits inside the tremor band and,
#: at the 5 ms frame period used by the validation platform INI, gives 25
#: samples per cycle.
_PROBE_TRANSLATION_AMPLITUDE_M = 0.001
_PROBE_ROTATION_AMPLITUDE_RAD = 0.05    # 2.865 deg -- large enough that the
                                        # induced inter-antenna phase shift
                                        # (~18.4 deg) is unambiguous
_PROBE_FREQUENCY_HZ = 8.0

#: Every accepted value of [profiles]/[synthetic] radar_motion_profile.
SUPPORTED_MOTION_PROFILES: Tuple[str, ...] = (
    "static",
    "handheld_light",
    "handheld_strong",
    "handheld_rotation_only",
    "probe_translation_x",
    "probe_translation_y",
    "probe_translation_z",
    "probe_yaw",
    "mirror_target_micromotion",
    "cancel_target_micromotion",
)

#: Presets that read the resolved target list.
_DERIVED_PROFILES: Dict[str, float] = {
    # profile name -> sign applied to the target's displacement
    #  -1 : radar mirros the target, so the two produce the SAME range change
    #       (equivalence when the target is static; doubling when it moves)
    #  +1 : radar oppose the target, so the two range changes CANCEL
    "mirror_target_micromotion": -1.0,
    "cancel_target_micromotion": +1.0,
}

_PROBE_TRANSLATION_AXES: Dict[str, Tuple[float, float, float]] = {
    "probe_translation_x": (1.0, 0.0, 0.0),
    "probe_translation_y": (0.0, 1.0, 0.0),
    "probe_translation_z": (0.0, 0.0, 1.0),
}


def micro_motion_direction(target, radar_origin_m) -> Tuple[np.ndarray, str]:
    """Unit direction of a Target's micro-motion, in world coordinates.

    Replicates entities.Target._apply_target_micro_motion() exactly: the
    ``"x"``/``"y"``/``"z"`` axes are world unit axes, and anything else means
    ``"radial"``, which that method implements as ``base / norm(base)`` -- i.e.
    radial from the WORLD ORIGIN, not from the radar.

    That documented quirk (entities.py, the "assumes radar origin at [0, 0, 0]"
    note) is why this function needs the radar origin: a mirrored ego motion is
    only geometrically equivalent to the target's motion when both are measured
    along the same line, so a ``"radial"`` target combined with an off-origin
    radar is refused rather than silently producing a wrong equivalence.

    Args:
        target: a resolved entities.Target.
        radar_origin_m: array-like [3], the radar base position.

    Returns:
        (np.ndarray shape [3] unit direction, str axis label).
    """
    axis = str(getattr(target, "micro_motion_axis", "radial"))

    explicit = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    if axis in explicit:
        return np.array(explicit[axis], dtype=float), axis

    radar_origin = np.asarray(radar_origin_m, dtype=float).reshape(-1)

    if float(np.linalg.norm(radar_origin)) > 1e-12:
        raise ValueError(
            f"target {getattr(target, 'name', '?')!r} uses micro_motion_axis={axis!r}"
            f"(radial), which entities.Target._apply_target_micro_motion() implements as "
            f"radial from the WORLD ORIGIN, not from the radar. The radar base position is "
            f"{radar_origin.tolist()!r}, so 'radial' does not point along the radar "
            f"line-of-sight and a mirrored ego motion would not be equivalent to the "
            f"target's motion. Use micro_motion_axis in {{x, y, z}}, or place the radar at "
            f"the world origin (geometry_mode = 2d_legacy with radar_position_x_m = "
            f"radar_position_y_m = 0)."
        )

    base = np.asarray(target._get_center_position(0), dtype=float).reshape(-1)
    norm = float(np.linalg.norm(base))

    if norm <= 1e-12:
        raise ValueError(
            f"target {getattr(target, 'name', '?')!r} uses radial micro-motion but sits at "
            f"the origin, so its radial direction is undefined."
        )

    return base / norm, "radial"


def _derived_single_tone(profile: str, targets, radar_origin_m, t0_s: float) -> SingleToneMotion:
    """Build a mirror/cancel preset from the resolved target's own micro-motion."""
    sign = _DERIVED_PROFILES[profile]

    moving = [t for t in (targets or []) if bool(getattr(t, "enable_micro_motion", False))]

    if len(moving) != 1:
        raise ValueError(
            f"radar_motion_profile = {profile!r} derives its motion from the scenario's "
            f"target micro-motion, so exactly one target must have "
            f"enable_micro_motion = True; found {len(moving)}. Select a target_profile with "
            f"a single breathing target (e.g. breathing_human)."
        )

    target = moving[0]
    amplitude_m = float(getattr(target, "micro_motion_amplitude_m", 0.0))
    frequency_hz = float(getattr(target, "micro_motion_frequency_hz", 0.0))
    phase_rad = float(getattr(target, "micro_motion_phase_rad", 0.0))

    if amplitude_m <= 0.0:
        raise ValueError(
            f"radar_motion_profile = {profile!r} needs a positive "
            f"micro_motion_amplitude_m on target {getattr(targets, 'name', '?')!r}, got "
            f"{amplitude_m!r}."
        )

    velocity_xy = tuple(getattr(target, "velocity_xy_m_per_frame", (0.0, 0.0)))
    velocity_z = float(getattr(target, "velocity_z_m_per_frame", 0.0))

    if any(v != 0.0 for v in velocity_xy) or velocity_z != 0.0:
        raise ValueError(
            f"radar_motion_profile = {profile!r} requires a stationary target: a non-zero "
            f"velocity makes the micro-motion direction frame-dependent (and adds its own "
            f"range change), so the mirror would not be equivalent. Target "
            f"{getattr(target, 'name', '?')!r} has velocity_xy_m_per_frame={velocity_xy!r}, "
            f"velocity_z_m_per_frame={velocity_z!r}."
        )

    direction, axis_label = micro_motion_direction(target, radar_origin_m)

    return SingleToneMotion(
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        t0_s=t0_s,
        translation_amplitude_m=sign * amplitude_m,
        translation_direction=direction,
        provenance=(
            f"{profile}: drived from target {getattr(target, 'name', '?')!r}"
            f"(amplitude {amplitude_m!r} m, frequency {frequency_hz!r} Hz, phase "
            f"{phase_rad!r} rad, axis {axis_label!r}), sign {sign:+.0f}"
        ),
    )


def build_radar_motion_from_profile(
    profile: str,
    targets,
    radar_origin_m,
    slow_time_s,
    random_seed,
) -> RadarMotion:
    """Resolve a named radar_motion_profile into a RadarMotion.

    Args:
        profile (str): one of SUPPORTED_MOTION_PROFILES.
        targets: the resolved cfg.targets list (needed by the derived presets).
        radar_origin_m: array-like [3], the radar base position.
        slow_time_s: array-like [F], the simulation slow-time grid.
        random_seed (init): scenario seed; only the handheld_* presets use it.

    Returns:
        RadarMotion
    """
    name = (profile or "").strip().lower()

    if name not in SUPPORTED_MOTION_PROFILES:
        raise ValueError(
            f"unknown radar_motion_profile = {profile!r}; supported profiles are "
            f"{', '.join(SUPPORTED_MOTION_PROFILES)}"
        )

    if name == "static":
        return StaticMotion()

    # Everything below moves, so the grid must be well formed. This also yields
    # the Nyquist limit used to gate the single-tone presets.
    grid, snapshot_interval_s = validate_slow_time_grid(slow_time_s)
    nyquist_hz = 1.0 / (2.0 * snapshot_interval_s)
    t0_s = float(grid[0])

    if name in ("handheld_light", "handheld_strong", "handheld_rotation_only"):
        scale = 3.0 if name == "handheld_strong" else 1.0
        translation = (
            (0.0, 0.0, 0.0)
            if name == "handheld_rotation_only"
            else tuple(scale * v for v in _HANDHELD_TRANSLATION_RMS_M)
        )
        return build_handheld_jitter(
            translation_rms_local_m=translation,
            rotation_rms_local_rad=_HANDHELD_ROTATION_RMS_RAD,
            min_frequency_hz=_HANDHELD_BAND_HZ[0],
            max_frequency_hz=_HANDHELD_BAND_HZ[1],
            num_spectral_components=_HANDHELD_COMPONENTS,
            slow_time_s=grid,
            random_seed=random_seed,
        )

    if name in _PROBE_TRANSLATION_AXES or name == "probe_yaw":
        if not (_PROBE_FREQUENCY_HZ < nyquist_hz):
            raise ValueError(
                f"radar_motion_profile = {name!r} uses a {_PROBE_FREQUENCY_HZ!r} Hz tone, "
                f"which is not strictly below the slow-time Nyquist frequency "
                f"{nyquist_hz!r} Hz (snapshot interval {snapshot_interval_s!r} s). Shorten"
                f"[radar] period."
            )

        if name == "probe_yaw":
            return SingleToneMotion(
                frequency_hz=_PROBE_FREQUENCY_HZ,
                phase_rad=0.0,
                t0_s=t0_s,
                rotation_amplitude_rad=_PROBE_ROTATION_AMPLITUDE_RAD,
                rotation_axis=(0.0, 0.0, 1.0),
                provenance=f"{name}: yaw-only probe tone, peak amplitude "
                           f"{_PROBE_ROTATION_AMPLITUDE_RAD!r} rad"
            )

        return SingleToneMotion(
            frequency_hz=_PROBE_FREQUENCY_HZ,
            phase_rad=0.0,
            t0_s=t0_s,
            translation_amplitude_m=_PROBE_TRANSLATION_AMPLITUDE_M,
            translation_direction=_PROBE_TRANSLATION_AXES[name],
            provenance=f"{name}: single-axis translation probe tone, peak amplitude "
                       f"{_PROBE_TRANSLATION_AMPLITUDE_M!r} m",
        )
    
    motion = _derived_single_tone(name, targets, radar_origin_m, t0_s)

    if not (motion._frequency_hz < nyquist_hz):
        raise ValueError(
            f"radar_motion_profile = {name!r} derived a {motion._frequency_hz!r} Hz tone from "
            f"the target's micro_motion_frequency_hz, which is not strictly below the "
            f"slow-time Nyquist frequency {nyquist_hz!r} Hz."
        )

    return motion

