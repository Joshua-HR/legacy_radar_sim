import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.constants import speed_of_light
from numpy import interp, array, argsort, clip, cos, deg2rad, rad2deg, arctan2
import numpy as np


wrap_angle_deg = lambda angle_deg: ((angle_deg + 180.0) % 360.0) - 180.0

# --------------------------------------------------------------------------
# Measured 3D antenna pattern CSV loading.
#
# Expected CSV format (as exported by HFSS-style far-field sweeps, e.g.
# data/ant/260713/edge_dual_dipole_ant{0,1}_pattern.csv):
#   Phi[deg],Theta[deg],GainTotal
#   -180,0,0.284101
#   ...
# GainTotal is LINEAR amplitude gain (not dBi, not power/dB) - confirmed by
# every value being strictly positive with no dB-like negative entries, and
# the degenerate rows at Theta=0/180 (every phi maps to the same value,
# consistent with those being the +z/-z poles under the polar-angle-from-+z
# convention used everywhere in this file - see the geometry-convention
# block below). Converted to dBi at load time via 20*log10(gain_linear),
# matching this project's amplitude-not-power dBi convention (see
# _interpolate_pattern_gain_linear() and friends: dBi -> linear uses /20.0
# everywhere, never /10.0).
# --------------------------------------------------------------------------

def load_pattern_gains_dbi_2d_csv(
    csv_path,
) -> "Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[Tuple[float, ...], ...]]":
    """Load a single-antenna measured G(phi, theta) pattern from CSV.

    Args:
        csv_path: path to a CSV file with columns
        ``Phi[deg],Theta[deg],GainTotal`` (GainTotal = linear
        amplitude gain, one row per (phi, theta) grid point, full
        grid - every phi value must appear with every theta value
        exactly once).

    Returns:
        (pattern_phi_deg, pattern_theta_deg, pattern_gains_dbi_2d):
            phi/theta axes (sorted ascending) and the gain table in dBi,
            shape [len(pattern_phi_deg), len(pattern_theta_deg)], indexed
            as gains_dbi_2d[phi_idx][theta_idx] - matching the
            AntennaConfig.pattern_gains_dbi_2d convention exactly.

    Raises:
        FileNotFoundError: csv_path does not exist.
        ValueError: CSV is missing required columns, has a non-rectangular
            phi/theta grid (some combination missing or duplicated), or
            contains a non-positive GainTotal value (linear gain must be
            > 0 to take log10).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Antenna pattern CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"Phi[deg]", "Theta[deg]", "GainTotal"}.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                f"{csv_path}: expected columns 'Phi[deg],Theta[deg],GainTotal', "
                f"got {reader.fieldnames}"
            )
        rows = [
            (float(row["Phi[deg]"]), float(row["Theta[deg]"]), float(row["GainTotal"]))
            for row in reader
        ]
    
    if not rows:
        raise ValueError(f"{csv_path}: no data rows found.")

    phi_axis = tuple(sorted(set(phi for phi, _, _ in rows)))
    theta_axis = tuple(sorted(set(theta for _, theta, _ in rows)))

    gain_lookup = {}
    for phi, theta, gain_linear in rows:
        if gain_linear <= 0.0:
            raise ValueError(
                f"{csv_path}: GainTotal must be > 0 (linear gain), got "
                f"{gain_linear} at (phi={phi}, theta={theta})."
            )
        key = (phi, theta)
        if key in gain_lookup:
            raise ValueError(f"{csv_path}: duplicate row for (phi={phi}, theta={theta}).")
        gain_lookup[key] = gain_linear

    expected_count = len(phi_axis) * len(theta_axis)
    if len(gain_lookup) != expected_count:
        raise ValueError(
            f"{csv_path}: expected a full {len(phi_axis)}x{len(theta_axis)} = "
            f"{expected_count}-point phi/theta grid, got {len(gain_lookup)} rows "
            "(grid must be rectangular - every phi paired with every theta)."
        )

    gains_dbi_2d = tuple(
        tuple(20.0 * np.log10(gain_lookup[(phi, theta)]) for theta in theta_axis)
        for phi in phi_axis
    )

    return phi_axis, theta_axis, gains_dbi_2d


def load_pattern_gains_dbi_2d_per_ant_csv(
    csv_paths: "List[str]",
) -> "Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[Tuple[Tuple[float, ...], ...], ...]]":
    """Load one measured G(phi, theta) pattern CSV per antenna.

    All CSVs must share the same phi/theta axes (this project's
    AntennaConfig has a single pattern_phi_deg/pattern_theta_deg pair
    shared across antennas - only the gain values differ per antenna).

    Args:
        csv_paths: list of CSV paths, one per antenna index (index 0 ->
        antenna 0, etc.)

    Returns:
        (pattern_phi_deg, pattern_theta_deg, pattern_gains_dbi_2d_per_ant):
            shared phi/theta axes, and a tuple of per-antenna gain tables
            (each shape [len(pattern_phi_deg), len(pattern_theta_deg)]).

    Raises:
        ValueError: csv_paths is empty, or the CSVs' phi/theta axes
            don't all match exactly.
    """
    if not csv_paths:
        raise ValueError("csv_paths must contain at least one path.")

    phi_axis0, theta_axis0, gains0 = load_pattern_gains_dbi_2d_csv(csv_paths[0])
    gains_per_ant = [gains0]

    for csv_path in csv_paths[1:]:
        phi_axis, theta_axis, gains = load_pattern_gains_dbi_2d_csv(csv_path)
        if phi_axis != phi_axis0 or theta_axis != theta_axis0:
            raise ValueError(
                f"{csv_path}: phi/theta axes do not match {csv_paths[0]} - "
                "all per-antenna pattern CSVs must share the same grid."
            )
        gains_per_ant.append(gains)
    
    return phi_axis0, theta_axis0, tuple(gains_per_ant)


# --------------------------------------------------------------------------
# Common 3D geometry helpers.
#
# Project angle convention (matches the standard spherical convention used
# by antenna-pattern tools such as HFSS - theta is the polar angle measured
# from +z, NOT elevation-from-horizon):
#   phi_deg     = azimuth           = atan2(dy, dx)                     (x-y plane)
#   theta_deg   = polar angle       = atan2(sqrt(dx^2 + dy^2), dz)      (from +z axis)
#       theta_deg == 0      -> +z axis (straight up)
#       theta_deg == 90     -> in the x-y plane (horizon)
#       theta_deg == 180    -> -z axis (straight down)
# Preferred tuple/argumnet order everywhere is (phi, theta), i.e. azimuth
# before theta - matching the antenna pattern convention G(phi, theta).
#
# Corrected 2026-07-13: an earlier migration pass implemented theta_deg as
# elevation-from-horizon (atan2(dz, sqrt(dx^2+dy^2))), range -90..+90,
# 0=horizon) instead of the polar-angle-from-+z convention HFSS (and this
# project's own antenna-pattern CSV exports) actually use. The two are
# related bty theta_hfss = 90 - theta_elevation. See
# docs/2d_to_3d_migration.md Section 12.E for the full history.
#
# Internal geometry is always represented as xyz arrays of shape [3].
# Legacy 2D callers can keep passing (x, y) tuples; as_xyz()/distance_3d()/
# point_to_phi_theta_deg() upgrade them to 3D with z=0.0 automatically.
# --------------------------------------------------------------------------

def as_xyz(position_xy_m, z_m: float = 0.0) -> "np.ndarray":
    """Convert a legacy 2D (x, y) position (or an already-3D position) into xyz.

    Args:
        position_xy_m: array-like of length 2 (x, y) or length 3 (x, y, z)
        z_m: z value used only when position_xy_m is 2D.

    Returns:
        np.ndarray of shape [3].
    """
    arr = np.asarray(position_xy_m, dtype=float)

    if arr.shape[-1] == 2:
        return np.array([arr[0], arr[1], z_m], dtype=float)
    
    if arr.shape[-1] == 3:
        return arr.astype(float)

    raise ValueError(f"Position must have length 2 or 3, got shape {arr.shape}")


def vector_to_phi_theta_deg(vec) -> "tuple[float, float]":
    """Convert a displacement vector into (phi_deg, theta_deg).

    Args:
        vec: array-like [dx, dy] or [dx, dy, dz].

    Returns:
        (phi_deg, theta_deg): azimuth and polar angle (from +z) in degrees.
        If vec is the zero vector, atan2(0, 0) == 0, so this returns
        (phi_deg = 0.0, theta_deg = 0.0) - the degenerate "+z axis" directions.
    """
    v = np.asarray(vec, dtype=float)
    dx = float(v[0])
    dy = float(v[1])
    dz = float(v[2]) if v.shape[-1] >= 3 else 0.0

    horizontal_range = np.sqrt(dx ** 2 + dy ** 2)

    phi_deg = float(np.degrees(np.arctan2(dy, dx)))
    theta_deg = float(np.degrees(np.arctan2(horizontal_range, dz)))

    return phi_deg, theta_deg


def point_to_phi_theta_deg(origin_pos, point_pos) -> "tuple[float, float]":
    """Compute (phi_deg, theta_deg) of point_pos as seen from origin_pos."""
    origin_xyz = as_xyz(origin_pos)
    point_xyz = as_xyz(point_pos)
    return vector_to_phi_theta_deg(point_xyz - origin_xyz)


def vector_to_theta_phi_deg(vec) -> "tuple[float, float]":
    """Deprecated: prefer vector_to_phi_theta_deg() (phi, theta) order.

    Kept only for backward compatibility with older call sites; returns
    (theta_deg, phi_deg) - the reversed order.
    """
    phi_deg, theta_deg = vector_to_phi_theta_deg(vec)
    return theta_deg, phi_deg


def point_to_theta_phi_deg(origin_pos, point_pos) -> "tuple[float, float]":
    """Deprecated: prefer point_to_phi_theta_deg() (phi, theta) order.

    Kept only for backward compatibility with older call sites; returns
    (theta_deg, phi_deg) - the reversed order.
    """
    phi_deg, theta_deg = point_to_phi_theta_deg(origin_pos, point_pos)
    return theta_deg, phi_deg


def distance_3d(a, b) -> float:
    """Euclidean 3D distance between two positions (2D input upgraded to 3D)."""
    return float(np.linalg.norm(as_xyz(a) - as_xyz(b)))


def _debug_check_geometry_convention() -> None:
    """Lightweight sanity check for the phi/theta convention (not a test suite).

    phi_deg     = azimuth           = atan2(dy, dx)
    theta_deg   = polar angle       = atan2(sqrt(dx^2 + dy^2), dz)  (from +z axis)
    """
    # [1, 0, 0] lies in the x-y plane (dz=0) -> theta = 90 deg (horizon).
    phi0, theta0 = vector_to_phi_theta_deg([1.0, 0.0, 0.0])
    assert abs(theta0 - 90.0) < 1e-9 and abs(phi0) < 1e-9, (phi0, theta0)

    phi45, _ = vector_to_phi_theta_deg([1.0, 1.0, 0.0])
    assert abs(phi45 - 45.0) < 1e-9, phi45

    # [1, 0, 1]: horizontal_range == dz == 1 -> theta = 45 deg either way
    # (this vector happens to be the fixed point of theta_hfss = 90 - theta_elev).
    _, theta45 = vector_to_phi_theta_deg([1.0, 0.0, 1.0])
    assert abs(theta45 - 45.0) < 1e-9, theta45

    # [0, 0, 1] lies on +z axis -> theta = 0 deg exactly.
    _, theta_up = vector_to_phi_theta_deg([0.0, 0.0, 1.0])
    assert abs(theta_up) < 1e-9, theta_up


# Run once at import time - cheap, catches convention regressions immediately.
_debug_check_geometry_convention()


# --------------------------------------------------------------------------
# Default 3D pattern grid factories (used as dataclass default_factory)
#
# Convention
#   pattern_phi_deg     = azimuth axis
#   pattern_theta_deg   = polar angle axis (0=+z, 90=horizon, 180=-z)
#   pattern_gains_dbi_2d shape = [len(pattern_phi_deg), len(pattern_theta_deg)]
#   Indexing: pattern_gains_dbi_2d[phi_idx][theta_idx]
#   Pattern is G(phi, theta) - phi first, theta second, everywhere.
# --------------------------------------------------------------------------

def default_pattern_phi_deg_15() -> Tuple[float, ...]:
    """Azimuth grid: -180...180 deg in 15 deg steps (25 points)."""
    return tuple(float(x) for x in range(-180, 181, 15))


def default_pattern_theta_deg_15() -> Tuple[float, ...]:
    """Polar-angle grid (from +z): 0...180 deg in 15 det steps (13 points)."""
    return tuple(float(x) for x in range(0, 181, 15))


def default_isotropic_pattern_gains_dbi_2d() -> Tuple[Tuple[float, ...], ...]:
    """All-zero-dBi (isotropic) 25 x 13 gain table, shape [phi, theta]."""
    phi = default_pattern_phi_deg_15()
    theta = default_pattern_theta_deg_15()
    return tuple(tuple(0.0 for _ in theta) for _ in phi)


def default_pattern_phi_gains_dbi_zero() -> Tuple[float, ...]:
    """All-zero-dBi azimuth cut, aligned with default_pattern_phi_deg_15()."""
    return tuple(0.0 for _ in default_pattern_phi_deg_15())


def default_pattern_theta_gains_dbi_zero() -> Tuple[float, ...]:
    """All-zero-dbi polar-angle cut, aligned with default_pattern_theta_deg_15()."""
    return tuple(0.0 for _ in default_pattern_theta_deg_15())


@dataclass
class AntennaConfig:
    """
    Container for the antenna array setting."""

    num_antennas: int
    antenna_spacing_m: float = 0.019
    carrier_freq_hz: float = 7987200000 # Hz; IEEE 802.15.4a/z UWB Channel 9 center frequency (7987.2 MHz)
    c: float = speed_of_light
    enable_radiation_pattern: bool = True

    # --- Round-trip radar-equation path-loss-model ------------------------
    # Scattered-echo amplitude is the two-way (TX -> scatter -> RX) voltage
    # law lambda_m*sqrt(sigma)/((4*pi)**1.5 * Rt * Rr), anchored to a reference
    # distance, with a per-collider-type excess-loss exponent on top of the
    # physical n=2 term (see CIRSimulator._radar_equation_amplitude_factor).
    # n == 2.0 reduces exactly to the radar equation; n > 2.0 adds excess loss
    # (indoor multipath / clutter / re-reflection) beyond
    # path_loss_ref_distance_m. Uses the SAME carrier_freq_hz/lambda_m as the
    # phase term above - do not add a second frequency knob.
    # These exponents apply ONLY to scattered echoes. TX-RX coupling
    # (tx_rx_feedthrough, its ringing tail, radiation/pcb leakage) is a 
    # separate one-way family handled by LeakageConfig and never routed
    # through this model.
    target_path_loss_exp: float = 2.0       # unitless exponent, direct path to moving targets
    clutter_path_loss_exp: float = 2.3      # unitless exponent, direct path to static clutter/room objects
    reflection_path_loss_exp: float = 2.5   # unitless exponent, secondary/reflected path (all collider types)
    path_loss_ref_distance_m: float = 1.0   # meters; Friis/log-distance pivot distance d0
    path_loss_floor_m: float = 0.1          # meters; minimum effective distance (singularity guard, matched old inline max(d, 0.1))
    # Calibration multiplier = (4*pi)**1.5 * path_loss_ref_distance_m**2 / lambda_m,
    # evaluated ONCE at the DEFAULT carrier_freq_hz/path_loss_ref_distance_m above. This
    # keeps the legacy numeric convention (amplitude=1.0 at d=1m for sigma=1 m^2) so
    # noise_std, ADC clipping, quantization, the Q8.8 hex codec scale, and leakage
    # constants elsewhere in this codebase - all tuned against that O(1) convention -
    # stay self-consistent across the radar-equation migration.
    #
    # DO NOT recompute this per carrier frequency. It is a FIXED dimensionless
    # constant: the physical lambda_m**1 dependence now lives in
    # _radar_equation_amplitude_factor itself. Re-deriving it from the running
    # lambda_m (as the previous (4*pi*d0/lambda_m)**2 comment instructed) would cancel
    # that dependence and silently restore a lambda**0 amplitude law, breaking
    # cross-channel (e.g. Ch5 vs Ch9) voltage scaling.
    # Only path_loss_ref_distance_m changing warrants recomputing it
    # See docs/friis_path_loss_model.md for the derivation.
    path_loss_ref_gain: float = 1186.830373866022 # unitless calibration multiplier (see comment above)
    
    # pattern_mode:
    #   "isotropic"             = gain 1.0 (0 dBi) in evert direction, ignores phi/theta
    #   "cosine"                = simple azimuth-only cosine model
    #   "table"                 = legacy azimuth-only interpolated real antenna pattern
    #   "table_azimuth"         = alias for "table"
    #   "table_3d"              = full G(phi, theta) bilinear-interpolated 2D table
    #   "table_3d_separable"    = phi-cut dBi + theta-cut dBit (first approximation)
    pattern_mode: str = "table"

    # Legacy azimuth-only boresight, still used by "table"/"table_azimuth"/"cosine".
    boresight_deg: float = 0.0
    # 3D-mode boresight offsets. Default to 0.0 if not explicitly configured.
    boresight_phi_deg: float = 0.0
    # NOTE: theta_deg is the polar angle from +z (0=+z axis, 90=horizon,
    # 180=-z axis - see the geometry-convention block at the top of this
    # file). boresight_theta_deg=0.0 therefore means "boresight points
    # straight up (+z)", NOT "boresight points at the horizon" - that would
    # be boresight_theta_deg=90.0. This numeric default is unchanged from
    # before the 2026-07-13 theta-convention fix; it has not yet been set
    # to reflect any specific antenna's actual mounting orientation - do
    # not assume 0.0 is physically correct for a horizon-pointing antenna.
    boresight_theta_deg: float = 0.0

    pattern_exponent: float = 2.0
    min_gain: float = 0.08
    
    # --- Legacy azimuth-only table (angle in deg, gain in dBi) ------------
    # Kept for backward compatibility with pattern_mode == "table".
    pattern_angles_deg: Tuple[float, ...] = (
        -180, -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15,
        0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180
    )

    pattern_gains_dbi: Tuple[float, ...] = (
        -3.2, -3.0, -2.8, -2.5, -2.2, -1.8, -1.4, -1.1, -0.8, -0.5, -0.3, -0.1,
        0.0, -0.1, -0.3, -0.5, -0.8, -1.1, -1.4, -1.8, -2.2, -2.5, -2.8, -3.0, -3.2
    )

    # --- "table_3d": full 2D G(phi, theta) table --------------------------
    # Default is an all-zero-dbi isotropic sphere (25 phi x 13 theta points).
    # Can be populated directly in Python, or loaded from a measured CSV via
    # load_pattern_gains_dbi_2d_csv()/load_pattern_gains_dbi_2d_per_ant_csv()
    # (see create_default_config()'s [antenna] pattern_csv_ant* INI keys).
    pattern_phi_deg: Tuple[float, ...] = field(default_factory=default_pattern_phi_deg_15)
    pattern_theta_deg: Tuple[float, ...] = field(default_factory=default_pattern_theta_deg_15)
    pattern_gains_dbi_2d: Tuple[Tuple[float, ...], ...] = field(
        default_factory=default_isotropic_pattern_gains_dbi_2d
    )
    # Optional per-antenna override of pattern_gains_dbi_2d - a tuple of
    # per-antenna gain tables, each shape [len(pattern_phi_deg),
    # len(pattern_theta_deg)], sharing the single pattern_phi_deg/
    # pattern_theta_deg axes above. None (default) means "use
    # pattern_gains_dbi_2d for every antenna" (the pre-existing,
    # single-shared-table behavior). When set, index ant_idx selects that
    # antenna's table; ant_idx values beyond len(pattern_gains_dbi_2d_per_ant)
    # fall back to pattern_gains_dbi_2d (so a scenario can supply a real
    # measured pattern for antenna 0/1 while additional antennas silently
    # use the isotropic/default table rather than raising).
    pattern_gains_dbi_2d_per_ant: Optional[Tuple[Tuple[Tuple[float, ...], ...], ...]] = None

    # --- "table_3d_separable": independent azimuth/theta cuts -------------
    # gain_dbi = phi_cut(phi_deg) + theta_cut(theta_deg); both default fo
    # all-zero dBi, normalizaed to 0 dBi at boresight.
    pattern_phi_gains_dbi: Tuple[float, ...] = field(
        default_factory=default_pattern_phi_gains_dbi_zero
    )
    pattern_theta_gains_dbi: Tuple[float, ...] = field(
        default_factory=default_pattern_theta_gains_dbi_zero
    )

    def _interpolate_pattern_gain_dbi(self, rel_angle_deg: float) -> float:
        """
        Interpolate antenna gain from user-provided angle/gain table.

        Args:
            rel_angle_deg (float): angle in degree.

        Returns:
            float: gain in dBi.
        
        Raises:
            ValueError: "pattern_angles_deg and pattern_gains_dbi must have the same length.",
            ValueError: "pattern table must contain at least 2 points."
        """
        angles = array(self.pattern_angles_deg, dtype=float)
        gains_dbi = array(self.pattern_gains_dbi, dtype=float)

        if len(angles) != len(gains_dbi):
            raise ValueError("pattern_angles_deg and pattern_gains_dbi must have the same length.")

        if len(angles) < 2:
            raise ValueError("pattern table must contain at least 2 points.")

        sort_idx = argsort(angles)
        angles = angles[sort_idx]
        gains_dbi = gains_dbi[sort_idx]

        angle_clamped = clip(rel_angle_deg, angles[0], angles[-1])
        gain_dbi = interp(angle_clamped, angles, gains_dbi)

        return float(gain_dbi)

    def _interpolate_pattern_gain_linear(self, rel_angle_deg: float) -> float:
        """
        Interpolate antenna gain and convert from dBi to linear amplitude gain.

        Args:
            rel_angle_deg (float): angle in degree

        Returns:
            float: linear gain
        """
        gain_dbi = self._interpolate_pattern_gain_dbi(rel_angle_deg)

        # dBi is power-like gain, convert to amplitude scale for field contribution
        gain_linear = 10.0 ** (gain_dbi / 20.0)

        gain_linear = max(gain_linear, self.min_gain)
        return float(gain_linear)

    def _vector_to_phi_theta_deg(self, antenna_pos, point_pos) -> "tuple[float, float]":
        """Compute (phi_deg, theta_deg) of point_pos as seen from antenna_pos.

        phi_deg     = azimuth           = atan2(dy, dx)
        theta_deg   = polar angle       = atan2(sqrt(dx^2 + dy^2), dz)  (from +z axis)

        Accepts 2D (x, y) or 3D (x, y, z) positions; 2D inputs are treated
        as z=0.0. If point_pos == antenna_pos, dx=dy=dz=0 and aten2(0, 0)
        gives 0, so this returns (phi_deg=0.0, theta_deg=0.0) - acceptable
        degenerate case.
        """
        antenna_xyz = as_xyz(antenna_pos)
        point_xyz = as_xyz(point_pos)
        vec = point_xyz - antenna_xyz

        dx = float(vec[0])
        dy = float(vec[1])
        dz = float(vec[2]) if len(vec) >= 3 else 0.0

        horizontal_range = np.sqrt(dx * dx + dy * dy)

        phi_deg = float(np.rad2deg(np.arctan2(dy, dx)))
        theta_deg = float(np.rad2deg(np.arctan2(horizontal_range, dz)))

        return phi_deg, theta_deg

    @staticmethod
    def _interp_1d_dbi(x_deg: float, xs_deg, gains_dbi) -> float:
        """1D linear interpolation with sorting + clamping, returns dBi.

        Args:
            x_deg (float): query angle in degrees.
            xs_deg: array-like of axis angles (not required to be sorted).
            gains_dbi: array-like of gains in dBi, same length as xs_deg.

        Returns:
            float: interpolated gain in dBi.
        """
        xs = np.asarray(xs_deg, dtype=float)
        gains = np.asarray(gains_dbi, dtype=float)

        if xs.shape != gains.shape:
            raise ValueError(
                f"xs_deg and gains_dbi must have the same shape, got {xs.shape} vs {gains.shape}"
            )
        
        if len(xs) < 2:
            raise ValueError("interpolation axis must contain at least 2 points.")

        sort_idx = np.argsort(xs)
        xs = xs[sort_idx]
        gains = gains[sort_idx]

        x_clamped = float(np.clip(x_deg, xs[0], xs[-1]))
        return float(np.interp(x_clamped, xs, gains))

    def _pattern_gains_dbi_2d_for_ant(self, ant_idx: Optional[int]) -> "np.ndarray":
        """Resolve which G(phi, theta) gain table applies to antenna ant_idx.

        Falls back to the single shared pattern_gains_dbi_2d when
        pattern_gains_dbi_2d_per_ant is unset (None), ant_idx is None (no
        antenna context - e.g. the polar/heatmap plot helpers), or ant_idx
        is out of range for the per-antenna tuple.
        """
        if ant_idx is not None and self.pattern_gains_dbi_2d_per_ant is not None:
            if 0 <= ant_idx < len(self.pattern_gains_dbi_2d_per_ant):
                return np.asarray(self.pattern_gains_dbi_2d_per_ant[ant_idx], dtype=float)

        return np.asarray(self.pattern_gains_dbi_2d, dtype=float)

    def _interp_2d_gain_dbi(
        self, phi_deg: float, theta_deg: float, ant_idx: Optional[int] = None
    ) -> float:
        """Bilinear-interpolate the G(phi, theta) gain table at (phi_deg, theta_deg).

        The table used in pattern_gains_dbi_2d_per_ant[ant_idx] if set,
        otherwise the single shared pattern_gains_dbi_2d (see
        _pattern_gains_dbi_2d_for_ant()). Shape is always
        [len(pattern_phi_deg), len(pattern_theta_deg)], indexed as
        gain_map[phi_idx][theta_idx] - i.e. G(phi, theta), phi first.

        Args:
            phi_deg (float): azimuth query angle, will be wrapped to
                [-180, 180] then clamped to the phi axis range.
            theta_deg (float): polar-angle (from +z) query angle, clamped
                (not wrapped) to the theta axis range.
            ant_idx (Optional[int]): antenna index selecting a per-antenna
                table; None uses the shared table unconditionally.

        Returns:
            float: interpolated gain in dBi.
        """
        phi_axis = np.asarray(self.pattern_phi_deg, dtype=float)
        theta_axis = np.asarray(self.pattern_theta_deg, dtype=float)
        gain_map = self._pattern_gains_dbi_2d_for_ant(ant_idx)

        if phi_axis.ndim != 1 or theta_axis.ndim != 1:
            raise ValueError("pattern_phi_deg and pattern_theta_deg must be 1D.")

        if gain_map.shape != (len(phi_axis), len(theta_axis)):
            raise ValueError(
                "pattern gain table must have shape "
                f"[len(pattern_phi_deg), len(pattern_theta_deg)] = "
                f"[{len(phi_axis)}, {len(theta_axis)}], got {gain_map.shape} "
                f"(ant_idx={ant_idx})."
            )

        if len(phi_axis) < 2 or len(theta_axis) < 2:
            raise ValueError("pattern_phi_deg and pattern_theta_deg must each have at least 2 points.")

        # sort phi (reorders rows) and theta (reorders columns) independently.
        phi_sort_idx = np.argsort(phi_axis)
        phi_axis = phi_axis[phi_sort_idx]
        gain_map = gain_map[phi_sort_idx, :]

        theta_sort_idx = np.argsort(theta_axis)
        theta_axis = theta_axis[theta_sort_idx]
        gain_map = gain_map[:, theta_sort_idx]

        # Wrap phi to [-180, 180] first (azimuth is circular), then clamp to
        # the configured axis range (handles axes narrower than 360 deg).
        phi_wrapped = wrap_angle_deg(phi_deg)
        phi_clamped = float(np.clip(phi_wrapped, phi_axis[0], phi_axis[-1]))

        # Polar angle (from +z) is not circular - clamp only, do not wrap.
        theta_clamped = float(np.clip(theta_deg, theta_axis[0], theta_axis[-1]))

        p1 = int(np.searchsorted(phi_axis, phi_clamped, side="left"))
        p1 = min(max(p1, 1), len(phi_axis) - 1)
        p0 = p1 - 1

        t1 = int(np.searchsorted(theta_axis, theta_clamped, side="left"))
        t1 = min(max(t1, 1), len(theta_axis) - 1)
        t0 = t1 - 1

        phi_span = phi_axis[p1] - phi_axis[p0]
        phi_w = 0.0 if phi_span <= 0 else (phi_clamped - phi_axis[p0]) / phi_span

        theta_span = theta_axis[t1] - theta_axis[t0]
        theta_w = 0.0 if theta_span <= 0 else (theta_clamped - theta_axis[t0]) / theta_span

        # q_{phi_idx}{theta_idx}: gain[phi_idx, theta_idx]
        q00 = gain_map[p0, t0]
        q01 = gain_map[p0, t1]
        q10 = gain_map[p1, t0]
        q11 = gain_map[p1, t1]

        gain_dbi = (
            q00 * (1 - phi_w) * (1 - theta_w)
            + q10 * phi_w * (1 - theta_w)
            + q01 * (1 - phi_w) * theta_w
            + q11 * phi_w * theta_w
        )

        return float(gain_dbi)

    def _antenna_pattern_gain_3d_table(
        self, phi_deg: float, theta_deg: float, ant_idx: Optional[int] = None
    ) -> float:
        """"table_3d" mode: bilinear-interpolated G(phi, theta) -> linear amplitude gain.

        ant_idx selects a per-antenna gain table when
        pattern_gains_dbi_2d_per_ant is set (see
        AntennaConfig._pattern_gains_dbi_2d_for_ant()); otherwise every
        antenna uses the single shared pattern_gains_dbi_2d table.
        """
        rel_phi = wrap_angle_deg(phi_deg - self.boresight_phi_deg)
        rel_theta = theta_deg - self.boresight_theta_deg

        gain_dbi = self._interp_2d_gain_dbi(rel_phi, rel_theta, ant_idx=ant_idx)

        # dBi is power-like gain; CIR samples are complex amplitude, so /20.
        gain_linear = 10.0 ** (gain_dbi / 20.0)
        gain_linear = max(gain_linear, self.min_gain)
        return float(gain_linear)

    def _antenna_pattern_gain_3d_separable(self, phi_deg: float, theta_deg: float) -> float:
        """"table_3d_separable" mode: independent azimuth/theta dBi cuts, summed.

        Assumes both cuts are normalized to 0 dBi at boresight; adding the
        two dB cuts is a first approximation ahead of a full measured
        G(phi, theta) table.
        """
        rel_phi = wrap_angle_deg(phi_deg - self.boresight_phi_deg)
        rel_theta = theta_deg - self.boresight_theta_deg

        phi_gain_dbi = self._interp_1d_dbi(rel_phi, self.pattern_phi_deg, self.pattern_phi_gains_dbi)
        theta_gain_dbi = self._interp_1d_dbi(rel_theta, self.pattern_theta_deg, self.pattern_theta_gains_dbi)

        gain_dbi = phi_gain_dbi + theta_gain_dbi

        gain_linear = 10.0 ** (gain_dbi / 20.0)
        gain_linear = max(gain_linear, self.min_gain)
        return float(gain_linear)

    def _antenna_pattern_gain(self, antenna_pos, point_pos, ant_idx: Optional[int] = None) -> float:
        """
        Compute antenna amplitude gain toward a point.

        Supported pattern_mode values:
            "isotropic"                 - gain 1.0 (0 dBit) everywhere, ignores phi/theta.
            "cosine"                    - simple azimuth-only cosine model.
            "table"/"table_azimuth"     - legacy azimuth-only interpolated tavle,
                                        uses phi_deg only; theta_deg is ignored.
            "table_3d"                  - full G(phi, theta) bilinear-interpolated table.
            "table_3d_separable"        - phi-chut dBi, theta-cut dBi (first approximatation).

        Accepts both 2D (x, y) and 3D (x, y, z) antenna_pos/point_pos; 2D
        inputs are treated as z=0.0.

        Angle convention (phi first, theta second, everywhere; matches HFSS):
            phi_deg     = azimuth               = atan2(dy, dz)
            theta_deg   = polar angle           = atan2(sqrt(dx^2 + dy^2), dz)  (from +z)

        Args:
            antenna_pos (np.ndarray): antenna position, shape [2] or [3].
            point_pos (np.ndarray): point position, shape [2] or [3].
            ant_idx (Optional[int]): antenna index, used only by
                "table_3d" to select a per-antenna gain table when
                pattern_gains_dbi_2d_per_ant is set (see
                _pattern_gains_dbi_2d_for_ant()); every other pattern_mode
                ignores it (they have no per-antenna table concept).

        Returns:
            float: gain
        """
        if not self.enable_radiation_pattern:
            return 1.0

        if self.pattern_mode == "isotropic":
            # Baseline sanity-check mode: no table, no interpolation.
            return 1.0

        phi_deg, theta_deg = self._vector_to_phi_theta_deg(antenna_pos, point_pos)
        
        if self.pattern_mode == "table_3d":
            return self._antenna_pattern_gain_3d_table(phi_deg, theta_deg, ant_idx=ant_idx)

        if self.pattern_mode == "table_3d_separable":
            return self._antenna_pattern_gain_3d_separable(phi_deg, theta_deg)

        if self.pattern_mode in ("table", "table_azimuth"):
            # Legacy azimuth-only table: looks up phi_deg only.
            # theta_deg is ignored in this mode.
            rel_angle = wrap_angle_deg(phi_deg - self.boresight_deg)
            return self._interpolate_pattern_gain_linear(rel_angle)

        # Default = cosine model (azimuth-only)
        rel_angle = wrap_angle_deg(phi_deg - self.boresight_deg)
        c = cos(deg2rad(rel_angle))
        c = max(c, 0.0)

        gain = (c ** self.pattern_exponent)
        gain = max(gain, self.min_gain)

        return float(gain)
@dataclass
class CIRConfig:
    """Container for CIR configuration"""
    num_bins: int = 256
    bin_time_s: float = 1.0 / 998.4e6  # fs = 998.4 MHz → bin_time ≈ 1.0016 ns
    num_frames: int = 256
    noise_std: float = 0.0
    noise_std_per_ant: Optional[List[float]] = None
    # Optional per-antenna amplitdue-proportional noise term, added in
    # quadrature to noise_std_per_ant: sigma[a,k] = sqrt(noise_std_per_ant[a]**2
    # + (noise_amp_proportional_factor[a] * |cir_frame[a,k]|)**2). Models
    # frame-to-frame fluctuation that scales with tap amplitdue (e.g. TX-RX
    # feedthrough/ringing tail), observed in measured data but absent from the
    # flat noise floor alone. None (default) disables it - no behavior change.
    noise_amp_proportional_factor: Optional[List[float]] = None


@dataclass
class DUTConfig:
    """
    Metadata block to mimic real DUT config in log output
    """
    rawlog: int = 1
    device_type: str = "UA200"
    instrument: str = "IqGigPLUS"
    reset_timeout: int = 4
    response_timeout: int = 1
    baudrate: int = 921600
    dump_cir: int = 0



@dataclass
class RadarTestConfig:
    """
    Metadata block to mimic real radar config in log output
    Only affects saved log text, not core simulation physics
    """
    test_name: str = "RADAR1"
    com: str = "COM14"
    preamble_code: int = 25
    tpc: int = 20
    rx_gains: list[float] = field(default_factory=list)
    rx1_gain: int = 40
    rx2_gain: int = 40
    channel: int = 9    # decorative metadata only (echoed by exporter,py); does not drive AntennaConfig.carrier_freq_hz
    tx_psf: str = "default"
    rframe: int = 0
    prf: str = "HPRF"
    sic_enable: int = 1
    calibrate_gain: int = 0
    calibrate_sic: int = 0
    rx_accumulation: int = 52
    packets: int = 128
    cir_taps: int = 64
    offset: int = -2
    period: int = 20
    ant: str = "12-1"

    def __post_init__(self):
       self.rx_gains = [self.rx1_gain, self.rx2_gain]
@dataclass
class LeakageConfig:
    """
    Container for leakage and coupling effects that do not come from the external scene.
    """

    num_antennas: int

    enable_radiation_leakage: bool = False
    enable_pcb_leakage: bool = False
    enable_tx_rx_feedthrough: bool = False
    enable_early_leakage: bool = False

    radiation_leakage_factor: list[float] = field(default_factory=list)
    radiation_phase_offset: list[float] = field(default_factory=list)
    pcb_leakage_factor: list[float] = field(default_factory=list)
    pcb_phase_offset: list[float] = field(default_factory=list)
    # Direct leakage into each RX chain from TX pulse
    tx_rx_feedthrough_amp: list[float] = field(default_factory=list)
    tx_rx_feedthrough_decay: list[float] = field(default_factory=list)
    tx_rx_feedthrough_phase_rad: list[float] = field(default_factory=list)
    tx_rx_feedthrough_ripple_amp: list[float] = field(default_factory=list)
    tx_rx_feedthrough_ripple_freq: list[float] = field(default_factory=list)
    tx_rx_feedthrough_random_std: list[float] = field(default_factory=list)
    early_leakage_amplitude: list[float] = field(default_factory=list)
    early_leakage_decay: list[float] = field(default_factory=list)
    early_leakage_phase_rad: list[float] = field(default_factory=list)
    early_leakage_ripple_amp: list[float] = field(default_factory=list)
    early_leakage_ripple_freq: list[float] = field(default_factory=list)
    early_leakage_random_std: list[float] = field(default_factory=list)

    radiation_delay_bins: int = 0
    pcb_delay_bins: int = 0
    early_leakage_num_taps: int = 4

    # Which antenna carries TX physically
    tx_antenna_index: int = 0
    # Start tap and shape
    tx_rx_feedthrough_start_tap: int = 0
    tx_rx_feedthrough_num_taps: int = 3

    # Damped ringing tail that follows the tx_rx_feedthrough peak
    # (matched-filter / pulse-shaping ringing, not target reflection or
    # antenna-to-antenna radiation coupling).
    enable_tx_rx_feedthrough_ringing: bool = False
    tx_rx_feedthrough_ringing_start_tap: int = 0
    tx_rx_feedthrough_ringing_num_taps: int = 0


    def __post_init__(self):
        for _ in range(self.num_antennas):

            self.radiation_leakage_factor.append(0.0)
            self.radiation_phase_offset.append(0.0)
            self.pcb_leakage_factor.append(0.0)
            self.pcb_phase_offset.append(0.0)
            self.tx_rx_feedthrough_amp.append(0.0)
            self.tx_rx_feedthrough_decay.append(0.0)
            self.tx_rx_feedthrough_phase_rad.append(0.0)

            self.tx_rx_feedthrough_ripple_amp.append(0.10)
            self.tx_rx_feedthrough_ripple_freq.append(1.0)
            self.tx_rx_feedthrough_random_std.append(0.01)

            self.early_leakage_amplitude.append(0.0)

            self.early_leakage_decay.append(0.75)
            self.early_leakage_phase_rad.append(0.0)
            self.early_leakage_ripple_amp.append(0.15)
            self.early_leakage_ripple_freq.append(1.1)
            self.early_leakage_random_std.append(0.02)

@dataclass
class FrontendConfig:
    """
    Container class for receiver frontend behavior: include ADC clipping, quantization, frame drift, and pulse spread
    """

    num_antennas: int
    enable_adc_clipping: bool = True
    enable_quantization: bool = False
    enable_frame_drift: bool = True
    enable_pulse_spreading: bool = True

    adc_soft_limit: float = 8.0
    adc_hard_limit: float = 12.0
    quantization_bits: int = 9
    quantization_full_scale: float = 9.0
    quantization_dc_offset_i: float = 0.0
    quantization_dc_offset_q: float = 0.0
    frame_amplitude_drift_std: float = 0.03

    frame_phase_drift_std: list[float] = field(default_factory=list)

    frame_timing_jitter_std_bins: float = 0.08

    pulse_spread_kernel: Tuple[float, ...] = (
        0.10,
        0.22,
        0.36,
        0.22,
        0.10
    )
    def __post_init__(self):
        for _ in range(self.num_antennas):
            self.frame_phase_drift_std.append(0.2)

@dataclass
class SceneConfig:
    """
    Container for Scene / room configuration

    This block defines the simulated environment layout:
    - room size
    - radar position
    - radar placement mode
    - floor height reference
    - whether to auto-create default clutter
    """

    # Room dimensions in meters
    room_length_x_m: float = 4.0
    room_width_y_m: float = 4.0
    room_height_z_m: float = 2.6

    # Radar placement in the room
    radar_position_x_m: float = 0.0
    radar_position_y_m: float = 0.0
    radar_height_z_m: float = 1.2

    # Just a descriptive helper for your scenarios
    # Example: "center", "near_wall", "corner"
    radar_placement_mode: str = "center"

    # If True, default room clutter can be auto-generated
    enable_default_room_clutter: bool = True

    # Geometry mode controls how z coordinates and future antenna pattern
    # lookup are interpreted. Internal positions are always xyz [3] arrays
    # in both modes - this only controls default z values and whether
    # theta_deg is expected to differ from the flat-scene default of 90 deg
    # (horizon; see the geometry-convention block at the top of this file).
    #   "2d_legacy" - antenna/target/clutter z forced to 0.0 (old behavior).
    #   "3d"        - antenna z = radar_height_z_m; target/clutter z respected.
    geometry_mode: str = "2d_legacy"
    
@dataclass
class RadarMotionConfig:
    """Container for radar ego-motion (slow-time radar pose) configuration.

    Populated from the optional ``[radar_motion]`` INI section, which mirrors
    the reference implementation's ``radar.motion`` node. Every field is
    optional and the default is inert: ``type = "static"`` reproduces the
    pre-ego-motion engine bit-for-bit via a legacy fast path in
    CIRSimulator._antenna_positions_at_frame().

    The section itself is optional, but WITHIN the section the surface is
    strict: an unknown key, or a key that belongs to a different ``type``, is a
    hard error (ego_motion.validate_ini_keys). A typo must never silently
    disable the motion.

    The radar BASE pose is not configured here -- it is the existing
    ``[scene] radar_position_x_m / radar_position_y_m / radar_height_z_m``
    (position) with an identity base orientation, because this engine has no
    radar-orientation field. The ``boresight_*`` fields on AntennaConfig are an
    antenna-pattern lookup offset, NOT a radar body rotation, and are
    deliberately left untouched by ego motion. See docs/ego_motion_model.md

    Units follow the project convention (name carries the unit): metres,
    radians, hertz, seconds.
    """

    # "static" | "sampled_pose" | "handheld_jitter"
    type: str = "static"

    # --- named preset (an alternative to the keys below) ------------------
    # Non-empty selects a hardcoded preset by name from 
    # ego_motion.SUPPORTED_MOTION_PROFILES, resolved by
    # ego_motion.build_radar_motion_from_profile(). Empty (default) means
    # "use `type` and its parameters below".
    #
    # Set from the platform INI's [profile] radar_motion_profile, and
    # overriden by the wrapper's [synthetic] radar_motion_profile
    # (override-if-present, exactly like hardware_profile / room_profile /
    # target_profile). Combining a non-empty profile with a non-static `type`
    # is a hard error, not a silent shadowing.
    #
    # The `mirror_target_micromotion` / `cancel_target_micromotion` presets are
    # DERIVED: they read the resolved target's own micro-motion parameters, so
    # they need no duplicated literals. That is only possible because
    # CIRSimulator._setup_radar_motion() runs after _apply_target_profile().
    profile: str = ""

    # --- type = sampled_pose ----------------------------------------------
    # Trace path, RELATIVE to the scenario INI's directory. Absolute paths are
    # refused so scenarios stay portable. The file holds body-local pose
    # DELTAS relative to the base pose, not world poses.
    file: str = ""
    # Both of these accept exactly one value each; anything else is an error.
    reference: str = "base_pose_relative"
    extrapolation: str = "error"

    # --- type = handheld_jitter -------------------------------------------
    # Non-negative 3-vectors in the INITIAL radar body frame. An axis set to
    # 0.0 is exactly zero (never renormalised).
    translation_rms_local_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rms_local_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Requested band. Runtime validation is stricter than the schema:
    # 0 < min <= max and max < slow-time Nyquist (equality refused).
    min_frequency_hz: float = 0.0
    max_frequency_hz: float = 0.0
    # K, the number of sinusoidal components per axis. Must be > 0.
    num_spectral_components: int = 0

    # --- derived, not INI keys --------------------------------------------
    # Directory of the scenario INI that declared the motion, used only to
    # resolve a relative `file`. Set by create_default_config().
    scenario_dir: Optional[str] = None
    # Reproducibility metadata (motion type, seed derivation, requested band,
    # component frequency list, K, and the 6 per-axis achieved RMS values),
    # filled in by CIRSimulator once the trajectory is built. It rides into
    # cir_metadata.json through CIRMetadata.raw_config, so no artifact writer
    # or schema needs to change.
    metadata: Optional[Dict[str, object]] = None


@dataclass
class TargetOverrideConfig:
    """Container for the optional ``[target_override]`` INI section.

    Holds per-target kinematics *deviations* applied on top of whatever
    ``[profiles] target_profile`` resolved to, so a range / speed / breathing
    sweep can be expressed in config instead of by editing the hardcoded
    ``Target(...)`` literals in ``CIRSimulator._apply_target_profile()``.

    The section is optional and an absent section is a bit-identical no-op, but
    WITHIN the section the surface is strict: unknown keys, out-of-range target
    indices, and values the engine would silently ignore are hard errors
    (``target_override.parse_ini_items`` / ``validate_values``). Same rationale
    as RadarMotionConfig above.

    Units follow the project convention (the name carries the unit). Velocity is
    offered as ``velocity_*_m_per_s`` and converted to the engine's native
    ``velocity_*_m_per_frame`` using the resolved ``[radar] period``.
    """

    #: {target_index: {field: parsed value}}. Empty == section absent.
    entries: Dict[int, Dict[str, object]] = field(default_factory=dict)

    #: What was actually applied, including both velocity spellings where a
    #: conversions happened. Filled in by CIRSimulator; rides into
    #: cir_metadata.json through CIRMetadata.raw_config, so no artifact writer
    #: or schema needs to change. This is what makes a sweep point
    #: reconstructible from its own run directory.
    metadata: Optional[Dict[str, object]] = None

@dataclass
class BuildInfo:
    """
    Metadata to mimic firmware/build information in saved log.
    """
    name: str = "sim_ua200"
    chip_id: str = "SIMULATED1234567890ABCDEF00000000"
    git_tag: str = "sim-build"
    build_date: str = "2026-04-14 15:35:00"

@dataclass
class OptionalImpairments:
    """Container class for optional impairment
    """
    enable_gain_mismatch: bool = False
    antenna_gain_mismatch: Optional[List[float]] = None
    enable_phase_mismatch: bool = False
    antenna_phase_mismatch_rad: Optional[List[float]] = None
    enable_baseline_subtraction: bool = True
    save_baseline_frame: bool = True