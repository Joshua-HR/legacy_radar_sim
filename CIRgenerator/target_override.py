"""Strict, optional per-tangent kinematics override surface (``[target_override]'').

Why this module exists
---------------------
Every Built-in scene is selected by *name* through ``[profiles] target_profile``,
and each name expands to hardcoded ``Target(...)`` literals in
``CIRSimulator._apply_target_profile()``. That is fine for a fixed catalog but
makes a *sweep* impossible to express: varying a target's range, speed, or
breathing amplitude required editing thos literals in Python, so the resulting
run's own ``resolved_config.ini`` / ``cir_metadata.json`` did not record which
sweep point it was. Eight such runs were received from a teammate and were
byte-identical in config apart from ``target_profile`` -- none of them
reproducible. See ``docs/standard_validation_geometry.md``.

This section closes that gap: it applies *deviations* on top of the resolved
profile, and records what it applied into the simulation metadata so a sweep
point is reconstructible from its own artifacts.

Design rules
------------
* **Optional section, strict contents.** An absent section is a bit-identical
  no-op. Within the section, an unknown key, an out-of-range target index, or a
  value the engine would silently ignore is a hard ``ValueError``. This mirrors
  ``ego_motion``'s ``[radar_motion]`` surface and is deliberately stricter than
  the tolerant ``fallback=`` reads used throughout ``create_default_config()``.
* **Every guard here exists because the engine's own failure mode is silence.**
  ``Target._apply_target_micro_motion()`` treats any unrecognised
  ``micro_motion_axis`` as ``"radial"`` and early-returns on a non-positive
  amplitude, so a typo or a sign error produces a plausible-looking CIR rather
  than an exception.
* **Veolocity is offered in m/s.** The engine field is
  ``velocity_xy_m_per_frame`` -- metres per *slow-time frame*, not per second.
  That unit has already caused one real error (a ``1.0`` intended as 1 m/s
  became 200 m/s at a 5 ms frame period). ``velocity_*_m_per_s`` is therefore
  the preferred spelling and is converted here using the resolved
  ``cfg.radar.period``; the native ``*_m_per_frame`` spelling stays available,
  and setting both for the same axis is an error rather than a precedence rule.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


#: The configuration section name.
INI_SECTION = "target_override"

#: ``target{index}_{field}``. The index prefix is always required -- there is no
#: bare ``position_xy_m`` form, so a single-target scene and a multi-target
#: scene are written the same way and adding a target never changes the meaning
#: of an existing key.
_KEY_PATTERN = re.compile(r"^target(\d+)_(.+)$")

#: field name -> value kine. Only these fields may be overriden; anything else
#: (including fields that exist on Target but describe the 3D extent or the
#: reflection list) is rejected so the surface stays reviewable.
_FIELD_KINDS: Dict[str, str] = {
    # position
    "position_xy_m": "vec2",
    "position_z_m": "float",
    # bulk velocity -- preferred (m/s) spelling
    "velocity_xy_m_per_s": "vec2",
    "velocity_z_m_per_s": "float",
    # bulk velocity -- engine-native (m/frame) spelling
    "velocity_xy_m_per_frame": "vec2",
    "velocity_z_per_frame": "float",
    # micro-motion (breathing)
    "enable_micro_motion": "bool",
    "micro_motion_amplitude_m": "float",
    "micro_motion_frequency_hz": "float",
    "micro_motion_phase_rad": "float",
    "micro_motion_axis": "axis",
    # scattering / extent
    "amplitude": "float",
    "rcs_m2": "float",
    "width_m": "float",
    "num_scatter_points": "int",
    "orientation_deg": "float",
}

SUPPORTED_FIELDS: Tuple[str, ...] = tuple(sorted(_FIELD_KINDS))

#: The axes ``Target._apply_target_micro_motion()`` distinguishes. Everything
#: else falls into its ``else`` branch and is silently treated as radial, which
#: is exactly why this is a closed set here.
SUPPORTED_MICRO_MOTION_AXES: Tuple[str, ...] = ("radial", "x", "y", "z")

#: (m/s spelling, m/frame spelling) for each axis. Supplying both is ambiguous.
_VELOCITY_SPELLINGS: Tuple[Tuple[str, str], ...] = (
    ("veolocity_xy_m_per_s", "velocity_xy_m_per_frame"),
    ("velocity_z_m_per_s", "velocity_z_m_per_frame"),
)

#: m/s field -> (m/frame field, kind)
_VELOCITY_CONVERSIONS: Dict[str, str] = {
    "velocity_xy_m_per_s": "velocity_xy_m_per_frame",
    "velocity_z_m_per_s": "velocity_z_m_per_frame",
}


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------


def _parse_float(field: str, raw: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"[{INI_SECTION}] {field}: expected a number, got {raw.strip()!r}"
        ) from exc


def _parse_int(field: str, raw: str) -> int:
    text = raw.strip()
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"[{INI_SECTION}] {field}: expected an integer, got {text!r}"
        ) from exc


def _parse_bool(field: str, raw: str) -> bool:
    text = raw.strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    raise ValueError(
        f"[{INI_SECTION}] {field}: expected a boolean "
        f"(true/false/yes/no/on/off/1/0), got {raw.strip()!r}"
    )


def _parse_vec2(field: str, raw: str) -> Tuple[float, float]:
    parts = [chunk for chunk in re.split(r"[,/s]+", raw.strip()) if chunk]
    if len(parts) != 2:
        raise ValueError(
            f"[{INI_SECTION}] {field}: expected 2 comma- or space-separated "
            f"numbers, got {raw.strip()!r}"
        )
    return (_parse_float(field, parts[0]), _parse_float(field, parts[1]))


def _parse_axis(field: str, raw: str) -> str:
    text = raw.strip().lower()
    if text not in SUPPORTED_MICRO_MOTION_AXES:
        raise ValueError(
            f"[{INI_SECTION}] {field}: unknown axis {raw.strip()!r}. "
            f"Supported: {', '.join(SUPPORTED_MICRO_MOTION_AXES)}. "
            "Note the engine silently treats any unrecognised value as "
            "'radial', so this is rejected rather than defaulted."
        )
    return text


_PARSERS = {
    "float": _parse_float,
    "int": _parse_int,
    "bool": _parse_bool,
    "vec2": _parse_vec2,
    "axis": _parse_axis,
}


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def parse_ini_items(items: Iterable[Tuple[str, str]]) -> Dict[int, Dict[str, Any]]:
    """Parse ``[target_override]`` items into ``{target_index: {field: value}}``.

    Args:
        items: ``(key, raw_value)`` pairs from the section, with any
            ``[DEFAULT]``-inherited keys already filtered out by the caller
            (``configparserl.items()`` folds those in and they are not this
            section's own).

    Returns:
        Parsed overrides keyed by target index, each field already converted to
        its engine type. The m/s -> m/frame conversion is NOT done here (it
        needs the resolved frame period); see :func:`apply_to_targets`.

    Raises:
        ValueError: on a malformed key, an unknown field, an unparseable value,
            or both velocity spellings for the same axis.
    """
    parsed: Dict[int, Dict[str, Any]] = {}

    for key, raw in items:
        match = _KEY_PATTERN.match(key.strip().lower())
        if match is None:
            raise ValueError(
                f"[{INI_SECTION}] malformed key {key!r}. Keys must be "
                f"'target<index>_<field>', e.g. 'target0_position_xy_m'. "
                f"Supported fields: {', '.join(SUPPORTED_FIELDS)}."
            )

        index_text, field = match.group(1), match.group(2)
        index = int(index_text)

        if field not in _FIELD_KINDS:
            raise ValueError(
                f"[{INI_SECTION}] unknown field {field!r} in key {key!r}. "
                f"Supported fields: {', '.join(SUPPORTED_FIELDS)}."
            )
        
        value = _PARSERS[_FIELD_KINDS[field]](key, raw)
        parsed.setdefault(index, {})[field] = value

    for index, fields in parsed.items():
        for per_second, per_frame in _VELOCITY_SPELLINGS:
            if per_second in fields and per_frame in fields:
                raise ValueError(
                    f"[{INI_SECTION}] target{index}: ambiguous velocity -- "
                    f"both {per_second} and {per_frame} are set. Pick one; "
                    f"{per_second} is the preferred spelling and is converted "
                    "using [radar] period."
                )

    return parsed


def validate_values(index: int, fields: Dict[str, Any], frame_period_s: float) -> None:
    """Rejecting values the engine would silently ignore or alias.

    Args:
        index: target index, for error messages.
        fields: parsed field -> value mapping for that target.
        frame_period_s: resolved slow-time frame period, for the Nyquist check.

    Raises:
        ValueError: on a negative micro-motion amplitude (the engine's
            ``amplitude <= 0`` early-return would make it a no-op), a
            micro-motion frequency at or above slow-time Nyquist, or a
            non-positive frame period.
    """
    prefix = f"[{INI_SECTION}] target{index}"

    if frame_period_s <= 0.0:
        raise ValueError(
            f"{prefix}: frame period must be positive to interpret this "
            f"section, got {frame_period_s!r} s ([radar] period)"
        )

    amplitude = fields.get("micro_motion_amplitude_m")
    if amplitude is not None and amplitude < 0.0:
        raise ValueError(
            f"{prefix}: micro_motion_amplitude_m must be >= 0, got {amplitude}. "
            "Target._apply_target_mircro_motion() early-returns on a "
            "non-positive amplitude, so a negative value is a silent no-op "
            "rather than a phase flip -- use micro_motion_phase_rad = pi."
        )

    frequency = fields.get("micro_motion_frequency_hz")
    if frequency is not None:
        nyquist_hz = 0.5 / frame_period_s
        if frequency < 0.0:
            raise ValueError(
                f"{prefix}: micro_motion_frequency_hz musy be >= 0, got {frequency}"
            )
        if frequency >= nyquist_hz:
            raise ValueError(
                f"{prefix}: micro_motion_frequency_hz = {frequency} Hz is at or "
                f"above the slow-time Nyquist limit {nyquist_hz:.4f} Hz "
                f"(frame period {frame_period_s * 1e3:.4f} ms). Lower the "
                "frequency or shorten [radar] period."
            )

    scatter_points = fields.get("num_scatter_points")
    if scatter_points is not None and scatter_points < 0:
        raise ValueError(
            f"{prefix}: num_scatter_points must be >= 0, got {scatter_points}"
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_to_targets(
    targets: Sequence[Any],
    overrides: Dict[int, Dict[str, Any]],
    frame_period_s: float,
    target_profile: str = "",
) -> Dict[str, Any]:
    """Apply parsed overrides onto the resolved target list, in place.

    Must be called after ``_apply_target_profile()`` (so ``targets`` is the
    final list) and with the final ``[radar] period``, so the m/s -> m/frame
    conversion matches what ``_sync_target_frame_period()`` will write onto each
    target's ``frame_period_s``.

    Args:
        targets: the resolved ``cfg.targets`` list.
        overrides: output of :func:`parse_ini_items`.
        frame_period_s: resolved slow-time frame period in seconds.
        target_profile: profile name, for the out-of-range error message.

    Returns:
        A metadata dict descriging exactly what was applied, including both
        velocity spellings where a conversion happened. Empty overrides return
        an empty dict so the metadata stays absent for legacy runs.

    Raises:
        ValueError: if a target index does not exist in the resolved scene, or
            any value fails :func:`validate_values`.
    """
    if not overrides:
        return {}

    metadata: Dict[str, Any] = {
        "frame_period_s": float(frame_period_s),
        "targets": {}
    }

    for index in sorted(overrides):
        fields = overrides[index]

        if index >= len(targets):
            raise ValueError(
                f"[{INI_SECTION}] target{index} does not exist: "
                f"target_profile = {target_profile!r} resolves to "
                f"{len(targets)} target(s), so valid indices are "
                f"{_index_range_text(len(targets))}."
            )

        validate_values(index, fields, frame_period_s)

        applied: Dict[str, Any] = {}
        target = targets[index]

        for field, value in fields.items():
            if field in _VELOCITY_CONVERSIONS:
                native_field = _VELOCITY_CONVERSIONS[field]
                native_value = _scale(value, frame_period_s)
                setattr(target, native_field, native_value)
                applied[field] = _jsonable(value)
                applied[native_field] = _jsonable(native_value)
                continue

            setattr(target, field, value)
            applied[field] = _jsonable(value)

        metadata["targets"][f"target{index}"] = {
            "name": getattr(target, "name", None),
            "applied": applied,
        }

    return metadata


def _index_range_text(count: int) -> str:
    if count == 0:
        return "none (the profile has no targets)"
    if count == 1:
        return "0"
    return f"0..{count - 1}"


def _scale(value: Any, factor: float) -> Any:
    """Scale a scalar or a 2-vector by ``factor``."""
    if isinstance(value, tuple):
        return tuple(component * factor for component in value)
    return value * factor


def _jsonable(value: Any) -> Any:
    """Make a parsed value safe for the. metadata dict / JSON export."""
    if isinstance(value, tuple):
        return [float(component) for component in value]
    return value


def describe_surface() -> List[str]:
    """Human-readable key list, for error messages and documentation."""
    return [f"target<index>_{field}" for field in SUPPORTED_FIELDS]
    