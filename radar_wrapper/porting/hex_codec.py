"""Hex / Q8.8 codec - used for round-trip verification and optional hex replay."""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Saturate helpers
# ---------------------------------------------------------------------------

def saturate_int16(x: int) -> int:
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return x


# ---------------------------------------------------------------------------
# complex <-> hex
# ---------------------------------------------------------------------------

def complex_to_q8_8_hex(
    value: complex,
    scale: int = 256,
    overflow: str = "saturate",
) -> str:
    """Convert a complex value to an 8-char hex string representing two Q8.8 words.

    Format: ``<4 hex digits real><4 hex digits imag>``
    """
    real_q = int(round(value.real * scale))
    imag_q = int(round(value.imag * scale))

    if overflow == "saturate":
        real_q = saturate_int16(real_q)
        imag_q = saturate_int16(imag_q)
    elif overflow != "wrap":
        raise ValueError(f"Unsupported overflow mode: {overflow!r}")

    real_u = real_q & 0xFFFF
    imag_u = imag_q & 0xFFFF

    return f"{real_u:04x}{imag_u:04x}"


def frame_to_hex_string(
    frame: np.ndarray,
    scale: int = 256,
    overflow: str = "saturate",
) -> str:
    """Convert a 1-D complex array to space-separated hex words."""
    words: List[str] = []
    for val in frame:
        words.append(complex_to_q8_8_hex(val, scale=scale, overflow=overflow))
    return " ".join(words)


def q8_8_hex_to_complex(word: str, scale: int = 256) -> complex:
    """Decode a single 8-char hex word back to a complex value.

    The word must contain exactly 8 hex characters: first 4 for real, last 4 for imag.
    """
    real_hex = word[0:4]
    imag_hex = word[4:8]

    real_i = int(real_hex, 16)
    imag_i = int(imag_hex, 16)

    # 16-bit two's complement
    if real_i >= 0x8000:
        real_i -= 0x10000
    if imag_i >= 0x8000:
        imag_i -= 0x10000

    return complex(real_i / scale, imag_i / scale)