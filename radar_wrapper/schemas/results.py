"""Backend-independent detection schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Detection:
    frame_idx: int
    segment_idx: int
    range_cm: Optional[float]
    angle_deg: Optional[float]
    power_db: Optional[float]
    doppler_hz: Optional[float] = None
    backend: str = ""
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)