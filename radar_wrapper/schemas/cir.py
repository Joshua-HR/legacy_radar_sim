"""Standard CIR schema used across every wrapper component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class CIRMetadata:
    source: str
    num_frames: int
    num_antennas: int
    num_taps: int
    pri_s: float
    fast_time_resolution_ns: float
    channel: str
    center_frequency_hz: Optional[float] = None
    scenario_name: Optional[str] = None
    raw_config: Optional[Any] = None


@dataclass
class CIRData:
    data: np.ndarray
    metadata: CIRMetadata
    truth: Optional[Any] = None

    def validate(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(
                f"CIRData.data must be 3-D (frames, antennas, taps), "
                f"got {self.data.ndim} dimensions"
            )
        frames, antennas, taps = self.data.shape
        if frames != self.metadata.num_frames:
            raise ValueError(
                f"num_frames mismatch: data has {frames}, metadata says {self.metadata.num_frames}"
            )
        if antennas != self.metadata.num_antennas:
            raise ValueError(
                f"num_antennas mismatch: data has {antennas}, metadata says {self.metadata.num_antennas}"
            )
        if taps != self.metadata.num_taps:
            raise ValueError(
                f"num_taps mismatch: data has {taps}, metadata says {self.metadata.num_taps}"
            )
        if not np.iscomplexobj(self.data):
            raise ValueError(
                f"CIRData.data must be complex dtype, got {self.data.dtype}"
            )
            