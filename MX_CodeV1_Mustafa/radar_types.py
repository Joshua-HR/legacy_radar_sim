from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Detection:
    segment_id: int
    sequence_id: int
    range_cm: float
    velocity_mps: float
    doppler_hz: float
    angle_deg: float | None
    snr_db: float
    power_db: float
    tap_index: int
    doppler_index: int
    rx_balance_db: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroupedTarget:
    segment_id: int
    range_cm: float
    velocity_mps: float
    angle_deg: float | None
    snr_db: float
    power_db: float
    detections_count: int
    score: float
    human_like: bool
    unique_taps: int = 0
    range_span_cm: float = 0.0
    doppler_span_hz: float = 0.0
    doppler_hz: float = 0.0
    doppler_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentResult:
    segment_id: int
    last_sequence_id: int
    packets_seen: int
    complete_pairs: int
    detections: list[Detection]
    targets: list[GroupedTarget]
    human_detected: bool
    noise_floor_db: float
    max_power_db: float
    rx_health: dict[str, Any] = field(default_factory=dict)

    object_change_detected: bool = False
    motion_detected: bool = False
    person_candidate: bool = False
    state: str = "CLEAR"
    static_detection: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "last_sequence_id": self.last_sequence_id,
            "packets_seen": self.packets_seen,
            "complete_pairs": self.complete_pairs,
            "human_detected": self.human_detected,
            "object_change_detected": self.object_change_detected,
            "motion_detected": self.motion_detected,
            "person_candidate": self.person_candidate,
            "state": self.state,
            "static_detection": self.static_detection,
            "noise_floor_db": self.noise_floor_db,
            "max_power_db": self.max_power_db,
            "rx_health": self.rx_health,
            "detections": [d.to_dict() for d in self.detections],
            "targets": [t.to_dict() for t in self.targets],
        }