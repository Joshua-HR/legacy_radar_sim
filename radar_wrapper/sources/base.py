"""Abstract base class for CIR sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from radar_wrapper.schemas.cir import CIRData


class CIRSource(ABC):
    @abstractmethod
    def load(self) -> CIRData:
        """Return synthetic or measured CIR wrapped in the canonical schema."""
        raise NotImplementedError