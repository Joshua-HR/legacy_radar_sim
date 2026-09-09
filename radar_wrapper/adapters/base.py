"""Abstract base class for postprocessing adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from radar_wrapper.schemas.cir import CIRData


class PostprocessingAdapter(ABC):
    @abstractmethod
    def configure(self, cir: CIRData) -> None:
        """Initialise the backend processor using CIR metadata and config."""
        raise NotImplementedError

    @abstractmethod
    def run_replay(self, cir: CIRData):
        """Feed all CIR frames through the backend and return Detection list."""
        raise NotImplementedError