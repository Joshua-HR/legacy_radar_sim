"""Save CIR data and metadata as NPZ + JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from radar_wrapper.schemas.cir import CIRData


def save_cir_npz(cir: CIRData, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)

    # Main compressed array
    npz_path = run_dir / "cir_data.npz"
    np.savez_compressed(
        str(npz_path),
        cir=cir.data,
        metadata=np.array([cir.metadata.__dict__]),  # pickle-safe container
    )

    # Human-readable metadata
    with open(str(run_dir / "cir_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(cir.metadata.__dict__, fh, indent=2, default=str)

    # Optional truth (synthetic only)
    if cir.truth is not None:
        with open(str(run_dir / "truth.json"), "w", encoding="utf-8") as fh:
            json.dump(cir.truth, fh, indent=2, default=str)

    return run_dir