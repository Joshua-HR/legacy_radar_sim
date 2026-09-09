"""Save detection results as JSON and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from radar_wrapper.schemas.results import Detection


def save_results(
    detections: list[Detection],
    cir,
    cfg,
    run_dir: str | Path,
) -> tuple[Path, Path]:
    run_dir = Path(run_dir)

    det_dicts = [d.to_dict() for d in detections]

    json_path = run_dir / "detections.json"
    with open(str(json_path), "w", encoding="utf-8") as fh:
        json.dump(det_dicts, fh, indent=2, default=str)

    csv_path = run_dir / "detections.csv"
    write_detection_csv(det_dicts, csv_path)

    return json_path, csv_path


def write_detection_csv(det_dicts: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)

    if not det_dicts:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "frame_idx",
        "segment_idx",
        "range_cm",
        "angle_deg",
        "power_db",
        "doppler_hz",
        "backend",
    ]

    with open(str(output_path), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in det_dicts:
            writer.writerow(row)
            