"""Root entry-point for the radar replay pieplione

Usage::

    python run_radar_pipeline.py --config configs/pipelines/synthetic_mx_smoke.ini
"""

from __future__ import annotations

from pathlib import Path

from radar_wrapper.config import load_config, parse_args
from radar_wrapper.pipeline import RadarReplayPipeline


def main() -> None:
    args = parse_args()

    cfg = load_config(
        config_path=args.config,
        cli_args=args,
    )

    pipeline = RadarReplayPipeline(cfg)
    cir, detections = pipeline.run()

    print("===Radar replay finished===")
    print(f"CIR shape   : {cir.data.shape}")
    print(f"CIR dtype   : {cir.data.dtype}")
    print(f"Num frames  : {cir.metadata.num_frames}")
    print(f"Num antennas: {cir.metadata.num_antennas}")
    print(f"Num taps    : {cir.metadata.num_taps}")
    print(f"PRI         : {cir.metadata.pri_s * 1000.0:.1f} ms")
    print(f"Num detections: {len(detections)}")
    print(f"Run directory: {pipeline.run_dir}")


if __name__ == "__main__":
    main()
