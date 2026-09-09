"""Range-Angle scatter plot from wrapper Detection objects."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_range_angle(
    detections,
    run_dir: str | Path,
    show: bool = False,
    save: bool = True,
) -> None:
    valid = [d for d in detections if d.range_cm is not None and d.angle_deg is not None]

    if not valid:
        print("[range_angle] No valid detections to plot")
        return
    
    angles = [d.angle_deg for d in valid]
    ranges = [d.range_cm for d in valid]
    powers = [d.power_db if d.power_db is not None else 0.0 for d in valid]

    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(angles, ranges, c=powers, s=60, edgecolor="k", cmap="viridis")
    fig.colorbar(sc, ax=ax, label="Power [dB]")
    ax.set_title("Detected Targets: Range vs Angle")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Range [cm]")
    ax.grid(True)
    fig.tight_layout()

    plot_dir = Path(run_dir) / "plots"

    if save:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / "range_angle.png", dpi=150)
    
    if show:
        plt.show()

    plt.close(fig)
