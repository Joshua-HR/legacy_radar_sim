"""Range-Time scatter plot from wrapper Detection objects."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_range_time(
    detections,
    cir,
    run_dir: str | Path,
    show: bool = False,
    save: bool = True,
) -> None:
    valid = [d for d in detections if d.range_cm is not None]

    if not valid:
        print("[range_time] No valid detection to plot")
        return

    pri_s = cir.metadata.pri_s
    times = [d.frame_idx * pri_s for d in valid]
    ranges = [d.range_cm for d in valid]
    powers = [d.power_db if d.power_db is not None else 0.0 for d in valid]

    fig, ax = plt.subplots(figsize=(10,5))
    sc = ax.scatter(times, ranges, c=powers, s=50, edgecolor="k", cmap="viridis")
    fig.colorbar(sc, ax=ax, label="Power [dB]")
    ax.set_title("Detected Targets: Range vs Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Range [cm]")
    ax.grid(True)
    fig.tight_layout()

    plot_dir = Path(run_dir) / "plots"

    if save:
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / "range_time.png", dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    