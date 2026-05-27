#!/usr/bin/env python3
"""Small pie chart of OpenHands resolution status across all models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["resolved", "unresolved", "unsubmitted"]
COLORS = {
    "resolved": "#4daf4a",
    "unresolved": "#e41a1c",
    "unsubmitted": "#999999",
}
LABELS = {
    "resolved": "Resolved",
    "unresolved": "Unresolved",
    "unsubmitted": "Unsubmitted",
}


def load_resolution_counts(analysis_dir: Path) -> pd.Series:
    frames = [
        pd.read_csv(path)
        for path in sorted(analysis_dir.glob("*/trajectory_metrics.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No trajectory_metrics.csv under {analysis_dir}")
    counts = pd.concat(frames, ignore_index=True)["resolution"].value_counts()
    return counts.reindex(ORDER).fillna(0).astype(int)


def plot_resolution_pie(counts: pd.Series, output: Path) -> None:
    labels = [LABELS[k] for k in counts.index]
    sizes = counts.values
    colors = [COLORS[k] for k in counts.index]

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{int(round(pct * sizes.sum() / 100))}",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
        t.set_color("white")

    ax.set_title("OpenHands resolution status", fontsize=11, pad=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    for k, v in counts.items():
        print(f"  {LABELS[k]}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/OpenHands/analysis"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/openhands/resolution_status_pie.png"),
    )
    args = parser.parse_args()
    counts = load_resolution_counts(args.analysis_dir)
    plot_resolution_pie(counts, args.output)


if __name__ == "__main__":
    main()
