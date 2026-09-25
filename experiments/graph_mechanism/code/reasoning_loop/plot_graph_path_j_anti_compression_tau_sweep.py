"""Plot raw and all-subthreshold singular-floor accuracy across loop count."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = {
    "baseline": {"label": "raw J", "color": "#111827", "linewidth": 2.8},
    "all_floor0.1": {"label": r"$\tau=0.1$", "color": "#2563eb"},
    "all_floor0.2": {"label": r"$\tau=0.2$", "color": "#0891b2"},
    "all_floor0.3": {"label": r"$\tau=0.3$", "color": "#16a34a"},
    "all_floor0.4": {"label": r"$\tau=0.4$", "color": "#f59e0b"},
    "all_floor0.5": {"label": r"$\tau=0.5$", "color": "#dc2626"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values: dict[str, list[tuple[int, float]]] = {name: [] for name in SERIES}
    for row in rows:
        condition = row["condition"]
        if condition in values:
            values[condition].append((int(row["back_count"]), float(row["accuracy"])))

    expected = list(range(1, 25))
    for condition, points in values.items():
        points.sort()
        loops = [loop for loop, _ in points]
        if loops != expected:
            raise ValueError(f"{condition} has loop grid {loops}, expected {expected}")

    figure, axis = plt.subplots(figsize=(10.5, 6.3), dpi=220)
    for condition, style in SERIES.items():
        points = values[condition]
        axis.plot(
            [loop for loop, _ in points],
            [accuracy for _, accuracy in points],
            label=style["label"],
            color=style["color"],
            linewidth=style.get("linewidth", 2.1),
            marker="o",
            markersize=4.2,
        )

    axis.set_title("Singular-floor dose response over cumulative loops")
    axis.set_xlabel("Cumulative $J$ calls (loop count)")
    axis.set_ylabel("Final accuracy")
    axis.set_xlim(1, 24)
    axis.set_ylim(0.0, 1.025)
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.grid(alpha=0.22)
    axis.legend(loc="lower left", ncol=2, frameon=True, fontsize=9.5)
    axis.text(
        0.995,
        0.025,
        r"All singular values below $\tau$ are lifted to $\tau$" "\n"
        "D8L8 seed0; 5 graph seeds; 128 examples/seed",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color="#374151",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
