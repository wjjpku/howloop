"""Plot long-horizon accuracy for four bottom-singular-channel interventions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITIONS = {
    "bottom4_floor0.1": {
        "label": r"bottom-4 $V$ write directions, $\tau=0.1$",
        "color": "#2563eb",
        "linestyle": "-",
        "marker": "o",
    },
    "bottom8_floor0.1": {
        "label": r"bottom-8 $V$ write directions, $\tau=0.1$",
        "color": "#ea580c",
        "linestyle": "-",
        "marker": "o",
    },
    "bottom4_floor0.5": {
        "label": r"bottom-4 $V$ write directions, $\tau=0.5$",
        "color": "#2563eb",
        "linestyle": "--",
        "marker": "s",
    },
    "bottom8_floor0.5": {
        "label": r"bottom-8 $V$ write directions, $\tau=0.5$",
        "color": "#ea580c",
        "linestyle": "--",
        "marker": "s",
    },
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

    series: dict[str, list[tuple[int, float]]] = {name: [] for name in CONDITIONS}
    for row in rows:
        condition = row["condition"]
        if condition in series:
            series[condition].append((int(row["back_count"]), float(row["accuracy"])))

    expected_loops = list(range(1, 25))
    for condition, values in series.items():
        values.sort()
        loops = [loop for loop, _ in values]
        if loops != expected_loops:
            raise ValueError(f"{condition} has loop grid {loops}, expected {expected_loops}")

    figure, axis = plt.subplots(figsize=(10.5, 6.3), dpi=220)
    for condition, style in CONDITIONS.items():
        values = series[condition]
        axis.plot(
            [loop for loop, _ in values],
            [accuracy for _, accuracy in values],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.4,
            linewidth=2.1,
        )

    axis.set_title("Cumulative anti-compression effect across long loop trajectories")
    axis.set_xlabel("Cumulative $J$ calls (loop count)")
    axis.set_ylabel("Final accuracy")
    axis.set_xlim(1, 24)
    axis.set_ylim(0.0, 1.025)
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.grid(alpha=0.22)
    axis.legend(loc="lower left", frameon=True, fontsize=9)
    axis.text(
        0.995,
        0.025,
        "D8L8 seed0; 5 graph seeds; 128 examples/seed\n"
        "Color = bottom rank; line style = singular-value floor",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
