"""Create the compact mechanism figure for the J anti-compression experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.metrics.open(encoding="utf-8")))

    def value(k: int, condition: str, key: str) -> float:
        row = next(
            item
            for item in rows
            if int(item["back_count"]) == k and item["condition"] == condition
        )
        return float(row[key])

    ks = (1, 8, 12)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=190)

    age_conditions = (
        ("baseline", "baseline"),
        ("bottom4_zero", "bottom-4 -> 0"),
        ("bottom4_floor1", "bottom-4 -> 1"),
        ("random4_floor1_d0", "random-4 matched"),
        ("bottom8_floor1", "bottom-8 -> 1"),
        ("random8_floor1_d0", "random-8 matched"),
    )
    labels = [label for _, label in age_conditions]
    signed = [value(1, name, "post_J_age_signed_error") for name, _ in age_conditions]
    axes[0, 0].bar(np.arange(len(labels)), signed)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axes[0, 0].set(
        title="One J: age readout immediately after rollback",
        ylabel="predicted age - target age",
    )

    for name, label, marker in (
        ("baseline", "baseline", "o"),
        ("bottom4_floor1", "lift true bottom-4", "s"),
        ("random4_floor1_d0", "random-4 matched", "^"),
        ("bottom8_floor1", "lift true bottom-8", "D"),
        ("random8_floor1_d0", "random-8 matched", "v"),
    ):
        axes[0, 1].plot(
            ks,
            [value(k, name, "accuracy") for k in ks],
            marker=marker,
            label=label,
        )
    axes[0, 1].set(
        title="Internal phase error compounds across J reuse",
        xlabel="number of J calls",
        ylabel="final accuracy",
        ylim=(0, 1.03),
        xticks=ks,
    )
    axes[0, 1].legend(fontsize=8)

    for rank, marker in ((4, "o"), (8, "s"), (16, "^")):
        axes[1, 0].plot(
            ks,
            [value(k, f"bottom{rank}_zero", "accuracy") for k in ks],
            marker=marker,
            label=f"set bottom-{rank} to zero",
        )
    axes[1, 0].set(
        title="Exact-zero sanity check",
        xlabel="number of J calls",
        ylabel="final accuracy",
        ylim=(0, 1.03),
        xticks=ks,
    )
    axes[1, 0].legend(fontsize=8)

    for name, label, marker in (
        ("baseline", "baseline", "o"),
        ("bottom16_floor0.5", "true bottom-16 floor=0.5", "s"),
    ):
        axes[1, 1].plot(
            ks,
            [value(k, name, "accuracy") for k in ks],
            marker=marker,
            label=label,
        )
    random_names = [f"random16_floor0.5_d{draw}" for draw in range(3)]
    axes[1, 1].plot(
        ks,
        [
            float(np.mean([value(k, name, "accuracy") for name in random_names]))
            for k in ks
        ],
        marker="^",
        label="random rank-16, same delta norm",
    )
    axes[1, 1].set(
        title="Moderate anti-compression is direction-selective",
        xlabel="number of J calls",
        ylabel="final accuracy",
        ylim=(0, 1.03),
        xticks=ks,
    )
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
