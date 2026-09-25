"""Plot the gain-dose response for the weakest J singular channels."""

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
    targets = (0.1, 0.25, 0.5, 0.75, 1.0)

    def true_value(k: int, rank: int, target: float, key: str) -> float:
        suffix = f"{target:g}"
        name = f"bottom{rank}_floor{suffix}"
        row = next(
            value
            for value in rows
            if int(value["back_count"]) == k and value["condition"] == name
        )
        return float(row[key])

    def random_value(k: int, rank: int, target: float, key: str) -> float:
        selected = [
            value
            for value in rows
            if int(value["back_count"]) == k
            and value["mode"] == "random_matched"
            and int(value["rank"]) == rank
            and float(value["target_singular_floor"]) == target
        ]
        return float(np.mean([float(value[key]) for value in selected]))

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=190)
    for rank, color in ((4, "tab:blue"), (8, "tab:orange")):
        axes[0, 0].plot(
            targets,
            [true_value(1, rank, target, "post_J_age_signed_error") for target in targets],
            marker="o",
            color=color,
            label=f"true bottom-{rank}",
        )
        axes[0, 0].plot(
            targets,
            [random_value(1, rank, target, "post_J_age_signed_error") for target in targets],
            marker="s",
            linestyle="--",
            color=color,
            label=f"random-{rank} matched",
        )
        axes[0, 1].plot(
            targets,
            [true_value(1, rank, target, "post_J_probe_closer_to_source_fraction") for target in targets],
            marker="o",
            color=color,
            label=f"true bottom-{rank}",
        )
        axes[0, 1].plot(
            targets,
            [random_value(1, rank, target, "post_J_probe_closer_to_source_fraction") for target in targets],
            marker="s",
            linestyle="--",
            color=color,
            label=f"random-{rank} matched",
        )
        axes[1, 0].plot(
            targets,
            [true_value(8, rank, target, "accuracy") for target in targets],
            marker="o",
            color=color,
            label=f"true bottom-{rank}",
        )
        axes[1, 0].plot(
            targets,
            [random_value(8, rank, target, "accuracy") for target in targets],
            marker="s",
            linestyle="--",
            color=color,
            label=f"random-{rank} matched",
        )
        axes[1, 1].plot(
            targets,
            [true_value(1, rank, target, "answer_rms") for target in targets],
            marker="o",
            color=color,
            label=f"true bottom-{rank}",
        )
        axes[1, 1].plot(
            targets,
            [random_value(1, rank, target, "answer_rms") for target in targets],
            marker="s",
            linestyle="--",
            color=color,
            label=f"random-{rank} matched",
        )
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set(
        title="One J: old-age signal passes through",
        xlabel="singular-value floor",
        ylabel="predicted age - target age",
    )
    axes[0, 1].set(
        title="One J: state resembles source age",
        xlabel="singular-value floor",
        ylabel="fraction closer to source age",
        ylim=(0, 1.0),
    )
    axes[1, 0].set(
        title="Eight J calls: phase error compounds",
        xlabel="singular-value floor",
        ylabel="final accuracy",
        ylim=(0, 1.03),
    )
    axes[1, 1].set(
        title="One J: answer-state norm remains stable",
        xlabel="singular-value floor",
        ylabel="answer-token RMS",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
