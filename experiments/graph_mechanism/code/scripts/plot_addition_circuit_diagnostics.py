#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_readout_heatmaps(rows: list[dict[str, str]], out_dir: Path) -> None:
    lengths = sorted({int(row["length"]) for row in rows})
    steps = list(range(1, max(int(row["step"]) for row in rows) + 1))
    variants = ("raw", "J")
    metrics = (
        ("actual_answer_exact_match", "actual sum-digit EM"),
        ("actual_answer_token_accuracy", "actual sum-digit accuracy"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row_index, (metric, metric_label) in enumerate(metrics):
        for column_index, variant in enumerate(variants):
            matrix = np.full((len(lengths), len(steps)), np.nan)
            for row in rows:
                if row["variant"] != variant:
                    continue
                length_index = lengths.index(int(row["length"]))
                step_index = steps.index(int(row["step"]))
                matrix[length_index, step_index] = float(row[metric])
            axis = axes[row_index, column_index]
            image = axis.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            axis.set_title(f"{variant}: {metric_label}")
            axis.set_xlabel("readout after loop t")
            axis.set_ylabel("logical length n")
            axis.set_xticks(range(len(steps)), labels=steps)
            axis.set_yticks(range(len(lengths)), labels=lengths)
            for length_index, length in enumerate(lengths):
                target_step = length + 1
                if target_step in steps:
                    axis.scatter(
                        [steps.index(target_step)],
                        [length_index],
                        marker="s",
                        facecolors="none",
                        edgecolors="white",
                        linewidths=1.4,
                        s=65,
                    )
            figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    figure.suptitle(
        "Fixed-n10 Addition readout dynamics: white square marks T(n)=n+1",
        fontsize=14,
    )
    figure.savefig(out_dir / "readout_dynamics.png", dpi=180)
    figure.savefig(out_dir / "readout_dynamics.pdf")
    plt.close(figure)


def plot_carry_recovery(rows: list[dict[str, str]], out_dir: Path) -> None:
    selected_groups = (
        ("operand_chain_slots", "operand carry-chain slots"),
        ("target_answer_slot", "target answer slot"),
        ("all_actual_answer_slots", "all answer slots"),
    )
    colors = {1: "#4c78a8", 3: "#f58518", 5: "#54a24b"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (group, label) in zip(axes, selected_groups, strict=True):
        for span in (1, 3, 5):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["position_group"] == group
                    and row["donor_mode"] == "paired"
                    and int(row["carry_span"]) == span
                ),
                key=lambda row: int(row["completed_steps"]),
            )
            axis.plot(
                [int(row["completed_steps"]) for row in selected],
                [float(row["normalized_logit_recovery"]) for row in selected],
                marker="o",
                markersize=3,
                color=colors[span],
                label=f"carry span {span}",
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_title(label)
        axis.set_xlabel("patch boundary after loop")
        axis.set_ylabel("normalized clean-logit recovery")
        axis.set_xticks(range(1, 12))
        axis.set_ylim(-0.1, 1.1)
    axes[0].legend(frameon=False)
    figure.suptitle("Clean-carry activation moves from operand chain into the target answer slot")
    figure.savefig(out_dir / "carry_state_migration.png", dpi=180)
    figure.savefig(out_dir / "carry_state_migration.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_readout_heatmaps(read_csv(args.trajectory), args.out_dir)
    plot_carry_recovery(read_csv(args.patching), args.out_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--patching", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
