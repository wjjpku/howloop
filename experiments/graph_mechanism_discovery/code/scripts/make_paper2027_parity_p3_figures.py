#!/usr/bin/env python3
"""Render the audited three-seed P3 phase-causality appendix figure."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


PLOT_CONDITIONS = (
    "phase_rotate_pi_answer",
    "random2d_rotation_pi_answer",
    "phase_answer",
)
LABELS = ("$\\pi$ phase\nrotation", "matched random\n$\\pi$ rotation", "full phase\nreplacement*")
COLORS = ("#c0392b", "#7f8c8d", "#2874a6")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p3-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser.parse_args()


def plot_metric(axis: Axes, rows: list[dict[str, str]], metric: str, title: str) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    for index, (condition, color) in enumerate(zip(PLOT_CONDITIONS, COLORS)):
        selected = sorted(grouped[condition], key=lambda row: int(row["backbone_seed"]))
        values = [float(row[metric]) for row in selected]
        offsets = (-0.08, 0.0, 0.08)
        axis.scatter(
            [index + offsets[i] for i in range(len(values))],
            values,
            color=color,
            s=28,
            zorder=3,
        )
        axis.plot([index - 0.16, index + 0.16], [sum(values) / len(values)] * 2, color=color, lw=2)
    axis.axhline(0.0, color="0.5", lw=0.8)
    axis.axhline(1.0, color="0.75", lw=0.8, ls="--")
    axis.set_xticks(range(len(PLOT_CONDITIONS)), LABELS)
    axis.set_ylim(-0.15, 1.08)
    axis.set_ylabel("recovery toward donor target")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)


def render(p3_dir: Path, figure_dir: Path) -> None:
    summary = json.loads((p3_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete_registered_three_seed_null":
        raise ValueError("P3 summary is not an eligible completed null result")
    causal = read_csv(p3_dir / "causal_conditions.csv")
    skip = read_csv(p3_dir / "skip_and_dynamics.csv")
    if {row["condition"] for row in causal} < set(PLOT_CONDITIONS):
        raise ValueError("P3 causal table lacks a required plotted condition")
    if len(skip) != 3:
        raise ValueError("P3 figure requires exactly three deep backbone rows")

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    plot_metric(axes[0], causal, "phase_translation_recovery", "Phase-coordinate recovery")
    plot_metric(axes[1], causal, "margin_translation_recovery", "Readout-margin recovery")

    for index, (key, label, color) in enumerate(
        (
            ("mlp_skip_fraction_closer_to_previous_phase", "MLP skip", "#1f77b4"),
            ("attention_skip_fraction_closer_to_previous_phase", "attention skip", "#7f7f7f"),
        )
    ):
        values = [float(row[key]) for row in sorted(skip, key=lambda row: int(row["backbone_seed"]))]
        offsets = (-0.08, 0.0, 0.08)
        axes[2].scatter([index + offsets[i] for i in range(len(values))], values, color=color, s=28, zorder=3)
        axes[2].plot([index - 0.16, index + 0.16], [sum(values) / len(values)] * 2, color=color, lw=2)
    axes[2].set_xticks((0, 1), ("MLP skip", "attention skip"))
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("fraction closer to preceding clean phase")
    axes[2].set_title("Within-call phase-role diagnostic")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("P3 held-out phase interventions (three frozen input-once backbones)", fontsize=11)
    fig.text(
        0.5,
        -0.035,
        "* Full replacement edits the answer-token residual and is a positive patching control, not causal evidence for the plane.",
        ha="center",
        fontsize=8,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"parity_p3_phase_causality.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    render(args.p3_dir, args.figure_dir)


if __name__ == "__main__":
    main()
