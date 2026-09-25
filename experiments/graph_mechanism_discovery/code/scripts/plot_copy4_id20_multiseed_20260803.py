#!/usr/bin/env python3
"""Aggregate the three Copy4 J(1--20) runs and plot their horizon curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    values: dict[str, dict[int, dict[str, Any]]] = {"raw": {}, "full": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant not in values:
                continue
            length = int(row["length"])
            values[variant][length] = {
                "examples": int(row["examples"]),
                "successes": int(row["exact_successes"]),
                "em": float(row["exact_match"]),
                "ce": float(row["answer_cross_entropy"]),
                "margin": float(row["mean_sequence_min_margin"]),
            }
    return values


def aggregate_range(
    runs: list[dict[str, dict[int, dict[str, Any]]]],
    variant: str,
    lower: int,
    upper: int,
) -> dict[str, int | float]:
    rows = [
        run[variant][length]
        for run in runs
        for length in sorted(run[variant])
        if lower <= length <= upper
    ]
    successes = sum(row["successes"] for row in rows)
    examples = sum(row["examples"] for row in rows)
    return {
        "length_seed_points": len(rows),
        "successes": successes,
        "examples": examples,
        "micro_em": successes / examples,
    }


def first_below(
    rows: dict[int, dict[str, Any]], threshold: float, minimum: int = 20
) -> int | None:
    for length in sorted(rows):
        if length >= minimum and rows[length]["em"] < threshold:
            return length
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, action="append", required=True)
    parser.add_argument("--anchor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.dense) != 3 or len(args.anchor) != 3:
        raise ValueError("exactly three dense and three anchor CSVs are required")

    dense = [read_csv(path) for path in args.dense]
    anchors = [read_csv(path) for path in args.anchor]
    lengths = sorted(dense[0]["raw"])
    if any(sorted(run["raw"]) != lengths or sorted(run["full"]) != lengths for run in dense):
        raise ValueError("dense length grids differ")
    x = np.asarray(lengths)
    matrices = {
        variant: np.asarray(
            [[run[variant][length]["em"] for length in lengths] for run in dense]
        )
        for variant in ("raw", "full")
    }
    delta = matrices["full"] - matrices["raw"]

    colors = {"raw": "#bd5038", "full": "#087e8b"}
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), sharex=True)
    ax = axes[0]
    ax.axvspan(1, 20, color=colors["full"], alpha=0.07, label="J train range n=1–20")
    for variant, label in (("raw", "Raw backbone"), ("full", "J(1–20), 5,376 steps")):
        mean = matrices[variant].mean(axis=0)
        low = matrices[variant].min(axis=0)
        high = matrices[variant].max(axis=0)
        ax.fill_between(x, low, high, color=colors[variant], alpha=0.12)
        ax.plot(x, mean, color=colors[variant], lw=1.8, label=f"{label}, 3-seed mean")
    ax.axhline(0.5, color="#777", lw=0.8, ls=":")
    ax.set_ylabel("Strict sequence EM")
    ax.set_ylim(-0.035, 1.035)
    ax.set_title("Copy4: does J trained only on n=1–20 move the horizon?")
    ax.grid(axis="y", color="#ddd", lw=0.7)
    ax.legend(loc="lower left", fontsize=8.7)

    ax = axes[1]
    ax.axvspan(1, 20, color=colors["full"], alpha=0.07)
    ax.axhline(0, color="#555", lw=0.8)
    for seed in range(3):
        ax.plot(x, delta[seed], lw=0.9, alpha=0.38, label=f"seed {seed}" if seed == 0 else None)
    ax.plot(x, delta.mean(axis=0), color=colors["full"], lw=1.8, label="3-seed mean gain")
    ax.set_ylabel("EM gain (J − raw)")
    ax.set_xlabel("Logical length n = target loop count")
    ax.set_xlim(1, 100)
    ax.set_ylim(-1.035, 1.035)
    ax.grid(axis="y", color="#ddd", lw=0.7)
    ax.legend(loc="lower right", fontsize=8.7)
    fig.text(
        0.01,
        0.008,
        "Three official Copy4 backbone seeds; rank-48 diagonal+low-rank J; identity initialization; "
        "anchor=1; post-final J; final answer CE; 64 matched examples per length and seed.",
        fontsize=8.1,
        color="#444",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.095, hspace=0.14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, Any] = {
        "task": "copy4",
        "backbone_seeds": [0, 1, 2],
        "controller_training_lengths": [1, 20],
        "controller_updates": 5376,
        "dense_examples_per_length_per_seed": 64,
        "anchor_examples_per_length_per_seed": 512,
        "dense_ranges": {},
        "first_below_threshold_from_n20": {},
        "selected_dense_lengths": {},
        "selected_anchor_lengths": {},
    }
    for lower, upper in ((1, 20), (21, 30), (31, 40), (41, 50), (51, 100)):
        summary["dense_ranges"][f"{lower}-{upper}"] = {
            variant: aggregate_range(dense, variant, lower, upper)
            for variant in ("raw", "full")
        }
    for threshold in (0.95, 0.90, 0.75, 0.50):
        summary["first_below_threshold_from_n20"][f"{threshold:.2f}"] = {
            variant: [first_below(run[variant], threshold) for run in dense]
            for variant in ("raw", "full")
        }
    for length in (20, 25, 30, 35, 40, 45, 50, 60, 75, 100):
        summary["selected_dense_lengths"][str(length)] = {
            variant: [run[variant][length]["em"] for run in dense]
            for variant in ("raw", "full")
        }
    for length in (20, 25, 30, 35, 40, 45, 50, 60):
        summary["selected_anchor_lengths"][str(length)] = {
            variant: {
                "successes": sum(run[variant][length]["successes"] for run in anchors),
                "examples": sum(run[variant][length]["examples"] for run in anchors),
                "micro_em": sum(run[variant][length]["successes"] for run in anchors)
                / sum(run[variant][length]["examples"] for run in anchors),
            }
            for variant in ("raw", "full")
        }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
