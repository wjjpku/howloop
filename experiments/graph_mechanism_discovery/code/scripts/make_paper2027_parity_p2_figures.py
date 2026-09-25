#!/usr/bin/env python3
"""Render appendix figures for the preregistered Parity P2 gate and fits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=260)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def gate_plot(rows: list[dict[str, str]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.7, 2.9), constrained_layout=True)
    seeds = [int(row["seed"]) for row in rows]
    first = [float(row["first_diseased_length"]) if row["first_diseased_length"] else np.nan for row in rows]
    colors = ["#d62728" if np.isfinite(value) else "#7f7f7f" for value in first]
    axis.scatter(seeds, np.nan_to_num(first, nan=0), color=colors, s=60)
    for seed, value, row in zip(seeds, first, rows, strict=True):
        text = f"n={int(value)}" if np.isfinite(value) else "no disease"
        axis.annotate(text, (seed, 0 if not np.isfinite(value) else value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    axis.set(xlabel="prespecified deep backbone seed", ylabel="first prospective disease length", xticks=seeds, ylim=(-8, max([value for value in first if np.isfinite(value)] + [10]) + 25))
    axis.grid(axis="y", alpha=.2)
    save(figure, path)


def horizon_plot(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["controller"]].append(row)
    figure, axis = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    style = {
        "raw": ("#222222", "raw frozen backbone", 2.3),
        "rank48_seed1": ("#0072B2", "rank-48 fit 1", 1.8),
        "rank48_seed2": ("#56B4E9", "rank-48 fit 2", 1.5),
        "rank128_seed1": ("#009E73", "rank-128 fit 1", 1.4),
        "rank128_seed2": ("#8CCB9B", "rank-128 fit 2", 1.2),
        "dense_seed1": ("#D55E00", "dense fit 1", 1.4),
        "dense_seed2": ("#E69F00", "dense fit 2", 1.2),
    }
    for label, (color, readable, linewidth) in style.items():
        if label not in grouped:
            continue
        series = sorted(grouped[label], key=lambda item: int(item["length"]))
        axis.plot(
            [int(item["length"]) for item in series],
            [float(item["exact_match"]) for item in series],
            marker="o", markersize=3.5, linewidth=linewidth, color=color,
            label=readable,
        )
    axis.set(xlabel="logical length; registered readout T(n)=n", ylabel="exact sequence accuracy", ylim=(-.03, 1.03))
    axis.grid(alpha=.2); axis.legend(frameon=False)
    save(figure, path)


def components_plot(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    # Every registered component control is shown at its own largest common
    # evaluation length.  The caption must state this summary convention; it
    # is not a substitute for the full horizon trajectory above.
    labels = sorted(grouped)
    values = []
    for label in labels:
        # Use the largest registered length for a compact component comparison.
        selected = max(grouped[label], key=lambda item: int(item["length"]))
        values.append(float(selected["exact_match"]))
    figure, axis = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
    colors = ["#999999" if label == "full_executor_off" else "#D55E00" for label in labels]
    axis.bar(range(len(labels)), values, color=colors)
    axis.set(xticks=range(len(labels)), xticklabels=labels, ylabel="exact match at largest P2 evaluation length", ylim=(-.03, 1.03))
    axis.tick_params(axis="x", rotation=30); axis.grid(axis="y", alpha=.2)
    save(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.aggregate_dir / "summary.json").read_text(encoding="utf-8"))
    gate_plot(read_csv(args.aggregate_dir / "gate_table.csv"), args.figure_dir / "parity_p2_prospective_gate.png")
    if summary["disease_positive_backbones"]:
        # Include all six preregistered controller fits.  Capacity rows omit
        # raw because raw is already identically recorded in the primary file.
        horizon_rows = read_csv(args.aggregate_dir / "primary_horizon.csv") + read_csv(args.aggregate_dir / "capacity_horizon.csv")
        horizon_plot(horizon_rows, args.figure_dir / "parity_p2_all_capacity_horizon.png")
        components_plot(read_csv(args.aggregate_dir / "component_horizon.csv"), args.figure_dir / "parity_p2_component_controls.png")


if __name__ == "__main__":
    main()
