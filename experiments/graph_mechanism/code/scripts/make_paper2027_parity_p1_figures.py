#!/usr/bin/env python3
"""Render corrected input-once Parity P1 figures from registered artifacts."""

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
    figure.savefig(path, dpi=260); figure.savefig(path.with_suffix(".pdf")); plt.close(figure)


def endpoint_figure(rows: list[dict[str, str]], summary: dict, path: Path) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["length"])].append(float(row["exact_match"]))
    lengths = np.asarray(sorted(grouped))
    values = [np.asarray(grouped[int(length)]) for length in lengths]
    statistics = summary["statistics"]
    mean = np.asarray([statistics[f"raw_exact_match_length_{int(length)}"]["estimate"] for length in lengths])
    low = np.asarray([statistics[f"raw_exact_match_length_{int(length)}"]["ci_low"] for length in lengths])
    high = np.asarray([statistics[f"raw_exact_match_length_{int(length)}"]["ci_high"] for length in lengths])
    figure, axis = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    axis.plot(lengths, mean, color="#1f77b4", marker="o", label="raw input-once Parity")
    axis.fill_between(lengths, low, high, color="#1f77b4", alpha=.18, label="nested 95% bootstrap CI")
    axis.axvspan(1, 20, color="black", alpha=.06, label="training lengths")
    axis.set(xlabel="logical length n; readout at T(n)=n", ylabel="exact sequence accuracy", ylim=(-.03, 1.03))
    axis.grid(alpha=.2); axis.legend(frameon=False, fontsize=8)
    save(figure, path)


def phase_figure(rows: list[dict[str, str]], path: Path) -> None:
    seeds = [int(row["seed"]) for row in rows]
    period = np.asarray([float(row["period_calls"]) for row in rows])
    energy = np.asarray([float(row["shared_phase_plane_energy_fraction"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
    axes[0].axhline(4, color="black", linestyle="--", linewidth=1)
    axes[0].scatter(seeds, period, color="#d62728")
    axes[0].set(xlabel="deep backbone seed", ylabel="held-out fitted period (calls)")
    axes[0].grid(alpha=.2)
    axes[1].scatter(seeds, energy, color="#2ca02c", label="phase-plane energy")
    axes[1].scatter(seeds, [float(row["transition_r2"]) for row in rows], color="#9467bd", label="one-step transition $R^2$")
    axes[1].set(xlabel="deep backbone seed", ylabel="held-out metric", ylim=(-.03, 1.03))
    axes[1].grid(alpha=.2); axes[1].legend(frameon=False, fontsize=8)
    save(figure, path)


def results_tex(summary: dict, path: Path) -> None:
    result = summary["statistics"]
    endpoint = result.get("raw_exact_match_length_100", {"estimate": float("nan")})
    auc = result["raw_exact_match_auc_lengths_24_to_500"]
    macros = {
        "ParityConfirmatoryBackbones": str(len(summary["backbone_seeds"])),
        "ParityRawLengthHundredAccuracy": f"{endpoint['estimate']:.3f}",
        "ParityRawAUC": f"{auc['estimate']:.3f}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--results-tex", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.aggregate_dir / "summary.json").read_text(encoding="utf-8"))
    endpoint_figure(read_csv(args.aggregate_dir / "endpoint_by_seed.csv"), summary, args.figure_dir / "parity_p1_input_once_horizon.png")
    phase_figure(read_csv(args.aggregate_dir / "phase_deep_seed_table.csv"), args.figure_dir / "parity_p1_phase_deep_seeds.png")
    results_tex(summary, args.results_tex)


if __name__ == "__main__":
    main()
