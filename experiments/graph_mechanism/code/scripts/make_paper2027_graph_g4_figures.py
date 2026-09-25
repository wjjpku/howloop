#!/usr/bin/env python3
"""Render the G4 main and appendix figures from final-lock cluster tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def call_metric(rows: list[dict[str, str]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        denominator = float(row["strict_examples"] if metric.startswith("strict_") else row["examples"])
        if denominator:
            grouped[int(row["call"])].append(float(row[metric]) / denominator)
    return {call: float(np.mean(values)) for call, values in grouped.items()}


def mean_and_seed_band(per_seed: list[dict[int, float]], calls: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray([[values.get(call, np.nan) for call in calls] for values in per_seed], dtype=float)
    mean = np.nanmean(array, axis=0)
    low = np.nanquantile(array, .025, axis=0) if len(per_seed) > 1 else mean
    high = np.nanquantile(array, .975, axis=0) if len(per_seed) > 1 else mean
    return mean, low, high


def mean_replicas(replicas: list[dict[int, float]], calls: list[int]) -> dict[int, float]:
    """Average nested controller fits before forming a backbone-level band.

    A controller replica is an optimisation repeat within one frozen backbone,
    not an additional independently trained backbone.  The figure therefore
    represents one curve per backbone here.  The companion aggregate uses the
    registered backbone -> replica -> graph bootstrap for inferential CIs.
    """
    if not replicas:
        raise ValueError("need at least one controller replica")
    # Strict successor excludes call 8 because the target coincides with the
    # endpoint there.  Keep that undefined point absent rather than turning it
    # into an artificial zero or failing figure generation.
    shared_calls = set(calls)
    for replica in replicas:
        shared_calls &= set(replica)
    return {
        call: float(np.mean([replica[call] for replica in replicas]))
        for call in sorted(shared_calls)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dir",
        default="g4_evaluation",
        help="named evaluation subtree paired with the aggregate",
    )
    parser.add_argument("--max-call", type=int, default=64)
    args = parser.parse_args()
    evaluation_root = args.root / args.evaluation_dir
    seeds = sorted(int(path.parents[1].name.removeprefix("seed")) for path in evaluation_root.glob("seed*/raw/permutation_clusters.csv"))
    if not seeds:
        raise ValueError("no G4 raw evaluations found")
    calls = list(range(1, args.max_call + 1))
    raw_successor, full_successor = [], []
    raw_hold, full_hold = [], []
    controls: dict[str, list[dict[int, float]]] = defaultdict(list)
    modes = ("executor_off", "batch_shuffle", "D_only", "no_AB", "identity_D", "AB_only", "no_bias", "mean_D", "shuffle_D", "spectrum_matched_random_delta")
    for seed in seeds:
        base = evaluation_root / f"seed{seed}"
        raw_rows = read_csv(base / "raw" / "permutation_clusters.csv")
        raw_successor.append(call_metric(raw_rows, "strict_successor_correct"))
        raw_hold.append(call_metric(raw_rows, "endpoint_hold_correct"))
        replica_successor, replica_hold = [], []
        for replica in (1, 2):
            rows = read_csv(base / f"rank48_seed{replica}" / "full" / "permutation_clusters.csv")
            replica_successor.append(call_metric(rows, "strict_successor_correct"))
            replica_hold.append(call_metric(rows, "endpoint_hold_correct"))
        full_successor.append(mean_replicas(replica_successor, calls))
        full_hold.append(mean_replicas(replica_hold, calls))
        for mode in modes:
            controls[mode].append(call_metric(read_csv(base / "rank48_seed1" / mode / "permutation_clusters.csv"), "strict_successor_correct"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.65), sharex=True, sharey=True, constrained_layout=True)
    for axis, raw, full, title in ((axes[0], raw_successor, full_successor, "Strict successor"), (axes[1], raw_hold, full_hold, "Endpoint hold")):
        for values, label, color in ((raw, "raw frozen backbone", "#222222"), (full, "D+AB+b controller", "#0072B2")):
            mean, low, high = mean_and_seed_band(values, calls)
            axis.plot(calls, mean, color=color, linewidth=2.2, label=label)
            axis.fill_between(calls, low, high, color=color, alpha=.16, linewidth=0)
        axis.axvline(8, color="#555555", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="recurrent call t", ylim=(-.03, 1.03), xlim=(1, args.max_call))
        axis.grid(alpha=.18)
    axes[0].set_ylabel("rate (seed 2.5–97.5% band)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(args.out_dir / "g4_final_lock_raw_vs_controller.png", dpi=260)
    figure.savefig(args.out_dir / "g4_final_lock_raw_vs_controller.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    key_modes = ("full", "executor_off", "batch_shuffle", "D_only", "no_AB", "identity_D", "AB_only")
    palette = {"full": "#0072B2", "executor_off": "#999999", "batch_shuffle": "#D55E00", "D_only": "#009E73", "no_AB": "#CC79A7", "identity_D": "#E69F00", "AB_only": "#56B4E9"}
    values_by_mode: dict[str, list[dict[int, float]]] = {"full": full_successor, **controls}
    for mode in key_modes:
        mean, _, _ = mean_and_seed_band(values_by_mode[mode], calls)
        axis.plot(calls, mean, color=palette[mode], linewidth=2.0 if mode == "full" else 1.35, label=mode.replace("_", " "))
    axis.axvline(8, color="#555555", linestyle="--", linewidth=1)
    axis.set(xlabel="recurrent call t", ylabel="strict-successor rate", xlim=(1, args.max_call), ylim=(-.03, 1.03), title="G4 component and executor controls (appendix)")
    axis.grid(alpha=.18); axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(args.out_dir / "g4_final_lock_controls.png", dpi=260)
    figure.savefig(args.out_dir / "g4_final_lock_controls.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
