#!/usr/bin/env python3
"""Aggregate locked G1 Graph phenotypes with graph-permutation clustering."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def hierarchical_bootstrap_mean(
    per_seed_values: Sequence[np.ndarray], *, draws: int, seed: int
) -> dict[str, float | int]:
    """Bootstrap a cluster mean after resampling top-level backbone seeds."""
    if not per_seed_values or any(values.ndim != 1 or not len(values) for values in per_seed_values):
        raise ValueError("each included backbone needs a nonempty one-dimensional cluster vector")
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    generator = np.random.default_rng(seed)
    n_backbones = len(per_seed_values)
    sampled = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        width = min(200, draws - cursor)
        selected_backbones = generator.integers(0, n_backbones, size=(width, n_backbones))
        estimate = np.zeros(width, dtype=np.float64)
        for slot in range(n_backbones):
            selected = selected_backbones[:, slot]
            for backbone in np.unique(selected):
                mask = selected == backbone
                values = per_seed_values[int(backbone)]
                cluster_index = generator.integers(0, len(values), size=(int(mask.sum()), len(values)))
                estimate[mask] += values[cluster_index].mean(axis=1)
        sampled[cursor : cursor + width] = estimate / n_backbones
        cursor += width
    point = float(np.mean([values.mean() for values in per_seed_values]))
    return {
        "point_estimate": point,
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "bootstrap_draws": draws,
        "backbone_count": n_backbones,
    }


def seed_call_table(rows: Sequence[dict[str, str]]) -> dict[int, dict[int, list[dict[str, str]]]]:
    table: dict[int, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        table[int(row["permutation"])][int(row["call"])].append(row)
    return table


def per_permutation_auc(
    rows: Sequence[dict[str, str]], *, metric: str, first_call: int, last_call: int
) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        call = int(row["call"])
        if first_call <= call <= last_call:
            numerator = float(row[metric])
            denominator = float(row["strict_examples"] if metric.startswith("strict_") else row["examples"])
            grouped[int(row["permutation"])].append(numerator / denominator if denominator else np.nan)
    if not grouped:
        raise ValueError("no rows in selected AUC window")
    values = np.asarray([np.nanmean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("selected G1 AUC has no strict examples for some graph permutations")
    return values


def summarize_seed(seed: int, rows: Sequence[dict[str, str]], *, qualification: float) -> dict[str, Any]:
    endpoint_rows = [row for row in rows if int(row["call"]) == 8]
    if not endpoint_rows:
        raise ValueError(f"seed {seed} lacks trained-call metrics")
    endpoint_accuracy = sum(float(row["moving_successor_correct"]) for row in endpoint_rows) / sum(
        float(row["examples"]) for row in endpoint_rows
    )
    strict_successor = per_permutation_auc(
        rows, metric="strict_successor_correct", first_call=9, last_call=128
    )
    endpoint_hold = per_permutation_auc(
        rows, metric="endpoint_hold_correct", first_call=9, last_call=128
    )
    if endpoint_hold.mean() >= 0.80 and strict_successor.mean() <= 0.20:
        phenotype = "terminal_hold"
    elif strict_successor.mean() >= 0.80 and endpoint_hold.mean() <= 0.20:
        phenotype = "successor_continuation"
    else:
        phenotype = "mixed_or_drifting"
    return {
        "backbone_seed": seed,
        "call8_exact_match": endpoint_accuracy,
        "endpoint_qualified": bool(endpoint_accuracy >= qualification),
        "post_horizon_strict_successor_auc_9_128": float(strict_successor.mean()),
        "post_horizon_endpoint_hold_auc_9_128": float(endpoint_hold.mean()),
        "phenotype": phenotype,
        "permutation_count": int(len(strict_successor)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty seed summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(seed_rows: Sequence[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    x = [int(row["backbone_seed"]) for row in seed_rows]
    y = [
        float(row["post_horizon_strict_successor_auc_9_128"])
        if row.get("post_horizon_strict_successor_auc_9_128") is not None
        else np.nan
        for row in seed_rows
    ]
    colors = ["#0072B2" if bool(row.get("endpoint_qualified")) else "#999999" for row in seed_rows]
    axis.scatter(x, y, c=colors, s=42)
    axis.set(xlabel="independent backbone seed", ylabel="strict-successor AUC (calls 9-128)", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qualification", type=float, default=0.99)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026094001)
    args = parser.parse_args()
    if not 0.0 < args.qualification <= 1.0:
        raise ValueError("qualification threshold must lie in (0, 1]")

    manifest_status: dict[int, str] = {}
    for manifest in sorted(args.root.glob("manifests/backbone_seed*.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_status[int(payload["backbone_seed"])] = str(payload.get("status", "unknown"))
    loaded: dict[int, list[dict[str, str]]] = {}
    for csv_path in sorted(args.root.glob("evaluation/seed*/permutation_clusters.csv")):
        seed = int(csv_path.parent.name.removeprefix("seed"))
        loaded[seed] = read_csv(csv_path)
    if not loaded and not manifest_status:
        raise FileNotFoundError("no G1 manifests or permutation-cluster tables found")
    seed_rows = [
        {
            **summarize_seed(seed, rows, qualification=args.qualification),
            "manifest_status": manifest_status.get(seed, "evaluation_without_manifest"),
        }
        for seed, rows in sorted(loaded.items())
    ]
    for seed, status in sorted(manifest_status.items()):
        if seed not in loaded:
            seed_rows.append(
                {
                    "backbone_seed": seed,
                    "manifest_status": status,
                    "call8_exact_match": None,
                    "endpoint_qualified": False,
                    "post_horizon_strict_successor_auc_9_128": None,
                    "post_horizon_endpoint_hold_auc_9_128": None,
                    "phenotype": "not_evaluated",
                    "permutation_count": 0,
                }
            )
    seed_rows.sort(key=lambda row: int(row["backbone_seed"]))
    included = [row for row in seed_rows if bool(row["endpoint_qualified"])]
    outcome: dict[str, Any] = {
        "status": "complete",
        "protocol_id": "paper2027.graph.g1.phenotype.v1",
        "qualification_threshold": args.qualification,
        "trained_backbone_count_with_evaluation": len(seed_rows),
        "endpoint_qualified_backbone_count": len(included),
        "seed_rows": seed_rows,
        "phenotype_counts_qualified": {
            label: sum(row["phenotype"] == label for row in included)
            for label in ("terminal_hold", "successor_continuation", "mixed_or_drifting")
        },
    }
    if included:
        included_seeds = [int(row["backbone_seed"]) for row in included]
        outcome["post_horizon_strict_successor_auc_9_128"] = hierarchical_bootstrap_mean(
            [
                per_permutation_auc(loaded[seed], metric="strict_successor_correct", first_call=9, last_call=128)
                for seed in included_seeds
            ],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )
        outcome["post_horizon_endpoint_hold_auc_9_128"] = hierarchical_bootstrap_mean(
            [
                per_permutation_auc(loaded[seed], metric="endpoint_hold_correct", first_call=9, last_call=128)
                for seed in included_seeds
            ],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed + 1,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "seed_summary.csv", seed_rows)
    plot(seed_rows, args.out_dir / "seed_summary.png")
    (args.out_dir / "summary.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
