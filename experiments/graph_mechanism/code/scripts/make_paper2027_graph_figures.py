#!/usr/bin/env python3
"""Render main-text and appendix Graph figures from locked confirmatory data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped_curve(rows: Iterable[dict[str, str]], *, x_key: str, y_key: str, mode: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if mode is not None and row.get("mode") != mode:
            continue
        grouped[int(row[x_key])].append(float(row[y_key]))
    if not grouped:
        raise ValueError(f"no data for mode={mode!r}")
    x = np.asarray(sorted(grouped))
    data = [np.asarray(grouped[int(value)]) for value in x]
    return x, np.asarray([values.mean() for values in data]), np.asarray([values.std(ddof=0) for values in data])


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=260)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def graph_g1(root: Path, seeds: list[int], figure_dir: Path) -> Path:
    rows = []
    for seed in seeds:
        rows.extend(read_csv(root / "evaluation" / f"seed{seed}" / "aggregate_calls.csv"))
    figure, axis = plt.subplots(figsize=(7.4, 3.9), constrained_layout=True)
    for key, label, color in (
        ("endpoint_hold_accuracy", "hold trained endpoint $f^8(s)$", "#1f77b4"),
        ("strict_successor_accuracy", "strict current successor $f^t(s)$", "#d62728"),
    ):
        x, mean, std = grouped_curve(rows, x_key="call", y_key=key)
        axis.plot(x, mean, color=color, label=label)
        axis.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std), color=color, alpha=.18)
    axis.axvline(8, color="black", linestyle="--", linewidth=1, label="trained call")
    axis.set(xlabel="recurrent call", ylabel="rate across backbone seeds", ylim=(-.03, 1.03))
    axis.grid(alpha=.2); axis.legend(frameon=False, ncol=2, fontsize=8)
    path = figure_dir / "graph_g1_locked_phenotype.png"; save(figure, path); return path


def graph_g2(root: Path, seeds: list[int], figure_dir: Path) -> Path:
    rows = []
    for seed in seeds:
        path = root / "g2_interface" / f"seed{seed}" / "aggregate.csv"
        if path.is_file():
            rows.extend(read_csv(path))
    figure, axis = plt.subplots(figsize=(7.4, 3.9), constrained_layout=True)
    for mode, label, color in (
        ("matched_young", "same graph, same current, young interface", "#2ca02c"),
        ("raw_late_next", "unpatched late interface", "#1f77b4"),
        ("wrong_current", "same graph, wrong current", "#ff7f0e"),
        ("cross_graph", "cross-graph donor", "#d62728"),
        ("norm_matched_random", "norm-matched random donor", "#9467bd"),
    ):
        x, mean, std = grouped_curve(rows, x_key="source_call", y_key="strict_next_successor_accuracy", mode=mode)
        axis.plot(x, mean, marker="o", color=color, label=label)
        axis.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std), color=color, alpha=.15)
    axis.set(xlabel="late receiver call $t$", ylabel="strict $f^{t+1}(s)$ accuracy after one frozen executor", ylim=(-.03, 1.03))
    axis.grid(alpha=.2); axis.legend(frameon=False, ncol=1, fontsize=7)
    path = figure_dir / "graph_g2_matched_interface.png"; save(figure, path); return path


def graph_g3(root: Path, seeds: list[int], figure_dir: Path) -> Path:
    raw_rows, full_rows = [], []
    for seed in seeds:
        raw_path = root / "g3_evaluation" / f"seed{seed}" / "rank48_seed1" / "raw" / "aggregate_calls.csv"
        if not raw_path.is_file():
            continue
        raw_rows.extend(read_csv(raw_path))
        for replica in (1, 2):
            full_rows.extend(read_csv(root / "g3_evaluation" / f"seed{seed}" / f"rank48_seed{replica}" / "full" / "aggregate_calls.csv"))
    figure, axis = plt.subplots(figsize=(7.4, 3.9), constrained_layout=True)
    for rows, label, color in ((raw_rows, "raw frozen recurrence", "#1f77b4"), (full_rows, "$D+AB+b$ controller", "#2ca02c")):
        x, mean, std = grouped_curve(rows, x_key="call", y_key="strict_successor_accuracy")
        axis.plot(x, mean, color=color, label=label)
        axis.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std), color=color, alpha=.18)
    axis.axvline(8, color="black", linestyle="--", linewidth=1, label="trained call")
    axis.axvspan(17, 64, color="black", alpha=.05, label="pre-registered OOD window")
    axis.set(xlabel="controlled recurrent call", ylabel="strict successor accuracy", ylim=(-.03, 1.03))
    axis.grid(alpha=.2); axis.legend(frameon=False, ncol=2, fontsize=8)
    path = figure_dir / "graph_g3_locked_controller.png"; save(figure, path); return path


def graph_components(root: Path, figure_dir: Path) -> Path:
    rows = read_csv(root / "aggregate_confirmatory" / "g3_component_table.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(float(row["strict_successor_auc_17_64"]))
    labels = list(grouped)
    mean = [np.mean(grouped[label]) for label in labels]
    std = [np.std(grouped[label]) for label in labels]
    figure, axis = plt.subplots(figsize=(8.0, 3.7), constrained_layout=True)
    axis.bar(range(len(labels)), mean, yerr=std, color="#7f7f7f", capsize=2)
    axis.set(xticks=range(len(labels)), xticklabels=labels, ylabel="strict successor AUC (calls 17–64)", ylim=(0, 1.03))
    axis.tick_params(axis="x", rotation=30); axis.grid(axis="y", alpha=.2)
    path = figure_dir / "graph_g3_component_controls.png"; save(figure, path); return path


def results_tex(summary: dict[str, Any], path: Path) -> None:
    statistics = summary["statistics"]
    macros = {
        "GraphConfirmatoryBackbones": str(summary["complete_backbones"]),
        "GraphEndpointQualifiedBackbones": str(summary["endpoint_qualified_backbones"]),
        "GraphGOnePostHorizonAUC": f"{statistics['g1_posthorizon_strict_successor_auc_17_64']['estimate']:.3f}",
        "GraphGTwoMatchedEffect": f"{statistics['g2_matched_young_minus_raw_next_strict_accuracy_calls_16_32_64']['estimate']:.3f}",
        "GraphGThreeControllerEffect": f"{statistics['g3_full_minus_raw_strict_successor_auc_17_64']['estimate']:.3f}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--results-tex", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 112)))
    args = parser.parse_args()
    summary = json.loads((args.root / "aggregate_confirmatory" / "summary.json").read_text(encoding="utf-8"))
    qualified = [int(row["seed"]) for row in read_csv(args.root / "aggregate_confirmatory" / "backbone_table.csv") if str(row["qualified"]).lower() == "true"]
    outputs = {
        "graph_g1": str(graph_g1(args.root, args.seeds, args.figure_dir)),
        "graph_g2": str(graph_g2(args.root, qualified, args.figure_dir)),
        "graph_g3": str(graph_g3(args.root, qualified, args.figure_dir)),
        "graph_g3_components": str(graph_components(args.root, args.figure_dir)),
    }
    results_tex(summary, args.results_tex)
    print(json.dumps({"status": "complete", "outputs": outputs, "results_tex": str(args.results_tex)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
