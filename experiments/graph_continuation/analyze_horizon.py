#!/usr/bin/env python3
"""Recover the registered 12-backbone continuation curve and horizon AUCs.

The statistical hierarchy matches the locked aggregate: resample backbones,
then one controller replica within each selected backbone, then graph clusters.
No backbone, controller, graph, or call is selected by outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODES = {
    "full": {"label": "+ boundary map", "color": "#D97706", "replicas": (1, 2)},
    "raw": {"label": "raw backbone", "color": "#4B5563", "replicas": (None,)},
    "executor_off": {"label": "executor off", "color": "#9CA3AF", "replicas": (1,)},
    "batch_shuffle": {"label": "batch-shuffled map", "color": "#3B82F6", "replicas": (1,)},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_path(root: Path, backbone: int, mode: str, replica: int | None) -> Path:
    base = root / f"seed{backbone}"
    if mode == "raw":
        return base / "raw" / "permutation_clusters.csv"
    return base / f"rank48_seed{replica}" / mode / "permutation_clusters.csv"


def graph_call_values(rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    by_call: dict[int, dict[int, float]] = defaultdict(dict)
    permutations = sorted({int(row["permutation"]) for row in rows})
    if permutations != list(range(len(permutations))):
        raise ValueError("graph permutations must be contiguous from zero")
    for row in rows:
        call = int(row["call"])
        denominator = float(row["strict_examples"])
        if denominator > 0:
            by_call[call][int(row["permutation"])] = float(row["strict_successor_correct"]) / denominator
    calls: dict[int, np.ndarray] = {}
    for call, mapping in by_call.items():
        values = np.full(len(permutations), np.nan, dtype=float)
        for key, value in mapping.items():
            values[key] = value
        calls[call] = values
    return calls


def load_mode(root: Path, backbones: list[int], mode: str) -> dict[int, np.ndarray]:
    """Return call -> [backbone, replica, graph] with balanced graph counts."""
    per_backbone: list[list[dict[int, np.ndarray]]] = []
    for backbone in backbones:
        replicas = []
        for replica in MODES[mode]["replicas"]:
            path = csv_path(root, backbone, mode, replica)
            if not path.is_file():
                raise FileNotFoundError(path)
            replicas.append(graph_call_values(read_rows(path)))
        per_backbone.append(replicas)
    common_calls = sorted(set.intersection(*(set(replica) for seed in per_backbone for replica in seed)))
    output = {}
    for call in common_calls:
        arrays = [[replica[call] for replica in seed] for seed in per_backbone]
        shape = arrays[0][0].shape
        if any(value.shape != shape for seed in arrays for value in seed):
            raise ValueError(f"unbalanced graph clusters at call {call}")
        output[call] = np.asarray(arrays, dtype=float)
    return output


def hierarchical_bootstrap(values: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    """Bootstrap [backbone, replica, graph] and return a mean and percentile CI."""
    n_backbone, n_replica, n_graph = values.shape
    samples = np.empty(draws, dtype=float)
    batch_size = 250
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        batch = stop - start
        selected = rng.integers(0, n_backbone, size=(batch, n_backbone))
        replicas = rng.integers(0, n_replica, size=(batch, n_backbone))
        graphs = rng.integers(0, n_graph, size=(batch, n_backbone, n_graph))
        sampled = values[selected[:, :, None], replicas[:, :, None], graphs]
        samples[start:stop] = np.nanmean(np.nanmean(sampled, axis=2), axis=1)
    per_backbone = np.nanmean(values, axis=(1, 2))
    return {
        "point_estimate": float(per_backbone.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def backbone_bootstrap(values: np.ndarray, draws: int, rng: np.random.Generator) -> dict[str, float]:
    """Descriptive curve CI over the independent-backbone population."""
    per_backbone = np.nanmean(values, axis=(1, 2))
    selected = rng.integers(0, len(per_backbone), size=(draws, len(per_backbone)))
    samples = per_backbone[selected].mean(axis=1)
    return {
        "point_estimate": float(per_backbone.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def horizon_auc(mode_data: dict[int, np.ndarray], first: int, last: int) -> np.ndarray:
    calls = [call for call in sorted(mode_data) if first <= call <= last]
    if calls != list(range(first, last + 1)):
        raise ValueError(f"missing calls in [{first}, {last}]")
    return np.nanmean(np.stack([mode_data[call] for call in calls], axis=0), axis=0)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(curves: dict[str, list[dict[str, float]]], aucs: dict[str, dict[str, dict[str, float]]], out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig, (ax, bar) = plt.subplots(1, 2, figsize=(6.85, 2.25), gridspec_kw={"width_ratios": [2.1, 1.0]})
    ax.axvspan(8.5, 16.5, color="#FDE68A", alpha=0.28, lw=0, label="trained calls")
    for mode in ("full", "batch_shuffle", "executor_off", "raw"):
        rows = curves[mode]
        x = np.asarray([row["call"] for row in rows])
        y = np.asarray([row["point_estimate"] for row in rows])
        lo = np.asarray([row["ci95_low"] for row in rows])
        hi = np.asarray([row["ci95_high"] for row in rows])
        style = MODES[mode]
        width = 1.8 if mode == "full" else 1.0
        alpha = 1.0 if mode == "full" else 0.9
        ax.plot(x, y, color=style["color"], lw=width, alpha=alpha, label=style["label"])
        if mode == "full":
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.16, lw=0)
    ax.axvline(16.5, color="#B45309", lw=0.8, ls="--")
    ax.text(10.7, 0.94, "training-covered", ha="center", va="top", color="#92400E", fontsize=7)
    ax.text(17.4, 0.94, "strict OOD", ha="left", va="top", color="#4B5563", fontsize=7)
    ax.set(xlim=(9, 64), ylim=(0, 1.02), xlabel="executor call", ylabel="strict successor accuracy")
    ax.set_xticks([9, 16, 24, 32, 48, 64])
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.legend(frameon=False, loc="upper right", ncol=2, handlelength=2.3)
    ax.set_title("a  Continuation decays beyond the trained horizon", loc="left", fontweight="bold")

    spans = [("9–16", "covered"), ("17–128", "OOD")]
    modes = ("full", "batch_shuffle", "executor_off", "raw")
    x = np.arange(len(spans))
    offsets = np.linspace(-0.27, 0.27, len(modes))
    for offset, mode in zip(offsets, modes):
        values = [aucs[mode][key]["point_estimate"] for _, key in spans]
        lower = [values[i] - aucs[mode][key]["ci95_low"] for i, (_, key) in enumerate(spans)]
        upper = [aucs[mode][key]["ci95_high"] - values[i] for i, (_, key) in enumerate(spans)]
        bar.bar(x + offset, values, width=0.16, color=MODES[mode]["color"], label=MODES[mode]["label"])
        bar.errorbar(x + offset, values, yerr=[lower, upper], fmt="none", ecolor="#374151", elinewidth=0.6, capsize=1.5)
    bar.set_xticks(x, [label for label, _ in spans])
    bar.set_ylim(0, 1.02)
    bar.set_ylabel("mean strict accuracy")
    bar.grid(axis="y", color="#E5E7EB", lw=0.6)
    bar.set_title("b  Horizon-separated AUC", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026092101)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    backbones = list(range(100, 112))
    curves: dict[str, list[dict[str, float]]] = {}
    aucs: dict[str, dict[str, dict[str, float]]] = {}
    rng = np.random.default_rng(args.bootstrap_seed)
    for mode in MODES:
        data = load_mode(args.root, backbones, mode)
        curves[mode] = []
        for call in range(9, 129):
            result = backbone_bootstrap(data[call], args.draws, rng)
            curves[mode].append({"mode": mode, "call": call, **result})
        aucs[mode] = {
            "covered": hierarchical_bootstrap(horizon_auc(data, 9, 16), args.draws, rng),
            "OOD": hierarchical_bootstrap(horizon_auc(data, 17, 128), args.draws, rng),
            "all": hierarchical_bootstrap(horizon_auc(data, 9, 128), args.draws, rng),
        }
    curve_rows = [row for mode in MODES for row in curves[mode]]
    save_csv(args.out / "call_curve.csv", curve_rows)
    summary = {
        "status": "complete",
        "backbone_count": len(backbones),
        "controller_replicas_full": 2,
        "graph_clusters_per_replica": 512,
        "bootstrap_draws": args.draws,
        "statistical_hierarchy": "backbone -> controller replica -> graph permutation",
        "horizon_auc": aucs,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot(curves, aucs, args.out / "horizon_decomposition")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
