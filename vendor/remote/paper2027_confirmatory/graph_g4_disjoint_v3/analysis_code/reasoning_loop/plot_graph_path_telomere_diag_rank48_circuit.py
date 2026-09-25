from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_stage(root: Path, out: Path) -> None:
    rows = _read(root / "enhanced_discovery_seed211_n512_v2/stage_readout.csv")
    selected = {
        ("J", "loop_input"): "J loop input",
        ("J", "B1_post_mlp"): "J after Block 1",
        ("J", "B2_post_attention"): "J after B2 attention",
        ("J", "B2_post_mlp"): "J after B2 MLP",
        ("exact_H7", "B2_post_mlp"): "exact H7 control",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, label in selected.items():
        data = sorted(
            (r for r in rows if (r["run"], r["stage"]) == key),
            key=lambda r: int(r["cycle"]),
        )
        ax.plot(
            [int(r["cycle"]) for r in data],
            [float(r["accuracy"]) for r in data],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set(xlabel="Continuation cycle", ylabel="Next-node accuracy", ylim=(-0.03, 1.04))
    ax.set_title("Where the next-hop answer is computed")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    _save(fig, out / "01_stage_readout.png")


def _plot_attention(root: Path, out: Path) -> None:
    rows = _read(
        root / "enhanced_discovery_seed211_n512_v2/attention_semantics.csv"
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for head in range(4):
        data = sorted(
            (
                r
                for r in rows
                if r["run"] == "J"
                and r["block"] == "2"
                and r["head"] == str(head)
                and r["key_role"] == "current_destination"
            ),
            key=lambda r: int(r["cycle"]),
        )
        ax.plot(
            [int(r["cycle"]) for r in data],
            [float(r["attention"]) for r in data],
            marker="o",
            linewidth=2,
            label=f"B2 H{head}",
        )
    ax.set(
        xlabel="Continuation cycle",
        ylabel="Attention to current node's destination",
        ylim=(-0.02, 0.43),
    )
    ax.set_title("The primary lookup head loses its target late")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "02_lookup_attention.png")


def _plot_necessity(root: Path, out: Path) -> None:
    rows = _read(root / "necessity_seed211_n512/component_hybrids.csv")
    components = [
        "B2.H0.context_answer",
        "B2.H1.context_answer",
        "B2.H2.context_answer",
        "B2.H3.context_answer",
        "B2.mlp_out_answer",
    ]
    labels = ["H0", "H1", "H2", "H3", "MLP"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    width = 0.36
    x = np.arange(len(components))
    for ax, cycle in zip(axes, (1, 32, 64), strict=True):
        for offset, (condition, label, color) in enumerate(
            (
                ("zero_in_J", "zero", "#4C78A8"),
                ("shuffled_J_into_J", "shuffled", "#E45756"),
            )
        ):
            values = [
                float(
                    next(
                        r["accuracy"]
                        for r in rows
                        if int(r["cycle"]) == cycle
                        and r["condition"] == condition
                        and r["component"] == component
                    )
                )
                for component in components
            ]
            ax.bar(x + (offset - 0.5) * width, values, width, label=label, color=color)
        ax.set_title(f"cycle {cycle}")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Accuracy after intervention")
    axes[-1].legend(frameon=False)
    fig.suptitle("Native component necessity and information specificity", y=1.03)
    _save(fig, out / "03_component_necessity.png")


def _plot_repairs(root: Path, out: Path) -> None:
    records: list[dict[str, Any]] = []
    for seed in (211, 311, 411):
        path = root / f"candidate_seed{seed}_n512/candidate_circuit.csv"
        if not path.exists():
            continue
        rows = [r for r in _read(path) if int(r["cycle"]) == 64]
        for name, condition, candidate in (
            (
                "fixed 64 MLP",
                "exact_H7_into_J_leave_H0_out",
                "selected_fixed_MLP_only",
            ),
            (
                "H0 + fixed 64",
                "exact_H7_into_J",
                "selected_H0_plus_fixed_MLP",
            ),
        ):
            row = next(
                r
                for r in rows
                if r["condition"] == condition and r["candidate"] == candidate
            )
            records.append({"seed": seed, "kind": name, "accuracy": float(row["accuracy"])})
        random = [
            float(r["accuracy"])
            for r in rows
            if r["condition"] == "exact_H7_into_J_size_matched_random"
        ]
        records.append(
            {"seed": seed, "kind": "random set mean", "accuracy": float(np.mean(random))}
        )
    kinds = ["fixed 64 MLP", "H0 + fixed 64", "random set mean"]
    colors = {211: "#4C78A8", 311: "#F58518", 411: "#54A24B"}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for seed in sorted({int(r["seed"]) for r in records}):
        values = [
            next(r["accuracy"] for r in records if r["seed"] == seed and r["kind"] == kind)
            for kind in kinds
        ]
        ax.plot(kinds, values, marker="o", linewidth=2, color=colors[seed], label=f"J seed {seed}")
    ax.set(ylabel="Cycle-64 accuracy", ylim=(0.70, 1.02))
    ax.set_title("A fixed 64-neuron repair channel transfers across J seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out / "04_fixed_neuron_repair.png")


def _plot_modes(root: Path, out: Path) -> None:
    rows = _read(
        root / "enhanced_discovery_seed211_n512_v2/J_mode_ablation.csv"
    )
    values: dict[int, dict[int, float]] = {1: {}, 64: {}}
    for row in rows:
        cycle = int(row["cycle"])
        if cycle in values and row.get("condition", "") == "":
            values[cycle][int(row["mode"])] = float(row["margin_drop"])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    modes = sorted(set(values[1]) & set(values[64]))
    x = np.array([values[1][mode] for mode in modes])
    y = np.array([values[64][mode] for mode in modes])
    ax.scatter(x, y, alpha=0.65, s=30)
    for mode in sorted(modes, key=lambda m: max(values[1][m], values[64][m]), reverse=True)[:8]:
        ax.annotate(str(mode), (values[1][mode], values[64][mode]), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Margin loss when mode is removed at cycle 1",
        ylabel="Margin loss when mode is removed at cycle 64",
    )
    ax.set_title("J modes change role with effective time")
    ax.grid(alpha=0.2)
    _save(fig, out / "05_J_mode_role_shift.png")


def _plot_failure_split(root: Path, out: Path) -> None:
    rows = _read(root / "seed211_cycle64_attention_v3/per_example.csv")
    correct = [
        float(r["B2H0_current_destination_attention"])
        for r in rows
        if r["run"] == "J"
        and r["correct"] == "1"
        and int(r["current_cycle_length"]) > 1
    ]
    incorrect = [
        float(r["B2H0_current_destination_attention"])
        for r in rows
        if r["run"] == "J"
        and r["correct"] == "0"
        and int(r["current_cycle_length"]) > 1
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.boxplot(
        [correct, incorrect],
        tick_labels=[f"correct\n(n={len(correct)})", f"incorrect\n(n={len(incorrect)})"],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#72B7B2"},
        medianprops={"color": "black"},
    )
    ax.set_ylabel("B2 H0 attention to current destination")
    ax.set_title("Late errors coincide with lookup loss")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, out / "06_failure_attention_split.png")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot_stage(args.root, args.out_dir)
    _plot_attention(args.root, args.out_dir)
    _plot_necessity(args.root, args.out_dir)
    _plot_repairs(args.root, args.out_dir)
    _plot_modes(args.root, args.out_dir)
    _plot_failure_split(args.root, args.out_dir)
    result = {
        "status": "complete",
        "source_root": str(args.root),
        "figures": sorted(path.name for path in args.out_dir.glob("*.png")),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the rank-48 J circuit audit.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
