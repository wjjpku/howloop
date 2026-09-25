#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("auc_1_24", "AUC loops 1–24"),
    ("auc_25_48", "AUC loops 25–48"),
    ("auc_49_64", "AUC loops 49–64"),
    ("accuracy_loop64", "Accuracy at loop 64"),
)
PLACEMENTS = ("loop_boundary", "pre_block2")
COLORS = {"loop_boundary": "#0072B2", "pre_block2": "#D55E00"}
LABELS = {
    "loop_boundary": "Loop boundary J",
    "pre_block2": "Pre-Block2 interface",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(run_root: Path) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    audits: dict[str, dict] = {}
    for placement in PLACEMENTS:
        audit = load_json(
            run_root / f"strict_unseen_{placement}" / "summary.json"
        )
        train = load_json(run_root / placement / "summary.json")
        backbone = int(train["backbone_parameter_count"])
        audits[placement] = audit
        for rank_text, aggregate in sorted(
            audit["rank_aggregates"].items(), key=lambda item: int(item[0])
        ):
            rank = int(rank_text)
            labels = aggregate["labels"]
            parameter_count = int(
                audit["operators"][labels[0]]["parameter_count"]
            )
            row = {
                "placement": placement,
                "rank": rank,
                "parameter_count": parameter_count,
                "controller_to_backbone_ratio": parameter_count / backbone,
            }
            for metric, _ in METRICS:
                row[f"{metric}_mean"] = aggregate[metric]["mean"]
                row[f"{metric}_std_population"] = aggregate[metric][
                    "std_population"
                ]
            rows.append(row)
    return rows, audits


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_capacity(
    path: Path,
    rows: list[dict],
    mlp_root: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
        for placement in PLACEMENTS:
            selected = [r for r in rows if r["placement"] == placement]
            ranks = np.asarray([r["rank"] for r in selected])
            means = np.asarray([r[f"{metric}_mean"] for r in selected])
            stds = np.asarray(
                [r[f"{metric}_std_population"] for r in selected]
            )
            axis.errorbar(
                ranks,
                means,
                yerr=stds,
                color=COLORS[placement],
                marker="o",
                linewidth=2,
                capsize=3,
                label=LABELS[placement],
            )
            mlp = load_json(
                mlp_root / f"strict_unseen_{placement}_w512" / "summary.json"
            )
            width_aggregate = mlp["width_aggregates"]["512"]
            if metric == "accuracy_loop64":
                baseline = float(
                    np.mean(
                        [
                            mlp["strictly_unseen"]["curves"][label][
                                "accuracy_by_cycle"
                            ][-1]
                            for label in width_aggregate["labels"]
                        ]
                    )
                )
            else:
                baseline = width_aggregate[metric]["mean"]
            axis.axhline(
                baseline,
                color=COLORS[placement],
                linestyle="--",
                linewidth=1,
                alpha=0.55,
            )
        axis.set_title(title)
        axis.set_ylim(-0.02, 1.03)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("LoRA rank")
    axes[1, 1].set_xlabel("LoRA rank")
    axes[0, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        axis.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Curriculum-trained low-rank J on strictly unseen permutations",
        y=0.995,
    )
    axes[0, 0].legend(handles[:2], labels[:2], loc="lower right")
    fig.text(
        0.5,
        0.95,
        "Error bars: population SD across three paired gauge seeds; dashed: width-512 MLP",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_curves(path: Path, audits: dict[str, dict]) -> None:
    selected_ranks = (8, 16, 32, 64, 128)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    palette = plt.cm.viridis(np.linspace(0.08, 0.92, len(selected_ranks)))
    for axis, placement in zip(axes, PLACEMENTS, strict=True):
        audit = audits[placement]
        for rank, color in zip(selected_ranks, palette, strict=True):
            aggregate = audit["rank_aggregates"].get(str(rank))
            if aggregate is None:
                continue
            curves = np.asarray(
                [
                    audit["strictly_unseen"]["curves"][label][
                        "accuracy_by_cycle"
                    ]
                    for label in aggregate["labels"]
                ]
            )
            cycles = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            axis.plot(cycles, mean, color=color, label=f"rank {rank}")
            axis.fill_between(
                cycles,
                mean - std,
                mean + std,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_title(LABELS[placement])
        axis.set_xlabel("Controlled continuation loop")
        axis.grid(alpha=0.2)
        axis.set_xlim(1, 64)
        axis.set_ylim(-0.02, 1.03)
    axes[0].set_ylabel("Strictly unseen accuracy")
    axes[1].legend(loc="lower left", ncol=2, fontsize=8)
    fig.suptitle("Closed-loop accuracy after curriculum training")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mlp-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, audits = collect_rows(args.run_root)
    write_csv(args.out_dir / "rank_capacity_metrics.csv", rows)
    plot_capacity(args.out_dir / "rank_capacity.png", rows, args.mlp_root)
    plot_curves(args.out_dir / "closed_loop_curves.png", audits)


if __name__ == "__main__":
    main()
