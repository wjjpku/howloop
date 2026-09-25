"""Plot the behavioral and matrix-level WSD comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def label_from_path(path: str) -> str:
    if "paired_fullbank_init" in path:
        return "warm r48"
    prefix = "WSD" if "wsd_identity_2x" in path else "old"
    rank = "r64" if "/r64_s16/" in path else "r48"
    return f"{prefix} {rank}"


def main() -> None:
    args = parse_args()
    eval_path = args.root / "wsd_matched_eval_fixed_h1" / "checkpoint_accuracy.csv"
    evaluation: dict[str, dict[str, float]] = {}
    with eval_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            evaluation.setdefault(label_from_path(row["artifact"]), {})[row["split"]] = float(
                row["accuracy_mean"]
            )

    metric_path = args.root / "matrix_analysis_best_controllers" / "bank_metrics.csv"
    with metric_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    metric_label = {
        "wsd_identity_r48": "WSD r48",
        "wsd_identity_r64": "WSD r64",
        "warm_start_r48": "warm r48",
    }

    product_path = args.root / "matrix_analysis_best_controllers" / "product_comparisons.csv"
    with product_path.open(newline="", encoding="utf-8") as handle:
        products = list(csv.DictReader(handle))

    diagonal_artifacts = {
        "old r64": args.root / "pure_identity/r64_s16/focused5_h1/age_specific_j_bank_after_T05_R5.pt",
        "WSD r64": args.root / "wsd_identity_2x/r64_s16/focused5_h1/age_specific_j_bank_after_T05_R5.pt",
        "warm r48": args.root / "paired_fullbank_init/r48_s16/focused5_h1/age_specific_j_bank_after_T05_R5.pt",
    }

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    split_names = (
        "single_J",
        "focused_mixture_a",
        "focused_mixture_b",
        "hard_boundary_T05_R5",
    )
    split_labels = ("single", "mix A", "mix B", "five J")
    model_order = ("old r48", "WSD r48", "old r64", "WSD r64", "warm r48")
    x = np.arange(len(split_names))
    width = 0.15
    for index, label in enumerate(model_order):
        values = [100 * evaluation[label][split] for split in split_names]
        axes[0, 0].bar(x + (index - 2) * width, values, width, label=label)
    axes[0, 0].set_xticks(x, split_labels)
    axes[0, 0].set_ylim(75, 100)
    axes[0, 0].set_ylabel("accuracy (%)")
    axes[0, 0].set_title("Matched fixed-H1 behavior")
    axes[0, 0].legend(fontsize=8, ncol=2)

    component_names = (
        "diagonal_update_norm",
        "shared_AB_norm",
        "mean_stage_U_norm",
        "bias_norm",
    )
    component_labels = ("diag-1", "shared AB", "stage AiBi", "bias")
    x = np.arange(len(component_names))
    width = 0.24
    for index, row in enumerate(metrics):
        values = [float(row[name]) for name in component_names]
        axes[0, 1].bar(
            x + (index - 1) * width,
            values,
            width,
            label=metric_label[row["bank"]],
        )
    axes[0, 1].set_xticks(x, component_labels)
    axes[0, 1].set_ylabel("Frobenius / L2 norm")
    axes[0, 1].set_title("Gauge-invariant component sizes")
    axes[0, 1].legend(fontsize=8)

    for label, artifact in diagonal_artifacts.items():
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        diagonal = payload["state_dict"]["shared_diagonal_scale"].float().numpy()
        axes[1, 0].plot(np.sort(diagonal), label=label)
    axes[1, 0].axhline(1, color="black", linewidth=0.7)
    axes[1, 0].set_xlabel("sorted hidden coordinate")
    axes[1, 0].set_ylabel("diagonal coefficient")
    axes[1, 0].set_title("Learned coordinate retention")
    axes[1, 0].legend(fontsize=8)

    target_pairs = {
        "wsd_identity_r48 -> warm_start_r48": "WSD r48 vs warm",
        "wsd_identity_r64 -> warm_start_r48": "WSD r64 vs warm",
    }
    for pair, label in target_pairs.items():
        selected = [row for row in products if row["comparison"] == pair]
        axes[1, 1].plot(
            [int(row["steps"]) for row in selected],
            [float(row["product_update_cosine"]) for row in selected],
            "o-",
            label=label,
        )
    axes[1, 1].set_xticks(range(1, 6))
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("consecutive J product length")
    axes[1, 1].set_ylabel("update cosine")
    axes[1, 1].set_title("Different matrices, similar behavior")
    axes[1, 1].legend(fontsize=8)

    figure.suptitle("WSD learns a new family of compositional rollback matrices", fontsize=16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
