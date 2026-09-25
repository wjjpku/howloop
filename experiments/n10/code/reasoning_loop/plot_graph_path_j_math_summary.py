"""Render a compact summary of the seven-J mathematical interventions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--functional-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    matrix = json.loads((args.matrix_dir / "summary.json").read_text())
    task_rows = read_csv(args.functional_dir / "task_intervention_accuracy.csv")
    age_rows = read_csv(args.functional_dir / "age_probe_interventions.csv")
    inverse_rows = read_csv(args.functional_dir / "inverse_and_cycle_metrics.csv")
    manifold_rows = read_csv(args.functional_dir / "natural_manifold_metrics.csv")

    by_variant: dict[str, dict[str, float]] = {}
    for row in task_rows:
        by_variant.setdefault(row["variant"], {})[row["split"]] = float(
            row["accuracy_mean"]
        )

    def mixed(name: str) -> float:
        values = by_variant[name]
        return (values["focused_mixture_a"] + values["focused_mixture_b"]) / 2

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    figure.suptitle(
        "Seven learned J maps: one shared interface plus stage-specific control",
        fontsize=16,
    )

    shared = matrix["shared_generator"]["uncentered_pc1_energy_fraction"]
    names = ["shared component", "stage differences"]
    axes[0, 0].bar(names, [100 * shared, 100 * (1 - shared)], color=["#4c78a8", "#f58518"])
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_ylabel("parameter energy (%)")
    axes[0, 0].set_title("A. Tiny stage differences in parameter space")
    for index, value in enumerate([shared, 1 - shared]):
        axes[0, 0].text(index, 100 * value + 2, f"{100 * value:.2f}%", ha="center")
    axes[0, 0].text(
        0.02, 0.78,
        f"pairwise cosine = {matrix['shared_generator']['pairwise_delta_cosine_mean']:.3f}\n"
        f"bias cosine = {matrix['shared_generator']['bias_pairwise_cosine_mean']:.5f}",
        transform=axes[0, 0].transAxes,
        va="top",
    )

    control_names = [
        "learned J", "shared only", "wrong stage", "hidden regression", "forward pseudoinverse"
    ]
    control_keys = [
        "full", "shared_mean", "cyclic_stage_residual",
        "direct_hidden_regression", "fitted_forward_pseudoinverse",
    ]
    x = np.arange(len(control_names))
    width = 0.38
    axes[0, 1].bar(x - width / 2, [mixed(key) for key in control_keys], width, label="mixed")
    axes[0, 1].bar(
        x + width / 2,
        [by_variant[key]["hard_boundary_T05_R5"] for key in control_keys],
        width,
        label="five consecutive J",
    )
    axes[0, 1].set_xticks(x, control_names, rotation=18, ha="right")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].set_title("B. Correct stage control is causal")
    axes[0, 1].legend()

    stage_ranks = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    stage_accuracy = [
        mixed(f"mean_plus_stage_residual_rank_{rank}") for rank in stage_ranks
    ]
    axes[1, 0].plot(stage_ranks, stage_accuracy, "o-", label="mixed")
    axes[1, 0].plot(
        stage_ranks,
        [by_variant[f"mean_plus_stage_residual_rank_{rank}"]["hard_boundary_T05_R5"] for rank in stage_ranks],
        "o-",
        label="five consecutive J",
    )
    axes[1, 0].axhline(mixed("full"), color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xscale("symlog", linthresh=1)
    axes[1, 0].set_ylim(0.7, 1)
    axes[1, 0].set_xlabel("rank retained in stage-specific residual")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_title("C. Stage control is task-low-rank")
    axes[1, 0].legend()

    full_age = [row for row in age_rows if row["variant"] == "full"]
    ages = [int(row["source_age"]) for row in full_age]
    shifts = [float(row["predicted_age_shift"]) for row in full_age]
    axes[1, 1].plot(ages, shifts, "o-", color="#e45756", label="learned J")
    axes[1, 1].axhline(-1, color="black", linestyle="--", label="literal one-age rollback")
    axes[1, 1].set_xlabel("source loop age")
    axes[1, 1].set_ylabel("linear-probe age shift")
    axes[1, 1].set_title("D. J is not a uniform minus-one age operator")
    axes[1, 1].legend()

    direct_r2 = np.mean([float(row["direct_regression_r2"]) for row in inverse_rows])
    full_r2 = np.mean([float(row["full_r2"]) for row in manifold_rows])
    axes[1, 1].text(
        0.02,
        0.05,
        f"young-state R²: hidden regression {direct_r2:.3f} vs learned J {full_r2:.3f}\n"
        f"mixed accuracy: {mixed('direct_hidden_regression'):.3f} vs {mixed('full'):.3f}",
        transform=axes[1, 1].transAxes,
        va="bottom",
    )

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
