"""Summarize the 3x3 shared-rank/stage-rank product-J ablation grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SHARED_RANKS = (64, 48, 32)
STAGE_RANKS = (24, 16, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def learned_accuracy(summary: dict[str, Any], split: str) -> float:
    matches = [
        row
        for row in summary["random_trajectory_summary"]
        if row["split"] == split and row["condition"] == "learned"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one learned row for {split}, found {len(matches)}")
    return float(matches[0]["accuracy_mean"])


def mean_initial_weight_error(single_summary: dict[str, Any]) -> float:
    values = [
        float(row["source_vs_expanded_weight_relative_error"])
        for row in single_summary["initialization_metrics"]
    ]
    return float(np.mean(values))


def training_window_metrics(path: Path) -> tuple[float, float, float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["stage"] == "T05_R5"]
    by_round: dict[int, list[dict[str, str]]] = {}
    for row in selected:
        by_round.setdefault(int(row["round"]), []).append(row)
    round_rows = []
    for round_index in sorted(by_round):
        batch_rows = by_round[round_index]
        round_rows.append(
            {
                "accuracy": float(
                    np.mean([float(row["accuracy"]) for row in batch_rows])
                ),
                "loss": float(np.mean([float(row["loss"]) for row in batch_rows])),
            }
        )
    if len(round_rows) < 24:
        raise ValueError(f"expected at least 24 T05_R5 rounds in {path}")
    first = round_rows[:12]
    last = round_rows[-12:]
    return (
        float(np.mean([row["accuracy"] for row in first])),
        float(np.mean([row["loss"] for row in first])),
        float(np.mean([row["accuracy"] for row in last])),
        float(np.mean([row["loss"] for row in last])),
    )


def load_cell(
    grid_root: Path,
    baseline_root: Path,
    shared_rank: int,
    stage_rank: int,
) -> dict[str, Any]:
    if (shared_rank, stage_rank) == (48, 16):
        focused_root = baseline_root
        single_root = baseline_root.parent / "single_r48_s16"
        source = "existing_baseline"
    else:
        focused_root = grid_root / f"focused5_r{shared_rank}_s{stage_rank}"
        single_root = grid_root / f"single_r{shared_rank}_s{stage_rank}"
        source = "new_grid_run"

    focused = json.loads((focused_root / "summary.json").read_text(encoding="utf-8"))
    single = json.loads((single_root / "summary.json").read_text(encoding="utf-8"))
    for payload, curriculum in ((single, "single_back"), (focused, "focused5")):
        if payload["status"] != "complete" or payload["curriculum"] != curriculum:
            raise ValueError(f"invalid {curriculum} result for r{shared_rank}/s{stage_rank}")
        # The original r48/s16 baseline predates this field in summary.json;
        # its checkpoint metadata and training report both record product.
        if payload.get("rollback_composition", "product") != "product":
            raise ValueError("rank grid must use exact product composition")
        if payload["J_bank"]["rank"] != shared_rank:
            raise ValueError("shared rank metadata mismatch")
        if payload["J_bank"]["stage_rank"] != stage_rank:
            raise ValueError("stage rank metadata mismatch")

    mixture_a = learned_accuracy(focused, "focused_mixture_a")
    mixture_b = learned_accuracy(focused, "focused_mixture_b")
    first_accuracy, first_loss, tail_accuracy, tail_loss = training_window_metrics(
        focused_root / "training_trajectories.csv"
    )
    parameter_count = int(focused["J_bank"]["parameter_count"])
    expected_parameters = 512 + 512 * shared_rank + 7 * 512 * stage_rank
    if parameter_count != expected_parameters:
        raise ValueError(
            f"parameter count mismatch: {parameter_count} != {expected_parameters}"
        )
    return {
        "shared_rank": shared_rank,
        "stage_rank": stage_rank,
        "effective_update_rank_upper_bound": shared_rank + stage_rank,
        "parameter_count": parameter_count,
        "reduction_vs_full_affine": 1.0 - parameter_count / 460_544,
        "initial_weight_relative_error_mean": mean_initial_weight_error(single),
        "single_J_accuracy": learned_accuracy(focused, "single_J"),
        "focused_mixture_a_accuracy": mixture_a,
        "focused_mixture_b_accuracy": mixture_b,
        "mixed_accuracy": (mixture_a + mixture_b) / 2.0,
        "hard_five_accuracy": learned_accuracy(focused, "hard_boundary_T05_R5"),
        "first12_round_training_accuracy": first_accuracy,
        "first12_round_training_loss": first_loss,
        "tail12_training_accuracy": tail_accuracy,
        "tail12_training_loss": tail_loss,
        "training_accuracy_change": tail_accuracy - first_accuracy,
        "training_loss_change": tail_loss - first_loss,
        "peak_cuda_allocated_mib": float(focused["peak_cuda_allocated_mib"]),
        "source": source,
        "checkpoint": str(focused_root / "age_specific_j_bank_after_T05_R5.pt"),
    }


def focused_root_for(
    grid_root: Path,
    baseline_root: Path,
    rank_pair: tuple[int, int],
) -> Path:
    if rank_pair == (48, 16):
        return baseline_root
    return grid_root / f"focused5_r{rank_pair[0]}_s{rank_pair[1]}"


def trajectory_values(root: Path, splits: tuple[str, ...]) -> np.ndarray:
    with (root / "random_trajectory_evaluation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["condition"] == "learned" and row["split"] in splits
        ]
    split_order = {split: index for index, split in enumerate(splits)}
    rows.sort(key=lambda row: (split_order[row["split"]], int(row["trajectory"])))
    return np.array([float(row["accuracy"]) for row in rows])


def paired_comparisons(
    grid_root: Path, baseline_root: Path
) -> list[dict[str, Any]]:
    pairs = (
        ((64, 24), (32, 8)),
        ((64, 24), (64, 16)),
        ((48, 24), (64, 8)),
        ((32, 24), (48, 8)),
        ((48, 16), (32, 8)),
    )
    split_groups = {
        "mixed": ("focused_mixture_a", "focused_mixture_b"),
        "hard_five": ("hard_boundary_T05_R5",),
    }
    rng = np.random.default_rng(830001)
    rows: list[dict[str, Any]] = []
    for evaluation, splits in split_groups.items():
        for left, right in pairs:
            left_values = trajectory_values(
                focused_root_for(grid_root, baseline_root, left), splits
            )
            right_values = trajectory_values(
                focused_root_for(grid_root, baseline_root, right), splits
            )
            difference = left_values - right_values
            indices = rng.integers(
                0, len(difference), size=(20_000, len(difference))
            )
            bootstrap = difference[indices].mean(axis=1)
            low, high = np.quantile(bootstrap, (0.025, 0.975))
            rows.append(
                {
                    "evaluation": evaluation,
                    "left_shared_rank": left[0],
                    "left_stage_rank": left[1],
                    "right_shared_rank": right[0],
                    "right_stage_rank": right[1],
                    "paired_accuracy_difference": float(difference.mean()),
                    "bootstrap_95_low": float(low),
                    "bootstrap_95_high": float(high),
                    "trajectories": len(difference),
                    "bootstrap_resamples": 20_000,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    lookup = {(row["shared_rank"], row["stage_rank"]): row for row in rows}
    return np.array(
        [
            [float(lookup[(shared, stage)][key]) for stage in STAGE_RANKS]
            for shared in SHARED_RANKS
        ]
    )


def annotate_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    percentage: bool,
    color_map: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = axis.imshow(values, cmap=color_map, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_title(title)
    axis.set_xticks(range(len(STAGE_RANKS)), labels=STAGE_RANKS)
    axis.set_yticks(range(len(SHARED_RANKS)), labels=SHARED_RANKS)
    axis.set_xlabel("stage rank")
    axis.set_ylabel("shared rank")
    midpoint = (float(np.nanmin(values)) + float(np.nanmax(values))) / 2
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = f"{100 * value:.2f}%" if percentage else f"{value:.1f}k"
            color = "white" if value < midpoint else "black"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def make_figure(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    annotate_heatmap(
        axes[0, 0],
        matrix(rows, "parameter_count") / 1000,
        title="Trainable J parameters",
        percentage=False,
        color_map="Blues",
    )
    annotate_heatmap(
        axes[0, 1],
        matrix(rows, "initial_weight_relative_error_mean"),
        title="Initial weight reconstruction error",
        percentage=True,
        color_map="magma_r",
    )
    for axis, key, title in (
        (axes[0, 2], "single_J_accuracy", "Single-J accuracy"),
        (axes[1, 0], "mixed_accuracy", "Mixed F/J accuracy"),
        (axes[1, 1], "hard_five_accuracy", "Five-consecutive-J accuracy"),
    ):
        annotate_heatmap(
            axis,
            matrix(rows, key),
            title=title,
            percentage=True,
            color_map="viridis",
            vmin=0.85,
            vmax=1.0,
        )

    pareto_axis = axes[1, 2]
    for row in rows:
        pareto_axis.scatter(
            row["parameter_count"] / 1000,
            100 * row["mixed_accuracy"],
            s=55 + 4 * row["stage_rank"],
            color=plt.cm.viridis((row["shared_rank"] - 32) / 32),
            edgecolor="black",
        )
        pareto_axis.annotate(
            f'{row["shared_rank"]}/{row["stage_rank"]}',
            (row["parameter_count"] / 1000, 100 * row["mixed_accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    pareto_axis.set_title("Parameter / mixed-accuracy trade-off")
    pareto_axis.set_xlabel("trainable J parameters (thousands)")
    pareto_axis.set_ylabel("mixed F/J accuracy (%)")
    pareto_axis.grid(alpha=0.25)

    figure.suptitle(
        "D8L8 seed0 product-J rank grid: shared diagonal + shared LoRA + stage LoRA",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        load_cell(args.grid_root, args.baseline_root, shared_rank, stage_rank)
        for shared_rank in SHARED_RANKS
        for stage_rank in STAGE_RANKS
    ]
    write_csv(args.out_dir / "rank_grid_summary.csv", rows)
    write_csv(
        args.out_dir / "rank_grid_paired_comparisons.csv",
        paired_comparisons(args.grid_root, args.baseline_root),
    )
    make_figure(rows, args.out_dir / "rank_grid_summary.png")
    (args.out_dir / "rank_grid_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
