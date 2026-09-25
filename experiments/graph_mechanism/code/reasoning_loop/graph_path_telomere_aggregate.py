from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


MODEL_ORDER = (
    "D8_L6_seed6",
    "D8_L8_seed0",
    "D8_L8_seed1",
    "D8_L8_seed2",
    "D8_L8_seed5",
)
COMPONENT_ORDER = (
    "B1.H0",
    "B1.H1",
    "B1.H2",
    "B1.H3",
    "B2.H0",
    "B2.H1",
    "B2.H2",
    "B2.H3",
    "B1.MLP",
    "B2.MLP",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lookup_closed(
    rows: list[dict[str, str]],
    *,
    condition: str,
    extra_loop: int,
    target_kind: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["condition"] == condition
        and int(row["extra_loop"]) == extra_loop
        and row["target_kind"] == target_kind
    ]
    if len(matches) != 1:
        raise ValueError("closed-loop lookup is not unique")
    return matches[0]


def _dominant_b2_head(
    executor_rows: list[dict[str, str]],
) -> tuple[str, float, float]:
    candidates = [
        row for row in executor_rows if row["component"].startswith("B2.H")
    ]
    winner = max(
        candidates,
        key=lambda row: (
            min(
                float(row["zero_accuracy_drop"]),
                float(row["shuffle_accuracy_drop"]),
            ),
            min(
                float(row["zero_margin_drop"]),
                float(row["shuffle_margin_drop"]),
            ),
        ),
    )
    return (
        winner["component"],
        float(winner["zero_accuracy_drop"]),
        float(winner["shuffle_accuracy_drop"]),
    )


def aggregate(
    *,
    formal_dir: Path,
    grid_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    formal = json.loads((formal_dir / "combined_summary.json").read_text())
    grid = json.loads((grid_dir / "combined_summary.json").read_text())
    if tuple(formal["models"]) != MODEL_ORDER or tuple(grid["models"]) != MODEL_ORDER:
        raise ValueError("model order or membership is unexpected")

    model_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    executor_rows_all: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        formal_summary = formal["models"][model_name]
        grid_summary = grid["models"][model_name]
        phase_decode = _read_csv(
            formal_dir / model_name / "phase_decode_rows.csv"
        )
        all_phase = next(
            row for row in phase_decode if row["group"] == "all"
        )
        formal_closed = _read_csv(
            formal_dir / model_name / "closed_loop_rows.csv"
        )
        executor_rows = _read_csv(
            grid_dir / model_name / "renewed_executor_rows.csv"
        )
        dominant_head, zero_drop, shuffle_drop = _dominant_b2_head(
            executor_rows
        )
        best = grid_summary["best_closed_loop_control"]
        baseline_curve = [
            float(
                _lookup_closed(
                    formal_closed,
                    condition="baseline",
                    extra_loop=extra_loop,
                    target_kind="phase_programmed",
                )["accuracy"]
            )
            for extra_loop in range(1, 5)
        ]
        model_rows.append(
            {
                "model": model_name,
                "trajectory": "->".join(
                    str(value)
                    for value in formal_summary[
                        "trajectory_positions_including_initial"
                    ]
                ),
                "all_state_age_correlation": float(
                    all_phase["heldout_age_correlation"]
                ),
                "all_state_monotone_fraction": float(
                    all_phase["heldout_monotone_fraction"]
                ),
                "centroid_once_gain": formal_summary["causal_tests"][
                    "centroid_gain_over_baseline"
                ],
                "rank1_once_gain": formal_summary["causal_tests"][
                    "rank1_gain_over_baseline"
                ],
                "best_closed_loop_group": best["group"],
                "best_closed_loop_reference_age": best["reference_age"],
                "best_closed_loop_jump": best["programmed_jump"],
                "best_closed_loop_mean_accuracy": best["mean_accuracy"],
                "best_closed_loop_mean_specificity": best[
                    "mean_specificity"
                ],
                "dominant_renewed_executor": dominant_head,
                "dominant_zero_accuracy_drop": zero_drop,
                "dominant_shuffle_accuracy_drop": shuffle_drop,
                "cuda_peak_reserved_gib": formal_summary[
                    "cuda_peak_reserved_gib"
                ],
            }
        )
        for phase in grid_summary["best_matched_phase_effect_by_age"]:
            phase_rows.append({"model": model_name, **phase})
        for extra_loop, (
            accuracy,
            shuffled,
            baseline_accuracy,
        ) in enumerate(
            zip(
                best["accuracy_by_extra_loop"],
                best["shuffled_accuracy_by_extra_loop"],
                baseline_curve,
                strict=True,
            ),
            start=1,
        ):
            closed_rows.append(
                {
                    "model": model_name,
                    "reference_age": best["reference_age"],
                    "programmed_jump": best["programmed_jump"],
                    "group": best["group"],
                    "extra_loop": extra_loop,
                    "matched_accuracy": accuracy,
                    "shuffled_accuracy": shuffled,
                    "baseline_accuracy": baseline_accuracy,
                }
            )
        for row in executor_rows:
            executor_rows_all.append(
                {
                    "model": model_name,
                    "component": row["component"],
                    "baseline_accuracy": float(row["baseline_accuracy"]),
                    "zero_accuracy_drop": float(
                        row["zero_accuracy_drop"]
                    ),
                    "shuffle_accuracy_drop": float(
                        row["shuffle_accuracy_drop"]
                    ),
                    "zero_margin_drop": float(row["zero_margin_drop"]),
                    "shuffle_margin_drop": float(
                        row["shuffle_margin_drop"]
                    ),
                    "strongly_used": row["strongly_used"],
                }
            )
    _write_csv(out_dir / "model_summary.csv", model_rows)
    _write_csv(out_dir / "phase_jump_by_age.csv", phase_rows)
    _write_csv(out_dir / "closed_loop_best_rows.csv", closed_rows)
    _write_csv(out_dir / "renewed_executor_rows.csv", executor_rows_all)
    _plot_phase_jump(phase_rows, out_dir / "phase_jump_control.png")
    _plot_closed_loop(closed_rows, out_dir / "overloop_control.png")
    _plot_executor(executor_rows_all, out_dir / "renewed_executor.png")
    summary = {
        "models": model_rows,
        "artifacts": {
            "phase_jump_control": "phase_jump_control.png",
            "overloop_control": "overloop_control.png",
            "renewed_executor": "renewed_executor.png",
        },
        "plot_missing_value_policy": (
            "No matrix cell is missing; executor heatmap clips negative "
            "accuracy drops to zero and uses light gray for zero."
        ),
    }
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _plot_phase_jump(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.6), sharey=True)
    colors = {1: "#2878B5", 2: "#D04A35"}
    for ax, model_name in zip(axes, MODEL_ORDER, strict=True):
        selected = [row for row in rows if row["model"] == model_name]
        for jump in (1, 2):
            curve = [row for row in selected if row["reference_jump"] == jump]
            if curve:
                ax.scatter(
                    [row["reference_age"] for row in curve],
                    [row["matched_accuracy"] for row in curve],
                    s=70,
                    color=colors[jump],
                    label=f"{jump}-hop phase",
                    zorder=3,
                )
                ax.scatter(
                    [row["reference_age"] for row in curve],
                    [row["shuffled_accuracy"] for row in curve],
                    s=38,
                    facecolors="none",
                    edgecolors=colors[jump],
                    marker="s",
                    label=f"{jump}-hop shuffled",
                    zorder=2,
                )
        if not selected:
            ax.set_facecolor("#EFEFEF")
            ax.text(
                0.5,
                0.5,
                "no reliably\nresolved phase",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#666666",
                fontsize=9,
            )
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("reference age")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("future-target accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=4,
            fontsize=8,
        )
    fig.suptitle(
        "Same graph + same current node: early phase controls jump size",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_closed_loop(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.6), sharey=True)
    for ax, model_name in zip(axes, MODEL_ORDER, strict=True):
        curve = [row for row in rows if row["model"] == model_name]
        x = [row["extra_loop"] for row in curve]
        ax.plot(
            x,
            [row["matched_accuracy"] for row in curve],
            marker="o",
            color="#D04A35",
            label="matched phase renewal",
        )
        ax.plot(
            x,
            [row["shuffled_accuracy"] for row in curve],
            marker="s",
            color="#7A7A7A",
            linestyle="--",
            label="shuffled delta",
        )
        ax.plot(
            x,
            [row["baseline_accuracy"] for row in curve],
            marker=".",
            color="#2878B5",
            linestyle=":",
            label="no renewal",
        )
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("extra loop")
        ax.set_xticks((1, 2, 3, 4))
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("programmed continuation accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        fontsize=8,
    )
    fig.suptitle(
        "Closed-loop renewal can defeat the endpoint attractor",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_executor(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = {
        (row["model"], row["component"]): row
        for row in rows
    }
    values = np.zeros((len(MODEL_ORDER), len(COMPONENT_ORDER)))
    for model_index, model in enumerate(MODEL_ORDER):
        for component_index, component in enumerate(COMPONENT_ORDER):
            row = lookup[(model, component)]
            values[model_index, component_index] = max(
                0.0,
                min(
                    row["zero_accuracy_drop"],
                    row["shuffle_accuracy_drop"],
                ),
            )
    cmap = LinearSegmentedColormap.from_list(
        "gray_red",
        ("#E6E6E6", "#F2A58F", "#A51E2D"),
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(COMPONENT_ORDER)), COMPONENT_ORDER, rotation=45)
    ax.set_yticks(
        range(len(MODEL_ORDER)),
        [model.replace("_", " ") for model in MODEL_ORDER],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white" if values[row, column] >= 0.55 else "#222222"
                ),
            )
    ax.set_title(
        "Renewed-step executor: min(zero drop, shuffled-patch drop)"
    )
    fig.colorbar(image, ax=ax, label="accuracy drop")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the graph-path telomere/overloop experiment."
    )
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(
        formal_dir=args.formal_dir,
        grid_dir=args.grid_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
