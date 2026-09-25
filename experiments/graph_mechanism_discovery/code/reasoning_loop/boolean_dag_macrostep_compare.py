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


CONDITIONS = {
    "conditioned": "conditioned_seed0",
    "no_instruction": "no_instruction_seed0_serial",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_matrix(rows: list[dict[str, str]], key: str) -> tuple[list[int], np.ndarray]:
    phases = sorted({int(row["phase"]) for row in rows})
    matrix = np.full((len(phases), 4), np.nan)
    phase_index = {phase: index for index, phase in enumerate(phases)}
    for row in rows:
        matrix[phase_index[int(row["phase"])], int(row["increment"]) - 1] = float(
            row[key]
        )
    return phases, matrix


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    phases: list[int],
    title: str,
) -> None:
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    ax.set(
        title=title,
        xlabel="requested increment d",
        ylabel="prefix cumulative depth",
        xticks=range(4),
        xticklabels=[1, 2, 3, 4],
        yticks=range(len(phases)),
        yticklabels=phases,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 3 else "black",
                )
    return image


def _metrics_for_condition(root: Path, run_name: str) -> dict[str, Any]:
    run = root / run_name
    train = _read_json(run / "summary.json")
    evaluation = _read_json(run / "eval" / "summary.json")
    manifest = _read_json(run / "data_manifest.json")
    phase_rows = _read_csv(run / "eval" / "phase_increment.csv")
    same_total = _read_csv(run / "eval" / "same_total.csv")

    progress_errors = [
        abs(float(row["realized_progress"]) - int(row["increment"]))
        for row in phase_rows
    ]
    stable_errors = [
        abs(float(row["realized_progress"]) - int(row["increment"]))
        for row in phase_rows
        if int(row["phase"]) >= 5
    ]
    request_ranges = []
    for phase in sorted({int(row["phase"]) for row in phase_rows}):
        realized = [
            float(row["mean_realized_depth"])
            for row in phase_rows
            if int(row["phase"]) == phase
        ]
        request_ranges.append(max(realized) - min(realized))

    result: dict[str, Any] = {
        "run_name": run_name,
        "manifest_sha256": manifest["first_batch_sha256"],
        "elapsed_sec": train["elapsed_sec"],
        "parameter_count": train["parameter_count"],
        "train_state_accuracy": train["final_metrics"]["state_accuracy"],
        "train_exact_state_accuracy": train["final_metrics"]["exact_state_accuracy"],
        "train_frontier_accuracy": train["final_metrics"]["frontier_accuracy"],
        "phase_mean_state_accuracy": evaluation["phase_increment_mean_state_accuracy"],
        "phase_min_state_accuracy": evaluation["phase_increment_min_state_accuracy"],
        "realized_progress_mae": float(np.mean(progress_errors)),
        "realized_progress_mae_phase_ge_5": float(np.mean(stable_errors)),
        "mean_request_sensitivity": float(np.mean(request_ranges)),
        "heldout_mean_final_state_accuracy": evaluation[
            "heldout_mean_final_state_accuracy"
        ],
        "overloop_mean_final_state_accuracy": evaluation[
            "overloop_mean_final_state_accuracy"
        ],
        "interchange_follows_swapped": evaluation[
            "interchange_mean_swapped_follows_swapped"
        ],
        "interchange_keeps_clean": evaluation[
            "interchange_mean_swapped_keeps_clean"
        ],
        "interchange_causal_margin_shift": evaluation[
            "interchange_mean_causal_logit_margin_shift"
        ],
    }
    for total in (4, 8):
        rows = [row for row in same_total if int(row["total_depth"]) == total]
        shortest = min(rows, key=lambda row: int(row["program_length"]))
        composed = [row for row in rows if int(row["program_length"]) > 1]
        result[f"depth_{total}_shortest_program"] = shortest["program"]
        result[f"depth_{total}_shortest_state_accuracy"] = float(
            shortest["state_accuracy"]
        )
        result[f"depth_{total}_composed_mean_state_accuracy"] = float(
            np.mean([float(row["state_accuracy"]) for row in composed])
        )
    return result


def _plot_comparison(root: Path, out_dir: Path) -> None:
    phase = {
        condition: _read_csv(root / run / "eval" / "phase_increment.csv")
        for condition, run in CONDITIONS.items()
    }
    interchange = {
        condition: _read_csv(root / run / "eval" / "instruction_interchange.csv")
        for condition, run in CONDITIONS.items()
    }
    overloop = {
        condition: _read_csv(root / run / "eval" / "overloop_programs_by_loop.csv")
        for condition, run in CONDITIONS.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("white")
    images = []
    for ax, condition, title in (
        (axes[0, 0], "conditioned", "Conditioned: realized progress minus request"),
        (
            axes[0, 1],
            "no_instruction",
            "No instruction: realized progress minus request",
        ),
    ):
        phases, realized = _phase_matrix(phase[condition], "realized_progress")
        error = realized - np.arange(1, 5)[None, :]
        images.append(_annotated_heatmap(ax, error, phases=phases, title=title))

    ax = axes[1, 0]
    colors = {"conditioned": "#0072B2", "no_instruction": "#D55E00"}
    for condition in CONDITIONS:
        rows = interchange[condition]
        phases = [int(row["phase"]) for row in rows]
        ax.plot(
            phases,
            [float(row["swapped_affected_follows_swapped"]) for row in rows],
            marker="o",
            color=colors[condition],
            label=f"{condition}: follows swapped",
        )
        ax.plot(
            phases,
            [float(row["swapped_affected_keeps_clean"]) for row in rows],
            linestyle="--",
            color=colors[condition],
            label=f"{condition}: keeps clean",
        )
    ax.set(
        title="Causal instruction interchange",
        xlabel="prefix cumulative depth",
        ylabel="accuracy on oracle-difference nodes",
        ylim=(-0.03, 1.03),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    program = "1+2+3+4+3+3"
    for condition in CONDITIONS:
        rows = [row for row in overloop[condition] if row["program"] == program]
        loops = [int(row["loop"]) for row in rows]
        ax.plot(
            loops,
            [int(row["cumulative_depth"]) for row in rows],
            color="black",
            linewidth=2,
            label="target" if condition == "conditioned" else None,
        )
        ax.plot(
            loops,
            [float(row["mean_realized_depth"]) for row in rows],
            marker="o",
            color=colors[condition],
            label=f"{condition}: realized",
        )
    ax.axvline(4.5, color="#777777", linestyle=":", label="trained-loop boundary")
    ax.set(
        title=f"Overloop behavior: {program}",
        xlabel="loop",
        ylabel="wavefront depth",
        xticks=range(1, 7),
        ylim=(0, 16.5),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.suptitle("Boolean-DAG macro-step comparison", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.07, top=0.93, hspace=0.3, wspace=0.25)
    colorbar_ax = fig.add_axes([0.94, 0.56, 0.015, 0.31])
    fig.colorbar(images[0], cax=colorbar_ax, label="depth error")
    fig.savefig(out_dir / "macrostep_comparison.png", dpi=190, facecolor="white")
    fig.savefig(out_dir / "macrostep_comparison.svg", facecolor="white")
    plt.close(fig)


def _plot_training(root: Path, out_dir: Path) -> None:
    histories = {
        condition: _read_csv(root / run / "history.csv")
        for condition, run in CONDITIONS.items()
    }
    colors = {"conditioned": "#0072B2", "no_instruction": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor("white")
    for condition, rows in histories.items():
        steps = [int(row["step"]) for row in rows]
        axes[0].plot(
            steps,
            [float(row["train_loss"]) for row in rows],
            color=colors[condition],
            label=condition,
        )
        axes[1].plot(
            steps,
            [float(row["eval_state_accuracy"]) for row in rows],
            color=colors[condition],
            label=condition,
        )
        axes[2].plot(
            steps,
            [float(row["eval_frontier_accuracy"]) for row in rows],
            color=colors[condition],
            label=condition,
        )
    for ax, title, ylabel in (
        (axes[0], "Training objective", "loss"),
        (axes[1], "Cumulative state", "accuracy"),
        (axes[2], "New frontier", "accuracy"),
    ):
        ax.set(title=title, xlabel="optimizer step", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "training_comparison.png", dpi=190, facecolor="white")
    fig.savefig(out_dir / "training_comparison.svg", facecolor="white")
    plt.close(fig)


def compare(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        condition: _metrics_for_condition(root, run)
        for condition, run in CONDITIONS.items()
    }
    metrics["manifest_match"] = (
        metrics["conditioned"]["manifest_sha256"]
        == metrics["no_instruction"]["manifest_sha256"]
    )
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    rows = []
    for condition in CONDITIONS:
        rows.append({"condition": condition, **metrics[condition]})
    with (out_dir / "comparison_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_comparison(root, out_dir)
    _plot_training(root, out_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Boolean-DAG macro-step runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.root / "comparison"
    compare(args.root, out_dir)


if __name__ == "__main__":
    main()
