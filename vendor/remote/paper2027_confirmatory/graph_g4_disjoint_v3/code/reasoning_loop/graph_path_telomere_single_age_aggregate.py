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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    *,
    primary_path: Path,
    replica_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "primary": json.loads(primary_path.read_text(encoding="utf-8")),
        "replica": json.loads(replica_path.read_text(encoding="utf-8")),
    }
    curve_rows = []
    transition_rows = []
    for run_name, summary in runs.items():
        for condition, curve in summary["control_curves"].items():
            for extra_loop, accuracy in enumerate(curve, start=1):
                curve_rows.append(
                    {
                        "run": run_name,
                        "condition": condition,
                        "extra_loop": extra_loop,
                        "accuracy": accuracy,
                    }
                )
        for row in summary["heldout_relative_mse_by_transition"]:
            transition_rows.append({"run": run_name, **row})
    _write_csv(out_dir / "single_age_curves.csv", curve_rows)
    _write_csv(out_dir / "adjacent_age_mse.csv", transition_rows)

    main_curves = np.asarray(
        [runs[name]["accuracy_by_extra_loop"] for name in runs]
    )
    result = {
        "model": runs["primary"]["name"],
        "checkpoint": runs["primary"]["checkpoint"],
        "single_operator": True,
        "matrix_shape": runs["primary"]["homogeneous_matrix_shape"],
        "target_age": runs["primary"]["target_age"],
        "initial_applications": runs["primary"][
            "initial_rewind_applications"
        ],
        "cycle_applications": runs["primary"][
            "cycle_rewind_applications"
        ],
        "primary_accuracy_by_extra_loop": runs["primary"][
            "accuracy_by_extra_loop"
        ],
        "replica_accuracy_by_extra_loop": runs["replica"][
            "accuracy_by_extra_loop"
        ],
        "mean_accuracy_by_extra_loop": main_curves.mean(axis=0).tolist(),
        "primary_mean_accuracy": runs["primary"]["mean_accuracy"],
        "replica_mean_accuracy": runs["replica"]["mean_accuracy"],
        "control_mean_accuracy": {
            run_name: {
                condition: float(np.mean(curve))
                for condition, curve in summary["control_curves"].items()
            }
            for run_name, summary in runs.items()
        },
    }
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    loops = np.arange(1, main_curves.shape[1] + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(
        loops,
        main_curves[0],
        marker="o",
        linewidth=2.2,
        label="single J: primary",
    )
    axes[0].plot(
        loops,
        main_curves[1],
        marker="o",
        linewidth=2.2,
        label="single J: replica",
    )
    for condition, label in (
        ("under_rewind", "J x5 (under)"),
        ("over_rewind", "J x7 (over)"),
        ("batch_shuffled", "shuffled"),
        ("baseline", "no J"),
    ):
        axes[0].plot(
            loops,
            runs["primary"]["control_curves"][condition],
            linestyle="--",
            linewidth=1.3,
            label=label,
        )
    axes[0].axhline(
        1 / 8,
        color="0.5",
        linewidth=1.0,
        linestyle=":",
        label="chance (1/8)",
    )
    axes[0].set(
        xlabel="extra recurrent loop",
        ylabel="collision-controlled accuracy",
        title="D8L8-seed1 finite lifespan extension",
        xticks=loops,
        ylim=(-0.03, 1.04),
    )
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)

    for run_name, marker in (("primary", "o"), ("replica", "s")):
        rows = runs[run_name]["heldout_relative_mse_by_transition"]
        labels = [
            f"{row['source_age']}→{row['target_age']}" for row in rows
        ]
        values = [row["relative_mse"] for row in rows]
        axes[1].plot(
            labels,
            values,
            marker=marker,
            linewidth=2.0,
            label=run_name,
        )
    axes[1].set(
        xlabel="one-age transition",
        ylabel="held-out relative MSE",
        title="One J is least exact at young execution ages",
        ylim=(0, 0.52),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(out_dir / "single_age_J_results.png", dpi=180)
    plt.close(figure)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--replica", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(
        primary_path=args.primary,
        replica_path=args.replica,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
