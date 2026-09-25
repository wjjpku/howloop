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


CONTROL_NAMES = (
    "outer_G0_inner_LN",
    "outer_G1_inner_LN",
    "outer_G4_inner_LN",
    "outer_G16_inner_LN",
)


def build_control_rows(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = set(CONTROL_NAMES).difference(summaries)
    if missing:
        raise ValueError(f"missing outer-control summaries: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for name in CONTROL_NAMES:
        summary = summaries[name]
        accuracy = np.asarray(summary["root_accuracy"], dtype=float)[:4].mean(axis=0)
        dynamics = summary["dynamics_rows"]
        for loop_index, mean_accuracy in enumerate(accuracy, start=1):
            loop_rows = [
                row
                for row in dynamics
                if int(row["loop"]) == loop_index and int(row["depth"]) <= 4
            ]
            state_norm = float(np.mean([row["state_norm"] for row in loop_rows]))
            update_norm = float(
                np.mean([row["effective_update_norm"] for row in loop_rows])
            )
            rows.append(
                {
                    "condition": name,
                    "loop": loop_index,
                    "mean_id_accuracy": float(mean_accuracy),
                    "state_norm": state_norm,
                    "effective_update_norm": update_norm,
                    "relative_update_norm": update_norm / max(state_norm, 1e-12),
                    "tangential_fraction": float(
                        np.mean([row["tangential_fraction"] for row in loop_rows])
                    ),
                }
            )
    return rows


def write_control_artifacts(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    trained_horizon: int,
) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "outer_control_loop30.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in CONTROL_NAMES:
        condition_rows = [row for row in rows if row["condition"] == name]
        loops = [int(row["loop"]) for row in condition_rows]
        axes[0].plot(
            loops,
            [row["mean_id_accuracy"] for row in condition_rows],
            linewidth=1.8,
            label=name,
        )
        axes[1].plot(
            loops,
            [row["state_norm"] for row in condition_rows],
            linewidth=1.8,
            label=name,
        )
        axes[2].plot(
            loops,
            [row["relative_update_norm"] for row in condition_rows],
            linewidth=1.8,
            label=name,
        )
    axes[0].set(
        title="Answer stability",
        xlabel="loop",
        ylabel="mean root accuracy, depths 1-4",
        ylim=(0, 1.02),
    )
    axes[1].set(
        title="Residual-stream magnitude",
        xlabel="loop",
        ylabel="mean hidden-state norm",
        yscale="log",
    )
    axes[2].set(
        title="Relative recurrent step size",
        xlabel="loop",
        ylabel="effective update norm / state norm",
        yscale="log",
    )
    for ax in axes:
        ax.axvline(trained_horizon, color="black", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "outer_control_loop30.png", dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the no-outer control report.")
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    summaries = {
        name: json.loads(
            (args.result_root / "analyses_loop30" / name / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for name in CONTROL_NAMES
    }
    rows = build_control_rows(summaries)
    write_control_artifacts(rows=rows, out_dir=args.result_root, trained_horizon=4)


if __name__ == "__main__":
    main()
