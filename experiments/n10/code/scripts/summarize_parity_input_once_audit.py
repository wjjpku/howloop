"""Create compact cross-seed tables and figures for the input-once Parity audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def summarize(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    far_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for seed in range(3):
        raw_root = args.root / "raw_evaluation" / f"seed{seed}"
        far = json.loads((raw_root / "far_horizon" / "summary.json").read_text())
        for row in far["rows"]:
            far_rows.append(
                {
                    "backbone_seed": seed,
                    "length": int(row["length"]),
                    "target_step": int(row["target_steps"]),
                    "exact_match": float(row["exact_match"]),
                    "mean_sequence_min_margin": float(
                        row["mean_sequence_min_margin"]
                    ),
                    "examples": int(row["examples"]),
                }
            )
        phase = json.loads(
            (raw_root / "four_phase" / "summary.json").read_text()
        )
        dynamics = phase["dynamics"]["raw"]["heldout_evaluation"]
        origin = phase["phase_origin_locking"][0]
        phase_rows.append(
            {
                "backbone_seed": seed,
                "period_calls": float(dynamics["polar_rotation_period_calls"]),
                "transition_r2": float(dynamics["transition_r2"]),
                "phase_plane_energy_fraction": float(
                    dynamics["shared_phase_plane_energy_fraction"]
                ),
                "period4_harmonic_energy_fraction": float(
                    dynamics["harmonic_energy_fraction_by_period"]["4"]
                ),
                "readout_in_phase_plane_energy_fraction": float(
                    dynamics["readout_direction_in_phase_plane_energy_fraction"]
                ),
                "phase_origin_slope_per_length": float(
                    origin["unwrapped_angle_slope_per_length"]
                ),
                "phase_origin_fit_r2": float(origin["unwrapped_linear_fit_r2"]),
                "mlp_skip_fraction_closer_to_previous_phase": float(
                    phase["mlp_skip_hidden_lag"][
                        "fraction_closer_to_clean_previous_phase"
                    ]
                ),
            }
        )
        heat = json.loads(
            (raw_root / "loop_depth_heatmap" / "summary.json").read_text()
        )
        for length, row in heat["variants"]["raw"]["per_length"].items():
            offset_rows.append(
                {
                    "backbone_seed": seed,
                    "length": int(length),
                    "best_offset": int(row["best_offset"]),
                    "target_accuracy": float(row["target_accuracy"]),
                    "best_accuracy": float(row["best_parity_token_accuracy"]),
                }
            )
        for row in read_csv(
            args.root
            / "input_path_ablation"
            / f"seed{seed}"
            / "input_path_ablation.csv"
        ):
            ablation_rows.append(
                {
                    "backbone_seed": seed,
                    "condition": row["condition"],
                    "length": int(row["length"]),
                    "relative_depth": int(row["relative_depth"]),
                    "accuracy": float(row["accuracy"]),
                    "input_injection_rms": float(row["input_injection_rms"]),
                }
            )

    write_csv(args.out_dir / "far_horizon_across_seeds.csv", far_rows)
    write_csv(args.out_dir / "phase_summary_across_seeds.csv", phase_rows)
    write_csv(args.out_dir / "best_offset_across_seeds.csv", offset_rows)
    write_csv(args.out_dir / "input_path_ablation_across_seeds.csv", ablation_rows)

    figure, axis = plt.subplots(figsize=(8, 5))
    for seed in range(3):
        curve = sorted(
            [row for row in far_rows if row["backbone_seed"] == seed],
            key=lambda row: row["length"],
        )
        axis.plot(
            [row["length"] for row in curve],
            [row["exact_match"] for row in curve],
            marker="o",
            label=f"seed {seed}",
        )
    axis.axvspan(1, 20, color="gray", alpha=0.12, label="backbone train lengths")
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axis.set(xlabel="logical length n", ylabel="exact-match accuracy", ylim=(-0.02, 1.02))
    axis.set_title("Input-once Parity: raw accuracy at registered T(n)=n")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.out_dir / "far_horizon_across_seeds.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for seed in range(3):
        curve = sorted(
            [row for row in offset_rows if row["backbone_seed"] == seed],
            key=lambda row: row["length"],
        )
        axes[0].plot(
            [row["length"] for row in curve],
            [row["best_offset"] for row in curve],
            label=f"seed {seed}",
        )
        axes[1].plot(
            [row["length"] for row in curve],
            [row["target_accuracy"] for row in curve],
            label=f"seed {seed}",
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("best loop offset t*-n")
    axes[0].set_title("Length-dependent phase drift")
    axes[1].set(xlabel="logical length n", ylabel="accuracy at T(n)=n", ylim=(-0.02, 1.02))
    for axis in axes:
        axis.axvline(20, color="gray", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(args.out_dir / "best_offset_and_target_accuracy.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    labels = {
        "registered_initial_only": "input once (registered)",
        "legacy_every_loop": "clean replay every call",
        "single_clean_replay_after_endpoint": "one clean replay at n+1",
        "single_shuffled_replay_after_endpoint": "one shuffled replay at n+1",
        "no_input": "no token entry",
    }
    for seed, axis in enumerate(axes):
        selected = [
            row
            for row in ablation_rows
            if row["backbone_seed"] == seed and row["length"] == 100
        ]
        for condition, label in labels.items():
            curve = sorted(
                [row for row in selected if row["condition"] == condition],
                key=lambda row: row["relative_depth"],
            )
            axis.plot(
                [row["relative_depth"] for row in curve],
                [row["accuracy"] for row in curve],
                marker="o",
                markersize=3,
                label=label,
            )
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"seed {seed}, n=100")
        axis.grid(alpha=0.2)
        axis.set_xlabel("relative depth d=t-n")
    axes[0].set_ylabel("accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Explicit input-edge ablation")
    figure.tight_layout(rect=(0, 0.15, 1, 0.94))
    figure.savefig(args.out_dir / "input_path_ablation_n100.png", dpi=190)
    plt.close(figure)


def main() -> None:
    summarize(parse_args())


if __name__ == "__main__":
    main()
