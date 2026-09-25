from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MODES = ("raw", "full", "no_AB", "identity_D", "full_executor_off")
MODE_SOURCES = {
    "raw": ("full", "raw"),
    "full": ("full", "full"),
    "no_AB": ("no_AB", "no_AB"),
    "identity_D": ("identity_D", "identity_D"),
    "full_executor_off": ("full_executor_off", "full_executor_off"),
}
COLORS = {
    "raw": "#5f6b73",
    "full": "#d62728",
    "no_AB": "#ff7f0e",
    "identity_D": "#2ca02c",
    "full_executor_off": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--backbone-train-max-length", type=int, default=19)
    parser.add_argument("--j-train-max-length", type=int, default=20)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete summary: {path}")
    return payload


def load_seed(root: Path, seed: int) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    summaries = {
        directory: load_summary(root / f"seed{seed}" / directory / "summary.json")
        for directory in ("full", "no_AB", "identity_D", "full_executor_off")
    }
    reference = summaries["full"]
    for directory, payload in summaries.items():
        for key in ("checkpoint", "checkpoint_step", "task", "target_loop_rule", "seed"):
            if payload[key] != reference[key]:
                raise ValueError(f"seed {seed}: {directory} mismatches {key}")
        if payload.get("controller_post_final_j") is not True:
            raise ValueError(f"seed {seed}: {directory} did not use post-final J")

    by_summary: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for directory, payload in summaries.items():
        by_summary[directory] = {
            (int(row["length"]), str(row["variant"])): row
            for row in payload["rows"]
        }
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for mode, (directory, variant) in MODE_SOURCES.items():
        source = by_summary[directory]
        for length in reference["lengths"]:
            rows[(int(length), mode)] = dict(source[(int(length), variant)])

    raw_reference = by_summary["full"]
    for directory, source in by_summary.items():
        for length in reference["lengths"]:
            first = raw_reference[(int(length), "raw")]
            second = source[(int(length), "raw")]
            for metric in ("exact_match", "answer_token_accuracy"):
                if abs(float(first[metric]) - float(second[metric])) > 1e-12:
                    raise ValueError(
                        f"seed {seed}: raw mismatch in {directory}, length {length}, {metric}"
                    )
    return rows, reference


def frontier(rows: dict[tuple[int, str], dict[str, Any]], mode: str, threshold: float) -> int | None:
    passing = [
        length
        for length, candidate_mode in rows
        if candidate_mode == mode and float(rows[(length, mode)]["exact_match"]) >= threshold
    ]
    return max(passing) if passing else None


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: dict[int, dict[tuple[int, str], dict[str, Any]]] = {}
    metadata: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        seed_rows[seed], metadata[seed] = load_seed(args.root, seed)

    lengths = sorted({length for rows in seed_rows.values() for length, _ in rows})
    flat_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for length in lengths:
            for mode in MODES:
                row = seed_rows[seed][(length, mode)]
                flat_rows.append(
                    {
                        "backbone_seed": seed,
                        "length": length,
                        "target_steps": int(row["target_steps"]),
                        "mode": mode,
                        "examples": int(row["examples"]),
                        "exact_match": float(row["exact_match"]),
                        "answer_token_accuracy": float(row["answer_token_accuracy"]),
                        "answer_cross_entropy": float(row["answer_cross_entropy"]),
                        "mean_sequence_min_margin": float(row["mean_sequence_min_margin"]),
                    }
                )
    write_csv(args.out_dir / "matched_component_rows.csv", flat_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for length in lengths:
        for mode in MODES:
            exact = [
                float(seed_rows[seed][(length, mode)]["exact_match"])
                for seed in args.seeds
            ]
            accuracy = [
                float(seed_rows[seed][(length, mode)]["answer_token_accuracy"])
                for seed in args.seeds
            ]
            margin = [
                float(seed_rows[seed][(length, mode)]["mean_sequence_min_margin"])
                for seed in args.seeds
            ]
            exact_mean, exact_std = mean_std(exact)
            accuracy_mean, accuracy_std = mean_std(accuracy)
            margin_mean, margin_std = mean_std(margin)
            aggregate_rows.append(
                {
                    "length": length,
                    "mode": mode,
                    "seed_count": len(args.seeds),
                    "exact_match_mean": exact_mean,
                    "exact_match_std": exact_std,
                    "exact_match_min": min(exact),
                    "exact_match_max": max(exact),
                    "answer_token_accuracy_mean": accuracy_mean,
                    "answer_token_accuracy_std": accuracy_std,
                    "mean_sequence_min_margin_mean": margin_mean,
                    "mean_sequence_min_margin_std": margin_std,
                }
            )
    write_csv(args.out_dir / "matched_component_aggregate.csv", aggregate_rows)

    aggregate_index = {
        (int(row["length"]), str(row["mode"])): row for row in aggregate_rows
    }
    seed_summary: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows = seed_rows[seed]
        mean_delta = float(
            np.mean(
                [
                    float(rows[(length, "full")]["exact_match"])
                    - float(rows[(length, "raw")]["exact_match"])
                    for length in lengths
                ]
            )
        )
        seed_summary.append(
            {
                "backbone_seed": seed,
                "raw_q90_frontier": frontier(rows, "raw", 0.90),
                "full_q90_frontier": frontier(rows, "full", 0.90),
                "raw_q50_frontier": frontier(rows, "raw", 0.50),
                "full_q50_frontier": frontier(rows, "full", 0.50),
                "mean_anchor_delta_em": mean_delta,
            }
        )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(
        "Copy4 frozen-backbone J: matched three-seed component controls",
        fontsize=16,
    )
    for axis_index, axis in enumerate(axes.flat):
        backbone_label = (
            f"backbone train max = {args.backbone_train_max_length}"
            if axis_index == 0
            else "_nolegend_"
        )
        j_label = (
            f"J train max = {args.j_train_max_length}"
            if axis_index == 0
            else "_nolegend_"
        )
        axis.axvspan(1, args.j_train_max_length, color="#dbe7f3", alpha=0.45)
        axis.axvline(
            args.backbone_train_max_length,
            color="#4c78a8",
            linestyle=":",
            linewidth=1.2,
            label=backbone_label,
        )
        axis.axvline(
            args.j_train_max_length,
            color="#4c78a8",
            linestyle="--",
            linewidth=1.2,
            label=j_label,
        )
        axis.grid(alpha=0.25)

    axis = axes[0, 0]
    for mode in ("raw", "full"):
        mean = np.asarray(
            [aggregate_index[(length, mode)]["exact_match_mean"] for length in lengths]
        )
        low = np.asarray(
            [aggregate_index[(length, mode)]["exact_match_min"] for length in lengths]
        )
        high = np.asarray(
            [aggregate_index[(length, mode)]["exact_match_max"] for length in lengths]
        )
        axis.plot(lengths, mean, marker="o", color=COLORS[mode], label=mode)
        axis.fill_between(lengths, low, high, color=COLORS[mode], alpha=0.15)
    axis.set(title="Mean EM; band = seed range", xlabel="logical length", ylabel="EM")
    axis.set_ylim(-0.03, 1.03)
    axis.legend()

    axis = axes[0, 1]
    for mode in ("no_AB", "identity_D", "full_executor_off"):
        mean = [aggregate_index[(length, mode)]["exact_match_mean"] for length in lengths]
        axis.plot(lengths, mean, marker="o", color=COLORS[mode], label=mode)
    axis.plot(
        lengths,
        [aggregate_index[(length, "raw")]["exact_match_mean"] for length in lengths],
        color=COLORS["raw"],
        linestyle="--",
        label="raw",
    )
    axis.set(title="Component and executor controls", xlabel="logical length", ylabel="EM")
    axis.set_ylim(-0.03, 1.03)
    axis.legend()

    axis = axes[1, 0]
    for seed in args.seeds:
        delta = [
            float(seed_rows[seed][(length, "full")]["exact_match"])
            - float(seed_rows[seed][(length, "raw")]["exact_match"])
            for length in lengths
        ]
        axis.plot(lengths, delta, marker="o", label=f"seed {seed}")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(title="Per-backbone J effect", xlabel="logical length", ylabel="full EM - raw EM")
    axis.legend()

    axis = axes[1, 1]
    for mode in ("raw", "full", "no_AB", "identity_D"):
        mean = [
            aggregate_index[(length, mode)]["mean_sequence_min_margin_mean"]
            for length in lengths
        ]
        axis.plot(lengths, mean, marker="o", color=COLORS[mode], label=mode)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(title="Sequence minimum margin", xlabel="logical length", ylabel="mean minimum margin")
    axis.legend()

    figure.savefig(args.out_dir / "copy4_matched_component_controls.png", dpi=180)
    figure.savefig(args.out_dir / "copy4_matched_component_controls.pdf")
    plt.close(figure)

    mode_anchor_means = {
        mode: float(
            np.mean(
                [
                    float(seed_rows[seed][(length, mode)]["exact_match"])
                    for seed in args.seeds
                    for length in lengths
                ]
            )
        )
        for mode in MODES
    }
    summary = {
        "status": "complete",
        "backbone_seeds": list(args.seeds),
        "lengths": lengths,
        "backbone_train_max_length": args.backbone_train_max_length,
        "j_train_max_length": args.j_train_max_length,
        "examples_per_seed_length": int(metadata[args.seeds[0]]["examples_per_length"]),
        "evaluation_seed": int(metadata[args.seeds[0]]["seed"]),
        "post_final_j": True,
        "mode_anchor_mean_em": mode_anchor_means,
        "seed_summary": seed_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Copy4 matched component-control report",
        "",
        f"Three backbone seeds ({', '.join(map(str, args.seeds))}); backbone train max 19; J train range 1-20; 512 matched examples per length; post-final J enabled.",
        "",
        "| seed | raw q90 | full q90 | raw q50 | full q50 | mean anchor delta EM |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summary:
        lines.append(
            f"| {row['backbone_seed']} | {row['raw_q90_frontier']} | {row['full_q90_frontier']} | "
            f"{row['raw_q50_frontier']} | {row['full_q50_frontier']} | {row['mean_anchor_delta_em']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean EM across all registered anchor points",
            "",
            "| mode | mean EM |",
            "|---|---:|",
        ]
    )
    for mode in MODES:
        lines.append(f"| {mode} | {mode_anchor_means[mode]:.4f} |")
    lines.extend(
        [
            "",
            "The raw row is verified identical across all component runs. `no_AB` tests the diagonal-plus-bias path; `identity_D` retains AB and bias while forcing D=I; executor-off tests whether J alone can replace the frozen recurrent executor.",
            "",
            "![matched controls](copy4_matched_component_controls.png)",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
