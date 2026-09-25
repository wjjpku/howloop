from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _group(job: str) -> str:
    return job.rsplit("_seed", 1)[0]


def _seed(job: str) -> int:
    return int(job.rsplit("_seed", 1)[1])


def _rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["rows"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prefix_horizon(curve: dict[int, float], threshold: float) -> int:
    horizon = 10
    for length in range(11, max(curve) + 1):
        if curve.get(length, 0.0) < threshold:
            break
        horizon = length
    return horizon


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    data = list(values)
    return mean(data), pstdev(data) if len(data) > 1 else 0.0


def aggregate(j_root: Path, out_dir: Path) -> dict[str, Any]:
    curve_rows: list[dict[str, Any]] = []
    job_summaries: list[dict[str, Any]] = []
    for job_dir in sorted(path for path in j_root.iterdir() if path.is_dir()):
        required = [
            job_dir / "balanced_carry_dense_l1to50" / "summary.json",
            job_dir / "balanced_carry_anchor512" / "summary.json",
            job_dir / "full_answer_audit256" / "summary.json",
            job_dir / "control_full_256" / "summary.json",
            job_dir / "control_no_AB_256" / "summary.json",
            job_dir / "control_identity_D_256" / "summary.json",
            job_dir / "control_no_bias_256" / "summary.json",
            job_dir / "control_executor_off_256" / "summary.json",
        ]
        if not all(path.exists() for path in required):
            continue
        job = job_dir.name
        group = _group(job)
        seed = _seed(job)
        datasets = {
            "balanced_dense": required[0],
            "balanced_anchor": required[1],
            "full_answer": required[2],
            "control_full": required[3],
            "control_no_AB": required[4],
            "control_identity_D": required[5],
            "control_no_bias": required[6],
            "control_executor_off": required[7],
        }
        per_dataset: dict[str, list[dict[str, Any]]] = {}
        for dataset, path in datasets.items():
            per_dataset[dataset] = _rows(path)
            for row in per_dataset[dataset]:
                curve_rows.append(
                    {
                        "group": group,
                        "job": job,
                        "seed": seed,
                        "dataset": dataset,
                        "variant": row["variant"],
                        "length": int(row["length"]),
                        "exact_match": float(row["exact_match"]),
                        "answer_cross_entropy": float(row["answer_cross_entropy"]),
                        "mean_sequence_min_margin": float(
                            row["mean_sequence_min_margin"]
                        ),
                        "examples": int(row["examples"]),
                    }
                )
        dense = per_dataset["balanced_dense"]
        raw = {
            int(row["length"]): float(row["exact_match"])
            for row in dense
            if row["variant"] == "raw"
        }
        full = {
            int(row["length"]): float(row["exact_match"])
            for row in dense
            if row["variant"] == "full"
        }
        job_summaries.append(
            {
                "group": group,
                "job": job,
                "seed": seed,
                "raw_id1to10_mean": mean(raw[length] for length in range(1, 11)),
                "j_id1to10_mean": mean(full[length] for length in range(1, 11)),
                "raw_ood11to50_mean": mean(raw[length] for length in range(11, 51)),
                "j_ood11to50_mean": mean(full[length] for length in range(11, 51)),
                "raw_prefix_horizon_90": _prefix_horizon(raw, 0.90),
                "j_prefix_horizon_90": _prefix_horizon(full, 0.90),
                "raw_prefix_horizon_80": _prefix_horizon(raw, 0.80),
                "j_prefix_horizon_80": _prefix_horizon(full, 0.80),
            }
        )

    if not job_summaries:
        raise ValueError("no complete J jobs found")

    grouped_jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in job_summaries:
        grouped_jobs[row["group"]].append(row)
    group_summaries: list[dict[str, Any]] = []
    for group, members in sorted(grouped_jobs.items()):
        row: dict[str, Any] = {
            "group": group,
            "seeds": "/".join(str(item["seed"]) for item in sorted(members, key=lambda x: x["seed"])),
            "seed_count": len(members),
        }
        for field in (
            "raw_id1to10_mean",
            "j_id1to10_mean",
            "raw_ood11to50_mean",
            "j_ood11to50_mean",
            "raw_prefix_horizon_90",
            "j_prefix_horizon_90",
            "raw_prefix_horizon_80",
            "j_prefix_horizon_80",
        ):
            center, spread = _mean_std(float(item[field]) for item in members)
            row[field] = center
            row[field + "_std"] = spread
        group_summaries.append(row)

    aggregate_curve_rows: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in curve_rows:
        buckets[
            (row["group"], row["dataset"], row["variant"], row["length"])
        ].append(row["exact_match"])
    for (group, dataset, variant, length), values in sorted(buckets.items()):
        center, spread = _mean_std(values)
        aggregate_curve_rows.append(
            {
                "group": group,
                "dataset": dataset,
                "variant": variant,
                "length": length,
                "mean_exact_match": center,
                "std_exact_match": spread,
                "seed_count": len(values),
            }
        )

    control_rows: list[dict[str, Any]] = []
    control_sources = (
        ("raw", "control_full", "raw"),
        ("full", "control_full", "full"),
        ("no_AB", "control_no_AB", "no_AB"),
        ("identity_D", "control_identity_D", "identity_D"),
        ("no_bias", "control_no_bias", "no_bias"),
        ("executor_off", "control_executor_off", "full_executor_off"),
    )
    for group in sorted(grouped_jobs):
        for length in (15, 20, 40):
            row: dict[str, Any] = {"group": group, "length": length}
            for label, dataset, variant in control_sources:
                match = next(
                    item
                    for item in aggregate_curve_rows
                    if item["group"] == group
                    and item["dataset"] == dataset
                    and item["variant"] == variant
                    and item["length"] == length
                )
                row[label + "_mean"] = match["mean_exact_match"]
                row[label + "_std"] = match["std_exact_match"]
            control_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "all_curves.csv", curve_rows)
    _write_csv(out_dir / "aggregate_curves.csv", aggregate_curve_rows)
    _write_csv(out_dir / "per_job_summary.csv", job_summaries)
    _write_csv(out_dir / "per_group_summary.csv", group_summaries)
    _write_csv(out_dir / "control_summary.csv", control_rows)
    result = {
        "status": "complete",
        "j_root": str(j_root),
        "groups": [row["group"] for row in group_summaries],
        "job_count": len(job_summaries),
        "controller_protocol": (
            "rank-48 J=diag(D)+AB with bias; identity initialization; anchor=1; "
            "same J before loops 2 onward; no post-final J; balanced final-carry "
            "CE on k=1..10 at T(k)=k+1; WSD 2048/2816/512"
        ),
        "claim_boundary": (
            "Primary accuracy is a balanced final-carry circuit metric. Full-answer "
            "audits and executor-off/component controls are reported separately; "
            "carry extension alone is not full Addition length generalization."
        ),
        "per_group": group_summaries,
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    groups = [row["group"] for row in group_summaries]
    cols = 2
    rows_count = math.ceil(len(groups) / cols)
    figure, axes = plt.subplots(rows_count, cols, figsize=(12, 3.7 * rows_count), squeeze=False)
    for axis, group in zip(axes.flat, groups):
        for variant, color, label in (("raw", "#777777", "raw"), ("full", "#d55e00", "J")):
            points = [
                row
                for row in aggregate_curve_rows
                if row["group"] == group
                and row["dataset"] == "balanced_dense"
                and row["variant"] == variant
            ]
            x = np.array([row["length"] for row in points])
            y = np.array([row["mean_exact_match"] for row in points])
            s = np.array([row["std_exact_match"] for row in points])
            axis.plot(x, y, color=color, linewidth=2, label=label)
            axis.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), color=color, alpha=0.15)
        axis.axvline(10, color="black", linestyle="--", linewidth=1)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(group)
        axis.set_xlabel("logical length / registered loops = n+1")
        axis.set_ylabel("balanced final-carry accuracy")
        axis.grid(alpha=0.2)
        axis.legend()
    for axis in list(axes.flat)[len(groups):]:
        axis.axis("off")
    figure.suptitle("Addition carry circuit: raw vs learned recurrent interface J")
    figure.tight_layout()
    figure.savefig(out_dir / "carry_length_curves.png", dpi=180)
    figure.savefig(out_dir / "carry_length_curves.pdf")
    plt.close(figure)

    x = np.arange(len(groups))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(9, 1.6 * len(groups)), 5.5))
    raw_h = [row["raw_prefix_horizon_90"] for row in group_summaries]
    j_h = [row["j_prefix_horizon_90"] for row in group_summaries]
    axis.bar(x - width / 2, raw_h, width, label="raw", color="#777777")
    axis.bar(x + width / 2, j_h, width, label="J", color="#d55e00")
    axis.axhline(10, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, groups, rotation=25, ha="right")
    axis.set_ylabel("90% prefix horizon (dense lengths 11–50)")
    axis.set_title("Three-seed carry horizon")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "carry_horizon90.png", dpi=180)
    figure.savefig(out_dir / "carry_horizon90.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    labels = ["raw", "full", "no_AB", "identity_D", "no_bias", "executor_off"]
    control_colors = ["#777777", "#d55e00", "#4c78a8", "#54a24b", "#b279a2", "#e45756"]
    width = 0.12
    for axis, length in zip(axes, (15, 20, 40)):
        selected = [row for row in control_rows if row["length"] == length]
        for offset_index, (label, color) in enumerate(zip(labels, control_colors)):
            offset = (offset_index - (len(labels) - 1) / 2) * width
            axis.bar(
                np.arange(len(groups)) + offset,
                [row[label + "_mean"] for row in selected],
                width,
                color=color,
                label=label,
            )
        axis.set_xticks(np.arange(len(groups)), groups, rotation=25, ha="right")
        axis.set_title(f"n={length}")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("balanced final-carry accuracy")
    axes[-1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    figure.suptitle("Matched-sample component and executor controls")
    figure.tight_layout()
    figure.savefig(out_dir / "carry_controls.png", dpi=180, bbox_inches="tight")
    figure.savefig(out_dir / "carry_controls.pdf", bbox_inches="tight")
    plt.close(figure)

    report = [
        "# Addition architecture × J report",
        "",
        result["controller_protocol"] + ".",
        "",
        "| group | seeds | raw ID | J ID | raw OOD | J OOD | raw H90 | J H90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_summaries:
        report.append(
            f"| {row['group']} | {row['seed_count']} | {row['raw_id1to10_mean']:.3f} "
            f"| {row['j_id1to10_mean']:.3f} | {row['raw_ood11to50_mean']:.3f} "
            f"| {row['j_ood11to50_mean']:.3f} | {row['raw_prefix_horizon_90']:.1f} "
            f"| {row['j_prefix_horizon_90']:.1f} |"
        )
    report.extend([
        "",
        "## Matched component controls at n=20",
        "",
        "| group | raw | full J | no AB | D=I | no bias | executor off |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in [item for item in control_rows if item["length"] == 20]:
        report.append(
            f"| {row['group']} | {row['raw_mean']:.3f} | {row['full_mean']:.3f} "
            f"| {row['no_AB_mean']:.3f} | {row['identity_D_mean']:.3f} "
            f"| {row['no_bias_mean']:.3f} | {row['executor_off_mean']:.3f} |"
        )
    report.extend(["", "Boundary: " + result["claim_boundary"], ""])
    (out_dir / "REPORT.md").write_text("\n".join(report))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(aggregate(args.j_root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
