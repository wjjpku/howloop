from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _group(job: str) -> str:
    return job.rsplit("_seed", 1)[0]


def aggregate(root: Path, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/spectrum/spectrum_summary.json")):
        value = json.loads(path.read_text())
        job = path.parents[1].name
        rows.append(
            {
                "group": _group(job),
                "job": job,
                "seed": int(job.rsplit("_seed", 1)[1]),
                "D_mean": value["D"]["mean"],
                "D_std": value["D"]["std"],
                "D_delta_l2": value["D"]["delta_from_one"]["l2_norm"],
                "A_l2": value["A"]["l2_norm"],
                "B_l2": value["B"]["l2_norm"],
                "bias_l2": value["bias"]["l2_norm"],
                "AB_frobenius": value["AB"]["frobenius_norm"],
                "AB_operator_norm": value["AB"]["operator_norm"],
                "AB_numerical_rank": value["AB"]["numerical_rank_at_1e-6_relative"],
                "J_spectral_radius": value["J"]["spectral_radius"],
                "J_operator_norm": value["J"]["operator_norm"],
                "J_minimum_singular": value["J"]["minimum_singular_value"],
                "J_nonnormality": value["J"]["nonnormality_relative"],
                "delta_J_frobenius": value["delta_J"]["frobenius_norm"],
                "delta_J_operator_norm": value["delta_J"]["operator_norm"],
                "delta_J_stable_rank": value["delta_J"]["stable_rank"],
                "AB_delta_fraction": value["delta_J"]["AB_frobenius_fraction"],
                "D_delta_fraction": value["delta_J"]["diagonal_delta_frobenius_fraction"],
            }
        )
    if not rows:
        raise ValueError("no spectrum summaries found")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_job.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    group_rows: list[dict[str, Any]] = []
    metric_fields = list(rows[0])[3:]
    for group, members in sorted(groups.items()):
        item: dict[str, Any] = {"group": group, "seed_count": len(members)}
        for field in metric_fields:
            values = [float(row[field]) for row in members]
            item[field + "_mean"] = mean(values)
            item[field + "_std"] = pstdev(values) if len(values) > 1 else 0.0
        group_rows.append(item)
    with (out_dir / "per_group.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)

    labels = [row["group"] for row in group_rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("D_delta_l2_mean", "||D-I||₂"),
        ("AB_operator_norm_mean", "||AB||₂"),
        ("delta_J_stable_rank_mean", "stable rank(J-I)"),
        ("J_spectral_radius_mean", "spectral radius(J), diagnostic"),
    )
    for axis, (field, title) in zip(axes.flat, panels):
        axis.bar(x, [row[field] for row in group_rows], color="#4c78a8")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Learned recurrent interface spectra (J in isolation)")
    figure.savefig(out_dir / "spectrum_comparison.png", dpi=180)
    figure.savefig(out_dir / "spectrum_comparison.pdf")
    plt.close(figure)

    result = {
        "status": "complete",
        "row_vector_convention": "h_out=h diag(D)+(hA)B+b=hJ+b",
        "groups": group_rows,
        "interpretation_boundary": (
            "J-only spectra are recurrent-dynamics diagnostics, not a causal "
            "stability proof for the trajectory-dependent composition F after J."
        ),
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(aggregate(args.root, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
