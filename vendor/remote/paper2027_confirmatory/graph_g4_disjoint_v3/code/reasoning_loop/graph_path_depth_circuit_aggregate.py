from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    selected = [value for value in values if math.isfinite(value)]
    return statistics.mean(selected) if selected else float("nan")


def finite_sample_std(values: list[float]) -> float:
    selected = [value for value in values if math.isfinite(value)]
    return statistics.stdev(selected) if len(selected) > 1 else 0.0


def target_accuracy(
    rows: list[dict[str, str]],
    *,
    loop: int,
    target_position: int,
) -> float:
    row = next(
        row
        for row in rows
        if int(row["loop"]) == loop
        and int(row["path_position"]) == target_position
    )
    return float(row["accuracy"])


def transplant_means(
    rows: list[dict[str, str]],
    *,
    endpoint_position: int,
) -> tuple[float, float]:
    pretarget = [
        float(row["next_step_accuracy"])
        for row in rows
        if row["donor_resolved"] == "True"
        and int(row["donor_best_path_position"]) < endpoint_position
    ]
    endpoint = [
        float(row["next_step_accuracy"])
        for row in rows
        if row["donor_resolved"] == "True"
        and int(row["donor_best_path_position"]) == endpoint_position
    ]
    return finite_mean(pretarget), finite_mean(endpoint)


def classify_mechanism(
    *,
    target_loop16: float,
    skip_endpoint: float,
    pretarget_next: float,
    endpoint_next: float,
) -> str:
    reusable = math.isfinite(pretarget_next) and pretarget_next >= 0.60
    endpoint_stop = (
        not math.isfinite(endpoint_next)
        or endpoint_next < 0.30
    )
    if reusable and endpoint_stop and target_loop16 >= 0.90:
        return "goal_gated_reusable_transition"
    if reusable:
        return "partially_open_reusable_transition"
    if skip_endpoint >= 0.80 and target_loop16 >= 0.90:
        return "compressed_endpoint_attractor"
    return "compressed_phase_program"


def aggregate(root: Path) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    role_accumulator: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for family_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        family = family_dir.name
        for run_dir in sorted(family_dir.glob("*_seed*")):
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            query_depth = int(summary["query_depth"])
            trained_loops = int(summary["trained_loops"])
            baseline_rows = read_rows(run_dir / "baseline_path_rows.csv")
            target_loop16 = target_accuracy(
                baseline_rows,
                loop=16,
                target_position=query_depth,
            )
            skip_rows = [
                row
                for row in summary["schedule"]
                if row["condition"].startswith("skip_")
            ]
            repeat_rows = [
                row
                for row in summary["schedule"]
                if row["condition"].startswith("repeat_")
            ]
            skip_endpoint = finite_mean(
                [float(row["endpoint_accuracy"]) for row in skip_rows]
            )
            repeat_endpoint = finite_mean(
                [float(row["endpoint_accuracy"]) for row in repeat_rows]
            )
            pretarget_next, endpoint_next = transplant_means(
                read_rows(run_dir / "state_transplant_rows.csv"),
                endpoint_position=query_depth,
            )
            mechanism = classify_mechanism(
                target_loop16=target_loop16,
                skip_endpoint=skip_endpoint,
                pretarget_next=pretarget_next,
                endpoint_next=endpoint_next,
            )
            run_rows.append(
                {
                    "family": family,
                    "run": summary["name"],
                    "checkpoint_step": summary["checkpoint_step"],
                    "query_depth": query_depth,
                    "trained_loops": trained_loops,
                    "trained_endpoint_accuracy": summary["baseline"][
                        "trained_endpoint_accuracy"
                    ],
                    "target_accuracy_loop16": target_loop16,
                    "skip_endpoint_accuracy": skip_endpoint,
                    "repeat_endpoint_accuracy": repeat_endpoint,
                    "pretarget_cross_graph_next_accuracy": pretarget_next,
                    "endpoint_cross_graph_next_accuracy": endpoint_next,
                    "selected_head_count": len(
                        summary["attention_head_circuit"]["selected"]
                    ),
                    "selected_branch_count": len(
                        summary["branch_circuit"]["selected"]
                    ),
                    "mechanism": mechanism,
                }
            )

            branch_ablation = {
                row["site"]: float(row["accuracy_drop"])
                for row in read_rows(
                    run_dir / "effective_branch_ablation_rows.csv"
                )
            }
            patch_rows = {
                row["site"]: row
                for row in read_rows(
                    run_dir / "effective_branch_patching_rows.csv"
                )
            }
            for site, accuracy_drop in branch_ablation.items():
                role = role_accumulator[(family, site)]
                role["accuracy_drop"].append(accuracy_drop)
                role["patch_in"].append(
                    float(patch_rows[site]["patch_in_recovery"])
                )
                role["shuffle"].append(
                    float(patch_rows[site]["shuffle_recovery"])
                )

    family_rows: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in run_rows}):
        selected = [row for row in run_rows if row["family"] == family]
        mechanism_counts: dict[str, int] = defaultdict(int)
        for row in selected:
            mechanism_counts[str(row["mechanism"])] += 1
        family_rows.append(
            {
                "family": family,
                "seed_count": len(selected),
                "endpoint_accuracy_mean": finite_mean(
                    [float(row["trained_endpoint_accuracy"]) for row in selected]
                ),
                "target_loop16_mean": finite_mean(
                    [float(row["target_accuracy_loop16"]) for row in selected]
                ),
                "target_loop16_std": finite_sample_std(
                    [float(row["target_accuracy_loop16"]) for row in selected]
                ),
                "skip_endpoint_mean": finite_mean(
                    [float(row["skip_endpoint_accuracy"]) for row in selected]
                ),
                "pretarget_next_mean": finite_mean(
                    [
                        float(row["pretarget_cross_graph_next_accuracy"])
                        for row in selected
                    ]
                ),
                "endpoint_next_mean": finite_mean(
                    [
                        float(row["endpoint_cross_graph_next_accuracy"])
                        for row in selected
                    ]
                ),
                "mechanism_counts": json.dumps(
                    dict(sorted(mechanism_counts.items())),
                    ensure_ascii=False,
                ),
            }
        )

    role_rows: list[dict[str, Any]] = []
    for (family, site), metrics in sorted(role_accumulator.items()):
        patch_in_mean = finite_mean(metrics["patch_in"])
        shuffle_mean = finite_mean(metrics["shuffle"])
        role_rows.append(
            {
                "family": family,
                "site": site,
                "accuracy_drop_mean": finite_mean(metrics["accuracy_drop"]),
                "accuracy_drop_std": finite_sample_std(metrics["accuracy_drop"]),
                "patch_in_mean": patch_in_mean,
                "shuffle_mean": shuffle_mean,
                "patch_specificity_mean": patch_in_mean - shuffle_mean,
            }
        )
    return {
        "runs": run_rows,
        "families": family_rows,
        "effective_branch_roles": role_rows,
    }


def plot_fingerprint(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        ("target_accuracy_loop16", "target@L16"),
        ("skip_endpoint_accuracy", "skip endpoint"),
        ("repeat_endpoint_accuracy", "repeat endpoint"),
        ("pretarget_cross_graph_next_accuracy", "pretarget next"),
        ("endpoint_cross_graph_next_accuracy", "endpoint next"),
        ("selected_head_count", "heads / 8"),
    ]
    values = []
    for row in rows:
        current = []
        for key, _ in columns:
            value = float(row[key])
            if key == "selected_head_count":
                value /= 8.0
            current.append(value)
        values.append(current)
    matrix = np.asarray(values, dtype=np.float32)
    plt.figure(figsize=(8.2, max(4.8, 0.55 * len(rows) + 2.0)))
    image = plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="score")
    plt.xticks(
        range(len(columns)),
        [label for _, label in columns],
        rotation=35,
        ha="right",
    )
    plt.yticks(
        range(len(rows)),
        [str(row["run"]) for row in rows],
    )
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            plt.text(x, y, label, ha="center", va="center", color="white", fontsize=8)
    plt.title("Graph-depth circuit mechanism fingerprints")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = aggregate(args.analysis_root)
    write_rows(args.out_dir / "run_mechanism_fingerprints.csv", result["runs"])
    write_rows(args.out_dir / "family_overview.csv", result["families"])
    write_rows(
        args.out_dir / "effective_branch_roles.csv",
        result["effective_branch_roles"],
    )
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_fingerprint(
        result["runs"],
        args.out_dir / "mechanism_fingerprints.png",
    )


if __name__ == "__main__":
    main()
