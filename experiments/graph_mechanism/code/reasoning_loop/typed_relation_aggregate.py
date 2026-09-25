from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


RUN_PATTERN = re.compile(
    r"^typed_(?P<arm>.+)_N(?P<nodes>\d+)_d(?P<width>\d+)_seed(?P<seed>\d+)$"
)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _discover(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        run_name = str(summary.get("run_name", path.parent.name))
        match = RUN_PATTERN.fullmatch(run_name)
        if match is None:
            continue
        full = summary["evaluation"]["full"]
        aligned = summary["evaluation"]["aligned"]
        swapped = summary["evaluation"]["swapped"]
        transplant = summary["transplant"]
        endpoint = float(full["trained_endpoint_accuracy"])
        extra = full.get("extra_loop_endpoint_accuracy")
        chance = 1.0 / int(summary["node_count"])
        virtual_depth_pass = (
            endpoint >= 0.99
            and float(full["loop1_first_relation_accuracy"]) >= 0.99
            and extra is not None
            and endpoint - float(extra) >= 0.50
            and float(aligned["trained_endpoint_accuracy"]) >= 0.99
            and float(swapped["trained_endpoint_accuracy"]) <= chance + 0.05
            and transplant.get("status") == "ok"
            and float(transplant["hybrid_full_accuracy"]) >= 0.90
            and float(transplant["hybrid_aligned_accuracy"]) >= 0.90
            and float(transplant["shuffled_workspace_aligned_accuracy"])
            <= chance + 0.05
        )
        rows.append(
            {
                "run_name": run_name,
                "arm": match.group("arm"),
                "node_count": int(summary["node_count"]),
                "width": int(summary["d_model"]),
                "seed": int(summary["seed"]),
                "architecture": summary["architecture"],
                "objective": summary["objective"],
                "anchor_weight": float(summary["anchor_weight"]),
                "train_loops": int(summary["train_loops"]),
                "parameter_count": int(summary["parameter_count"]),
                "best_step": int(summary["best_step"]),
                "chance": chance,
                "endpoint_accuracy": endpoint,
                "loop1_first_relation_accuracy": float(
                    full["loop1_first_relation_accuracy"]
                ),
                "extra_loop_endpoint_accuracy": (
                    float(extra) if extra is not None else None
                ),
                "extra_loop_damage": (
                    endpoint - float(extra) if extra is not None else None
                ),
                "aligned_endpoint_accuracy": float(
                    aligned["trained_endpoint_accuracy"]
                ),
                "swapped_endpoint_accuracy": float(
                    swapped["trained_endpoint_accuracy"]
                ),
                "hybrid_full_accuracy": (
                    float(transplant["hybrid_full_accuracy"])
                    if transplant.get("status") == "ok"
                    else None
                ),
                "hybrid_aligned_accuracy": (
                    float(transplant["hybrid_aligned_accuracy"])
                    if transplant.get("status") == "ok"
                    else None
                ),
                "shuffled_workspace_accuracy": (
                    float(transplant["shuffled_workspace_aligned_accuracy"])
                    if transplant.get("status") == "ok"
                    else None
                ),
                "virtual_depth_pass": virtual_depth_pass,
                "summary_path": str(path.resolve()),
            }
        )
    return rows


def _aggregate_arms(
    rows: list[dict[str, Any]],
    *,
    min_stable_seeds: int,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for arm, group in sorted(grouped.items()):
        endpoints = [float(row["endpoint_accuracy"]) for row in group]
        successes = sum(bool(row["virtual_depth_pass"]) for row in group)
        result[arm] = {
            "seed_count": len({int(row["seed"]) for row in group}),
            "seeds": sorted(int(row["seed"]) for row in group),
            "architecture": sorted({str(row["architecture"]) for row in group}),
            "objective": sorted({str(row["objective"]) for row in group}),
            "anchor_weights": sorted(
                {float(row["anchor_weight"]) for row in group}
            ),
            "endpoint_mean": _mean(endpoints),
            "endpoint_min": min(endpoints),
            "endpoint_max": max(endpoints),
            "loop1_mean": _mean(
                [float(row["loop1_first_relation_accuracy"]) for row in group]
            ),
            "extra_loop_damage_mean": _mean(
                [
                    float(row["extra_loop_damage"])
                    for row in group
                    if row["extra_loop_damage"] is not None
                ]
            ),
            "hybrid_full_mean": _mean(
                [
                    float(row["hybrid_full_accuracy"])
                    for row in group
                    if row["hybrid_full_accuracy"] is not None
                ]
            ),
            "virtual_depth_success_count": successes,
            "virtual_depth_success_rate": successes / len(group),
            "stable_virtual_depth_pass": (
                len({int(row["seed"]) for row in group})
                >= min_stable_seeds
                and successes == len(group)
            ),
        }
    return result


def _gates(
    rows: list[dict[str, Any]],
    arms: dict[str, dict[str, Any]],
    *,
    min_stable_seeds: int,
) -> dict[str, Any]:
    chance_rows = [
        row
        for row in rows
        if int(row["train_loops"]) == 1
        and str(row["objective"]) == "final"
    ]
    final_two_rows = [
        row
        for row in rows
        if int(row["train_loops"]) == 2
        and str(row["architecture"]) == "looped"
        and str(row["objective"]) == "final"
    ]

    def stable_failure(group: list[dict[str, Any]]) -> bool:
        return (
            len({int(row["seed"]) for row in group}) >= min_stable_seeds
            and all(
                float(row["endpoint_accuracy"])
                <= float(row["chance"]) + 0.05
                for row in group
            )
        )

    successful_anchor_weights = [
        float(result["anchor_weights"][0])
        for arm, result in arms.items()
        if result["objective"] == ["anchor"]
        and len(result["anchor_weights"]) == 1
        and result["stable_virtual_depth_pass"]
    ]
    staged_success = any(
        result["objective"] == ["staged"]
        and result["stable_virtual_depth_pass"]
        for result in arms.values()
    )
    one_loop_shortage = stable_failure(chance_rows)
    final_only_shortage = stable_failure(final_two_rows)
    min_weight = (
        min(successful_anchor_weights)
        if successful_anchor_weights
        else None
    )
    if (
        one_loop_shortage
        and final_only_shortage
        and (min_weight is not None or staged_success)
    ):
        classification = (
            "stage_supervision_induces_phase_specific_virtual_depth"
        )
    elif any(
        result["stable_virtual_depth_pass"] for result in arms.values()
    ):
        classification = "phase_specific_virtual_depth_without_full_controls"
    elif rows:
        classification = "no_stable_virtual_depth_at_tested_conditions"
    else:
        classification = "insufficient_evidence"
    return {
        "min_stable_seeds": min_stable_seeds,
        "one_loop_depth_shortage": one_loop_shortage,
        "two_loop_final_only_identifiability_shortage": final_only_shortage,
        "minimum_stable_anchor_weight": min_weight,
        "staged_objective_success": staged_success,
        "classification": classification,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    dose_rows = [
        row
        for row in rows
        if row["objective"] in {"final", "anchor", "staged"}
        and int(row["train_loops"]) == 2
        and row["architecture"] == "looped"
    ]
    if not dose_rows:
        return
    x_values: list[float] = []
    endpoint: list[float] = []
    loop1: list[float] = []
    for row in dose_rows:
        if row["objective"] == "final":
            weight = 0.0
        elif row["objective"] == "staged":
            weight = 1.0
        else:
            weight = float(row["anchor_weight"])
        x_values.append(weight)
        endpoint.append(float(row["endpoint_accuracy"]))
        loop1.append(float(row["loop1_first_relation_accuracy"]))
    order = sorted(range(len(x_values)), key=lambda index: x_values[index])
    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    axis.scatter(
        [x_values[index] for index in order],
        [endpoint[index] for index in order],
        label="endpoint g(f(x))",
        alpha=0.8,
    )
    axis.scatter(
        [x_values[index] for index in order],
        [loop1[index] for index in order],
        label="loop-1 f(x)",
        alpha=0.8,
    )
    axis.axhline(1.0 / int(rows[0]["node_count"]), color="0.5", linestyle=":")
    axis.set(
        xlabel="intermediate-stage loss weight",
        ylabel="accuracy",
        xscale="symlog",
        ylim=(-0.02, 1.03),
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def aggregate_typed_runs(
    root: Path,
    *,
    out_dir: Path | None = None,
    min_stable_seeds: int = 5,
) -> dict[str, Any]:
    if min_stable_seeds < 1:
        raise ValueError("min_stable_seeds must be positive")
    rows = _discover(root)
    arms = _aggregate_arms(rows, min_stable_seeds=min_stable_seeds)
    gates = _gates(
        rows,
        arms,
        min_stable_seeds=min_stable_seeds,
    )
    result = {
        "root": str(root.resolve()),
        "run_count": len(rows),
        "seed_rows": rows,
        "arms": arms,
        "gates": gates,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "aggregate.json").write_text(
            json.dumps(result, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        _write_csv(rows, out_dir / "seed_rows.csv")
        _plot(rows, out_dir / "supervision_dose.png")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate typed-relation virtual-depth experiments."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-stable-seeds", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = aggregate_typed_runs(
        args.root,
        out_dir=args.out_dir,
        min_stable_seeds=args.min_stable_seeds,
    )
    print(json.dumps(result["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
