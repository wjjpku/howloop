from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _seed_from_summary(summary: dict[str, Any], path: Path) -> int:
    if "seed" in summary:
        return int(summary["seed"])
    config = summary.get("config")
    if isinstance(config, dict) and "seed" in config:
        return int(config["seed"])
    candidates = (
        str(summary.get("run_name", "")),
        str(summary.get("run_dir", "")),
        path.parent.name,
    )
    matches = [
        match
        for candidate in candidates
        for match in re.findall(r"seed(\d+)", candidate)
    ]
    return int(matches[-1]) if matches else -1


def aggregate_access(root: Path) -> dict[str, Any]:
    runs = []
    for path in sorted(root.glob("*/scorecard.json")):
        scorecard = json.loads(path.read_text(encoding="utf-8"))
        sweep = {
            row["name"]: float(row["accuracy"])
            for row in scorecard["visibility_sweep"]["rows"]
        }
        prefix = [
            sweep.get(f"prefix_{loop}")
            for loop in range(1, 7)
        ]
        minimum_prefix = next(
            (
                loop
                for loop, accuracy in enumerate(prefix, start=1)
                if accuracy is not None and accuracy >= 0.90
            ),
            None,
        )
        reset = scorecard["reset_map"]
        sequential = next(
            (
                item
                for item in scorecard["transplants"]
                if item["target_condition"] == "sequential"
            ),
            None,
        )

        def transplant_value(index: int, key: str) -> float | None:
            if sequential is None or len(sequential["rows"]) <= index:
                return None
            row = sequential["rows"][index]
            values = row.get("covered_accuracy") or row["accuracy"]
            return values.get(key)

        components = scorecard["components"]["rows"]
        runs.append(
            {
                "run_name": path.parent.name,
                "condition": scorecard["condition"],
                "seed": int(scorecard["seed"]),
                "native_accuracy": float(reset["baseline_accuracy"]),
                "prefix_1_accuracy": prefix[0],
                "prefix_3_accuracy": prefix[2],
                "prefix_5_accuracy": prefix[4],
                "prefix_6_accuracy": prefix[5],
                "minimum_prefix_loops_90": minimum_prefix,
                "reset_after_first_reveal_drop": float(
                    reset["baseline_accuracy"]
                    - reset["individual_rows"][1]["accuracy"]
                ),
                "sequential_hybrid_loop1": transplant_value(0, "hybrid"),
                "sequential_hybrid_loop2": transplant_value(1, "hybrid"),
                "attention_hybrid_loop1": float(
                    components[0]["attention_patch_accuracy"]["hybrid"]
                ),
                "attention_hybrid_loop2": float(
                    components[1]["attention_patch_accuracy"]["hybrid"]
                ),
                "attention_hybrid_loop3": float(
                    components[2]["attention_patch_accuracy"]["hybrid"]
                ),
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[row["condition"]].append(row)
    summaries = {}
    for condition, rows in grouped.items():
        summaries[condition] = {
            "seed_count": len(rows),
            "metrics": {
                key: _mean_std(
                    [
                        float(row[key])
                        for row in rows
                        if isinstance(row.get(key), (int, float))
                    ]
                )
                for key in (
                    "native_accuracy",
                    "prefix_1_accuracy",
                    "prefix_3_accuracy",
                    "prefix_5_accuracy",
                    "prefix_6_accuracy",
                    "reset_after_first_reveal_drop",
                    "sequential_hybrid_loop1",
                    "sequential_hybrid_loop2",
                    "attention_hybrid_loop1",
                    "attention_hybrid_loop2",
                    "attention_hybrid_loop3",
                )
            },
        }
    return {"runs": runs, "conditions": summaries}


def aggregate_mixed_runs(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "mode_conditioning" not in summary or "final_metrics" not in summary:
            continue
        accuracies = {
            condition: float(values["final_accuracy"])
            for condition, values in summary["final_metrics"].items()
        }
        rows.append(
            {
                "run_name": summary["run_name"],
                "arm": "mode" if summary["mode_conditioning"] else "nomode",
                "seed": int(summary["seed"]),
                "accuracy": accuracies,
                "minimum_accuracy": min(accuracies.values()),
            }
        )
    arms = {}
    for arm in ("nomode", "mode"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "seed_count": len(arm_rows),
            "behavior_pass": (
                len(arm_rows) >= 3
                and all(row["minimum_accuracy"] >= 0.99 for row in arm_rows)
            ),
            "minimum_accuracy": (
                min(row["minimum_accuracy"] for row in arm_rows)
                if arm_rows
                else None
            ),
            "runs": arm_rows,
        }
    return {"runs": rows, "arms": arms}


def aggregate_graph_controls(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if {
            "target_depth",
            "active_macro_steps",
            "final_metrics",
        } <= summary.keys():
            rows.append(
                {
                    "run_name": summary["run_name"],
                    "family": "compressor",
                    "seed": int(summary["seed"]),
                    "target_depth": int(summary["target_depth"]),
                    "active_macro_steps": int(summary["active_macro_steps"]),
                    "accuracy": float(
                        summary["final_metrics"]["final_target_accuracy"]
                    ),
                }
            )
        elif summary.get("architecture") == "standard":
            rows.append(
                {
                    "run_name": summary.get("run_dir", path.parent.name),
                    "family": "standard_transition",
                    "seed": _seed_from_summary(summary, path),
                    "accuracy": float(
                        summary["final_metrics"]["final_target_acc"]
                    ),
                }
            )
    compressor = [row for row in rows if row["family"] == "compressor"]
    return {
        "runs": rows,
        "compressor_gate": {
            "seed_count": len(compressor),
            "all_below_0_8": (
                len(compressor) >= 3
                and all(row["accuracy"] < 0.80 for row in compressor)
            ),
            "accuracies": [row["accuracy"] for row in compressor],
        },
    }
