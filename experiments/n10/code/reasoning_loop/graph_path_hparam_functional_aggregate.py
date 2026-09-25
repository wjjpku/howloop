from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasoning_loop.graph_path_hparam_screen import wilson_interval


def parse_named_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, raw_path = text.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    return name, Path(raw_path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_greater(value: str | float | None, threshold: float) -> bool:
    if value is None:
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric > threshold


def metric_row(
    rows: Sequence[dict[str, str]],
    *,
    site: str,
    role: str,
) -> dict[str, str] | None:
    return next(
        (
            row
            for row in rows
            if row["site"] == site and row["role_probe"] == role
        ),
        None,
    )


def localize_executor(
    rows: Sequence[dict[str, str]],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for site in sorted({row["site"] for row in rows}):
        output = metric_row(rows, site=site, role="attention_output_answer")
        current = metric_row(rows, site=site, role="pattern_current_edge")
        random = metric_row(rows, site=site, role="pattern_random_edge")
        mlp = metric_row(rows, site=site, role="mlp_output_answer")
        if output is None or current is None or random is None:
            continue
        accuracy_specificity = (
            float(current["accuracy_drop"]) - float(random["accuracy_drop"])
        )
        margin_specificity = (
            float(current["margin_drop"]) - float(random["margin_drop"])
        )
        attention_necessary = (
            finite_greater(output["accuracy_drop"], 0.10)
            or finite_greater(output["margin_drop"], 1.0)
        )
        edge_specific = (
            finite_greater(accuracy_specificity, 0.10)
            or finite_greater(margin_specificity, 1.0)
        )
        if not attention_necessary or not edge_specific:
            continue
        block = int(current["block"])
        mlp_accuracy_drop = (
            float(mlp["accuracy_drop"]) if mlp is not None else float("nan")
        )
        mlp_margin_drop = (
            float(mlp["margin_drop"]) if mlp is not None else float("nan")
        )
        candidates.append(
            {
                "site": site,
                "block": block,
                "attention_output_accuracy_drop": float(
                    output["accuracy_drop"]
                ),
                "attention_output_margin_drop": float(output["margin_drop"]),
                "edge_accuracy_specificity": accuracy_specificity,
                "edge_margin_specificity": margin_specificity,
                "mlp_output_accuracy_drop": mlp_accuracy_drop,
                "mlp_output_margin_drop": mlp_margin_drop,
                "mlp_writer_support": (
                    finite_greater(mlp_accuracy_drop, 0.10)
                    or finite_greater(mlp_margin_drop, 1.0)
                ),
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            row["edge_accuracy_specificity"],
            row["edge_margin_specificity"],
            row["attention_output_accuracy_drop"],
        ),
    )


def summarize_seed(
    *,
    config: str,
    behavior: dict[str, Any],
    functional_dir: Path,
) -> dict[str, Any]:
    name = str(behavior["name"])
    run_dir = functional_dir / name
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    transition = summary["functional_tests"]["transition"]
    heldout_next = float(transition.get("heldout_next_accuracy", 0.0))
    portable_pass = (
        transition.get("status") == "passed_heldout_transition_gate"
        and heldout_next >= 0.80
    )
    executor = localize_executor(
        read_csv(run_dir / "transition_function_rows.csv")
    )
    executor_pass = executor is not None
    component_loop_pass = (
        bool(behavior["stride_pass"])
        and portable_pass
        and executor_pass
    )
    return {
        "config": config,
        "name": name,
        "initialization_seed": behavior.get("initialization_seed"),
        "data_seed": behavior.get("data_seed"),
        "physical_blocks": behavior["config"]["n_layers"],
        "endpoint_accuracy": behavior["endpoint_accuracy"],
        "matched_step_min_accuracy": behavior[
            "matched_step_min_accuracy"
        ],
        "stride_pass": bool(behavior["stride_pass"]),
        "strong_stride_pass": bool(behavior["strong_stride_pass"]),
        "heldout_next_accuracy": heldout_next,
        "portable_transition_pass": portable_pass,
        "executor_localized_pass": executor_pass,
        "component_loop_pass": component_loop_pass,
        "executor_site": executor["site"] if executor is not None else "",
        "executor_block": executor["block"] if executor is not None else None,
        "edge_accuracy_specificity": (
            executor["edge_accuracy_specificity"]
            if executor is not None
            else None
        ),
        "edge_margin_specificity": (
            executor["edge_margin_specificity"]
            if executor is not None
            else None
        ),
        "attention_output_accuracy_drop": (
            executor["attention_output_accuracy_drop"]
            if executor is not None
            else None
        ),
        "mlp_writer_support": (
            executor["mlp_writer_support"]
            if executor is not None
            else False
        ),
        "mlp_output_accuracy_drop": (
            executor["mlp_output_accuracy_drop"]
            if executor is not None
            else None
        ),
    }


def aggregate_config(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("config must contain at least one seed")
    total = len(rows)
    counts = {
        "stride": sum(bool(row["stride_pass"]) for row in rows),
        "strong_stride": sum(bool(row["strong_stride_pass"]) for row in rows),
        "portable_transition": sum(
            bool(row["portable_transition_pass"]) for row in rows
        ),
        "localized_executor": sum(
            bool(row["executor_localized_pass"]) for row in rows
        ),
        "component_loop": sum(
            bool(row["component_loop_pass"]) for row in rows
        ),
        "mlp_writer": sum(bool(row["mlp_writer_support"]) for row in rows),
    }
    passing_blocks = [
        int(row["executor_block"])
        for row in rows
        if row["component_loop_pass"] and row["executor_block"] is not None
    ]
    block_counts = Counter(passing_blocks)
    stable_block = (
        block_counts.most_common(1)[0][0] if block_counts else None
    )
    stable_block_count = (
        block_counts[stable_block] if stable_block is not None else 0
    )
    return {
        "seed_count": total,
        **{
            f"{name}_success_count": count
            for name, count in counts.items()
        },
        **{
            f"{name}_success_rate": count / total
            for name, count in counts.items()
        },
        **{
            f"{name}_wilson95": wilson_interval(count, total)
            for name, count in counts.items()
        },
        "matched_step_min_accuracy_mean": sum(
            float(row["matched_step_min_accuracy"]) for row in rows
        )
        / total,
        "heldout_next_accuracy_mean": sum(
            float(row["heldout_next_accuracy"]) for row in rows
        )
        / total,
        "stable_executor_block": stable_block,
        "stable_executor_block_support": stable_block_count,
        "executor_block_counts": dict(sorted(block_counts.items())),
    }


def select_trends(
    aggregates: dict[str, dict[str, Any]],
    *,
    baseline: str,
) -> list[str]:
    if baseline not in aggregates:
        raise ValueError(f"missing baseline config {baseline}")
    reference = aggregates[baseline]
    selected: list[str] = []
    for config, result in aggregates.items():
        if config == baseline:
            continue
        component_gain = (
            int(result["component_loop_success_count"])
            - int(reference["component_loop_success_count"])
        )
        portable_gain = (
            int(result["portable_transition_success_count"])
            - int(reference["portable_transition_success_count"])
        )
        stride_gain = (
            int(result["stride_success_count"])
            - int(reference["stride_success_count"])
        )
        continuous_gain = (
            float(result["matched_step_min_accuracy_mean"])
            - float(reference["matched_step_min_accuracy_mean"])
        )
        if (
            component_gain >= 2
            or portable_gain >= 2
            or stride_gain >= 2
            or (
                max(component_gain, portable_gain, stride_gain) >= 1
                and continuous_gain >= 0.20
            )
        ):
            selected.append(config)
    return selected


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate behavior and component gates for a hyperparameter sweep."
    )
    parser.add_argument(
        "--behavior",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument(
        "--functional",
        action="append",
        type=parse_named_path,
        required=True,
    )
    parser.add_argument("--baseline", default="baseline_b2")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    behavior_paths = dict(args.behavior)
    functional_paths = dict(args.functional)
    if behavior_paths.keys() != functional_paths.keys():
        raise ValueError("behavior and functional configs must match")
    seed_rows: list[dict[str, Any]] = []
    config_results: dict[str, dict[str, Any]] = {}
    for config in sorted(behavior_paths):
        behavior_payload = json.loads(
            behavior_paths[config].read_text(encoding="utf-8")
        )
        current_rows = [
            summarize_seed(
                config=config,
                behavior=run,
                functional_dir=functional_paths[config],
            )
            for run in behavior_payload["runs"]
        ]
        seed_rows.extend(current_rows)
        config_results[config] = aggregate_config(current_rows)
    selected = select_trends(config_results, baseline=args.baseline)
    payload = {
        "baseline": args.baseline,
        "configs": config_results,
        "trend_candidates": selected,
        "selection_rule": (
            "gain >=2/5 on component, portable, or stride frequency; or gain "
            ">=1/5 plus >=0.20 mean minimum matched-step accuracy"
        ),
        "seed_rows": seed_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "seed_rows.csv", seed_rows)
    config_rows = [
        {"config": config, **result}
        for config, result in config_results.items()
    ]
    write_csv(args.out_dir / "config_rows.csv", config_rows)
    print(
        json.dumps(
            {
                "configs": config_results,
                "trend_candidates": selected,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
