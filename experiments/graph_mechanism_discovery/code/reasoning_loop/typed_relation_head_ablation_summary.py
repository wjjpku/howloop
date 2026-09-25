from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _signature(row: dict[str, Any], n_heads: int) -> tuple[bool, ...]:
    bits: list[bool] = []
    for heads, keep_mlp in zip(
        row["kept_heads"], row["kept_mlps"], strict=True
    ):
        head_set = set(heads)
        bits.extend(head in head_set for head in range(n_heads))
        bits.append(bool(keep_mlp))
    return tuple(bits)


def _shapley_values(
    row_by_signature: dict[tuple[bool, ...], dict[str, Any]],
    metric: str,
) -> list[float]:
    component_count = len(next(iter(row_by_signature)))
    denominator = math.factorial(component_count)
    values: list[float] = []
    for component in range(component_count):
        total = 0.0
        other_components = [
            index for index in range(component_count) if index != component
        ]
        for subset_size in range(component_count):
            weight = (
                math.factorial(subset_size)
                * math.factorial(component_count - subset_size - 1)
                / denominator
            )
            for subset in itertools.combinations(other_components, subset_size):
                without = [False] * component_count
                for index in subset:
                    without[index] = True
                with_component = without.copy()
                with_component[component] = True
                total += weight * (
                    float(row_by_signature[tuple(with_component)][metric])
                    - float(row_by_signature[tuple(without)][metric])
                )
        values.append(total)
    return values


def _mean_pair_interaction(
    row_by_signature: dict[tuple[bool, ...], dict[str, Any]],
    metric: str,
    left: int,
    right: int,
) -> float:
    component_count = len(next(iter(row_by_signature)))
    other_components = [
        index for index in range(component_count) if index not in {left, right}
    ]
    effects: list[float] = []
    for subset_size in range(len(other_components) + 1):
        for subset in itertools.combinations(other_components, subset_size):
            neither = [False] * component_count
            for index in subset:
                neither[index] = True
            left_only = neither.copy()
            left_only[left] = True
            right_only = neither.copy()
            right_only[right] = True
            both = left_only.copy()
            both[right] = True
            effects.append(
                float(row_by_signature[tuple(both)][metric])
                - float(row_by_signature[tuple(left_only)][metric])
                - float(row_by_signature[tuple(right_only)][metric])
                + float(row_by_signature[tuple(neither)][metric])
            )
    return sum(effects) / len(effects)


def summarize_parameter_heads(
    rows: list[dict[str, Any]],
    *,
    attention: dict[str, Any],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("circuit rows are required")
    n_heads = int(rows[0]["n_heads"])
    component_count = 2 * (n_heads + 1)
    row_by_signature = {
        _signature(row, n_heads): row
        for row in rows
    }
    if len(row_by_signature) != 2**component_count:
        raise ValueError("a complete two-loop circuit enumeration is required")
    full_signature = (True,) * component_count
    baseline = row_by_signature[full_signature]
    shapley_accuracy = _shapley_values(row_by_signature, "endpoint_accuracy")
    shapley_margin = _shapley_values(row_by_signature, "mean_logit_margin")
    result: list[dict[str, Any]] = []
    for head in range(n_heads):
        loop1_index = head
        loop2_index = n_heads + 1 + head
        loop1_leave_out = list(full_signature)
        loop1_leave_out[loop1_index] = False
        loop2_leave_out = list(full_signature)
        loop2_leave_out[loop2_index] = False
        both_leave_out = list(loop1_leave_out)
        both_leave_out[loop2_index] = False
        only_attention_head = [False] * component_count
        only_attention_head[loop1_index] = True
        only_attention_head[n_heads] = True
        only_attention_head[loop2_index] = True
        only_attention_head[-1] = True
        loop1_row = row_by_signature[tuple(loop1_leave_out)]
        loop2_row = row_by_signature[tuple(loop2_leave_out)]
        both_row = row_by_signature[tuple(both_leave_out)]
        result.append(
            {
                "head": head,
                "baseline_accuracy": float(baseline["endpoint_accuracy"]),
                "baseline_margin": float(baseline["mean_logit_margin"]),
                "loop1_leave_out_accuracy": float(loop1_row["endpoint_accuracy"]),
                "loop1_leave_out_drop": float(baseline["endpoint_accuracy"])
                - float(loop1_row["endpoint_accuracy"]),
                "loop2_leave_out_accuracy": float(loop2_row["endpoint_accuracy"]),
                "loop2_leave_out_drop": float(baseline["endpoint_accuracy"])
                - float(loop2_row["endpoint_accuracy"]),
                "both_loops_leave_out_accuracy": float(
                    both_row["endpoint_accuracy"]
                ),
                "both_loops_leave_out_drop": float(baseline["endpoint_accuracy"])
                - float(both_row["endpoint_accuracy"]),
                "only_attention_head_accuracy": float(
                    row_by_signature[tuple(only_attention_head)]["endpoint_accuracy"]
                ),
                "shapley_accuracy_loop1": shapley_accuracy[loop1_index],
                "shapley_accuracy_loop2": shapley_accuracy[loop2_index],
                "shapley_margin_loop1": shapley_margin[loop1_index],
                "shapley_margin_loop2": shapley_margin[loop2_index],
                "cross_loop_interaction_accuracy": _mean_pair_interaction(
                    row_by_signature,
                    "endpoint_accuracy",
                    loop1_index,
                    loop2_index,
                ),
                "typed_selectivity_loop1": float(
                    attention["typed_edge_selectivity"][0][head]
                ),
                "typed_selectivity_loop2": float(
                    attention["typed_edge_selectivity"][1][head]
                ),
                "correct_relation_attention_loop1": float(
                    attention["correct_relation_attention"][0][head]
                ),
                "correct_relation_attention_loop2": float(
                    attention["correct_relation_attention"][1][head]
                ),
            }
        )
    return result


def analyze_head_ablation_runs(
    run_dirs: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    per_seed: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        circuit_rows: list[dict[str, Any]] = []
        with (run_dir / "effective_circuits.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                circuit_rows.append(
                    {
                        **row,
                        "n_heads": int(row["n_heads"]),
                        "kept_heads": json.loads(row["kept_heads"]),
                        "kept_mlps": json.loads(row["kept_mlps"]),
                        "endpoint_accuracy": float(row["endpoint_accuracy"]),
                        "mean_logit_margin": float(row["mean_logit_margin"]),
                    }
                )
        head_rows = summarize_parameter_heads(
            circuit_rows,
            attention=summary["phase"]["attention"],
        )
        for row in head_rows:
            per_seed.append(
                {
                    "seed": int(summary["model_seed"]),
                    **row,
                }
            )

    aggregate: list[dict[str, Any]] = []
    for head in sorted({int(row["head"]) for row in per_seed}):
        selected = [row for row in per_seed if int(row["head"]) == head]
        aggregate_row: dict[str, Any] = {
            "head": head,
            "seed_count": len(selected),
        }
        numeric_fields = [
            key
            for key, value in selected[0].items()
            if key not in {"seed", "head"} and isinstance(value, (int, float))
        ]
        for field in numeric_fields:
            values = [float(row[field]) for row in selected]
            aggregate_row[f"{field}_mean"] = statistics.mean(values)
            aggregate_row[f"{field}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        aggregate.append(aggregate_row)

    result = {
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (out_dir / "per_seed_head_ablation.csv", per_seed),
        (out_dir / "aggregate_head_ablation.csv", aggregate),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result
