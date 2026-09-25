from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasoning_loop.graph_path_global_depth_aggregate import (
    _save_plots,
    _write_seed_rows,
    parse_run_spec,
    summarize_variable_seed,
)


def _finite_values(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            return []
        numeric = float(value)
        if not math.isfinite(numeric):
            return []
        values.append(numeric)
    return values


def _all_at_least(
    rows: Sequence[dict[str, Any]],
    key: str,
    threshold: float,
) -> bool:
    values = _finite_values(rows, key)
    return bool(values) and min(values) >= threshold


def _minimum_or_none(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> float | None:
    values = _finite_values(rows, key)
    return min(values) if values else None


def _maximum_or_none(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> float | None:
    values = _finite_values(rows, key)
    return max(values) if values else None


def evaluate_horizon_gcd_gates(
    seed_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        arm = str(row.get("arm", ""))
        if not arm:
            raise ValueError("every seed row must include an arm")
        grouped[arm].append(dict(row))

    arm_results: dict[str, dict[str, Any]] = {}
    for arm, rows in sorted(grouped.items()):
        seed_count = len({int(row["seed"]) for row in rows})
        trained_endpoint_pass = (
            _all_at_least(rows, "trained_native_mean_accuracy", 0.995)
            and _all_at_least(rows, "trained_native_min_accuracy", 0.99)
        )
        interpolation_pass = (
            _all_at_least(rows, "interpolation_native_mean_accuracy", 0.98)
            and _all_at_least(rows, "interpolation_native_min_accuracy", 0.95)
        )
        extrapolation_pass = (
            _all_at_least(rows, "extrapolation_native_mean_accuracy", 0.90)
            and _all_at_least(rows, "extrapolation_margin_over_shortcut_min", 0.50)
        )
        depth1_pass = _all_at_least(rows, "heldout_depth1_accuracy", 0.95)
        temporal_pass = (
            _all_at_least(rows, "skip_target_mean_accuracy", 0.95)
            and _all_at_least(rows, "repeat_target_mean_accuracy", 0.95)
            and _all_at_least(rows, "temporal_alternative_margin_min", 0.20)
        )
        interface_pass = (
            _all_at_least(rows, "transplant_delta1_to_3_mean_accuracy", 0.90)
            and _all_at_least(rows, "transplant_alternative_margin_min", 0.20)
        )
        endpoint_behavior_pass = (
            trained_endpoint_pass and interpolation_pass and extrapolation_pass
        )
        primitive_operator_behavior_pass = (
            endpoint_behavior_pass and depth1_pass and temporal_pass
        )
        stable_primitive_operator_pass = (
            seed_count >= 5 and primitive_operator_behavior_pass
        )

        odd_gaps = [
            float(row["trained_native_mean_accuracy"])
            - float(row["heldout_odd_mean_accuracy"])
            for row in rows
            if row.get("trained_native_mean_accuracy") is not None
            and row.get("heldout_odd_mean_accuracy") is not None
        ]
        odd_gap_min = min(odd_gaps) if len(odd_gaps) == len(rows) and rows else None
        odd_gap_pass = odd_gap_min is not None and odd_gap_min >= 0.20
        stable_macro_behavior_pass = (
            seed_count >= 5 and trained_endpoint_pass and odd_gap_pass
        )

        arm_results[arm] = {
            "seed_count": seed_count,
            "trained_endpoint_pass": trained_endpoint_pass,
            "interpolation_pass": interpolation_pass,
            "extrapolation_pass": extrapolation_pass,
            "depth1_pass": depth1_pass,
            "temporal_pass": temporal_pass,
            "interface_pass": interface_pass,
            "endpoint_behavior_pass": endpoint_behavior_pass,
            "primitive_operator_behavior_pass": primitive_operator_behavior_pass,
            "stable_primitive_operator_pass": stable_primitive_operator_pass,
            "stable_macro_behavior_pass": stable_macro_behavior_pass,
            "minimums": {
                key: _minimum_or_none(rows, key)
                for key in (
                    "trained_native_mean_accuracy",
                    "trained_native_min_accuracy",
                    "interpolation_native_mean_accuracy",
                    "interpolation_native_min_accuracy",
                    "extrapolation_native_mean_accuracy",
                    "extrapolation_margin_over_shortcut_min",
                    "heldout_depth1_accuracy",
                    "heldout_odd_mean_accuracy",
                    "heldout_odd_min_accuracy",
                    "skip_target_mean_accuracy",
                    "repeat_target_mean_accuracy",
                    "temporal_alternative_margin_min",
                    "transplant_delta1_to_3_mean_accuracy",
                    "transplant_alternative_margin_min",
                )
            },
            "trained_minus_heldout_odd_gap_min": odd_gap_min,
            "temporal_target_min": min(
                value
                for value in (
                    _minimum_or_none(rows, "skip_target_mean_accuracy"),
                    _minimum_or_none(rows, "repeat_target_mean_accuracy"),
                )
                if value is not None
            )
            if _minimum_or_none(rows, "skip_target_mean_accuracy") is not None
            and _minimum_or_none(rows, "repeat_target_mean_accuracy") is not None
            else None,
            "temporal_target_max": max(
                value
                for value in (
                    _maximum_or_none(rows, "skip_target_mean_accuracy"),
                    _maximum_or_none(rows, "repeat_target_mean_accuracy"),
                )
                if value is not None
            )
            if _maximum_or_none(rows, "skip_target_mean_accuracy") is not None
            and _maximum_or_none(rows, "repeat_target_mean_accuracy") is not None
            else None,
        }

    coprime = arm_results.get("coprime_no_d1")
    even = arm_results.get("even_no_d1")
    causal_contrast = None
    if (
        coprime
        and even
        and coprime["temporal_target_min"] is not None
        and even["temporal_target_max"] is not None
    ):
        causal_contrast = (
            float(coprime["temporal_target_min"])
            - float(even["temporal_target_max"])
        )

    if (
        coprime
        and coprime["stable_primitive_operator_pass"]
        and even
        and even["stable_primitive_operator_pass"]
    ):
        classification = "primitive_successor_inductive_bias"
    elif (
        coprime
        and coprime["stable_primitive_operator_pass"]
        and even
        and even["seed_count"] >= 5
        and even["trained_endpoint_pass"]
        and (
            even["stable_macro_behavior_pass"]
            or (causal_contrast is not None and causal_contrast >= 0.20)
        )
    ):
        classification = "gcd_conditioned_mechanism"
    elif coprime and coprime["stable_primitive_operator_pass"]:
        classification = (
            "coprime_sufficiency_pending_even_control"
            if not even or even["seed_count"] < 5
            else "coprime_sufficiency_even_optimization_failure"
        )
    elif (
        coprime
        and coprime["primitive_operator_behavior_pass"]
        and coprime["seed_count"] < 5
    ):
        classification = "insufficient_seed_stability"
    elif coprime and coprime["endpoint_behavior_pass"]:
        classification = "endpoint_fitting_without_operator_semantics"
    elif arm_results:
        classification = "no_primitive_operator_evidence"
    else:
        classification = "insufficient_evidence"

    return {
        "arms": arm_results,
        "coprime_minus_even_temporal_target_contrast": causal_contrast,
        "classification": classification,
    }


def summarize_horizon_seed(
    summary: dict[str, Any],
    *,
    arm: str,
    seed: int,
    temporal: dict[str, Any] | None,
) -> dict[str, Any]:
    row = summarize_variable_seed(
        summary,
        arm=arm,
        seed=seed,
        temporal=temporal,
    )
    train_depths = {int(depth) for depth in summary["train_depths"]}
    max_train_depth = int(summary["max_train_depth"])
    native = list(row["native_accuracy_by_depth"])
    heldout_odd_depths = [
        depth
        for depth in range(1, max_train_depth + 1)
        if depth % 2 == 1 and depth not in train_depths
    ]
    if not heldout_odd_depths:
        raise ValueError("horizon-GCD analysis requires held-out odd depths")
    heldout_odd = [float(native[depth - 1]) for depth in heldout_odd_depths]
    row.update(
        {
            "train_depths": sorted(train_depths),
            "heldout_odd_depths": heldout_odd_depths,
            "heldout_depth1_accuracy": float(native[0]),
            "heldout_odd_mean_accuracy": sum(heldout_odd) / len(heldout_odd),
            "heldout_odd_min_accuracy": min(heldout_odd),
        }
    )
    return row


def _claim_ledger(gates: dict[str, Any]) -> list[dict[str, str]]:
    classification = gates["classification"]
    primitive_status = (
        "supported"
        if classification
        in {
            "gcd_conditioned_mechanism",
            "primitive_successor_inductive_bias",
            "coprime_sufficiency_pending_even_control",
            "coprime_sufficiency_even_optimization_failure",
        }
        else "preliminary"
        if classification == "insufficient_seed_stability"
        else "not_supported"
    )
    gcd_status = (
        "supported"
        if classification == "gcd_conditioned_mechanism"
        else "rejected_as_necessary"
        if classification == "primitive_successor_inductive_bias"
        else "not_established"
    )
    return [
        {
            "claim": "global endpoints can induce a primitive operator without direct depth-1 labels",
            "status": primitive_status,
            "evidence_boundary": "requires held-out depth 1, other unseen horizons, skip/repeat, and five coprime-arm seeds",
        },
        {
            "claim": "the horizon gcd controls whether primitive or macro dynamics emerge",
            "status": gcd_status,
            "evidence_boundary": "requires a trained even-only arm with odd-horizon or causal failure relative to the coprime arm",
        },
        {
            "claim": "even-only success proves primitive-step identifiability from the objective",
            "status": "rejected",
            "evidence_boundary": "even-only supervision is compatible with architecture or optimization bias toward a primitive root",
        },
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate no-depth-1 horizon-GCD ablation results."
    )
    parser.add_argument("--run", action="append", required=True, help="ARM:SEED=summary.json")
    parser.add_argument("--temporal", action="append", default=[], help="ARM:SEED=summary.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporal_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in args.temporal:
        arm, seed, path = parse_run_spec(spec)
        temporal_by_key[(arm, seed)] = json.loads(path.read_text(encoding="utf-8"))

    seed_rows: list[dict[str, Any]] = []
    for spec in args.run:
        arm, seed, path = parse_run_spec(spec)
        summary = json.loads(path.read_text(encoding="utf-8"))
        seed_rows.append(
            summarize_horizon_seed(
                summary,
                arm=arm,
                seed=seed,
                temporal=temporal_by_key.get((arm, seed)),
            )
        )
    seed_rows.sort(key=lambda row: (row["arm"], row["seed"]))
    gates = evaluate_horizon_gcd_gates(seed_rows)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps({"seed_rows": seed_rows, "gates": gates}, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "claim_ledger.json").write_text(
        json.dumps(_claim_ledger(gates), indent=2),
        encoding="utf-8",
    )
    _write_seed_rows(args.out_dir / "seed_rows.csv", seed_rows)
    _save_plots(args.out_dir, seed_rows)


if __name__ == "__main__":
    main()
