from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analytic_start_shortcut(node_count: int, depth: int) -> float:
    if node_count < 1 or depth < 1:
        raise ValueError("node_count and depth must be positive")
    return (
        sum(depth % cycle_length == 0 for cycle_length in range(1, node_count + 1))
        / node_count
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


def evaluate_claim_gates(
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
        behavior_pass = (
            _all_at_least(rows, "trained_native_mean_accuracy", 0.995)
            and _all_at_least(rows, "trained_native_min_accuracy", 0.99)
        )
        interpolation_not_applicable = arm == "dense" and not _finite_values(
            rows,
            "interpolation_native_mean_accuracy",
        )
        interpolation_pass = interpolation_not_applicable or (
            _all_at_least(rows, "interpolation_native_mean_accuracy", 0.98)
            and _all_at_least(rows, "interpolation_native_min_accuracy", 0.95)
        )
        extrapolation_pass = (
            _all_at_least(rows, "extrapolation_native_mean_accuracy", 0.90)
            and _all_at_least(rows, "extrapolation_margin_over_shortcut_min", 0.50)
        )
        temporal_pass = (
            _all_at_least(rows, "skip_target_mean_accuracy", 0.95)
            and _all_at_least(rows, "repeat_target_mean_accuracy", 0.95)
            and _all_at_least(rows, "temporal_alternative_margin_min", 0.20)
        )
        interface_pass = (
            _all_at_least(
                rows,
                "transplant_delta1_to_3_mean_accuracy",
                0.90,
            )
            and _all_at_least(rows, "transplant_alternative_margin_min", 0.20)
        )
        stable_algorithm_pass = (
            seed_count >= 5
            and behavior_pass
            and interpolation_pass
            and extrapolation_pass
            and temporal_pass
            and interface_pass
        )
        arm_results[arm] = {
            "seed_count": seed_count,
            "behavior_pass": behavior_pass,
            "interpolation_pass": interpolation_pass,
            "interpolation_not_applicable": interpolation_not_applicable,
            "extrapolation_pass": extrapolation_pass,
            "temporal_pass": temporal_pass,
            "interface_pass": interface_pass,
            "stable_algorithm_pass": stable_algorithm_pass,
            "minimums": {
                key: _minimum_or_none(rows, key)
                for key in (
                    "trained_native_mean_accuracy",
                    "trained_native_min_accuracy",
                    "interpolation_native_mean_accuracy",
                    "interpolation_native_min_accuracy",
                    "extrapolation_native_mean_accuracy",
                    "extrapolation_margin_over_shortcut_min",
                    "skip_target_mean_accuracy",
                    "repeat_target_mean_accuracy",
                    "temporal_alternative_margin_min",
                    "transplant_delta1_to_3_mean_accuracy",
                    "transplant_alternative_margin_min",
                )
            },
        }

    sparse = arm_results.get("sparse")
    dense = arm_results.get("dense")
    stable_emergence = bool(sparse and sparse["stable_algorithm_pass"])
    if stable_emergence:
        classification = "sparse_global_endpoints_induce_operator"
    elif dense and dense["stable_algorithm_pass"]:
        classification = "coverage_qualified_operator"
    elif sparse and sparse["behavior_pass"]:
        if (
            sparse["seed_count"] < 5
            and sparse["interpolation_pass"]
            and sparse["extrapolation_pass"]
            and sparse["temporal_pass"]
            and sparse["interface_pass"]
        ):
            classification = "insufficient_seed_stability"
        elif (
            sparse["interpolation_pass"]
            and sparse["extrapolation_pass"]
            and sparse["temporal_pass"]
            and not sparse["interface_pass"]
        ):
            classification = "algorithmic_operator_without_stable_interface"
        else:
            classification = "endpoint_fitting_without_algorithm"
    elif any(result["behavior_pass"] for result in arm_results.values()):
        classification = "endpoint_fitting_without_algorithm"
    elif arm_results:
        classification = "final_only_failed"
    else:
        classification = "insufficient_evidence"

    return {
        "arms": arm_results,
        "stable_emergence": stable_emergence,
        "classification": classification,
    }


def parse_run_spec(text: str) -> tuple[str, int, Path]:
    if "=" not in text or ":" not in text.split("=", 1)[0]:
        raise ValueError(f"run spec must be ARM:SEED=/path/to/summary.json, got {text!r}")
    identity, path_text = text.split("=", 1)
    arm, seed_text = identity.split(":", 1)
    if not arm:
        raise ValueError("arm must not be empty")
    return arm, int(seed_text), Path(path_text)


def summarize_temporal(temporal: dict[str, Any]) -> dict[str, float]:
    schedule = temporal["schedule"]
    condition_names = list(schedule["condition_names"])
    final_slot = np.asarray(schedule["final_slot_accuracy"], dtype=np.float64)
    if final_slot.ndim != 2 or final_slot.shape[0] != len(condition_names):
        raise ValueError("temporal schedule has inconsistent shapes")
    skip_indices = [
        index for index, name in enumerate(condition_names) if name.startswith("skip_")
    ]
    repeat_indices = [
        index for index, name in enumerate(condition_names) if name.startswith("repeat_")
    ]
    if not skip_indices or len(skip_indices) != len(repeat_indices):
        raise ValueError("temporal schedule must contain paired skip and repeat conditions")
    base_loops = len(skip_indices)
    skip_target_position = base_loops - 1
    repeat_target_position = base_loops + 1

    def target_mean_and_margin(
        row_indices: list[int],
        target_position: int,
    ) -> tuple[float, float]:
        target_values: list[float] = []
        margins: list[float] = []
        for row_index in row_indices:
            row = final_slot[row_index]
            if target_position >= row.shape[0]:
                raise ValueError("temporal schedule does not cover the target path position")
            target = float(row[target_position])
            alternatives = np.delete(row, target_position)
            target_values.append(target)
            margins.append(target - float(alternatives.max()))
        return float(np.mean(target_values)), float(np.min(margins))

    skip_mean, skip_margin = target_mean_and_margin(
        skip_indices,
        skip_target_position,
    )
    repeat_mean, repeat_margin = target_mean_and_margin(
        repeat_indices,
        repeat_target_position,
    )

    cross_time = temporal["cross_time"]
    transplant = np.asarray(
        cross_time["mean_transplant_acc_by_delta"],
        dtype=np.float64,
    )
    receiver_path = np.asarray(
        cross_time["receiver_path_acc"],
        dtype=np.float64,
    )
    if receiver_path.ndim != 3:
        raise ValueError("receiver_path_acc must have donor, receiver, and delta axes")
    receiver = receiver_path.mean(axis=(0, 1))
    donor_final = np.asarray(
        cross_time["mean_donor_final_acc_by_delta"],
        dtype=np.float64,
    )
    available_delta = min(3, len(transplant) - 1, len(receiver) - 1, len(donor_final) - 1)
    if available_delta < 1:
        raise ValueError("cross-time summary must contain at least delta 1")
    target_slice = transplant[1 : available_delta + 1]
    receiver_slice = receiver[1 : available_delta + 1]
    donor_slice = donor_final[1 : available_delta + 1]
    transplant_margin = target_slice - np.maximum(receiver_slice, donor_slice)
    return {
        "skip_target_mean_accuracy": skip_mean,
        "repeat_target_mean_accuracy": repeat_mean,
        "temporal_alternative_margin_min": min(skip_margin, repeat_margin),
        "transplant_delta1_to_3_mean_accuracy": float(target_slice.mean()),
        "transplant_alternative_margin_min": float(transplant_margin.min()),
    }


def summarize_variable_seed(
    summary: dict[str, Any],
    *,
    arm: str,
    seed: int,
    temporal: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = summary["final_metrics"]
    extrapolation_depths = [
        int(depth)
        for depth in summary.get(
            "extrapolation_depths",
            metrics.get("extrapolation_depths", []),
        )
    ]
    native = list(metrics["native_accuracy_by_depth"])
    shortcut = list(metrics["analytic_start_shortcut_by_depth"])
    margins = [
        float(native[depth - 1]) - float(shortcut[depth - 1])
        for depth in extrapolation_depths
    ]
    row: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "run_name": summary.get("run_name", ""),
        "trained_native_mean_accuracy": metrics["trained_native_mean_accuracy"],
        "trained_native_min_accuracy": metrics["trained_native_min_accuracy"],
        "interpolation_native_mean_accuracy": metrics.get(
            "interpolation_native_mean_accuracy"
        ),
        "interpolation_native_min_accuracy": metrics.get(
            "interpolation_native_min_accuracy"
        ),
        "extrapolation_native_mean_accuracy": metrics.get(
            "extrapolation_native_mean_accuracy"
        ),
        "extrapolation_native_min_accuracy": metrics.get(
            "extrapolation_native_min_accuracy"
        ),
        "extrapolation_margin_over_shortcut_min": min(margins) if margins else None,
        "native_accuracy_by_depth": native,
        "analytic_start_shortcut_by_depth": shortcut,
    }
    row.update(
        summarize_temporal(temporal)
        if temporal is not None
        else {
            "skip_target_mean_accuracy": None,
            "repeat_target_mean_accuracy": None,
            "temporal_alternative_margin_min": None,
            "transplant_delta1_to_3_mean_accuracy": None,
            "transplant_alternative_margin_min": None,
        }
    )
    return row


def _write_seed_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty seed rows")
    scalar_keys = [
        key
        for key in rows[0]
        if key not in {"native_accuracy_by_depth", "analytic_start_shortcut_by_depth"}
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def _save_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in rows:
        native = np.asarray(row["native_accuracy_by_depth"], dtype=np.float64)
        ax.plot(
            np.arange(1, len(native) + 1),
            native,
            marker="o",
            label=f"{row['arm']} seed {row['seed']}",
        )
    ax.set_xlabel("depth / recurrent applications")
    ax.set_ylabel("native accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "native_depth_accuracy.png", dpi=180)
    plt.close(fig)

    labels = [f"{row['arm']}:{row['seed']}" for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(rows)), 5))
    width = 0.35
    skip = [
        float("nan")
        if row["skip_target_mean_accuracy"] is None
        else row["skip_target_mean_accuracy"]
        for row in rows
    ]
    repeat = [
        float("nan")
        if row["repeat_target_mean_accuracy"] is None
        else row["repeat_target_mean_accuracy"]
        for row in rows
    ]
    ax.bar(x - width / 2, skip, width, label="skip -> f^7")
    ax.bar(x + width / 2, repeat, width, label="repeat -> f^9")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_gate_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(rows)), 5))
    transplant = [
        float("nan")
        if row["transplant_delta1_to_3_mean_accuracy"] is None
        else row["transplant_delta1_to_3_mean_accuracy"]
        for row in rows
    ]
    ax.bar(x, transplant)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("cross-context transplant accuracy, delta 1-3")
    fig.tight_layout()
    fig.savefig(out_dir / "transplant_accuracy.png", dpi=180)
    plt.close(fig)


def _claim_ledger(gates: dict[str, Any]) -> list[dict[str, str]]:
    classification = gates["classification"]
    native_operator_status = (
        "supported"
        if classification
        in {
            "sparse_global_endpoints_induce_operator",
            "algorithmic_operator_without_stable_interface",
        }
        else "coverage_qualified"
        if classification == "coverage_qualified_operator"
        else "preliminary"
        if classification == "insufficient_seed_stability"
        else "not_supported"
    )
    portable_interface_status = (
        "supported"
        if classification == "sparse_global_endpoints_induce_operator"
        else "coverage_qualified"
        if classification == "coverage_qualified_operator"
        else "not_stable"
        if classification == "algorithmic_operator_without_stable_interface"
        else "not_supported"
    )
    return [
        {
            "claim": "multiple global endpoints induce a native-context successor operator",
            "status": native_operator_status,
            "evidence_boundary": "requires endpoint behavior, unseen horizons, skip/repeat interventions, and five seeds",
        },
        {
            "claim": "the learned operator exposes a stable portable state interface",
            "status": portable_interface_status,
            "evidence_boundary": "additionally requires cross-context state transplant across five seeds",
        },
        {
            "claim": "endpoint accuracy alone establishes algorithm reuse",
            "status": "rejected",
            "evidence_boundary": "skip/repeat is required for native operator semantics; transplant is required only for portability",
        },
        {
            "claim": "the component circuit is unique",
            "status": "not_tested",
            "evidence_boundary": "requires alternative-circuit search and minimality",
        },
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate global-depth supervision behavior and causal diagnostics."
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
            summarize_variable_seed(
                summary,
                arm=arm,
                seed=seed,
                temporal=temporal_by_key.get((arm, seed)),
            )
        )
    seed_rows.sort(key=lambda row: (row["arm"], row["seed"]))
    gates = evaluate_claim_gates(seed_rows)
    aggregate = {
        "seed_rows": seed_rows,
        "gates": gates,
    }
    (args.out_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    _write_seed_rows(args.out_dir / "seed_rows.csv", seed_rows)
    (args.out_dir / "claim_ledger.json").write_text(
        json.dumps(_claim_ledger(gates), indent=2),
        encoding="utf-8",
    )
    _save_plots(args.out_dir, seed_rows)


if __name__ == "__main__":
    main()
