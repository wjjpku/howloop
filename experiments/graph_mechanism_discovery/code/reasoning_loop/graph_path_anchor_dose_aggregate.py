from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _row_at_least(
    row: dict[str, Any],
    key: str,
    threshold: float,
) -> bool:
    value = row.get(key)
    if value is None:
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric >= threshold


def _seed_operator_pass(row: dict[str, Any]) -> bool:
    return all(
        _row_at_least(row, key, threshold)
        for key, threshold in (
            ("native_depth1_accuracy", 0.99),
            ("native_depth1_to_8_min_accuracy", 0.99),
            ("extrapolation_native_mean_accuracy", 0.90),
            ("extrapolation_margin_over_shortcut_min", 0.50),
            ("skip_target_mean_accuracy", 0.95),
            ("repeat_target_mean_accuracy", 0.95),
            ("temporal_alternative_margin_min", 0.20),
        )
    )


def _seed_interface_pass(row: dict[str, Any]) -> bool:
    return all(
        _row_at_least(row, key, threshold)
        for key, threshold in (
            ("transplant_delta1_to_3_mean_accuracy", 0.90),
            ("transplant_alternative_margin_min", 0.20),
        )
    )


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate_anchor_dose_gates(
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
        probabilities = {
            float(row["primitive_probability"])
            for row in rows
            if row.get("primitive_probability") is not None
        }
        if len(probabilities) != 1:
            raise ValueError(f"arm {arm} must have one primitive probability")
        primitive_probability = probabilities.pop()
        seed_count = len({int(row["seed"]) for row in rows})
        native_pass = (
            _all_at_least(rows, "native_depth1_accuracy", 0.99)
            and _all_at_least(rows, "native_depth1_to_8_min_accuracy", 0.99)
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
            _all_at_least(rows, "transplant_delta1_to_3_mean_accuracy", 0.90)
            and _all_at_least(rows, "transplant_alternative_margin_min", 0.20)
        )
        operator_success_count = sum(_seed_operator_pass(row) for row in rows)
        interface_success_count = sum(_seed_interface_pass(row) for row in rows)
        operator_behavior_pass = operator_success_count == len(rows)
        stable_operator_pass = seed_count >= 5 and operator_behavior_pass
        transition_steps = sorted(
            int(row["first_all_native_step"])
            for row in rows
            if _seed_operator_pass(row)
            and row.get("first_all_native_step") is not None
        )
        arm_results[arm] = {
            "primitive_probability": primitive_probability,
            "seed_count": seed_count,
            "native_pass": native_pass,
            "extrapolation_pass": extrapolation_pass,
            "temporal_pass": temporal_pass,
            "interface_pass": interface_pass,
            "operator_behavior_pass": operator_behavior_pass,
            "stable_operator_pass": stable_operator_pass,
            "operator_success_count": operator_success_count,
            "operator_success_rate": operator_success_count / len(rows),
            "operator_success_wilson95": _wilson_interval(
                operator_success_count,
                len(rows),
            ),
            "interface_success_count": interface_success_count,
            "interface_success_rate": interface_success_count / len(rows),
            "interface_success_wilson95": _wilson_interval(
                interface_success_count,
                len(rows),
            ),
            "transition_step_median": (
                float(median(transition_steps)) if transition_steps else None
            ),
            "transition_step_range": (
                [transition_steps[0], transition_steps[-1]]
                if transition_steps
                else None
            ),
            "minimums": {
                key: _minimum_or_none(rows, key)
                for key in (
                    "native_depth1_accuracy",
                    "native_depth1_to_8_min_accuracy",
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

    zero_arms = [
        result
        for result in arm_results.values()
        if result["primitive_probability"] == 0.0
    ]
    anchored_success = [
        result
        for result in arm_results.values()
        if 0.0 < result["primitive_probability"] < 1.0
        and result["stable_operator_pass"]
    ]
    minimum_successful_anchor_probability = (
        min(result["primitive_probability"] for result in anchored_success)
        if anchored_success
        else None
    )
    zero_anchor_stable = any(result["stable_operator_pass"] for result in zero_arms)
    primitive_only_compositional_pass = any(
        result["primitive_probability"] == 1.0
        and result["stable_operator_pass"]
        for result in arm_results.values()
    )

    if anchored_success and zero_arms and not zero_anchor_stable:
        classification = "primitive_anchor_phase_transition"
    elif anchored_success:
        classification = "anchored_operator_without_zero_anchor_baseline"
    elif primitive_only_compositional_pass:
        classification = "primitive_only_compositional_sufficiency"
    elif any(
        result["operator_behavior_pass"] and result["seed_count"] < 5
        for result in arm_results.values()
    ):
        classification = "insufficient_seed_stability"
    elif arm_results:
        classification = "no_stable_operator_at_tested_doses"
    else:
        classification = "insufficient_evidence"

    return {
        "arms": arm_results,
        "zero_anchor_stable": zero_anchor_stable,
        "minimum_successful_anchor_probability": minimum_successful_anchor_probability,
        "primitive_only_compositional_pass": primitive_only_compositional_pass,
        "classification": classification,
    }


def primitive_probability(summary: dict[str, Any]) -> float:
    depths = [int(depth) for depth in summary["train_depths"]]
    if 1 not in depths:
        return 0.0
    weights = summary.get("train_depth_weights")
    if weights is None:
        return 1.0 / len(depths)
    return float(weights[depths.index(1)])


def first_all_native_step(
    history: Sequence[dict[str, Any]],
    *,
    max_native_depth: int = 8,
    threshold: float = 0.99,
) -> int | None:
    for row in history:
        native = list(row["native_accuracy_by_depth"])
        if len(native) >= max_native_depth and min(native[:max_native_depth]) >= threshold:
            return int(row["step"])
    return None


def summarize_anchor_seed(
    summary: dict[str, Any],
    *,
    arm: str,
    seed: int,
    temporal: dict[str, Any] | None,
    history: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    row = summarize_variable_seed(
        summary,
        arm=arm,
        seed=seed,
        temporal=temporal,
    )
    native = [float(value) for value in row["native_accuracy_by_depth"]]
    if len(native) < 8:
        raise ValueError("anchor-dose analysis requires native depths 1 through 8")
    histogram = {
        int(depth): int(count)
        for depth, count in summary.get("train_depth_histogram", {}).items()
    }
    total = sum(histogram.values())
    depth1_fraction = histogram.get(1, 0) / total if total else None
    row.update(
        {
            "primitive_probability": primitive_probability(summary),
            "realized_primitive_fraction": depth1_fraction,
            "native_depth1_accuracy": native[0],
            "native_depth1_to_8_min_accuracy": min(native[:8]),
            "first_all_native_step": (
                first_all_native_step(history) if history is not None else None
            ),
            "last_evaluated_step": (
                int(history[-1]["step"]) if history else None
            ),
        }
    )
    return row


def _save_anchor_dose_response(
    path: Path,
    rows: Sequence[dict[str, Any]],
    gates: dict[str, Any],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(dict(row))
    ordered = sorted(
        grouped,
        key=lambda arm: (
            float(gates["arms"][arm]["primitive_probability"]),
            arm,
        ),
    )
    x = np.arange(len(ordered), dtype=np.float64)
    labels = [
        f"{100 * float(gates['arms'][arm]['primitive_probability']):g}%\n{arm}"
        for arm in ordered
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.6 * len(ordered)), 4.4))

    success = np.asarray(
        [gates["arms"][arm]["operator_success_rate"] for arm in ordered],
        dtype=np.float64,
    )
    intervals = [
        gates["arms"][arm]["operator_success_wilson95"]
        for arm in ordered
    ]
    lower = success - np.asarray([interval[0] for interval in intervals])
    upper = np.asarray([interval[1] for interval in intervals]) - success
    axes[0].errorbar(
        x,
        success,
        yerr=np.vstack([lower, upper]),
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
    for index, arm in enumerate(ordered):
        result = gates["arms"][arm]
        axes[0].annotate(
            f"{result['operator_success_count']}/{result['seed_count']}",
            (x[index], success[index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set_title("Algorithmic reuse success")
    axes[0].set_ylabel("seed success fraction")
    axes[0].set_ylim(-0.05, 1.16)

    fallback_step = max(
        (
            int(row["last_evaluated_step"])
            for row in rows
            if row.get("last_evaluated_step") is not None
        ),
        default=20000,
    )
    for index, arm in enumerate(ordered):
        successful_steps = [
            int(row["first_all_native_step"])
            for row in grouped[arm]
            if _seed_operator_pass(row)
            and row.get("first_all_native_step") is not None
        ]
        failed = len(grouped[arm]) - len(successful_steps)
        if successful_steps:
            jitter = np.linspace(-0.08, 0.08, len(successful_steps))
            axes[1].scatter(
                x[index] + jitter,
                successful_steps,
                color="tab:blue",
                s=28,
            )
            axes[1].scatter(
                [x[index]],
                [median(successful_steps)],
                marker="_",
                s=180,
                linewidths=2,
                color="black",
            )
        if failed:
            axes[1].scatter(
                [x[index]],
                [fallback_step],
                marker="x",
                s=45,
                color="tab:red",
            )
            axes[1].annotate(
                f"{failed} fail",
                (x[index], fallback_step),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="tab:red",
            )
    axes[1].set_title("Transition time")
    axes[1].set_ylabel("first step with D1-D8 >= 0.99")
    axes[1].set_ylim(0, fallback_step * 1.15)

    for index, arm in enumerate(ordered):
        values = [
            float(row["transplant_delta1_to_3_mean_accuracy"])
            for row in grouped[arm]
            if row.get("transplant_delta1_to_3_mean_accuracy") is not None
        ]
        if not values:
            continue
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[2].scatter(
            x[index] + jitter,
            values,
            s=28,
            alpha=0.8,
        )
        axes[2].scatter(
            [x[index]],
            [float(np.mean(values))],
            marker="_",
            s=180,
            linewidths=2,
            color="black",
        )
    axes[2].axhline(0.90, color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_title("Portable interface")
    axes[2].set_ylabel("transplant accuracy, delta 1-3")
    axes[2].set_ylim(0, 1.02)

    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _claim_ledger(gates: dict[str, Any]) -> list[dict[str, str]]:
    classification = gates["classification"]
    return [
        {
            "claim": "a small primitive-label dose can select the algorithmic reuse basin",
            "status": (
                "supported"
                if classification == "primitive_anchor_phase_transition"
                else "preliminary"
                if classification == "insufficient_seed_stability"
                else "not_established"
            ),
            "evidence_boundary": "requires a failed zero-anchor arm and a five-seed successful nonzero dose",
        },
        {
            "claim": "primitive-step supervision alone yields compositional closure",
            "status": (
                "supported"
                if gates["primitive_only_compositional_pass"]
                else "not_established"
            ),
            "evidence_boundary": "requires unseen depths, extrapolation, and skip/repeat across five primitive-only seeds",
        },
        {
            "claim": "coprime global endpoints are mathematically insufficient",
            "status": "rejected",
            "evidence_boundary": "training failure establishes optimization difficulty, not non-identifiability",
        },
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate primitive-anchor dose experiments."
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

    rows: list[dict[str, Any]] = []
    for spec in args.run:
        arm, seed, path = parse_run_spec(spec)
        summary = json.loads(path.read_text(encoding="utf-8"))
        history_path = path.with_name("history.json")
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else None
        )
        rows.append(
            summarize_anchor_seed(
                summary,
                arm=arm,
                seed=seed,
                temporal=temporal_by_key.get((arm, seed)),
                history=history,
            )
        )
    rows.sort(key=lambda row: (float(row["primitive_probability"]), row["arm"], row["seed"]))
    gates = evaluate_anchor_dose_gates(rows)
    (args.out_dir / "aggregate.json").write_text(
        json.dumps({"seed_rows": rows, "gates": gates}, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "claim_ledger.json").write_text(
        json.dumps(_claim_ledger(gates), indent=2),
        encoding="utf-8",
    )
    _write_seed_rows(args.out_dir / "seed_rows.csv", rows)
    _save_plots(args.out_dir, rows)
    _save_anchor_dose_response(
        args.out_dir / "anchor_dose_response.png",
        rows,
        gates,
    )


if __name__ == "__main__":
    main()
