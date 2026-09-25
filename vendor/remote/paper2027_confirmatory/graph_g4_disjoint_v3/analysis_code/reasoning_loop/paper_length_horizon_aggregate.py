"""Aggregate logical-length horizon experiments for the paper-style task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_VARIANTS = (
    "raw",
    "full",
    "no_AB",
    "identity_D",
    "full_executor_off",
)
REQUIRED_TARGET_METRICS = (
    "target_step_exact_match",
    "target_step_answer_nll",
    "target_step_predictive_entropy",
)
REQUIRED_DIAGNOSTIC_METRICS = (
    "local_window_best_exact_match",
    "local_window_best_offset",
)
REQUIRED_CURVE_METRICS = REQUIRED_TARGET_METRICS + REQUIRED_DIAGNOSTIC_METRICS
DEFAULT_EVALUATION_LENGTHS = (
    20,
    30,
    40,
    50,
    60,
    75,
    84,
    100,
    120,
    150,
    200,
)
EXPECTED_CONTROLLER_TRAINING_BUDGET = {
    "dense_optimizer_updates": 80,
    "low_rank_optimizer_updates": 176,
    "total_optimizer_updates": 256,
    "dense_training_examples": 4352,
    "low_rank_training_examples": 7040,
    "total_training_examples": 11392,
}


@dataclass(frozen=True)
class HorizonProtocol:
    task: str
    baseline_variant: str
    train_maximum: int
    step_offset: int
    block_layers: int
    n_heads: int
    logical_maxima: tuple[int, int]
    evaluation_lengths: tuple[int, ...]

    @property
    def target_loop_rule(self) -> str:
        return "T(n)=n" if self.step_offset == 0 else f"T(n)=n+{self.step_offset}"

    @property
    def main_metric(self) -> str:
        loop = "n" if self.step_offset == 0 else f"n+{self.step_offset}"
        return f"EM(n, loop={loop})"


HORIZON_PROTOCOLS = {
    "parity": HorizonProtocol(
        task="parity",
        baseline_variant="released64",
        train_maximum=20,
        step_offset=0,
        block_layers=1,
        n_heads=64,
        logical_maxima=(20, 40),
        evaluation_lengths=DEFAULT_EVALUATION_LENGTHS,
    ),
    "addition": HorizonProtocol(
        task="addition",
        baseline_variant="official",
        train_maximum=19,
        step_offset=1,
        block_layers=3,
        n_heads=8,
        logical_maxima=(19, 40),
        evaluation_lengths=(19, 25, 30, 40, 50, 60, 75, 100),
    ),
    "copy": HorizonProtocol(
        task="copy",
        baseline_variant="official",
        train_maximum=19,
        step_offset=0,
        block_layers=2,
        n_heads=8,
        logical_maxima=(19, 40),
        evaluation_lengths=(19, 25, 30, 40, 50, 60, 75, 100),
    ),
    "sum_reverse": HorizonProtocol(
        task="sum_reverse",
        baseline_variant="official",
        train_maximum=19,
        step_offset=0,
        block_layers=2,
        n_heads=16,
        logical_maxima=(19, 40),
        evaluation_lengths=(19, 24, 30, 40, 50, 60, 75, 100),
    ),
}


def _protocol(task: str) -> HorizonProtocol:
    try:
        return HORIZON_PROTOCOLS[task]
    except KeyError as error:
        raise ValueError(f"unsupported horizon-study task: {task}") from error


def condition_summary_path(
    run_root: Path,
    *,
    task: str = "parity",
    seed: int,
    logical_maximum: int,
    controller_seed: int,
) -> Path:
    protocol = _protocol(task)
    label = (
        f"{task}_adaptive_step_{protocol.baseline_variant}_seed{seed}_rank48_"
        f"logical1to{logical_maximum}_seed{controller_seed}"
    )
    return run_root / "audits" / label / "summary.json"


def diagnosis_summary_path(
    run_root: Path, *, task: str, seed: int
) -> Path:
    protocol = _protocol(task)
    label = (
        f"{task}_adaptive_step_{protocol.baseline_variant}_seed{seed}"
    )
    return run_root / "diagnosis" / label / "summary.json"


def validate_ineligible_diagnosis(
    payload: dict[str, Any], *, task: str, seed: int, path: Path
) -> dict[str, Any]:
    """Validate a healthy registered extension before excluding it from J."""
    protocol = _protocol(task)
    task_payload = payload.get("task", {})
    model = payload.get("model", {})
    gate = payload.get("disease_gate", {})
    anchor_accuracy = gate.get("anchor_target_step_exact_match")
    minimum_endpoint = gate.get("minimum_endpoint_accuracy")
    checks = {
        "status": payload.get("status") == "complete",
        "checkpoint_step": payload.get("checkpoint_step") == 100001,
        "checkpoint_seed": (
            f"_{protocol.baseline_variant}_seed{seed}/final.pt"
            in str(payload.get("checkpoint", ""))
        ),
        "task.name": task_payload.get("name") == protocol.task,
        "task.train_max_length": (
            task_payload.get("train_max_length") == protocol.train_maximum
        ),
        "task.step_offset": task_payload.get("step_offset") == protocol.step_offset,
        "model.d_model": model.get("d_model") == 256,
        "model.n_heads": model.get("n_heads") == protocol.n_heads,
        "model.block_layers": model.get("block_layers") == protocol.block_layers,
        "supervision": payload.get("supervision") == "adaptive_step",
        "gate.passed": gate.get("passed") is False,
        "gate.target_telomere_failure": (
            gate.get("target_telomere_failure") is False
        ),
        "gate.anchor_length": gate.get("anchor_length") == protocol.train_maximum,
        "gate.anchor_healthy": (
            anchor_accuracy is not None
            and minimum_endpoint is not None
            and float(anchor_accuracy) >= float(minimum_endpoint)
        ),
        "gate.extension_length": gate.get("extension_length") is not None,
        "gate.extension_target": (
            gate.get("extension_target_step_exact_match") is not None
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{path}: failed ineligible-diagnosis invariants: "
            + ", ".join(failed)
        )
    return {
        "backbone_seed": seed,
        "reason": "no registered target-loop telomere failure",
        "diagnosis": str(path),
        "anchor_length": int(gate["anchor_length"]),
        "anchor_target_step_exact_match": float(anchor_accuracy),
        "extension_length": int(gate["extension_length"]),
        "extension_target_step_exact_match": float(
            gate["extension_target_step_exact_match"]
        ),
        "extension_local_window_best_exact_match": (
            None
            if gate.get("extension_local_window_best_exact_match") is None
            else float(gate["extension_local_window_best_exact_match"])
        ),
    }


def reliable_prefix_horizon(
    *,
    lengths: Sequence[int],
    values: Mapping[int, float],
    threshold: float,
) -> int | None:
    """Return the last length in the consecutive reliable evaluation prefix."""
    horizon: int | None = None
    for length in lengths:
        if float(values[length]) < threshold:
            break
        horizon = int(length)
    return horizon


def normalized_length_auc(
    *, lengths: Sequence[int], values: Mapping[int, float]
) -> float:
    """Return trapezoidal AUC divided by the evaluated length span."""
    if len(lengths) < 2:
        raise ValueError("length AUC requires at least two evaluation lengths")
    area = 0.0
    for left, right in zip(lengths, lengths[1:]):
        width = int(right) - int(left)
        if width <= 0:
            raise ValueError("evaluation lengths must be strictly increasing")
        area += width * (float(values[left]) + float(values[right])) / 2.0
    return area / (int(lengths[-1]) - int(lengths[0]))


def validate_condition_summary(
    payload: dict[str, Any],
    *,
    seed: int,
    logical_maximum: int,
    controller_seed: int,
    evaluation_seed: int,
    lengths: Sequence[int],
    minimum_examples: int,
    path: Path,
    task_name: str = "parity",
) -> None:
    """Reject an audit that is not the preregistered official task arm."""
    protocol = _protocol(task_name)
    task = payload.get("task", {})
    model = payload.get("model", {})
    placement = payload.get("loss_placement", {})
    checks = {
        "status": payload.get("status") == "complete",
        "paper": payload.get("paper") == "arXiv:2409.15647v5",
        "task.name": task.get("name") == protocol.task,
        "task.train_max_length": (
            task.get("train_max_length") == protocol.train_maximum
        ),
        "task.step_offset": task.get("step_offset") == protocol.step_offset,
        "model.d_model": model.get("d_model") == 256,
        "model.n_heads": model.get("n_heads") == protocol.n_heads,
        "model.block_layers": (
            model.get("block_layers") == protocol.block_layers
        ),
        "checkpoint_step": payload.get("checkpoint_step") == 100001,
        "backbone_seed": payload.get("backbone_seed") == seed,
        "backbone_training_precision": (
            payload.get("backbone_training_precision") == "fp32"
        ),
        "backbone_supervision": (
            payload.get("backbone_supervision") == "adaptive_step"
        ),
        "backbone_official_model_config": (
            payload.get("backbone_official_model_config") is True
        ),
        "backbone_official_source_commit": (
            payload.get("backbone_official_source_commit")
            == "33650c3c0cec3dd1d466b32489aaf30df4bda796"
        ),
        "target_loop_rule": (
            payload.get("target_loop_rule") == protocol.target_loop_rule
        ),
        "controller_rank": payload.get("controller_rank") == 48,
        "controller_seed": payload.get("controller_seed") == controller_seed,
        "controller_curriculum": (
            payload.get("controller_curriculum") == "logical_range"
        ),
        "controller_logical_max_length": (
            payload.get("controller_logical_max_length") == logical_maximum
        ),
        "controller_start_step": payload.get("controller_start_step") == 1,
        "controller_trained_logical_length_range": (
            payload.get("controller_trained_logical_length_range")
            == [1, logical_maximum]
        ),
        "controller_sampled_logical_length_range": (
            payload.get("controller_sampled_logical_length_range")
            == [max(1, 2 - protocol.step_offset), logical_maximum]
        ),
        "controller_sampled_logical_lengths": (
            payload.get("controller_sampled_logical_lengths")
            == list(
                range(
                    max(1, 2 - protocol.step_offset),
                    logical_maximum + 1,
                )
            )
        ),
        "controller_training_budget": (
            payload.get("controller_training_budget")
            == EXPECTED_CONTROLLER_TRAINING_BUDGET
        ),
        "controller_loss": (
            "CE only" in str(payload.get("controller_loss", ""))
            and "no intermediate" in str(payload.get("controller_loss", ""))
        ),
        "loss_placement.controller": (
            "CE only" in str(placement.get("controller", ""))
            and "no intermediate or state loss"
            in str(placement.get("controller", ""))
        ),
        "shared_physical_block_layers": (
            payload.get("shared_physical_block_layers")
            == protocol.block_layers
        ),
        "maximum_effective_evaluated_depth": (
            int(payload.get("maximum_effective_evaluated_depth", -1))
            >= (int(lengths[-1]) + protocol.step_offset)
            * protocol.block_layers
        ),
        "evaluation_seed": payload.get("evaluation_seed") == evaluation_seed,
        "lengths": payload.get("lengths") == list(lengths),
        "examples_per_length": (
            int(payload.get("examples_per_length", 0)) >= minimum_examples
        ),
    }
    audited_modes = set(payload.get("audited_controller_modes", ()))
    checks["audited_controller_modes"] = {
        "full",
        "no_AB",
        "identity_D",
    } <= audited_modes
    curves = payload.get("curves", {})
    checks["curve_variants"] = all(
        variant in curves for variant in REQUIRED_VARIANTS
    )
    checks["curve_lengths_and_metrics"] = all(
        str(length) in curves.get(variant, {})
        and all(
            curves[variant][str(length)].get(metric) is not None
            for metric in REQUIRED_CURVE_METRICS
        )
        for variant in REQUIRED_VARIANTS
        for length in lengths
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{path}: failed horizon-audit invariants: {', '.join(failed)}"
        )


def _row(
    *,
    seed: int,
    logical_maximum: int | None,
    variant: str,
    length: int,
    curve: Mapping[str, Any],
) -> dict[str, Any]:
    condition = "raw" if logical_maximum is None else f"J_1-{logical_maximum}"
    display_variant = (
        "executor_off" if variant == "full_executor_off" else variant
    )
    arm = (
        "raw"
        if logical_maximum is None
        else condition
        if variant == "full"
        else f"{condition}/{display_variant}"
    )
    return {
        "backbone_seed": seed,
        "condition": condition,
        "variant": display_variant,
        "arm": arm,
        "controller_logical_max_length": logical_maximum,
        "length": length,
        "target_step_exact_match": float(curve["target_step_exact_match"]),
        "target_step_answer_nll": float(curve["target_step_answer_nll"]),
        "target_step_predictive_entropy": float(
            curve["target_step_predictive_entropy"]
        ),
        "local_window_best_exact_match": float(
            curve["local_window_best_exact_match"]
        ),
        "local_window_best_offset": float(
            curve["local_window_best_offset"]
        ),
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["length"]))].append(row)
    aggregate: list[dict[str, Any]] = []
    for (arm, length), members in sorted(grouped.items()):
        result: dict[str, Any] = {
            "arm": arm,
            "length": length,
            "seeds": len(members),
        }
        for metric in REQUIRED_CURVE_METRICS:
            values = [float(member[metric]) for member in members]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate.append(result)
    return aggregate


def _curve_metrics(
    *,
    lengths: Sequence[int],
    exact_match: Mapping[int, float],
    answer_nll: Mapping[int, float],
    predictive_entropy: Mapping[int, float],
) -> dict[str, float]:
    return {
        "target_em_grid_mean": statistics.mean(
            float(exact_match[length]) for length in lengths
        ),
        "target_em_length_auc": normalized_length_auc(
            lengths=lengths, values=exact_match
        ),
        "answer_nll_grid_mean": statistics.mean(
            float(answer_nll[length]) for length in lengths
        ),
        "predictive_entropy_grid_mean": statistics.mean(
            float(predictive_entropy[length]) for length in lengths
        ),
    }


def _arm_curves_from_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[int, float]]]:
    result: dict[str, dict[str, dict[int, float]]] = {}
    for row in rows:
        arm = str(row["arm"])
        length = int(row["length"])
        arm_result = result.setdefault(
            arm, {metric: {} for metric in REQUIRED_CURVE_METRICS}
        )
        for metric in REQUIRED_CURVE_METRICS:
            arm_result[metric][length] = float(row[metric])
    return result


def phase_failure_mode(
    *,
    target_exact_match: float,
    nearby_best_exact_match: float,
    threshold: float,
) -> str:
    """Label failure type without allowing nearby accuracy to score success."""
    if float(target_exact_match) >= threshold:
        return "target_reliable"
    if float(nearby_best_exact_match) >= threshold:
        return "clock_drift"
    return "nearby_executor_exhaustion"


def _phase_profile(
    *,
    lengths: Sequence[int],
    metrics: Mapping[str, Mapping[int, float]],
    threshold: float,
) -> dict[str, Any]:
    target = metrics["target_step_exact_match"]
    nearby = metrics["local_window_best_exact_match"]
    modes = {
        int(length): phase_failure_mode(
            target_exact_match=target[int(length)],
            nearby_best_exact_match=nearby[int(length)],
            threshold=threshold,
        )
        for length in lengths
    }
    return {
        "target_reliable_prefix": reliable_prefix_horizon(
            lengths=lengths, values=target, threshold=threshold
        ),
        "nearby_reliable_prefix": reliable_prefix_horizon(
            lengths=lengths, values=nearby, threshold=threshold
        ),
        "clock_drift_lengths": [
            length for length in lengths if modes[int(length)] == "clock_drift"
        ],
        "nearby_executor_exhaustion_lengths": [
            length
            for length in lengths
            if modes[int(length)] == "nearby_executor_exhaustion"
        ],
    }


def _strict_horizon_order(
    *,
    horizons: Mapping[str, int | None],
    first_length: int,
    ordered_arms: Sequence[str],
) -> bool:
    numerical = {
        arm: first_length - 1 if value is None else int(value)
        for arm, value in horizons.items()
    }
    return all(
        numerical[left] < numerical[right]
        for left, right in zip(ordered_arms, ordered_arms[1:])
    )


def aggregate_horizon_study(
    *,
    task: str = "parity",
    run_root: Path,
    seeds: Sequence[int],
    logical_maxima: Sequence[int],
    controller_seed: int,
    evaluation_seed: int,
    lengths: Sequence[int],
    minimum_examples: int,
    allow_ineligible: bool = False,
) -> dict[str, Any]:
    """Load paired audits and compute main target-loop horizon evidence."""
    protocol = _protocol(task)
    if tuple(logical_maxima) != protocol.logical_maxima:
        raise ValueError(
            f"the preregistered {task} logical maxima must be "
            f"{protocol.logical_maxima}"
        )
    screened_seeds = tuple(int(seed) for seed in seeds)
    eligible_seeds: list[int] = []
    ineligible: list[dict[str, Any]] = []
    sources: list[str] = []
    for seed in screened_seeds:
        condition_paths = [
            condition_summary_path(
                run_root,
                task=task,
                seed=seed,
                logical_maximum=logical_maximum,
                controller_seed=controller_seed,
            )
            for logical_maximum in logical_maxima
        ]
        if all(path.exists() for path in condition_paths):
            eligible_seeds.append(seed)
            continue
        if not allow_ineligible:
            missing = next(path for path in condition_paths if not path.exists())
            raise FileNotFoundError(missing)
        if any(path.exists() for path in condition_paths):
            raise ValueError(
                f"seed {seed}: only part of the paired J audit exists"
            )
        diagnosis_path = diagnosis_summary_path(
            run_root, task=task, seed=seed
        )
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        ineligible.append(
            validate_ineligible_diagnosis(
                diagnosis, task=task, seed=seed, path=diagnosis_path
            )
        )
        sources.append(str(diagnosis_path))
    if not eligible_seeds:
        raise ValueError("no disease-positive backbone has paired J audits")

    payloads: dict[tuple[int, int], dict[str, Any]] = {}
    paired_raw_max_difference = 0.0
    for seed in eligible_seeds:
        reference_payload: dict[str, Any] | None = None
        for logical_maximum in logical_maxima:
            path = condition_summary_path(
                run_root,
                task=task,
                seed=seed,
                logical_maximum=logical_maximum,
                controller_seed=controller_seed,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_condition_summary(
                payload,
                seed=seed,
                logical_maximum=logical_maximum,
                controller_seed=controller_seed,
                evaluation_seed=evaluation_seed,
                lengths=lengths,
                minimum_examples=minimum_examples,
                path=path,
                task_name=task,
            )
            if reference_payload is None:
                reference_payload = payload
            else:
                if payload.get("checkpoint") != reference_payload.get("checkpoint"):
                    raise ValueError(
                        f"seed {seed}: logical conditions use different backbones"
                    )
                for length in lengths:
                    for metric in REQUIRED_CURVE_METRICS:
                        reference_value = float(
                            reference_payload["curves"]["raw"][str(length)][metric]
                        )
                        observed_value = float(
                            payload["curves"]["raw"][str(length)][metric]
                        )
                        difference = abs(reference_value - observed_value)
                        paired_raw_max_difference = max(
                            paired_raw_max_difference, difference
                        )
                        if difference > 1e-7:
                            raise ValueError(
                                f"seed {seed}: paired raw evaluation differs at "
                                f"length {length}, metric {metric}"
                            )
            payloads[(seed, logical_maximum)] = payload
            sources.append(str(path))

    rows: list[dict[str, Any]] = []
    for seed in eligible_seeds:
        raw_payload = payloads[(seed, logical_maxima[0])]
        for length in lengths:
            rows.append(
                _row(
                    seed=seed,
                    logical_maximum=None,
                    variant="raw",
                    length=length,
                    curve=raw_payload["curves"]["raw"][str(length)],
                )
            )
        for logical_maximum in logical_maxima:
            payload = payloads[(seed, logical_maximum)]
            for variant in REQUIRED_VARIANTS[1:]:
                for length in lengths:
                    rows.append(
                        _row(
                            seed=seed,
                            logical_maximum=logical_maximum,
                            variant=variant,
                            length=length,
                            curve=payload["curves"][variant][str(length)],
                        )
                    )

    per_seed_curves: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    per_seed_length_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for seed in eligible_seeds:
        seed_rows = [row for row in rows if row["backbone_seed"] == seed]
        curves = _arm_curves_from_rows(seed_rows)
        per_seed_curves[str(seed)] = curves
        per_seed_length_metrics[str(seed)] = {
            arm: _curve_metrics(
                lengths=lengths,
                exact_match=metrics["target_step_exact_match"],
                answer_nll=metrics["target_step_answer_nll"],
                predictive_entropy=metrics[
                    "target_step_predictive_entropy"
                ],
            )
            for arm, metrics in curves.items()
        }

    aggregate = _aggregate_rows(rows)
    mean_rows = [
        {
            "arm": row["arm"],
            "length": row["length"],
            **{
                metric: row[f"{metric}_mean"]
                for metric in REQUIRED_CURVE_METRICS
            },
        }
        for row in aggregate
    ]
    mean_curves = _arm_curves_from_rows(mean_rows)
    mean_length_metrics = {
        arm: _curve_metrics(
            lengths=lengths,
            exact_match=metrics["target_step_exact_match"],
            answer_nll=metrics["target_step_answer_nll"],
            predictive_entropy=metrics["target_step_predictive_entropy"],
        )
        for arm, metrics in mean_curves.items()
    }

    primary_arms = (
        "raw",
        *(f"J_1-{maximum}" for maximum in logical_maxima),
    )
    horizons: dict[str, Any] = {}
    phase_diagnostics: dict[str, Any] = {}
    for threshold in (0.90, 0.95):
        per_seed_horizons = {
            str(seed): {
                arm: reliable_prefix_horizon(
                    lengths=lengths,
                    values=per_seed_curves[str(seed)][arm][
                        "target_step_exact_match"
                    ],
                    threshold=threshold,
                )
                for arm in primary_arms
            }
            for seed in eligible_seeds
        }
        mean_horizons = {
            arm: reliable_prefix_horizon(
                lengths=lengths,
                values=mean_curves[arm]["target_step_exact_match"],
                threshold=threshold,
            )
            for arm in primary_arms
        }
        horizons[f"{threshold:.2f}"] = {
            "per_seed": per_seed_horizons,
            "mean_curve": mean_horizons,
            "strict_order_pass": _strict_horizon_order(
                horizons=mean_horizons,
                first_length=int(lengths[0]),
                ordered_arms=primary_arms,
            ),
            "strict_order_pass_per_seed": {
                seed: _strict_horizon_order(
                    horizons=values,
                    first_length=int(lengths[0]),
                    ordered_arms=primary_arms,
                )
                for seed, values in per_seed_horizons.items()
            },
        }
        phase_diagnostics[f"{threshold:.2f}"] = {
            "per_seed": {
                str(seed): {
                    arm: _phase_profile(
                        lengths=lengths,
                        metrics=per_seed_curves[str(seed)][arm],
                        threshold=threshold,
                    )
                    for arm in per_seed_curves[str(seed)]
                }
                for seed in eligible_seeds
            },
            "mean_curve": {
                arm: _phase_profile(
                    lengths=lengths,
                    metrics=metrics,
                    threshold=threshold,
                )
                for arm, metrics in mean_curves.items()
            },
        }

    return {
        "status": "complete",
        "paper": "arXiv:2409.15647v5",
        "task": task,
        "main_metric": protocol.main_metric,
        "horizon_definition": (
            "last length in the ordered evaluation prefix whose target-loop "
            "EM is at least q at every earlier grid point"
        ),
        "backbone_training": {
            "logical_length_range": [1, protocol.train_maximum],
            "target_loop_rule": protocol.target_loop_rule,
            "loss": (
                "answer-region CE only at sample-specific "
                f"{protocol.target_loop_rule}"
            ),
            "shared_physical_block_layers": protocol.block_layers,
            "maximum_effective_training_depth": (
                (protocol.train_maximum + protocol.step_offset)
                * protocol.block_layers
            ),
            "official_n_heads": protocol.n_heads,
            "official_baseline_variant": protocol.baseline_variant,
        },
        "controller_training_logical_maxima": list(logical_maxima),
        "controller": "J(h)=hD+(hA)B+b, rank 48",
        "controller_loss": "pure final task CE at T(n); no state loss",
        "evaluation_lengths": list(lengths),
        "examples_per_length_minimum": minimum_examples,
        "backbone_seeds": list(eligible_seeds),
        "screening": {
            "screened_backbone_seeds": list(screened_seeds),
            "eligible_backbone_seeds": list(eligible_seeds),
            "ineligible_backbone_seeds": [
                int(row["backbone_seed"]) for row in ineligible
            ],
            "telomere_positive_fraction": (
                len(eligible_seeds) / len(screened_seeds)
            ),
            "ineligible": ineligible,
            "method_effect_population": (
                "paired disease-positive frozen backbones only"
            ),
        },
        "controller_seed": controller_seed,
        "controller_training_budget": EXPECTED_CONTROLLER_TRAINING_BUDGET,
        "evaluation_seed": evaluation_seed,
        "sources": sources,
        "paired_raw_max_abs_difference": paired_raw_max_difference,
        "per_seed": rows,
        "aggregate": aggregate,
        "horizons": horizons,
        "phase_diagnostic_role": (
            "nearby-window accuracy only distinguishes clock drift from "
            "nearby executor exhaustion; it never replaces target-loop EM"
        ),
        "phase_diagnostics": phase_diagnostics,
        "length_metrics": {
            "per_seed": per_seed_length_metrics,
            "mean_curve": mean_length_metrics,
        },
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def markdown_report(study: Mapping[str, Any]) -> str:
    """Render a paper-facing report that keeps target-loop EM primary."""
    aggregate = {
        (str(row["arm"]), int(row["length"])): row
        for row in study["aggregate"]
    }
    logical_maxima = tuple(study["controller_training_logical_maxima"])
    primary_arms = ("raw", *(f"J_1-{value}" for value in logical_maxima))
    backbone = study["backbone_training"]
    lines = [
        f"# Logical-length horizon generalization of J on {study['task']}",
        "",
        "## Protocol",
        "",
        f"- Main metric: `{study['main_metric']}`; nearby or oracle loop selection is not used.",
        f"- Backbone: {study['task']}, logical lengths 1-{backbone['logical_length_range'][1]}, sample-specific final answer-region CE at {backbone['target_loop_rule']}.",
        "- Controller: `J(h)=hD+(hA)B+b`, rank 48, pure final task CE and no hidden-state loss.",
        f"- {backbone['shared_physical_block_layers']} shared physical Transformer block layer(s); maximum backbone training effective depth is {backbone['maximum_effective_training_depth']}.",
        f"- J conditions use identical optimizer-update budgets and logical training ranges 1-{logical_maxima[0]} versus 1-{logical_maxima[1]}.",
        (
            "- Baseline screening: "
            f"{len(study['screening']['eligible_backbone_seeds'])}/"
            f"{len(study['screening']['screened_backbone_seeds'])} frozen "
            "backbone seeds passed the registered telomere gate; paired J "
            "effects below are conditional on those disease-positive seeds."
        ),
        "",
        "## Reliable-prefix horizon",
        "",
        "The horizon is the last point in the ordered length grid for which every earlier target-loop EM also meets the threshold. A late accuracy revival cannot extend it.",
        "",
        f"| Threshold | {' | '.join(primary_arms)} | strict order |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for threshold in ("0.90", "0.95"):
        horizon = study["horizons"][threshold]
        mean = horizon["mean_curve"]
        values = " | ".join(_fmt(mean[arm]) for arm in primary_arms)
        lines.append(
            f"| {threshold} | {values} | "
            f"{'PASS' if horizon['strict_order_pass'] else 'FAIL'} |"
        )
    lines.extend(
        (
            "",
            "## Target-loop exact match",
            "",
            "Mean +/- sample standard deviation across frozen-backbone seeds.",
            "",
            f"| Length n | {' | '.join(primary_arms)} |",
            "|---:|---:|---:|---:|",
        )
    )
    for length in study["evaluation_lengths"]:
        cells = []
        for arm in primary_arms:
            row = aggregate[(arm, int(length))]
            cells.append(
                f"{row['target_step_exact_match_mean']:.3f} +/- "
                f"{row['target_step_exact_match_std']:.3f}"
            )
        lines.append(f"| {length} | {' | '.join(cells)} |")
    lines.extend(
        (
            "",
            "## Across-length metrics on the mean curves",
            "",
            "| Arm | normalized EM length AUC | mean answer NLL | mean predictive entropy |",
            "|:---|---:|---:|---:|",
        )
    )
    for arm in primary_arms:
        metrics = study["length_metrics"]["mean_curve"][arm]
        lines.append(
            f"| {arm} | {metrics['target_em_length_auc']:.3f} | "
            f"{metrics['answer_nll_grid_mean']:.3f} | "
            f"{metrics['predictive_entropy_grid_mean']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Causal component controls",
            "",
            "The audit also contains `no_AB`, `identity_D`, and executor-off arms for each J curriculum. These distinguish low-rank correction, diagonal contribution, and continued use of the frozen recurrent executor.",
            "",
            "| Arm | q=0.90 reliable prefix | normalized EM length AUC | AUC gap from full J |",
            "|:---|---:|---:|---:|",
        )
    )
    for logical_maximum in study["controller_training_logical_maxima"]:
        condition = f"J_1-{logical_maximum}"
        full_auc = study["length_metrics"]["mean_curve"][condition][
            "target_em_length_auc"
        ]
        for suffix in (None, "no_AB", "identity_D", "executor_off"):
            arm = condition if suffix is None else f"{condition}/{suffix}"
            exact_match = {
                int(length): float(
                    aggregate[(arm, int(length))][
                        "target_step_exact_match_mean"
                    ]
                )
                for length in study["evaluation_lengths"]
            }
            horizon = reliable_prefix_horizon(
                lengths=study["evaluation_lengths"],
                values=exact_match,
                threshold=0.90,
            )
            auc = study["length_metrics"]["mean_curve"][arm][
                "target_em_length_auc"
            ]
            lines.append(
                f"| {arm} | {_fmt(horizon)} | {auc:.3f} | "
                f"{full_auc - auc:+.3f} |"
            )
    lines.extend(
        (
            "",
            "Interpretation boundary: a successful result is a finite rightward shift of the reliable prefix, not elimination of recurrent aging or proof of indefinitely stable computation.",
        )
    )
    return "\n".join(lines) + "\n"


def write_study_outputs(study: dict[str, Any], out_dir: Path) -> None:
    _write_csv(out_dir / "per_seed.csv", study["per_seed"])
    _write_csv(out_dir / "aggregate.csv", study["aggregate"])
    _atomic_text(out_dir / "REPORT.md", markdown_report(study))
    _atomic_text(
        out_dir / "summary.json",
        json.dumps(study, indent=2, sort_keys=True) + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=tuple(HORIZON_PROTOCOLS), default="parity"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--logical-maxima", type=int, nargs="+"
    )
    parser.add_argument("--controller-seed", type=int, default=211001)
    parser.add_argument("--evaluation-seed", type=int, default=261001)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
    )
    parser.add_argument("--minimum-examples", type=int, default=512)
    parser.add_argument(
        "--allow-ineligible",
        action="store_true",
        help=(
            "accept a screened seed without J audits only when its validated "
            "diagnosis records a healthy, telomere-negative registered gate"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = _protocol(args.task)
    study = aggregate_horizon_study(
        task=args.task,
        run_root=args.run_root,
        seeds=tuple(args.seeds),
        logical_maxima=tuple(args.logical_maxima or protocol.logical_maxima),
        controller_seed=args.controller_seed,
        evaluation_seed=args.evaluation_seed,
        lengths=tuple(args.lengths or protocol.evaluation_lengths),
        minimum_examples=args.minimum_examples,
        allow_ineligible=args.allow_ineligible,
    )
    write_study_outputs(study, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
