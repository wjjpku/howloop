"""Find adversarial age probes with nearly unchanged natural-state R-squared."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.age_probe_adversarial import (
    adversarial_probe_endpoints,
    build_probe_geometry,
    fit_scaled_ridge,
    minimum_loss_probe_for_linear_targets,
    r2_score,
)
from reasoning_loop.analyze_graph_path_age_specific_j_functional_math import (
    AGES,
    collect_states,
    load_arrays,
    make_bank,
)
from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=826101)
    parser.add_argument("--optimization-examples", type=int, default=1024)
    parser.add_argument("--test-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--r2-epsilons", type=float, nargs="+", default=(0.0, 1e-5, 1e-4, 1e-3)
    )
    parser.add_argument("--primary-r2-epsilon", type=float, default=1e-4)
    parser.add_argument("--rank-rcond", type=float, default=1e-12)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.03)
    return parser.parse_args()


def natural_arrays(states: dict[int, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    chunks = [states[age][:, -1, :].detach().cpu().double().numpy() for age in range(1, 9)]
    x = np.concatenate(chunks, axis=0)
    y = np.concatenate(
        [np.full(chunks[age - 1].shape[0], float(age)) for age in range(1, 9)]
    )
    return x, y


@torch.no_grad()
def displacement_vectors(
    states: dict[int, torch.Tensor], bank, positions: tuple[int, ...]
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for age in AGES:
        source = states[age]
        after = bank.rollback(source, source_age=age, positions=positions)
        displacement = after[:, -1, :] - source[:, -1, :]
        vectors[f"age_{age}"] = displacement.mean(0).cpu().double().numpy()
    vectors["mean_ages_2_8"] = np.mean(
        [vectors[f"age_{age}"] for age in AGES], axis=0
    )
    return vectors


def prediction_metrics(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float
) -> dict[str, float]:
    prediction = x @ weight + bias
    return {
        "r2": r2_score(x, y, weight, bias),
        "mae": float(np.abs(prediction - y).mean()),
        "rounded_age_accuracy": float(
            (np.clip(np.rint(prediction), 1, 8) == y).mean()
        ),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else float("nan")


def interpolate_probe(
    start_weight: np.ndarray,
    start_bias: float,
    end_weight: np.ndarray,
    end_bias: float,
    fraction: float,
) -> tuple[np.ndarray, float]:
    return (
        start_weight + fraction * (end_weight - start_weight),
        float(start_bias + fraction * (end_bias - start_bias)),
    )


def certified_fraction(
    *,
    baseline_weight: np.ndarray,
    baseline_bias: float,
    endpoint_weight: np.ndarray,
    endpoint_bias: float,
    constraints: list[tuple[np.ndarray, np.ndarray, float]],
) -> float:
    """Largest path fraction whose natural R2 passes every supplied constraint."""

    def feasible(fraction: float) -> bool:
        weight, bias = interpolate_probe(
            baseline_weight, baseline_bias, endpoint_weight, endpoint_bias, fraction
        )
        return all(
            r2_score(x, y, weight, bias) >= threshold - 2e-10
            for x, y, threshold in constraints
        )

    if feasible(1.0):
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_ranges(rows: list[dict[str, Any]], epsilon: float, path: Path) -> None:
    selected = [
        row for row in rows
        if row["r2_epsilon"] == epsilon and row["objective"].startswith("age_")
    ]
    selected.sort(key=lambda row: int(row["objective"].split("_")[1]))
    ages = np.array([int(row["objective"].split("_")[1]) for row in selected])
    baseline = np.array([row["baseline_shift_test"] for row in selected])
    exact_min = np.array([row["exact_min_shift_optimization"] for row in selected])
    exact_max = np.array([row["exact_max_shift_optimization"] for row in selected])
    robust_min = np.array([row["certified_min_shift_test"] for row in selected])
    robust_max = np.array([row["certified_max_shift_test"] for row in selected])

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.fill_between(
        ages, exact_min, exact_max, color="#9ecae1", alpha=0.35,
        label="exact optimization-set interval",
    )
    axis.vlines(
        ages, robust_min, robust_max, color="#08519c", linewidth=5,
        label="optimization + held-out certified interval",
    )
    axis.scatter(ages, baseline, color="#d7301f", zorder=3, label="original ridge probe")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(-1.0, color="gray", linewidth=0.8, linestyle="--")
    axis.set_xlabel("source age a in J_a: H_a to H_{a-1}")
    axis.set_ylabel("mean probe shift T(J_a h) - T(h)")
    axis.set_title(f"Adversarial age-probe range, allowed R2 drop = {epsilon:g}")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.primary_r2_epsilon not in args.r2_epsilons:
        raise ValueError("primary R2 epsilon must occur in --r2-epsilons")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = tuple(range(cfg.seq_len))
    weights, biases, bank_payload = load_arrays(args.bank_artifact)
    bank = make_bank(weights, biases, dimension=cfg.d_model, device=device)

    optimization_states, _, _ = collect_states(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        examples=args.optimization_examples,
        batch_size=args.batch_size,
        seed=args.seed + 100,
    )
    test_states, _, _ = collect_states(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        examples=args.test_examples,
        batch_size=args.batch_size,
        seed=args.seed + 200,
    )
    optimization_x, optimization_y = natural_arrays(optimization_states)
    test_x, test_y = natural_arrays(test_states)
    optimization_q = displacement_vectors(optimization_states, bank, positions)
    test_q = displacement_vectors(test_states, bank, positions)

    baseline_weight, baseline_bias = fit_scaled_ridge(
        optimization_x, optimization_y, args.probe_ridge
    )
    geometry = build_probe_geometry(
        optimization_x, optimization_y, rcond=args.rank_rcond
    )
    baseline_optimization = prediction_metrics(
        optimization_x, optimization_y, baseline_weight, baseline_bias
    )
    baseline_test = prediction_metrics(test_x, test_y, baseline_weight, baseline_bias)
    ols_optimization = prediction_metrics(
        optimization_x, optimization_y, geometry.ols_weight, geometry.ols_bias
    )
    ols_test = prediction_metrics(
        test_x, test_y, geometry.ols_weight, geometry.ols_bias
    )

    rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    weight_artifact: dict[str, np.ndarray] = {
        "baseline_weight": baseline_weight,
        "baseline_bias": np.array(baseline_bias),
        "ols_weight": geometry.ols_weight,
        "ols_bias": np.array(geometry.ols_bias),
    }
    for objective_name in optimization_q:
        weight_artifact[f"optimization_displacement_{objective_name}"] = optimization_q[objective_name]
        weight_artifact[f"test_displacement_{objective_name}"] = test_q[objective_name]
    for objective_name, objective in optimization_q.items():
        for epsilon in args.r2_epsilons:
            optimization_threshold = baseline_optimization["r2"] - epsilon
            test_threshold = baseline_test["r2"] - epsilon
            endpoints = adversarial_probe_endpoints(
                geometry, objective, threshold_r2=optimization_threshold
            )
            if endpoints.status != "bounded":
                rows.append(
                    {
                        "objective": objective_name,
                        "r2_epsilon": epsilon,
                        "status": endpoints.status,
                        "baseline_r2_optimization": baseline_optimization["r2"],
                        "baseline_r2_test": baseline_test["r2"],
                        "threshold_r2_optimization": optimization_threshold,
                        "threshold_r2_test": test_threshold,
                        "ols_r2_optimization": ols_optimization["r2"],
                        "ols_r2_test": ols_test["r2"],
                        "baseline_shift_optimization": float(objective @ baseline_weight),
                        "baseline_shift_test": float(test_q[objective_name] @ baseline_weight),
                        "ols_shift_optimization": float(objective @ geometry.ols_weight),
                        "ols_shift_test": float(test_q[objective_name] @ geometry.ols_weight),
                        "exact_min_shift_optimization": float("-inf"),
                        "exact_max_shift_optimization": float("inf"),
                        "exact_min_shift_test": float("nan"),
                        "exact_max_shift_test": float("nan"),
                        "exact_min_r2_optimization": float("nan"),
                        "exact_max_r2_optimization": float("nan"),
                        "exact_min_r2_test": float("nan"),
                        "exact_max_r2_test": float("nan"),
                        "certified_min_fraction": float("nan"),
                        "certified_max_fraction": float("nan"),
                        "certified_min_shift_optimization": float("-inf"),
                        "certified_max_shift_optimization": float("inf"),
                        "certified_min_shift_test": float("nan"),
                        "certified_max_shift_test": float("nan"),
                        "certified_min_r2_optimization": float("nan"),
                        "certified_max_r2_optimization": float("nan"),
                        "certified_min_r2_test": float("nan"),
                        "certified_max_r2_test": float("nan"),
                        "min_weight_norm": float("nan"),
                        "max_weight_norm": float("nan"),
                        "min_cosine_to_baseline": float("nan"),
                        "max_cosine_to_baseline": float("nan"),
                        "null_objective_norm": endpoints.null_objective_norm,
                    }
                )
                continue

            constraints = [
                (optimization_x, optimization_y, optimization_threshold),
                (test_x, test_y, test_threshold),
            ]
            certified: dict[str, tuple[np.ndarray, float, float]] = {}
            for endpoint_name, endpoint_weight, endpoint_bias in (
                ("min", endpoints.min_weight, endpoints.min_bias),
                ("max", endpoints.max_weight, endpoints.max_bias),
            ):
                fraction = certified_fraction(
                    baseline_weight=baseline_weight,
                    baseline_bias=baseline_bias,
                    endpoint_weight=endpoint_weight,
                    endpoint_bias=endpoint_bias,
                    constraints=constraints,
                )
                certified_weight, certified_bias = interpolate_probe(
                    baseline_weight,
                    baseline_bias,
                    endpoint_weight,
                    endpoint_bias,
                    fraction,
                )
                certified[endpoint_name] = (
                    certified_weight, certified_bias, fraction
                )
                key = f"{objective_name}_eps_{epsilon:g}_{endpoint_name}"
                weight_artifact[f"{key}_exact_weight"] = endpoint_weight
                weight_artifact[f"{key}_exact_bias"] = np.array(endpoint_bias)
                weight_artifact[f"{key}_certified_weight"] = certified_weight
                weight_artifact[f"{key}_certified_bias"] = np.array(certified_bias)

            min_certified_weight, min_certified_bias, min_fraction = certified["min"]
            max_certified_weight, max_certified_bias, max_fraction = certified["max"]
            exact_min_opt = prediction_metrics(
                optimization_x, optimization_y, endpoints.min_weight, endpoints.min_bias
            )
            exact_max_opt = prediction_metrics(
                optimization_x, optimization_y, endpoints.max_weight, endpoints.max_bias
            )
            exact_min_test = prediction_metrics(
                test_x, test_y, endpoints.min_weight, endpoints.min_bias
            )
            exact_max_test = prediction_metrics(
                test_x, test_y, endpoints.max_weight, endpoints.max_bias
            )
            certified_min_opt = prediction_metrics(
                optimization_x, optimization_y, min_certified_weight, min_certified_bias
            )
            certified_max_opt = prediction_metrics(
                optimization_x, optimization_y, max_certified_weight, max_certified_bias
            )
            certified_min_test = prediction_metrics(
                test_x, test_y, min_certified_weight, min_certified_bias
            )
            certified_max_test = prediction_metrics(
                test_x, test_y, max_certified_weight, max_certified_bias
            )
            row = {
                "objective": objective_name,
                "r2_epsilon": epsilon,
                "status": endpoints.status,
                "baseline_r2_optimization": baseline_optimization["r2"],
                "baseline_r2_test": baseline_test["r2"],
                "threshold_r2_optimization": optimization_threshold,
                "threshold_r2_test": test_threshold,
                "ols_r2_optimization": ols_optimization["r2"],
                "ols_r2_test": ols_test["r2"],
                "baseline_shift_optimization": float(objective @ baseline_weight),
                "baseline_shift_test": float(test_q[objective_name] @ baseline_weight),
                "ols_shift_optimization": float(objective @ geometry.ols_weight),
                "ols_shift_test": float(test_q[objective_name] @ geometry.ols_weight),
                "exact_min_shift_optimization": endpoints.min_objective,
                "exact_max_shift_optimization": endpoints.max_objective,
                "exact_min_shift_test": float(test_q[objective_name] @ endpoints.min_weight),
                "exact_max_shift_test": float(test_q[objective_name] @ endpoints.max_weight),
                "exact_min_r2_optimization": exact_min_opt["r2"],
                "exact_max_r2_optimization": exact_max_opt["r2"],
                "exact_min_r2_test": exact_min_test["r2"],
                "exact_max_r2_test": exact_max_test["r2"],
                "certified_min_fraction": min_fraction,
                "certified_max_fraction": max_fraction,
                "certified_min_shift_optimization": float(objective @ min_certified_weight),
                "certified_max_shift_optimization": float(objective @ max_certified_weight),
                "certified_min_shift_test": float(test_q[objective_name] @ min_certified_weight),
                "certified_max_shift_test": float(test_q[objective_name] @ max_certified_weight),
                "certified_min_r2_optimization": certified_min_opt["r2"],
                "certified_max_r2_optimization": certified_max_opt["r2"],
                "certified_min_r2_test": certified_min_test["r2"],
                "certified_max_r2_test": certified_max_test["r2"],
                "min_weight_norm": float(np.linalg.norm(endpoints.min_weight)),
                "max_weight_norm": float(np.linalg.norm(endpoints.max_weight)),
                "min_cosine_to_baseline": cosine(endpoints.min_weight, baseline_weight),
                "max_cosine_to_baseline": cosine(endpoints.max_weight, baseline_weight),
                "null_objective_norm": endpoints.null_objective_norm,
            }
            rows.append(row)

            for evaluated_name, evaluated_q in test_q.items():
                stage_rows.append(
                    {
                        "optimized_objective": objective_name,
                        "r2_epsilon": epsilon,
                        "evaluated_objective": evaluated_name,
                        "baseline_shift_test": float(evaluated_q @ baseline_weight),
                        "exact_min_probe_shift_test": float(evaluated_q @ endpoints.min_weight),
                        "exact_max_probe_shift_test": float(evaluated_q @ endpoints.max_weight),
                        "certified_min_probe_shift_test": float(evaluated_q @ min_certified_weight),
                        "certified_max_probe_shift_test": float(evaluated_q @ max_certified_weight),
                    }
                )

    write_csv(args.out_dir / "adversarial_probe_ranges.csv", rows)
    write_csv(args.out_dir / "adversarial_probe_stage_matrix.csv", stage_rows)

    stage_names = [f"age_{age}" for age in AGES]
    joint_objectives = np.stack([optimization_q[name] for name in stage_names])
    joint_probe = minimum_loss_probe_for_linear_targets(
        geometry, joint_objectives, -np.ones(len(stage_names))
    )
    joint_optimization_metrics = prediction_metrics(
        optimization_x, optimization_y, joint_probe.weight, joint_probe.bias
    )
    joint_test_metrics = prediction_metrics(
        test_x, test_y, joint_probe.weight, joint_probe.bias
    )
    joint_rows = [
        {
            "objective": name,
            "target_shift": -1.0,
            "optimization_shift": float(optimization_q[name] @ joint_probe.weight),
            "test_shift": float(test_q[name] @ joint_probe.weight),
        }
        for name in stage_names
    ]
    write_csv(args.out_dir / "joint_minus_one_probe.csv", joint_rows)
    weight_artifact["joint_minus_one_weight"] = joint_probe.weight
    weight_artifact["joint_minus_one_bias"] = np.array(joint_probe.bias)
    np.savez_compressed(args.out_dir / "adversarial_probe_weights.npz", **weight_artifact)
    plot_ranges(
        rows,
        args.primary_r2_epsilon,
        args.out_dir / "adversarial_probe_ranges_primary.png",
    )

    positive_eigenvalues = geometry.gram_eigenvalues[: geometry.numerical_rank]
    primary_rows = [
        row for row in rows if row["r2_epsilon"] == args.primary_r2_epsilon
    ]
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_curriculum": bank_payload.get("curriculum"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "probe_domain": "answer-token hidden states at natural ages 1 through 8",
        "optimization_problem": (
            "min/max mean[T(J_a h)-T(h)] subject to natural-state R2 "
            ">= original ridge-probe R2 - epsilon"
        ),
        "optimization_examples_per_age": args.optimization_examples,
        "test_examples_per_age": args.test_examples,
        "probe_ridge": args.probe_ridge,
        "r2_epsilons": args.r2_epsilons,
        "primary_r2_epsilon": args.primary_r2_epsilon,
        "baseline_optimization_metrics": baseline_optimization,
        "baseline_test_metrics": baseline_test,
        "ols_optimization_metrics": ols_optimization,
        "ols_test_metrics": ols_test,
        "numerical_rank": geometry.numerical_rank,
        "dimension": cfg.d_model,
        "rank_rcond": args.rank_rcond,
        "gram_condition_number_on_retained_space": float(
            positive_eigenvalues[0] / positive_eigenvalues[-1]
        ),
        "primary_ranges": primary_rows,
        "joint_minus_one_probe": {
            "status": joint_probe.status,
            "constraint_residual_norm_on_optimization_displacements": joint_probe.constraint_residual_norm,
            "optimization_metrics": joint_optimization_metrics,
            "test_metrics": joint_test_metrics,
            "r2_drop_from_baseline_optimization": baseline_optimization["r2"] - joint_optimization_metrics["r2"],
            "r2_drop_from_baseline_test": baseline_test["r2"] - joint_test_metrics["r2"],
            "stage_shifts": joint_rows,
        },
        "certification": (
            "exact endpoints solve the optimization-set problem; certified endpoints "
            "are shrunk toward the original ridge probe until both optimization and "
            "independent test R2 satisfy their split-specific thresholds"
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
