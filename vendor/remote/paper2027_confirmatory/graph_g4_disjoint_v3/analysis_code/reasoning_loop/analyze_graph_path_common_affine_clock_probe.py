"""Test whether all age-specific affine J maps translate one scalar coordinate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_age_specific_j_functional_math import (
    AGES,
    collect_states,
    load_arrays,
    make_bank,
)
from reasoning_loop.common_affine_clock_probe import (
    exact_common_direction_test,
    minimum_residual_fixed_delta_probe,
    minimum_residual_mean_delta_probe,
    orient_probe,
    scale_probe_to_mean_delta,
)
from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--comparison-probes", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=826101)
    parser.add_argument("--optimization-examples", type=int, default=1024)
    parser.add_argument("--test-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-directions", type=int, default=10000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_r2(values: np.ndarray, targets: np.ndarray, calibration: np.ndarray) -> float:
    prediction = calibration[0] * values + calibration[1]
    centered = targets - targets.mean()
    return float(1.0 - np.sum((prediction - targets) ** 2) / np.sum(centered**2))


def collect_answer_values(
    states: dict[int, torch.Tensor], probe: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    per_age = {
        age: states[age][:, -1, :].cpu().double().numpy() @ probe
        for age in range(1, 9)
    }
    values = np.concatenate([per_age[age] for age in range(1, 9)])
    targets = np.concatenate(
        [np.full(per_age[age].shape[0], float(age)) for age in range(1, 9)]
    )
    return values, targets, per_age


def matrix_candidate_rows(
    *,
    candidates: dict[str, np.ndarray],
    weights: dict[int, np.ndarray],
    biases: dict[int, np.ndarray],
    random_median_unit_residual: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identity = np.eye(next(iter(weights.values())).shape[0])
    rows = []
    per_map = []
    for name, probe in candidates.items():
        norm = float(np.linalg.norm(probe))
        residuals = np.array(
            [np.linalg.norm((weights[age] - identity) @ probe) for age in AGES]
        )
        deltas = np.array([biases[age] @ probe for age in AGES])
        stack_residual = float(np.linalg.norm(residuals))
        rows.append(
            {
                "candidate": name,
                "probe_norm": norm,
                "stack_operator_residual": stack_residual,
                "unit_normalized_stack_residual": stack_residual / max(norm, 1e-15),
                "unit_residual_vs_random_median": (
                    stack_residual / max(norm, 1e-15) / random_median_unit_residual
                ),
                "mean_algebraic_delta": float(deltas.mean()),
                "std_algebraic_delta": float(deltas.std()),
                "min_algebraic_delta": float(deltas.min()),
                "max_algebraic_delta": float(deltas.max()),
                "max_per_map_operator_residual": float(residuals.max()),
            }
        )
        for age, residual, delta in zip(AGES, residuals, deltas):
            rowscale = np.linalg.norm(weights[age] - identity, ord="fro") / np.sqrt(probe.size)
            per_map.append(
                {
                    "candidate": name,
                    "source_age": age,
                    "algebraic_delta": float(delta),
                    "operator_residual": float(residual),
                    "operator_residual_per_probe_norm": float(residual / max(norm, 1e-15)),
                    "residual_relative_to_rms_map_action": float(
                        residual / max(norm * rowscale, 1e-15)
                    ),
                    "unit_hidden_worst_error_relative_to_delta": float(
                        residual / max(abs(delta), 1e-15)
                    ),
                }
            )
    return rows, per_map


@torch.no_grad()
def natural_rows(
    *,
    candidates: dict[str, np.ndarray],
    optimization_states: dict[int, torch.Tensor],
    test_states: dict[int, torch.Tensor],
    bank,
    positions: tuple[int, ...],
    biases: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    age_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}
    for name, probe in candidates.items():
        optimization_values, optimization_targets, optimization_per_age = collect_answer_values(
            optimization_states, probe
        )
        design = np.column_stack((optimization_values, np.ones_like(optimization_values)))
        calibration = np.linalg.lstsq(design, optimization_targets, rcond=None)[0]
        test_values, test_targets, test_per_age = collect_answer_values(test_states, probe)
        optimization_prediction = design @ calibration
        test_prediction = calibration[0] * test_values + calibration[1]
        anchor_bias = 1.0 - float(optimization_per_age[1].mean())
        summaries[name] = {
            "calibration_scale": float(calibration[0]),
            "calibration_bias": float(calibration[1]),
            "optimization_age_r2": scalar_r2(
                optimization_values, optimization_targets, calibration
            ),
            "test_age_r2": scalar_r2(test_values, test_targets, calibration),
            "optimization_age_mae": float(
                np.abs(optimization_prediction - optimization_targets).mean()
            ),
            "test_age_mae": float(np.abs(test_prediction - test_targets).mean()),
            "test_rounded_age_accuracy": float(
                (np.clip(np.rint(test_prediction), 1, 8) == test_targets).mean()
            ),
            "h1_anchor_bias": anchor_bias,
            "anchored_test_age_rmse": float(
                np.sqrt(np.mean((test_values + anchor_bias - test_targets) ** 2))
            ),
        }
        for split, per_age in (
            ("optimization", optimization_per_age),
            ("test", test_per_age),
        ):
            for age in range(1, 9):
                values = per_age[age]
                age_rows.append(
                    {
                        "candidate": name,
                        "split": split,
                        "age": age,
                        "raw_mean": float(values.mean()),
                        "raw_std": float(values.std()),
                        "h1_anchored_mean": float(values.mean() + anchor_bias),
                        "calibrated_mean": float(calibration[0] * values.mean() + calibration[1]),
                        "calibrated_std": float(abs(calibration[0]) * values.std()),
                    }
                )
        for split, states in (
            ("optimization", optimization_states),
            ("test", test_states),
        ):
            for age in AGES:
                source = states[age]
                after = bank.rollback(source, source_age=age, positions=positions)
                shift = (
                    (after[:, -1, :] - source[:, -1, :]).cpu().double().numpy() @ probe
                )
                algebraic_delta = float(biases[age] @ probe)
                state_term = shift - algebraic_delta
                shift_rows.append(
                    {
                        "candidate": name,
                        "split": split,
                        "source_age": age,
                        "algebraic_delta_from_bias": algebraic_delta,
                        "natural_shift_mean": float(shift.mean()),
                        "natural_shift_std": float(shift.std()),
                        "natural_best_constant_rmse": float(shift.std()),
                        "natural_shift_cv": float(
                            shift.std() / max(abs(shift.mean()), 1e-15)
                        ),
                        "state_dependent_term_mean": float(state_term.mean()),
                        "state_dependent_term_rmse": float(
                            np.sqrt(np.mean(state_term**2))
                        ),
                    }
                )
    return age_rows, shift_rows, summaries


def plot_diagnostics(
    *, shift_rows: list[dict[str, Any]], age_rows: list[dict[str, Any]], path: Path
) -> None:
    names = (
        "svd_direction_mean_delta_minus_one",
        "minimum_residual_mean_delta_minus_one",
        "minimum_residual_all_deltas_minus_one",
    )
    colors = ("#1b9e77", "#d95f02", "#7570b3")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, color in zip(names, colors):
        selected = [
            row for row in shift_rows
            if row["candidate"] == name and row["split"] == "test"
        ]
        selected.sort(key=lambda row: row["source_age"])
        ages = np.array([row["source_age"] for row in selected])
        means = np.array([row["natural_shift_mean"] for row in selected])
        stds = np.array([row["natural_shift_std"] for row in selected])
        axes[0].errorbar(ages, means, yerr=stds, marker="o", color=color, label=name)
        algebraic = np.array([row["algebraic_delta_from_bias"] for row in selected])
        axes[0].plot(ages, algebraic, linestyle="--", color=color, alpha=0.65)

        selected_age = [
            row for row in age_rows
            if row["candidate"] == name and row["split"] == "test"
        ]
        selected_age.sort(key=lambda row: row["age"])
        natural_ages = np.array([row["age"] for row in selected_age])
        calibrated = np.array([row["calibrated_mean"] for row in selected_age])
        axes[1].plot(natural_ages, calibrated, marker="o", color=color, label=name)
    axes[0].axhline(-1.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("source age")
    axes[0].set_ylabel("T(J_i h) - T(h)")
    axes[0].set_title("solid: natural mean +/- std; dashed: bias delta")
    axes[1].plot(range(1, 9), range(1, 9), color="black", linestyle="--", label="ideal age")
    axes[1].set_xlabel("natural age")
    axes[1].set_ylabel("affine-calibrated scalar mean")
    axes[1].set_title("does the matrix-derived direction read age?")
    axes[1].legend(fontsize=7)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    weights, biases, bank_payload = load_arrays(args.bank_artifact)
    weight_list = [weights[age] for age in AGES]
    bias_matrix = np.stack([biases[age] for age in AGES], axis=1)
    exact = exact_common_direction_test(weight_list)
    unit_probe = orient_probe(exact.unit_probe, bias_matrix, target_sign=-1.0)
    candidates = {
        "minimum_singular_unit": unit_probe,
        "svd_direction_mean_delta_minus_one": scale_probe_to_mean_delta(
            unit_probe, bias_matrix, -1.0
        ),
        "minimum_residual_mean_delta_minus_one": minimum_residual_mean_delta_probe(
            weight_list, bias_matrix, -1.0
        ),
        "minimum_residual_all_deltas_minus_one": minimum_residual_fixed_delta_probe(
            weight_list, bias_matrix, -np.ones(len(AGES))
        ),
    }
    comparison_loaded = []
    if args.comparison_probes is not None:
        comparison = np.load(args.comparison_probes)
        for source_key, target_name in (
            ("joint_minus_one_weight", "previous_natural_joint_minus_one"),
            ("baseline_weight", "previous_ridge_age_probe"),
        ):
            if source_key in comparison:
                candidates[target_name] = comparison[source_key].astype(np.float64)
                comparison_loaded.append(target_name)

    rng = np.random.default_rng(args.seed + 900)
    random_residuals = []
    for _ in range(args.random_directions):
        direction = rng.normal(size=unit_probe.shape)
        direction /= np.linalg.norm(direction)
        random_residuals.append(np.linalg.norm(exact.stacked_matrix @ direction))
    random_residuals = np.asarray(random_residuals)
    random_median = float(np.median(random_residuals))
    candidate_rows, per_map_rows = matrix_candidate_rows(
        candidates=candidates,
        weights=weights,
        biases=biases,
        random_median_unit_residual=random_median,
    )

    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    optimization_states, _, _ = collect_states(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=torch.device("cpu"),
        examples=args.optimization_examples,
        batch_size=args.batch_size,
        seed=args.seed + 100,
    )
    test_states, _, _ = collect_states(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=torch.device("cpu"),
        examples=args.test_examples,
        batch_size=args.batch_size,
        seed=args.seed + 200,
    )
    bank = make_bank(weights, biases, dimension=cfg.d_model, device=torch.device("cpu"))
    positions = tuple(range(cfg.seq_len))
    age_rows, shift_rows, natural_summaries = natural_rows(
        candidates=candidates,
        optimization_states=optimization_states,
        test_states=test_states,
        bank=bank,
        positions=positions,
        biases=biases,
    )

    leave_one_out = []
    full_unit = unit_probe / np.linalg.norm(unit_probe)
    for omitted_age in AGES:
        retained = [age for age in AGES if age != omitted_age]
        retained_weights = [weights[age] for age in retained]
        retained_biases = np.stack([biases[age] for age in retained], axis=1)
        retained_test = exact_common_direction_test(retained_weights)
        retained_unit = orient_probe(retained_test.unit_probe, retained_biases, target_sign=-1.0)
        retained_mean = minimum_residual_mean_delta_probe(
            retained_weights, retained_biases, -1.0
        )
        full_mean = candidates["minimum_residual_mean_delta_minus_one"]
        leave_one_out.append(
            {
                "omitted_age": omitted_age,
                "minimum_singular_abs_cosine_to_full": float(
                    abs(retained_unit @ full_unit) / np.linalg.norm(retained_unit)
                ),
                "minimum_singular_value_without_map": float(
                    retained_test.singular_values[-1]
                ),
                "mean_delta_probe_abs_cosine_to_full": float(
                    abs(retained_mean @ full_mean)
                    / (np.linalg.norm(retained_mean) * np.linalg.norm(full_mean))
                ),
            }
        )

    spectrum_rows = [
        {"descending_index": index + 1, "singular_value": float(value)}
        for index, value in enumerate(exact.singular_values)
    ]
    write_csv(args.out_dir / "singular_spectrum.csv", spectrum_rows)
    write_csv(args.out_dir / "matrix_candidate_metrics.csv", candidate_rows)
    write_csv(args.out_dir / "per_map_matrix_metrics.csv", per_map_rows)
    write_csv(args.out_dir / "natural_age_values.csv", age_rows)
    write_csv(args.out_dir / "natural_shift_metrics.csv", shift_rows)
    write_csv(args.out_dir / "leave_one_map_out.csv", leave_one_out)
    np.savez_compressed(args.out_dir / "common_affine_clock_probes.npz", **candidates)
    plot_diagnostics(
        shift_rows=shift_rows,
        age_rows=age_rows,
        path=args.out_dir / "common_affine_clock_diagnostics.png",
    )

    rconds = (1e-8, 1e-10, 1e-12, 1e-14)
    nullities = {
        f"rcond_{rcond:g}": int(
            np.sum(exact.singular_values <= rcond * exact.singular_values[0])
        )
        for rcond in rconds
    }
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
        "exact_equation": (
            "for row-state maps h -> h W_i + b_i: "
            "(W_i-I) omega=0 and delta_i=b_i omega for every i"
        ),
        "stacked_constraint_shape": list(exact.stacked_matrix.shape),
        "largest_singular_value": float(exact.singular_values[0]),
        "smallest_singular_value": float(exact.singular_values[-1]),
        "smallest_to_largest_singular_ratio": float(
            exact.singular_values[-1] / exact.singular_values[0]
        ),
        "condition_number": float(
            exact.singular_values[0] / exact.singular_values[-1]
        ),
        "numerical_nullities": nullities,
        "exact_nonzero_common_probe_exists": any(value > 0 for value in nullities.values()),
        "random_unit_residual": {
            "directions": args.random_directions,
            "minimum": float(random_residuals.min()),
            "p01": float(np.quantile(random_residuals, 0.01)),
            "median": random_median,
            "maximum": float(random_residuals.max()),
        },
        "candidate_matrix_metrics": candidate_rows,
        "natural_state_summaries": natural_summaries,
        "comparison_probes_loaded": comparison_loaded,
        "optimization_examples_per_age": args.optimization_examples,
        "test_examples_per_age": args.test_examples,
        "claim_level": (
            "matrix and representation diagnostic only; no causal clock claim without "
            "direction-selective or interchange intervention"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
