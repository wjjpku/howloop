"""Matrix-level mathematical analysis of seven age-specific affine J maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.stats as stats
import torch


AGES = tuple(range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-bank", type=Path, required=True)
    parser.add_argument("--control-bank", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_bank(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture = payload.get("map_architecture")
    state = payload["state_dict"]
    if architecture == "full_affine":
        weights = {
            age: state[f"maps.{age}.weight"].double().numpy() for age in AGES
        }
        biases = {
            age: state[f"maps.{age}.bias"].double().numpy() for age in AGES
        }
    elif architecture == "shared_diagonal_stage_lora":
        diagonal = torch.diag(state["shared_diagonal_scale"].double())
        shared = state["shared_A"].double() @ state["shared_B"].double()
        weights = {
            age: (
                diagonal
                + shared
                + state[f"stage_A.{age}"].double()
                @ state[f"stage_B.{age}"].double()
            ).numpy()
            for age in AGES
        }
        biases = {
            age: state["shared_bias"].double().numpy().copy() for age in AGES
        }
    elif architecture == "diagonal_lora":
        weights = {
            age: (
                torch.diag(state[f"maps.{age}.diagonal_scale"].double())
                + state[f"maps.{age}.A"].double()
                @ state[f"maps.{age}.B"].double()
            ).numpy()
            for age in AGES
        }
        biases = {
            age: state[f"maps.{age}.bias"].double().numpy() for age in AGES
        }
    else:
        raise ValueError(f"unsupported J architecture: {architecture}")
    return weights, biases, payload


def affine_delta(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return [W-I; b], so [h,1] @ delta is the hidden displacement."""
    return np.vstack((weight - np.eye(weight.shape[0]), bias[None, :]))


def augmented(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    dimension = weight.shape[0]
    result = np.eye(dimension + 1)
    result[:dimension, :dimension] = weight
    result[dimension, :dimension] = bias
    return result


def compose(maps: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    dimension = maps[0][0].shape[0]
    weight = np.eye(dimension)
    bias = np.zeros(dimension)
    for next_weight, next_bias in maps:
        bias = bias @ next_weight + next_bias
        weight = weight @ next_weight
    return weight, bias


def energy_rank(singular: np.ndarray, threshold: float) -> int:
    energy = singular**2
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), threshold) + 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix_metrics(weights: dict[int, np.ndarray], biases: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for age in AGES:
        weight, bias = weights[age], biases[age]
        dimension = weight.shape[0]
        identity = np.eye(dimension)
        delta = weight - identity
        singular = sla.svdvals(weight)
        delta_singular = sla.svdvals(delta)
        eigenvalues = sla.eigvals(weight)
        unitary, positive = sla.polar(weight)
        normal_commutator = weight.T @ weight - weight @ weight.T
        rows.append(
            {
                "source_age": age,
                "weight_frobenius": float(sla.norm(weight, "fro")),
                "delta_weight_frobenius": float(sla.norm(delta, "fro")),
                "delta_relative_to_identity": float(
                    sla.norm(delta, "fro") / sla.norm(identity, "fro")
                ),
                "bias_norm": float(sla.norm(bias)),
                "singular_min": float(singular[-1]),
                "singular_q10": float(np.quantile(singular, 0.1)),
                "singular_median": float(np.median(singular)),
                "singular_q90": float(np.quantile(singular, 0.9)),
                "singular_max": float(singular[0]),
                "fraction_singular_below_1": float(np.mean(singular < 1.0)),
                "fraction_singular_above_1": float(np.mean(singular > 1.0)),
                "mean_log_singular": float(np.mean(np.log(singular.clip(1e-300)))),
                "weight_stable_rank": float(np.sum(singular**2) / singular[0] ** 2),
                "delta_stable_rank": float(
                    np.sum(delta_singular**2) / delta_singular[0] ** 2
                ),
                "delta_energy_rank_50": energy_rank(delta_singular, 0.50),
                "delta_energy_rank_80": energy_rank(delta_singular, 0.80),
                "delta_energy_rank_90": energy_rank(delta_singular, 0.90),
                "delta_energy_rank_95": energy_rank(delta_singular, 0.95),
                "delta_energy_rank_99": energy_rank(delta_singular, 0.99),
                "spectral_radius": float(np.max(np.abs(eigenvalues))),
                "eigenvalue_angle_abs_median": float(
                    np.median(np.abs(np.angle(eigenvalues)))
                ),
                "eigenvalue_angle_abs_p90": float(
                    np.quantile(np.abs(np.angle(eigenvalues)), 0.9)
                ),
                "normalized_nonnormality": float(
                    sla.norm(normal_commutator, "fro")
                    / sla.norm(weight, "fro") ** 2
                ),
                "polar_rotation_distance": float(sla.norm(unitary - identity, "fro")),
                "polar_deformation_distance": float(sla.norm(positive - identity, "fro")),
                "distance_to_nearest_orthogonal_relative_to_delta": float(
                    sla.norm(weight - unitary, "fro")
                    / sla.norm(delta, "fro")
                ),
            }
        )
    return rows


def shared_analysis(weights: dict[int, np.ndarray], biases: dict[int, np.ndarray]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    deltas = np.stack(
        [affine_delta(weights[age], biases[age]).reshape(-1) for age in AGES]
    )
    norms = np.linalg.norm(deltas, axis=1)
    cosine = deltas @ deltas.T / np.outer(norms, norms)
    u, singular, vh = sla.svd(deltas, full_matrices=False)
    uncentered_fraction = singular**2 / np.sum(singular**2)
    mean_delta = deltas.mean(axis=0)
    denominator = float(mean_delta @ mean_delta)
    step_coefficients = deltas @ mean_delta / denominator
    generator_fit = np.outer(step_coefficients, mean_delta)
    residuals = deltas - generator_fit
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    centered_u, centered_singular, centered_vh = sla.svd(centered, full_matrices=False)
    centered_fraction = centered_singular**2 / np.sum(centered_singular**2)
    centered_scores = centered_u * centered_singular[None, :]
    ages = np.asarray(AGES, dtype=float)
    first_score = centered_scores[:, 0]
    regression = stats.linregress(ages, first_score)
    bias_matrix = np.stack([biases[age] for age in AGES])
    bias_norms = np.linalg.norm(bias_matrix, axis=1)
    bias_cosine = bias_matrix @ bias_matrix.T / np.outer(bias_norms, bias_norms)
    mean_bias = bias_matrix.mean(axis=0)
    result = {
        "pairwise_delta_cosine_min": float(
            cosine[np.triu_indices_from(cosine, k=1)].min()
        ),
        "pairwise_delta_cosine_mean": float(
            cosine[np.triu_indices_from(cosine, k=1)].mean()
        ),
        "uncentered_pc_energy_fractions": uncentered_fraction.tolist(),
        "uncentered_pc1_energy_fraction": float(uncentered_fraction[0]),
        "shared_generator_r2_uncentered": float(
            1.0 - np.sum((deltas - generator_fit) ** 2) / np.sum(deltas**2)
        ),
        "step_coefficients": {
            str(age): float(value) for age, value in zip(AGES, step_coefficients)
        },
        "step_coefficient_range": [
            float(step_coefficients.min()), float(step_coefficients.max())
        ],
        "stage_residual_relative_norm": {
            str(age): float(sla.norm(residual) / norm)
            for age, residual, norm in zip(AGES, residuals, norms)
        },
        "centered_pc_energy_fractions": centered_fraction.tolist(),
        "centered_pc1_scores": {
            str(age): float(value) for age, value in zip(AGES, first_score)
        },
        "centered_pc1_linear_age_r2": float(regression.rvalue**2),
        "centered_pc1_age_spearman": float(stats.spearmanr(ages, first_score).statistic),
        "bias_pairwise_cosine_min": float(
            bias_cosine[np.triu_indices_from(bias_cosine, k=1)].min()
        ),
        "bias_pairwise_cosine_mean": float(
            bias_cosine[np.triu_indices_from(bias_cosine, k=1)].mean()
        ),
        "bias_max_relative_deviation_from_mean": float(
            np.max(np.linalg.norm(bias_matrix - mean_bias, axis=1) / bias_norms)
        ),
    }
    return result, deltas, cosine, centered_scores, mean_delta


def commutator_rows(weights: dict[int, np.ndarray], biases: dict[int, np.ndarray]) -> tuple[list[dict[str, Any]], np.ndarray]:
    maps = {age: augmented(weights[age], biases[age]) for age in AGES}
    identity = np.eye(next(iter(maps.values())).shape[0])
    matrix = np.zeros((len(AGES), len(AGES)))
    rows = []
    for i, first_age in enumerate(AGES):
        for j, second_age in enumerate(AGES):
            first, second = maps[first_age], maps[second_age]
            raw = sla.norm(first @ second - second @ first, "fro")
            denominator = (
                sla.norm(first - identity, "fro")
                * sla.norm(second - identity, "fro")
            )
            normalized = float(raw / denominator) if denominator else 0.0
            matrix[i, j] = normalized
            if i < j:
                rows.append(
                    {
                        "first_age": first_age,
                        "second_age": second_age,
                        "commutator_frobenius": float(raw),
                        "normalized_commutator": normalized,
                    }
                )
    return rows, matrix


def product_rows(weights: dict[int, np.ndarray], biases: dict[int, np.ndarray], mean_delta: np.ndarray, step_coefficients: np.ndarray) -> list[dict[str, Any]]:
    dimension = next(iter(weights.values())).shape[0]
    mean_affine = mean_delta.reshape(dimension + 1, dimension)
    mean_weight = np.eye(dimension) + mean_affine[:dimension]
    mean_bias = mean_affine[dimension]
    scalar_maps = {}
    for age, coefficient in zip(AGES, step_coefficients):
        delta = coefficient * mean_affine
        scalar_maps[age] = (np.eye(dimension) + delta[:dimension], delta[dimension])
    rows = []
    for source_age in AGES:
        for steps in range(1, min(5, source_age - 1) + 1):
            correct_ages = list(range(source_age, source_age - steps, -1))
            correct = compose([(weights[age], biases[age]) for age in correct_ages])
            reverse = compose(
                [(weights[age], biases[age]) for age in reversed(correct_ages)]
            )
            shared = compose([(mean_weight, mean_bias)] * steps)
            scalar = compose([scalar_maps[age] for age in correct_ages])
            correct_aug = augmented(*correct)
            denominator = sla.norm(correct_aug, "fro")
            rows.append(
                {
                    "source_age": source_age,
                    "target_age": source_age - steps,
                    "rollback_steps": steps,
                    "correct_age_order": ",".join(map(str, correct_ages)),
                    "shared_power_relative_error": float(
                        sla.norm(correct_aug - augmented(*shared), "fro") / denominator
                    ),
                    "scalar_generator_product_relative_error": float(
                        sla.norm(correct_aug - augmented(*scalar), "fro") / denominator
                    ),
                    "reverse_order_relative_error": float(
                        sla.norm(correct_aug - augmented(*reverse), "fro") / denominator
                    ),
                }
            )
    return rows


def checkpoint_rows(
    primary_weights: dict[int, np.ndarray],
    primary_biases: dict[int, np.ndarray],
    control_weights: dict[int, np.ndarray],
    control_biases: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    primary_deltas = []
    control_deltas = []
    for age in AGES:
        first = affine_delta(primary_weights[age], primary_biases[age])
        second = affine_delta(control_weights[age], control_biases[age])
        primary_deltas.append(first.reshape(-1))
        control_deltas.append(second.reshape(-1))
        rows.append(
            {
                "source_age": age,
                "relative_affine_change": float(
                    sla.norm(first - second, "fro") / sla.norm(second, "fro")
                ),
                "affine_delta_cosine": float(
                    np.vdot(first, second) / (sla.norm(first) * sla.norm(second))
                ),
                "relative_weight_change": float(
                    sla.norm(primary_weights[age] - control_weights[age], "fro")
                    / sla.norm(control_weights[age], "fro")
                ),
                "relative_bias_change": float(
                    sla.norm(primary_biases[age] - control_biases[age])
                    / sla.norm(control_biases[age])
                ),
            }
        )
    primary = np.stack(primary_deltas)
    control = np.stack(control_deltas)
    primary_mean, control_mean = primary.mean(0), control.mean(0)
    primary_residual = primary - primary_mean
    control_residual = control - control_mean
    summary = {
        "common_delta_cosine": float(
            primary_mean @ control_mean
            / (sla.norm(primary_mean) * sla.norm(control_mean))
        ),
        "common_delta_relative_change": float(
            sla.norm(primary_mean - control_mean) / sla.norm(control_mean)
        ),
        "stage_residual_relative_change": float(
            sla.norm(primary_residual - control_residual) / sla.norm(control_residual)
        ),
        "parameter_change_fraction_in_common_component": float(
            7 * sla.norm(primary_mean - control_mean) ** 2
            / sla.norm(primary - control) ** 2
        ),
    }
    return rows, summary


def plot_geometry(
    *,
    weights: dict[int, np.ndarray],
    biases: dict[int, np.ndarray],
    cosine: np.ndarray,
    centered_scores: np.ndarray,
    commutators: np.ndarray,
    product_rows_data: list[dict[str, Any]],
    shared_summary: dict[str, Any],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    image = axes[0, 0].imshow(cosine, vmin=0.95, vmax=1.0, cmap="viridis")
    axes[0, 0].set_title("cosine of affine deltas")
    axes[0, 0].set_xticks(range(7), AGES)
    axes[0, 0].set_yticks(range(7), AGES)
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)

    axes[0, 1].plot(AGES, centered_scores[:, 0], "o-", label="centered PC1")
    axes[0, 1].plot(AGES, centered_scores[:, 1], "o-", label="centered PC2")
    axes[0, 1].axhline(0, color="black", linewidth=0.6)
    axes[0, 1].set_title("stage residual coordinates")
    axes[0, 1].set_xlabel("source age")
    axes[0, 1].legend()

    for age in AGES:
        singular = sla.svdvals(weights[age])
        axes[0, 2].semilogy(np.arange(1, len(singular) + 1), singular, alpha=0.75, label=f"J{age}")
    axes[0, 2].axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    axes[0, 2].set_title("singular spectra of W")
    axes[0, 2].set_xlabel("rank")
    axes[0, 2].legend(ncol=2, fontsize=8)

    image = axes[1, 0].imshow(commutators, vmin=0.0, cmap="magma")
    axes[1, 0].set_title("normalized affine commutator")
    axes[1, 0].set_xticks(range(7), AGES)
    axes[1, 0].set_yticks(range(7), AGES)
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)

    for field, label in (
        ("shared_power_relative_error", "mean map power"),
        ("scalar_generator_product_relative_error", "scalar generator"),
        ("reverse_order_relative_error", "reverse order"),
    ):
        values = []
        for step in range(1, 6):
            selected = [row[field] for row in product_rows_data if row["rollback_steps"] == step]
            values.append(np.mean(selected) if selected else np.nan)
        axes[1, 1].plot(range(1, 6), values, "o-", label=label)
    axes[1, 1].set_title("product error vs correct ordered product")
    axes[1, 1].set_xlabel("rollback steps")
    axes[1, 1].set_ylabel("relative Frobenius error")
    axes[1, 1].legend()

    fractions = shared_summary["uncentered_pc_energy_fractions"]
    axes[1, 2].bar(np.arange(1, 8), fractions)
    axes[1, 2].set_title("uncentered matrix-space energy")
    axes[1, 2].set_xlabel("component")
    axes[1, 2].set_ylabel("fraction")
    axes[1, 2].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_common_and_residuals(
    weights: dict[int, np.ndarray], biases: dict[int, np.ndarray], path: Path
) -> None:
    deltas = {age: affine_delta(weights[age], biases[age]) for age in AGES}
    mean = np.mean(list(deltas.values()), axis=0)
    residuals = {age: deltas[age] - mean for age in AGES}
    scale = np.quantile(np.abs(mean), 0.995)
    residual_scale = max(np.quantile(np.abs(value), 0.995) for value in residuals.values())
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    image = axes[0, 0].imshow(mean, cmap="coolwarm", vmin=-scale, vmax=scale, aspect="auto")
    axes[0, 0].set_title("shared mean [W-I; b]")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)
    for axis, age in zip(axes.flat[1:], AGES):
        image = axis.imshow(
            residuals[age], cmap="coolwarm", vmin=-residual_scale,
            vmax=residual_scale, aspect="auto"
        )
        axis.set_title(f"J{age} stage residual")
    figure.colorbar(image, ax=list(axes.flat[1:]), fraction=0.018, pad=0.02)
    figure.suptitle("Shared rollback update and age-specific corrections")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary_weights, primary_biases, primary_payload = load_bank(args.primary_bank)
    control_weights, control_biases, control_payload = load_bank(args.control_bank)
    matrix_rows_data = matrix_metrics(primary_weights, primary_biases)
    shared, deltas, cosine, centered_scores, mean_delta = shared_analysis(
        primary_weights, primary_biases
    )
    step_coefficients = np.asarray(
        [shared["step_coefficients"][str(age)] for age in AGES]
    )
    commutators, commutator_matrix = commutator_rows(primary_weights, primary_biases)
    products = product_rows(
        primary_weights, primary_biases, mean_delta, step_coefficients
    )
    checkpoint_comparison, checkpoint_summary = checkpoint_rows(
        primary_weights, primary_biases, control_weights, control_biases
    )
    write_csv(args.out_dir / "per_map_metrics.csv", matrix_rows_data)
    write_csv(args.out_dir / "pairwise_commutators.csv", commutators)
    write_csv(args.out_dir / "product_approximations.csv", products)
    write_csv(args.out_dir / "checkpoint_comparison.csv", checkpoint_comparison)
    plot_geometry(
        weights=primary_weights,
        biases=primary_biases,
        cosine=cosine,
        centered_scores=centered_scores,
        commutators=commutator_matrix,
        product_rows_data=products,
        shared_summary=shared,
        path=args.out_dir / "matrix_geometry.png",
    )
    plot_common_and_residuals(
        primary_weights, primary_biases, args.out_dir / "shared_and_stage_residuals.png"
    )
    summary = {
        "status": "complete",
        "primary_bank": str(args.primary_bank),
        "control_bank": str(args.control_bank),
        "model_context": {
            "backbone_loss_placement": primary_payload.get("backbone_loss_placement"),
            "trained_loop_count": primary_payload.get("trained_loop_count"),
            "shared_block_count": primary_payload.get("shared_block_count"),
            "effective_training_depth": primary_payload.get("effective_training_depth"),
        },
        "shared_generator": shared,
        "commutator": {
            "mean": float(np.mean([row["normalized_commutator"] for row in commutators])),
            "max": float(np.max([row["normalized_commutator"] for row in commutators])),
        },
        "product_approximation_by_steps": {
            str(step): {
                field: float(
                    np.mean([row[field] for row in products if row["rollback_steps"] == step])
                )
                for field in (
                    "shared_power_relative_error",
                    "scalar_generator_product_relative_error",
                    "reverse_order_relative_error",
                )
            }
            for step in range(1, 6)
            if any(row["rollback_steps"] == step for row in products)
        },
        "checkpoint_stability": checkpoint_summary,
        "control_model_context": {
            "curriculum": control_payload.get("curriculum"),
            "completed_stages": control_payload.get("completed_stages"),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
