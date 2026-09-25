"""Detailed operator analysis of the seven stage-specific affine J maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


AGES = tuple(range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--parent-bank-artifact", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rank-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def numerical_rank(singular_values: np.ndarray, tolerance: float) -> int:
    if not len(singular_values) or singular_values[0] == 0:
        return 0
    return int(np.sum(singular_values > singular_values[0] * tolerance))


def energy_fraction(singular_values: np.ndarray, rank: int) -> float:
    denominator = float(np.square(singular_values).sum())
    return float(np.square(singular_values[:rank]).sum() / denominator) if denominator else 0.0


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Mean squared canonical correlation; random expectation is rank/dimension."""
    if left.shape[1] != right.shape[1]:
        raise ValueError("subspaces must have the same rank")
    return float(np.linalg.norm(left.T @ right, ord="fro") ** 2 / left.shape[1])


def normalized_commutator(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.linalg.norm(left @ right - right @ left) / denominator)


def residual_normalized_commutator(
    left: np.ndarray, right: np.ndarray, identity: np.ndarray
) -> float:
    """Normalize order sensitivity by the learned updates rather than full maps."""
    denominator = np.linalg.norm(left - identity) * np.linalg.norm(right - identity)
    return float(np.linalg.norm(left @ right - right @ left) / denominator)


def homogeneous(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    dimension = weight.shape[0]
    result = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    result[:dimension, :dimension] = weight
    result[dimension, :dimension] = bias
    result[dimension, dimension] = 1.0
    return result


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def load_bank(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("map_architecture") != "shared_diagonal_stage_lora":
        raise ValueError("analysis requires shared_diagonal_stage_lora")
    state = payload["state_dict"]
    diagonal = state["shared_diagonal_scale"].double().numpy()
    shared = state["shared_A"].double().numpy() @ state["shared_B"].double().numpy()
    bias = state["shared_bias"].double().numpy()
    updates: dict[int, np.ndarray] = {}
    weights: dict[int, np.ndarray] = {}
    for age in AGES:
        update = (
            state[f"stage_A.{age}"].double().numpy()
            @ state[f"stage_B.{age}"].double().numpy()
        )
        updates[age] = update
        weights[age] = np.diag(diagonal) + shared + update
    return payload, diagonal, bias, updates, weights


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload, diagonal, bias, updates, weights = load_bank(args.bank_artifact)
    identity = np.eye(len(diagonal), dtype=np.float64)
    dimension = len(diagonal)

    per_matrix: list[dict[str, Any]] = []
    eigenvalues: dict[int, np.ndarray] = {}
    singular_values: dict[int, np.ndarray] = {}
    fixed_points: dict[int, np.ndarray] = {}
    for age in AGES:
        weight = weights[age]
        singular = np.linalg.svd(weight, compute_uv=False)
        eigen = np.linalg.eigvals(weight)
        eigenvalues[age] = eigen
        singular_values[age] = singular
        residual = weight - identity
        symmetric = (residual + residual.T) / 2
        skew = (residual - residual.T) / 2
        update_singular = np.linalg.svd(updates[age], compute_uv=False)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_abs_det = np.linalg.slogdet(weight)
        normality = np.linalg.norm(weight.T @ weight - weight @ weight.T) / np.linalg.norm(weight) ** 2
        henrici = math.sqrt(
            max(0.0, np.linalg.norm(weight) ** 2 - float(np.square(np.abs(eigen)).sum()))
        ) / np.linalg.norm(weight)
        fixed_matrix = identity - weight
        fixed_singular = np.linalg.svd(fixed_matrix, compute_uv=False)
        try:
            fixed = np.linalg.solve(fixed_matrix.T, bias)
        except np.linalg.LinAlgError:
            fixed = np.linalg.lstsq(fixed_matrix.T, bias, rcond=None)[0]
        fixed_points[age] = fixed
        per_matrix.append(
            {
                "source_age": age,
                "user_J": age - 1,
                "singular_max": float(singular[0]),
                "singular_median": float(np.median(singular)),
                "singular_min": float(singular[-1]),
                "condition_number": float(singular[0] / singular[-1]),
                "spectral_radius": float(np.abs(eigen).max()),
                "eigenvalue_abs_median": float(np.median(np.abs(eigen))),
                "eigenvalues_abs_gt_1": int(np.sum(np.abs(eigen) > 1)),
                "determinant_sign": float(sign),
                "log_abs_determinant": float(log_abs_det),
                "relative_distance_to_identity": float(np.linalg.norm(residual) / np.linalg.norm(identity)),
                "symmetric_residual_fraction": float(np.linalg.norm(symmetric) ** 2 / np.linalg.norm(residual) ** 2),
                "skew_residual_fraction": float(np.linalg.norm(skew) ** 2 / np.linalg.norm(residual) ** 2),
                "normalized_nonnormal_commutator": float(normality),
                "henrici_departure": float(henrici),
                "stage_update_frobenius": float(np.linalg.norm(updates[age])),
                "stage_update_numerical_rank": numerical_rank(update_singular, args.rank_tolerance),
                "fixed_point_norm": float(np.linalg.norm(fixed)),
                "fixed_point_equation_residual": float(np.linalg.norm(fixed @ fixed_matrix - bias)),
                "I_minus_W_condition": float(fixed_singular[0] / fixed_singular[-1]),
            }
        )
    write_csv(args.out_dir / "per_matrix_spectral_properties.csv", per_matrix)

    pairwise: list[dict[str, Any]] = []
    matrix_commutator = np.zeros((len(AGES), len(AGES)))
    affine_commutator = np.zeros_like(matrix_commutator)
    input_overlap = np.eye(len(AGES))
    output_overlap = np.eye(len(AGES))
    for left_age, right_age in combinations(AGES, 2):
        left = weights[left_age]
        right = weights[right_age]
        difference_singular = np.linalg.svd(left - right, compute_uv=False)
        left_u, left_s, left_vh = np.linalg.svd(updates[left_age], full_matrices=False)
        right_u, right_s, right_vh = np.linalg.svd(updates[right_age], full_matrices=False)
        left_rank = numerical_rank(left_s, args.rank_tolerance)
        right_rank = numerical_rank(right_s, args.rank_tolerance)
        input_cosines = np.linalg.svd(
            left_u[:, :left_rank].T @ right_u[:, :right_rank], compute_uv=False
        )
        output_cosines = np.linalg.svd(
            left_vh[:left_rank] @ right_vh[:right_rank].T, compute_uv=False
        )
        matrix_comm = normalized_commutator(left, right)
        residual_matrix_comm = residual_normalized_commutator(
            left, right, identity
        )
        affine_comm = normalized_commutator(
            homogeneous(left, bias), homogeneous(right, bias)
        )
        i, j = AGES.index(left_age), AGES.index(right_age)
        matrix_commutator[i, j] = matrix_commutator[j, i] = matrix_comm
        affine_commutator[i, j] = affine_commutator[j, i] = affine_comm
        input_overlap[i, j] = input_overlap[j, i] = float(np.mean(np.square(input_cosines)))
        output_overlap[i, j] = output_overlap[j, i] = float(np.mean(np.square(output_cosines)))
        left_delta = left - identity
        right_delta = right - identity
        pairwise.append(
            {
                "left_source_age": left_age,
                "right_source_age": right_age,
                "frobenius_distance": float(np.linalg.norm(left - right)),
                "relative_distance_to_mean_norm": float(
                    2 * np.linalg.norm(left - right) / (np.linalg.norm(left) + np.linalg.norm(right))
                ),
                "delta_cosine": float(
                    np.vdot(left_delta, right_delta).real
                    / (np.linalg.norm(left_delta) * np.linalg.norm(right_delta))
                ),
                "matrix_commutator": matrix_comm,
                "residual_normalized_matrix_commutator": residual_matrix_comm,
                "affine_commutator": affine_comm,
                "difference_numerical_rank": numerical_rank(difference_singular, args.rank_tolerance),
                "difference_top16_energy": energy_fraction(difference_singular, 16),
                "difference_top32_energy": energy_fraction(difference_singular, 32),
                "stage_input_subspace_overlap": float(np.mean(np.square(input_cosines))),
                "stage_output_subspace_overlap": float(np.mean(np.square(output_cosines))),
            }
        )
    write_csv(args.out_dir / "pairwise_matrix_relations.csv", pairwise)

    # A single minimum singular vector can rotate inside a cluster of small
    # singular values. Compare bottom-k subspaces instead of over-interpreting
    # one vector. In row-vector convention, columns of U are input directions.
    common_weight = weights[AGES[0]] - updates[AGES[0]]
    common_u, _, common_vh = np.linalg.svd(common_weight)
    parent_weights: dict[int, np.ndarray] | None = None
    if args.parent_bank_artifact is not None:
        _, _, _, _, parent_weights = load_bank(args.parent_bank_artifact)
    compression_ranks = (1, 2, 4, 8, 16, 24, 32, 48, 64)
    compression_rows: list[dict[str, Any]] = []
    bottom_input_overlap_matrices: dict[int, np.ndarray] = {}
    consensus_spectra: dict[int, np.ndarray] = {}
    for rank in compression_ranks:
        bottom_inputs = {
            age: np.linalg.svd(weights[age])[0][:, -rank:] for age in AGES
        }
        bottom_outputs = {
            age: np.linalg.svd(weights[age])[2][-rank:, :].T for age in AGES
        }
        top_inputs = {
            age: np.linalg.svd(weights[age])[0][:, :rank] for age in AGES
        }
        overlap_matrix = np.eye(len(AGES), dtype=np.float64)
        pair_bottom_input: list[float] = []
        pair_bottom_output: list[float] = []
        pair_top_input: list[float] = []
        for left_age, right_age in combinations(AGES, 2):
            value = subspace_overlap(
                bottom_inputs[left_age], bottom_inputs[right_age]
            )
            i, j = AGES.index(left_age), AGES.index(right_age)
            overlap_matrix[i, j] = overlap_matrix[j, i] = value
            pair_bottom_input.append(value)
            pair_bottom_output.append(
                subspace_overlap(
                    bottom_outputs[left_age], bottom_outputs[right_age]
                )
            )
            pair_top_input.append(
                subspace_overlap(top_inputs[left_age], top_inputs[right_age])
            )
        bottom_input_overlap_matrices[rank] = overlap_matrix
        consensus_projector = sum(
            value @ value.T for value in bottom_inputs.values()
        ) / len(AGES)
        consensus = np.linalg.eigvalsh(consensus_projector)[::-1]
        consensus_spectra[rank] = consensus
        parent_overlaps: list[float] = []
        if parent_weights is not None:
            for age in AGES:
                parent_u = np.linalg.svd(parent_weights[age])[0]
                parent_overlaps.append(
                    subspace_overlap(parent_u[:, -rank:], bottom_inputs[age])
                )
        compression_rows.append(
            {
                "bottom_rank": rank,
                "random_subspace_baseline": rank / dimension,
                "pairwise_bottom_input_overlap_mean": float(np.mean(pair_bottom_input)),
                "pairwise_bottom_input_overlap_min": float(np.min(pair_bottom_input)),
                "pairwise_bottom_input_overlap_max": float(np.max(pair_bottom_input)),
                "pairwise_bottom_output_overlap_mean": float(np.mean(pair_bottom_output)),
                "pairwise_top_input_overlap_mean_control": float(np.mean(pair_top_input)),
                "common_skeleton_bottom_input_overlap_mean": float(
                    np.mean(
                        [
                            subspace_overlap(value, common_u[:, -rank:])
                            for value in bottom_inputs.values()
                        ]
                    )
                ),
                "common_skeleton_bottom_output_overlap_mean": float(
                    np.mean(
                        [
                            subspace_overlap(value, common_vh[-rank:, :].T)
                            for value in bottom_outputs.values()
                        ]
                    )
                ),
                "consensus_eigenvalues_above_0_5": int(np.sum(consensus > 0.5)),
                "consensus_eigenvalues_above_0_8": int(np.sum(consensus > 0.8)),
                "consensus_participation_dimension": float(
                    np.square(consensus.sum()) / np.square(consensus).sum()
                ),
                "same_age_parent_to_stage24_bottom_input_overlap_mean": (
                    float(np.mean(parent_overlaps)) if parent_overlaps else ""
                ),
                "same_age_parent_to_stage24_bottom_input_overlap_min": (
                    float(np.min(parent_overlaps)) if parent_overlaps else ""
                ),
            }
        )
    write_csv(args.out_dir / "compressed_direction_subspaces.csv", compression_rows)

    product_rows: list[dict[str, Any]] = []
    product = identity.copy()
    product_bias = np.zeros(dimension, dtype=np.float64)
    product_singulars: list[np.ndarray] = []
    for steps, age in enumerate(range(8, 1, -1), start=1):
        product_bias = product_bias @ weights[age] + bias
        product = product @ weights[age]
        singular = np.linalg.svd(product, compute_uv=False)
        product_singulars.append(singular)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sign, log_abs_det = np.linalg.slogdet(product)
        product_rows.append(
            {
                "rollback_steps": steps,
                "ordered_source_ages": ",".join(str(value) for value in range(8, age - 1, -1)),
                "singular_max": float(singular[0]),
                "singular_median": float(np.median(singular)),
                "singular_min": float(singular[-1]),
                "condition_number": float(singular[0] / singular[-1]),
                "determinant_sign": float(sign),
                "log_abs_determinant": float(log_abs_det),
                "relative_distance_to_identity": float(np.linalg.norm(product - identity) / np.linalg.norm(identity)),
                "composed_bias_norm": float(np.linalg.norm(product_bias)),
            }
        )
    write_csv(args.out_dir / "ordered_product_properties.csv", product_rows)

    product_u = np.linalg.svd(product)[0]
    product_compression_rows: list[dict[str, Any]] = []
    for rank in (4, 8, 16, 32):
        product_bottom = product_u[:, -rank:]
        overlaps = []
        for age in AGES:
            map_u = np.linalg.svd(weights[age])[0]
            overlap = subspace_overlap(product_bottom, map_u[:, -rank:])
            overlaps.append(overlap)
            product_compression_rows.append(
                {
                    "bottom_rank": rank,
                    "source_age": age,
                    "user_J": age - 1,
                    "ordered_product_to_single_map_input_overlap": overlap,
                }
            )
        product_compression_rows.append(
            {
                "bottom_rank": rank,
                "source_age": "common_skeleton",
                "user_J": "",
                "ordered_product_to_single_map_input_overlap": subspace_overlap(
                    product_bottom, common_u[:, -rank:]
                ),
            }
        )
    write_csv(
        args.out_dir / "ordered_product_compressed_direction_overlap.csv",
        product_compression_rows,
    )

    flattened = np.stack([(weights[age] - identity).reshape(-1) for age in AGES])
    uncentered_singular = np.linalg.svd(flattened, compute_uv=False)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    centered_u, centered_singular, centered_vh = np.linalg.svd(centered, full_matrices=False)
    centered_scores = centered @ centered_vh.T
    centered_energy = np.square(centered_singular) / np.square(centered_singular).sum()
    common = flattened.mean(axis=0)
    coefficients = flattened @ common / np.vdot(common, common).real
    age_values = np.asarray(AGES, dtype=np.float64)
    pc1 = centered_scores[:, 0]
    if np.corrcoef(age_values, pc1)[0, 1] < 0:
        centered_scores *= -1
        pc1 = centered_scores[:, 0]
    age_spearman = float(np.corrcoef(ranks(age_values), ranks(pc1))[0, 1])
    linear_prediction = np.polyval(np.polyfit(age_values, pc1, 1), age_values)
    linear_r2 = float(1 - np.square(pc1 - linear_prediction).sum() / np.square(pc1 - pc1.mean()).sum())
    pca_rows = [
        {
            "source_age": age,
            "common_generator_coefficient": float(coefficients[index]),
            "centered_PC1_score": float(centered_scores[index, 0]),
            "centered_PC2_score": float(centered_scores[index, 1]),
            "stage_update_frobenius": float(np.linalg.norm(updates[age])),
            "fixed_point_norm": float(np.linalg.norm(fixed_points[age])),
        }
        for index, age in enumerate(AGES)
    ]
    write_csv(args.out_dir / "age_order_coordinates.csv", pca_rows)

    figure, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=180)
    ranks_axis = np.arange(1, dimension + 1)
    for age in AGES:
        axes[0, 0].semilogy(ranks_axis, singular_values[age], label=f"J{age - 1}")
    axes[0, 0].axhline(1, color="black", linestyle=":", linewidth=0.8)
    axes[0, 0].set(title="Singular spectra", xlabel="singular-value rank", ylabel="singular value")
    axes[0, 0].legend(ncol=2, fontsize=8)

    unit_circle = plt.Circle((0, 0), 1, fill=False, color="black", linestyle=":", linewidth=0.8)
    axes[0, 1].add_patch(unit_circle)
    for age in AGES:
        axes[0, 1].scatter(
            eigenvalues[age].real,
            eigenvalues[age].imag,
            s=8,
            alpha=0.45,
            label=f"J{age - 1}",
        )
    axes[0, 1].set(title="Eigenvalues (unit circle dotted)", xlabel="real", ylabel="imaginary", aspect="equal")

    steps_axis = np.arange(1, len(product_rows) + 1)
    axes[0, 2].semilogy(steps_axis, [row["singular_max"] for row in product_rows], marker="o", label="max")
    axes[0, 2].semilogy(steps_axis, [row["singular_median"] for row in product_rows], marker="o", label="median")
    axes[0, 2].semilogy(steps_axis, [row["singular_min"] for row in product_rows], marker="o", label="min")
    axes[0, 2].set(title="Ordered product J7…J1", xlabel="rollback maps composed", ylabel="singular value")
    axes[0, 2].legend()

    image = axes[1, 0].imshow(affine_commutator, vmin=0, vmax=max(0.05, affine_commutator.max()), cmap="magma")
    axes[1, 0].set(title="Normalized affine commutator", xticks=range(7), yticks=range(7))
    axes[1, 0].set_xticklabels([f"J{age - 1}" for age in AGES])
    axes[1, 0].set_yticklabels([f"J{age - 1}" for age in AGES])
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)

    image = axes[1, 1].imshow(output_overlap, vmin=0, vmax=1, cmap="viridis")
    axes[1, 1].set(title="Stage-update output-subspace overlap", xticks=range(7), yticks=range(7))
    axes[1, 1].set_xticklabels([f"J{age - 1}" for age in AGES])
    axes[1, 1].set_yticklabels([f"J{age - 1}" for age in AGES])
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].plot(AGES, centered_scores[:, 0], marker="o", label=f"centered PC1 ({centered_energy[0]:.1%})")
    axes[1, 2].plot(AGES, centered_scores[:, 1], marker="o", label=f"centered PC2 ({centered_energy[1]:.1%})")
    axes[1, 2].axhline(0, color="black", linewidth=0.7)
    axes[1, 2].set(title="Stage-specific age coordinates", xlabel="rollback source age", ylabel="score")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(args.out_dir / "seven_matrix_operator_properties.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    rank_axis = [row["bottom_rank"] for row in compression_rows]
    axes[0, 0].plot(
        rank_axis,
        [row["pairwise_bottom_input_overlap_mean"] for row in compression_rows],
        marker="o",
        label="bottom input (crushed)",
    )
    axes[0, 0].plot(
        rank_axis,
        [row["pairwise_bottom_output_overlap_mean"] for row in compression_rows],
        marker="o",
        label="bottom output",
    )
    axes[0, 0].plot(
        rank_axis,
        [row["pairwise_top_input_overlap_mean_control"] for row in compression_rows],
        marker="o",
        label="top input control",
    )
    axes[0, 0].plot(
        rank_axis,
        [row["random_subspace_baseline"] for row in compression_rows],
        linestyle=":",
        color="black",
        label="random baseline",
    )
    axes[0, 0].set(
        title="Shared compressed subspaces across J1…J7",
        xlabel="bottom-k singular directions",
        ylabel="mean squared subspace overlap",
        ylim=(0, 1.03),
    )
    axes[0, 0].legend()

    image = axes[0, 1].imshow(
        bottom_input_overlap_matrices[8], vmin=0, vmax=1, cmap="viridis"
    )
    axes[0, 1].set(
        title="Pairwise overlap of bottom-8 input directions",
        xticks=range(7),
        yticks=range(7),
    )
    axes[0, 1].set_xticklabels([f"J{age - 1}" for age in AGES])
    axes[0, 1].set_yticklabels([f"J{age - 1}" for age in AGES])
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046)

    for rank in (4, 8, 16, 32):
        axes[1, 0].plot(
            np.arange(1, 49),
            consensus_spectra[rank][:48],
            marker="o",
            markersize=3,
            label=f"bottom-{rank}",
        )
    axes[1, 0].axhline(0.8, color="black", linestyle=":", linewidth=0.8)
    axes[1, 0].set(
        title="Eigenvalues of mean compressed-subspace projector",
        xlabel="consensus direction rank",
        ylabel="presence across seven maps",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        rank_axis,
        [row["common_skeleton_bottom_input_overlap_mean"] for row in compression_rows],
        marker="o",
        label="current J vs common skeleton",
    )
    if parent_weights is not None:
        axes[1, 1].plot(
            rank_axis,
            [
                row["same_age_parent_to_stage24_bottom_input_overlap_mean"]
                for row in compression_rows
            ],
            marker="o",
            label="parent vs stage24, same J",
        )
    axes[1, 1].plot(
        rank_axis,
        [row["random_subspace_baseline"] for row in compression_rows],
        linestyle=":",
        color="black",
        label="random baseline",
    )
    axes[1, 1].set(
        title="Origin and training stability of crushed directions",
        xlabel="bottom-k singular directions",
        ylabel="subspace overlap",
        ylim=(0, 1.03),
    )
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(args.out_dir / "compressed_direction_subspaces.png", bbox_inches="tight")
    plt.close(figure)

    pair_ranks = [int(row["difference_numerical_rank"]) for row in pairwise]
    summary = {
        "status": "complete",
        "bank_artifact": str(args.bank_artifact),
        "architecture": "W_age = diag(d) + A_shared B_shared + A_age B_age; J_age(h)=h W_age+b",
        "dimension": dimension,
        "shared_rank": int(payload["rank"]),
        "stage_rank": int(payload["stage_rank"]),
        "shared_bias_norm": float(np.linalg.norm(bias)),
        "bias_identical_across_maps": True,
        "diagonal": {
            "minimum": float(diagonal.min()),
            "mean": float(diagonal.mean()),
            "median": float(np.median(diagonal)),
            "maximum": float(diagonal.max()),
            "entries_above_one": int(np.sum(diagonal > 1)),
        },
        "single_map": {
            "singular_max_range": [float(min(row["singular_max"] for row in per_matrix)), float(max(row["singular_max"] for row in per_matrix))],
            "singular_median_range": [float(min(row["singular_median"] for row in per_matrix)), float(max(row["singular_median"] for row in per_matrix))],
            "singular_min_range": [float(min(row["singular_min"] for row in per_matrix)), float(max(row["singular_min"] for row in per_matrix))],
            "condition_number_range": [float(min(row["condition_number"] for row in per_matrix)), float(max(row["condition_number"] for row in per_matrix))],
            "spectral_radius_range": [float(min(row["spectral_radius"] for row in per_matrix)), float(max(row["spectral_radius"] for row in per_matrix))],
            "log_abs_determinant_range": [float(min(row["log_abs_determinant"] for row in per_matrix)), float(max(row["log_abs_determinant"] for row in per_matrix))],
            "orientation_reversing_maps": [int(row["user_J"]) for row in per_matrix if row["determinant_sign"] < 0],
            "relative_distance_to_identity_mean": float(np.mean([row["relative_distance_to_identity"] for row in per_matrix])),
            "skew_residual_fraction_mean": float(np.mean([row["skew_residual_fraction"] for row in per_matrix])),
            "normalized_nonnormal_commutator_mean": float(np.mean([row["normalized_nonnormal_commutator"] for row in per_matrix])),
        },
        "pairwise": {
            "difference_rank_values": sorted(set(pair_ranks)),
            "difference_top16_energy_mean": float(np.mean([row["difference_top16_energy"] for row in pairwise])),
            "difference_top32_energy_minimum": float(min(row["difference_top32_energy"] for row in pairwise)),
            "matrix_commutator_mean": float(np.mean([row["matrix_commutator"] for row in pairwise])),
            "residual_normalized_matrix_commutator_mean": float(
                np.mean([row["residual_normalized_matrix_commutator"] for row in pairwise])
            ),
            "affine_commutator_mean": float(np.mean([row["affine_commutator"] for row in pairwise])),
            "stage_input_subspace_overlap_mean": float(np.mean([row["stage_input_subspace_overlap"] for row in pairwise])),
            "stage_output_subspace_overlap_mean": float(np.mean([row["stage_output_subspace_overlap"] for row in pairwise])),
            "random_rank16_subspace_overlap_expectation": 16 / dimension,
        },
        "compressed_directions": {
            "bottom4_pairwise_input_overlap_mean": float(
                compression_rows[compression_ranks.index(4)][
                    "pairwise_bottom_input_overlap_mean"
                ]
            ),
            "bottom8_pairwise_input_overlap_mean": float(
                compression_rows[compression_ranks.index(8)][
                    "pairwise_bottom_input_overlap_mean"
                ]
            ),
            "bottom16_pairwise_input_overlap_mean": float(
                compression_rows[compression_ranks.index(16)][
                    "pairwise_bottom_input_overlap_mean"
                ]
            ),
            "bottom16_random_baseline": 16 / dimension,
            "bottom16_common_skeleton_overlap_mean": float(
                compression_rows[compression_ranks.index(16)][
                    "common_skeleton_bottom_input_overlap_mean"
                ]
            ),
            "bottom16_parent_to_stage24_overlap_mean": (
                float(
                    compression_rows[compression_ranks.index(16)][
                        "same_age_parent_to_stage24_bottom_input_overlap_mean"
                    ]
                )
                if parent_weights is not None
                else None
            ),
            "bottom4_consensus_directions_above_0_8": int(
                compression_rows[compression_ranks.index(4)][
                    "consensus_eigenvalues_above_0_8"
                ]
            ),
            "bottom8_consensus_directions_above_0_8": int(
                compression_rows[compression_ranks.index(8)][
                    "consensus_eigenvalues_above_0_8"
                ]
            ),
        },
        "age_geometry": {
            "uncentered_PC1_energy": float(np.square(uncentered_singular[0]) / np.square(uncentered_singular).sum()),
            "centered_PC_energy": [float(value) for value in centered_energy],
            "PC1_age_spearman": age_spearman,
            "PC1_linear_age_R2": linear_r2,
            "common_generator_coefficient_range": [float(coefficients.min()), float(coefficients.max())],
        },
        "ordered_seven_product": product_rows[-1],
        "fixed_point_norm_range": [float(min(np.linalg.norm(value) for value in fixed_points.values())), float(max(np.linalg.norm(value) for value in fixed_points.values()))],
        "interpretation_boundary": (
            "These are ambient-space operator diagnostics. Functional success is measured separately on hidden states visited by the graph task; "
            "near-singularity or contraction in ambient space does not imply failure on that task manifold."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
