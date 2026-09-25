from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().double().flatten()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.double),
    )
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "rms": float(flat.square().mean().sqrt()),
        "min": float(flat.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(flat.max()),
        "l2_norm": float(flat.norm()),
    }


def load_affine(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    if "weight" in state:
        weight = state["weight"].double()
        parameterization = "dense_affine"
    else:
        diagonal = state["diagonal"].double()
        weight = torch.diag(diagonal) + state["A"].double() @ state["B"].double()
        parameterization = f"diagonal_plus_rank_{state['A'].shape[1]}"
    return weight, state["bias"].double(), {
        "artifact": str(path),
        "parameterization": parameterization,
        "snapshot_update": payload.get("snapshot_update"),
        "anchor_step": payload.get("anchor_step"),
        "post_final_j": payload.get("controller_post_final_j"),
        "initialization": payload.get("controller_initialization"),
        "seed": payload.get("seed"),
    }


def effective_rank_at(singular: torch.Tensor, fraction: float) -> int:
    cumulative = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-30)
    return int(torch.searchsorted(cumulative, fraction).item()) + 1


def matrix_stats(weight: torch.Tensor, bias: torch.Tensor) -> dict[str, Any]:
    dimension = weight.shape[0]
    identity = torch.eye(dimension, dtype=torch.double)
    delta = weight - identity
    mask = ~torch.eye(dimension, dtype=torch.bool)
    singular_w = torch.linalg.svdvals(weight)
    singular_delta = torch.linalg.svdvals(delta)
    eigenvalues = torch.linalg.eigvals(weight)
    commutator = weight.T @ weight - weight @ weight.T
    energy_indices = (1, 4, 8, 16, 32, 48, 64, 128, 256)
    energy = singular_delta.square()
    total_energy = energy.sum().clamp_min(1e-30)
    row_norms = delta.norm(dim=1)
    column_norms = delta.norm(dim=0)
    power_operator_norms: dict[str, float] = {}
    live_power = torch.eye(dimension, dtype=torch.double)
    for power in range(1, 13):
        live_power = live_power @ weight
        if power in {1, 2, 4, 8, 12}:
            power_operator_norms[str(power)] = float(torch.linalg.matrix_norm(live_power, ord=2))
    return {
        "dimension": dimension,
        "diagonal": tensor_stats(torch.diag(weight)),
        "diagonal_delta": tensor_stats(torch.diag(weight) - 1.0),
        "off_diagonal": tensor_stats(weight[mask]),
        "bias": tensor_stats(bias),
        "delta_identity": {
            "frobenius_norm": float(delta.norm()),
            "operator_norm": float(singular_delta[0]),
            "stable_rank": float(energy.sum() / energy[0].clamp_min(1e-30)),
            "effective_rank_50pct": effective_rank_at(singular_delta, 0.50),
            "effective_rank_90pct": effective_rank_at(singular_delta, 0.90),
            "effective_rank_95pct": effective_rank_at(singular_delta, 0.95),
            "effective_rank_99pct": effective_rank_at(singular_delta, 0.99),
            "energy_fraction_top_k": {
                str(k): float(energy[:k].sum() / total_energy)
                for k in energy_indices
                if k <= dimension
            },
            "singular_values": singular_delta.tolist(),
            "diagonal_frobenius_fraction": float(
                torch.diag(torch.diag(delta)).norm() / delta.norm().clamp_min(1e-30)
            ),
            "off_diagonal_frobenius_fraction": float(
                delta.masked_fill(~mask, 0.0).norm() / delta.norm().clamp_min(1e-30)
            ),
            "row_norm_cv": float(row_norms.std(unbiased=False) / row_norms.mean().clamp_min(1e-30)),
            "column_norm_cv": float(
                column_norms.std(unbiased=False) / column_norms.mean().clamp_min(1e-30)
            ),
        },
        "J": {
            "frobenius_norm": float(weight.norm()),
            "operator_norm": float(singular_w[0]),
            "minimum_singular_value": float(singular_w[-1]),
            "condition_number": float(singular_w[0] / singular_w[-1].clamp_min(1e-30)),
            "spectral_radius": float(eigenvalues.abs().max()),
            "eigenvalue_abs_median": float(eigenvalues.abs().median()),
            "eigenvalue_abs_q05": float(torch.quantile(eigenvalues.abs(), 0.05)),
            "eigenvalue_abs_q95": float(torch.quantile(eigenvalues.abs(), 0.95)),
            "eigenvalues_outside_unit_circle": int(eigenvalues.abs().gt(1.0).sum()),
            "nonnormality_relative": float(
                commutator.norm() / weight.norm().square().clamp_min(1e-30)
            ),
            "power_operator_norms_diagnostic_only": power_operator_norms,
            "singular_values": singular_w.tolist(),
            "eigenvalues_real_imag": [
                [float(value.real), float(value.imag)] for value in eigenvalues
            ],
        },
    }


def compare_matrices(
    trained: torch.Tensor,
    trained_bias: torch.Tensor,
    initial: torch.Tensor,
    initial_bias: torch.Tensor,
) -> dict[str, Any]:
    learned = trained - initial
    learned_bias = trained_bias - initial_bias
    identity = torch.eye(trained.shape[0], dtype=torch.double)
    initial_delta = initial - identity
    trained_delta = trained - identity
    singular = torch.linalg.svdvals(learned)
    cosine = torch.nn.functional.cosine_similarity(
        initial_delta.flatten(), trained_delta.flatten(), dim=0
    )
    projection = (trained_delta.flatten() @ initial_delta.flatten()) / initial_delta.square().sum().clamp_min(1e-30)
    residual_after_init_direction = trained_delta - projection * initial_delta
    return {
        "learned_weight_change": {
            "frobenius_norm": float(learned.norm()),
            "operator_norm": float(singular[0]),
            "stable_rank": float(singular.square().sum() / singular[0].square().clamp_min(1e-30)),
            "effective_rank_90pct": effective_rank_at(singular, 0.90),
            "effective_rank_95pct": effective_rank_at(singular, 0.95),
            "effective_rank_99pct": effective_rank_at(singular, 0.99),
            "singular_values": singular.tolist(),
        },
        "learned_bias_change": tensor_stats(learned_bias),
        "trained_delta_to_initial_delta_frobenius_ratio": float(
            trained_delta.norm() / initial_delta.norm().clamp_min(1e-30)
        ),
        "learned_change_to_initial_delta_frobenius_ratio": float(
            learned.norm() / initial_delta.norm().clamp_min(1e-30)
        ),
        "initial_trained_delta_cosine": float(cosine),
        "initial_direction_projection_coefficient": float(projection),
        "trained_delta_residual_after_initial_direction_fraction": float(
            residual_after_init_direction.norm() / trained_delta.norm().clamp_min(1e-30)
        ),
    }


def snapshot_trajectory(checkpoint_dir: Path, initial_weight: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    identity = torch.eye(initial_weight.shape[0], dtype=torch.double)
    mask = ~torch.eye(initial_weight.shape[0], dtype=torch.bool)
    for path in sorted(checkpoint_dir.glob("controller_*.pt")):
        weight, bias, meta = load_affine(path)
        update = int(meta["snapshot_update"] or path.stem.split("_")[-1])
        delta = weight - identity
        singular = torch.linalg.svdvals(delta)
        rows.append(
            {
                "update": update,
                "delta_frobenius": float(delta.norm()),
                "delta_operator": float(singular[0]),
                "movement_from_init_frobenius": float((weight - initial_weight).norm()),
                "diagonal_delta_rms": float(torch.diag(delta).square().mean().sqrt()),
                "off_diagonal_rms": float(weight[mask].square().mean().sqrt()),
                "bias_rms": float(bias.square().mean().sqrt()),
            }
        )
    return rows


def plot_summary(
    trained: torch.Tensor,
    initial: torch.Tensor,
    trained_stats: dict[str, Any],
    initial_stats: dict[str, Any],
    trajectory: list[dict[str, float]],
    out_path: Path,
) -> None:
    dimension = trained.shape[0]
    mask = ~torch.eye(dimension, dtype=torch.bool)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    axes[0, 0].hist(torch.diag(initial).numpy(), bins=40, alpha=0.6, label="init")
    axes[0, 0].hist(torch.diag(trained).numpy(), bins=40, alpha=0.6, label="trained")
    axes[0, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Diagonal of J")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hist(initial[mask].numpy(), bins=60, alpha=0.55, label="init")
    axes[0, 1].hist(trained[mask].numpy(), bins=60, alpha=0.55, label="trained")
    axes[0, 1].set_title("Off-diagonal entries")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].semilogy(
        initial_stats["delta_identity"]["singular_values"],
        label="init J-I",
    )
    axes[0, 2].semilogy(
        trained_stats["delta_identity"]["singular_values"],
        label="trained J-I",
    )
    axes[0, 2].set_title("Singular values of J-I")
    axes[0, 2].legend(frameon=False)

    eig = np.asarray(trained_stats["J"]["eigenvalues_real_imag"])
    axes[1, 0].scatter(eig[:, 0], eig[:, 1], s=12, alpha=0.65)
    axes[1, 0].add_patch(plt.Circle((0, 0), 1.0, fill=False, linestyle="--", color="black"))
    axes[1, 0].axhline(0, color="black", linewidth=0.5)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_aspect("equal", adjustable="datalim")
    axes[1, 0].set_title("Eigenvalues of trained J")

    singular = np.asarray(trained_stats["delta_identity"]["singular_values"])
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)
    axes[1, 1].plot(np.arange(1, len(cumulative) + 1), cumulative)
    for y in (0.5, 0.9, 0.95, 0.99):
        axes[1, 1].axhline(y, color="grey", linestyle="--", linewidth=0.6)
    axes[1, 1].set_ylim(0, 1.01)
    axes[1, 1].set_title("Cumulative energy of J-I")
    axes[1, 1].set_xlabel("rank")

    updates = [row["update"] for row in trajectory]
    axes[1, 2].plot(updates, [row["diagonal_delta_rms"] for row in trajectory], label="diag delta RMS")
    axes[1, 2].plot(updates, [row["off_diagonal_rms"] for row in trajectory], label="offdiag RMS")
    axes[1, 2].plot(updates, [row["bias_rms"] for row in trajectory], label="bias RMS")
    axes[1, 2].set_title("Parameter trajectory")
    axes[1, 2].set_xlabel("optimizer update")
    axes[1, 2].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--low-rank", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trained, trained_bias, trained_meta = load_affine(args.trained)
    initial, initial_bias, initial_meta = load_affine(args.initial)
    trained_stats = matrix_stats(trained, trained_bias)
    initial_stats = matrix_stats(initial, initial_bias)
    trajectory = snapshot_trajectory(args.checkpoint_dir, initial)
    result: dict[str, Any] = {
        "status": "complete",
        "row_vector_convention": "h_out = hW + b",
        "trained": {"meta": trained_meta, **trained_stats},
        "initial": {"meta": initial_meta, **initial_stats},
        "training_change": compare_matrices(trained, trained_bias, initial, initial_bias),
        "snapshot_trajectory": trajectory,
        "interpretation_boundary": (
            "This is an isolated affine-map diagnostic. The recurrent mechanism "
            "is governed by the trajectory-dependent Jacobian of F composed with J."
        ),
    }
    if args.low_rank is not None:
        low_rank, low_rank_bias, low_rank_meta = load_affine(args.low_rank)
        result["low_rank_control"] = {
            "meta": low_rank_meta,
            **matrix_stats(low_rank, low_rank_bias),
        }
    (args.out_dir / "matrix_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    plot_summary(
        trained,
        initial,
        trained_stats,
        initial_stats,
        trajectory,
        args.out_dir / "dense_fullrank_j_matrix.png",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
