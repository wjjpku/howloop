from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot zoomed diagnostics for a diagonal plus low-rank J."
    )
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-power", type=int, default=10)
    return parser.parse_args()


def quantiles(value: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor(
        [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
        dtype=value.dtype,
    )
    names = ("min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max")
    return {
        name: float(item)
        for name, item in zip(names, torch.quantile(value, levels), strict=True)
    }


def main() -> None:
    args = parse_args()
    payload = torch.load(args.controller, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].double()
    a_factor = state["A"].double()
    b_factor = state["B"].double()
    bias = state["bias"].double()

    low_rank = a_factor @ b_factor
    weight = torch.diag(diagonal) + low_rank
    identity = torch.eye(weight.shape[0], dtype=weight.dtype)
    delta = weight - identity
    singular_j = torch.linalg.svdvals(weight)
    singular_ab = torch.linalg.svdvals(low_rank)
    eigenvalues = torch.linalg.eigvals(weight)
    eigen_modulus = eigenvalues.abs()

    commutator = weight.T @ weight - weight @ weight.T
    weight_fro = torch.linalg.norm(weight)
    henrici = math.sqrt(
        max(0.0, float(weight_fro**2 - (eigen_modulus**2).sum()))
    ) / float(weight_fro)

    power_rows: list[dict[str, float]] = []
    power = identity
    for exponent in range(1, args.max_power + 1):
        power = power @ weight
        power_singular = torch.linalg.svdvals(power)
        power_rows.append(
            {
                "power": exponent,
                "operator_norm": float(power_singular.max()),
                "minimum_singular_value": float(power_singular.min()),
            }
        )

    ab_energy = singular_ab.square()
    cumulative_ab_energy = torch.cumsum(ab_energy, dim=0) / ab_energy.sum()
    delta_diagonal = diagonal - 1.0
    metrics = {
        "controller": str(args.controller.resolve()),
        "row_vector_convention": "J(h)=h@diag(D)+(h@A)@B+b",
        "snapshot_update": int(payload.get("snapshot_update", -1)),
        "dimension": int(weight.shape[0]),
        "rank": int(payload["rank"]),
        "diagonal": quantiles(diagonal),
        "diagonal_minus_one": quantiles(delta_diagonal),
        "diagonal_below_one": int((diagonal < 1.0).sum()),
        "diagonal_above_one": int((diagonal > 1.0).sum()),
        "diagonal_abs_delta_gt_1e_3": int((delta_diagonal.abs() > 1e-3).sum()),
        "diagonal_abs_delta_gt_2e_3": int((delta_diagonal.abs() > 2e-3).sum()),
        "diagonal_delta_frobenius_norm": float(torch.linalg.norm(delta_diagonal)),
        "AB_frobenius_norm": float(torch.linalg.norm(low_rank)),
        "AB_operator_norm": float(singular_ab.max()),
        "AB_top8_energy_fraction": float(cumulative_ab_energy[7]),
        "bias_l2_norm": float(torch.linalg.norm(bias)),
        "bias_rms": float(torch.sqrt(torch.mean(bias.square()))),
        "J_singular_values": quantiles(singular_j),
        "J_eigenvalue_moduli": quantiles(eigen_modulus),
        "J_condition_number": float(singular_j.max() / singular_j.min()),
        "J_spectral_radius": float(eigen_modulus.max()),
        "normalized_commutator": float(
            torch.linalg.norm(commutator) / weight_fro.square()
        ),
        "relative_henrici_departure": henrici,
        "powers": power_rows,
    }

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    figure.suptitle(
        "Addition LSB-causal-NoPE: trained J spectral diagnostics\n"
        "row-vector convention: J(h)=h diag(D)+(hA)B+b",
        fontsize=16,
    )

    axis = axes[0, 0]
    axis.hist(delta_diagonal.numpy(), bins=32, color="#4c78a8", alpha=0.9)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("Diagonal residual D - 1")
    axis.set_xlabel("Dᵢ - 1")
    axis.set_ylabel("count")

    axis = axes[0, 1]
    positive = singular_ab[singular_ab > 1e-12]
    indices = np.arange(1, positive.numel() + 1)
    axis.semilogy(indices, positive.numpy(), marker="o", markersize=3, label="σ(AB)")
    axis.set_title("Low-rank correction spectrum")
    axis.set_xlabel("singular-value index")
    axis.set_ylabel("singular value")
    energy_axis = axis.twinx()
    energy_axis.plot(
        indices,
        cumulative_ab_energy[: positive.numel()].numpy(),
        color="#f58518",
        label="cumulative energy",
    )
    energy_axis.set_ylabel("cumulative squared energy")
    energy_axis.set_ylim(0.0, 1.03)

    axis = axes[0, 2]
    ordered_j = torch.sort(singular_j).values
    ordered_eigen = torch.sort(eigen_modulus).values
    axis.plot(ordered_j.numpy(), label="σ(J)", color="#54a24b")
    axis.plot(ordered_eigen.numpy(), label="|λ(J)|", color="#e45756")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("J spectrum around identity")
    axis.set_xlabel("sorted index")
    axis.set_ylabel("magnitude")
    axis.legend()

    axis = axes[1, 0]
    axis.scatter(eigenvalues.real.numpy(), eigenvalues.imag.numpy(), s=14, alpha=0.8)
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.0, color="black", linestyle=":", linewidth=1)
    span = max(
        float((eigenvalues.real - 1.0).abs().max()),
        float(eigenvalues.imag.abs().max()),
    )
    span = max(0.05, span * 1.15)
    axis.set_xlim(1.0 - span, 1.0 + span)
    axis.set_ylim(-span, span)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Eigenvalues of J (zoomed)")
    axis.set_xlabel("real")
    axis.set_ylabel("imaginary")

    axis = axes[1, 1]
    powers = [row["power"] for row in power_rows]
    axis.plot(
        powers,
        [row["operator_norm"] for row in power_rows],
        marker="o",
        label="σmax(Jᵏ)",
    )
    axis.plot(
        powers,
        [row["minimum_singular_value"] for row in power_rows],
        marker="o",
        label="σmin(Jᵏ)",
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_title("Repeated J alone (diagnostic only)")
    axis.set_xlabel("k")
    axis.set_ylabel("singular value")
    axis.legend()

    axis = axes[1, 2]
    axis.axis("off")
    text = (
        f"D < 1 / D > 1: {metrics['diagonal_below_one']} / "
        f"{metrics['diagonal_above_one']}\n"
        f"median(D-1): {metrics['diagonal_minus_one']['median']:+.3e}\n"
        f"range(D-1): [{metrics['diagonal_minus_one']['min']:+.3e}, "
        f"{metrics['diagonal_minus_one']['max']:+.3e}]\n"
        f"||D-1||F / ||AB||F: "
        f"{metrics['diagonal_delta_frobenius_norm'] / metrics['AB_frobenius_norm']:.3%}\n"
        f"||AB||2 / ||AB||F: {metrics['AB_operator_norm']:.3f} / "
        f"{metrics['AB_frobenius_norm']:.3f}\n"
        f"AB top-8 energy: {metrics['AB_top8_energy_fraction']:.1%}\n"
        f"σ(J): [{metrics['J_singular_values']['min']:.3f}, "
        f"{metrics['J_singular_values']['max']:.3f}]\n"
        f"cond(J): {metrics['J_condition_number']:.3f}\n"
        f"ρ(J): {metrics['J_spectral_radius']:.3f}\n"
        f"relative Henrici departure: {henrici:.3%}\n"
        f"normalized commutator: {metrics['normalized_commutator']:.3e}"
    )
    axis.text(0.02, 0.98, text, va="top", family="monospace", fontsize=12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out_dir / "current_j_spectral_zoom.png", dpi=180)
    figure.savefig(args.out_dir / "current_j_spectral_zoom.pdf")
    (args.out_dir / "current_j_spectral_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
