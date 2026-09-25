from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _stats(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().double().flatten()
    quantiles = torch.quantile(
        flat, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.double)
    )
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(flat.max()),
        "l2_norm": float(flat.norm()),
    }


def analyze(artifact: Path, out_dir: Path, label: str) -> dict[str, object]:
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].double()
    left = state["A"].double()
    right = state["B"].double()
    bias = state["bias"].double()
    correction = left @ right
    identity = torch.eye(diagonal.numel(), dtype=torch.double)
    weight = torch.diag(diagonal) + correction
    delta = weight - identity
    correction_singular = torch.linalg.svdvals(correction)
    weight_singular = torch.linalg.svdvals(weight)
    delta_singular = torch.linalg.svdvals(delta)
    weight_eigenvalues = torch.linalg.eigvals(weight)
    correction_eigenvalues = torch.linalg.eigvals(right @ left)
    nonnormal_residual = weight.T @ weight - weight @ weight.T
    threshold = float(correction_singular[0]) * 1e-6
    summary: dict[str, object] = {
        "status": "complete",
        "source_artifact": str(artifact),
        "label": label,
        "row_vector_convention": "h_out=h diag(D)+(h A)B+b=h J+b",
        "J_definition": "J=diag(D)+AB",
        "dimension": int(diagonal.numel()),
        "rank_budget": int(left.shape[1]),
        "D": {
            **_stats(diagonal),
            "delta_from_one": _stats(diagonal - 1.0),
            "below_one": int(diagonal.lt(1.0).sum()),
            "above_one": int(diagonal.gt(1.0).sum()),
        },
        "A": _stats(left),
        "B": _stats(right),
        "bias": _stats(bias),
        "AB": {
            "frobenius_norm": float(correction.norm()),
            "operator_norm": float(correction_singular[0]),
            "numerical_rank_at_1e-6_relative": int(
                correction_singular.gt(threshold).sum()
            ),
            "singular_values": correction_singular.tolist(),
            "nonzero_eigenvalue_spectral_radius": float(
                correction_eigenvalues.abs().max()
            ),
        },
        "J": {
            "frobenius_norm": float(weight.norm()),
            "operator_norm": float(weight_singular[0]),
            "minimum_singular_value": float(weight_singular[-1]),
            "spectral_radius": float(weight_eigenvalues.abs().max()),
            "eigenvalue_abs_median": float(weight_eigenvalues.abs().median()),
            "nonnormality_relative": float(
                nonnormal_residual.norm() / weight.norm().square().clamp_min(1e-12)
            ),
            "singular_values": weight_singular.tolist(),
        },
        "delta_J": {
            "frobenius_norm": float(delta.norm()),
            "operator_norm": float(delta_singular[0]),
            "stable_rank": float(
                delta_singular.square().sum()
                / delta_singular[0].square().clamp_min(1e-12)
            ),
            "diagonal_delta_frobenius_fraction": float(
                (diagonal - 1.0).norm() / delta.norm().clamp_min(1e-12)
            ),
            "AB_frobenius_fraction": float(
                correction.norm() / delta.norm().clamp_min(1e-12)
            ),
        },
        "interpretation_boundary": (
            "These are diagnostics of J in isolation. Recurrent stability and "
            "causal mechanism depend on the local Jacobian of F composed with J "
            "along real trajectories."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spectrum_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes[0, 0].hist(diagonal.numpy(), bins=40, alpha=0.85)
    axes[0, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Diagonal D")
    axes[0, 0].set_xlabel("value")

    axes[0, 1].hist(left.flatten().numpy(), bins=50, alpha=0.65, label="A")
    axes[0, 1].hist(right.flatten().numpy(), bins=50, alpha=0.65, label="B")
    axes[0, 1].set_title("Low-rank factors")
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        np.arange(1, len(correction_singular) + 1),
        correction_singular.numpy(),
        marker=".",
        markersize=3,
    )
    axes[1, 0].axvline(left.shape[1], color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("AB singular values")
    axes[1, 0].set_xlabel("index")

    eigenvalues = weight_eigenvalues.numpy()
    axes[1, 1].scatter(eigenvalues.real, eigenvalues.imag, s=13, alpha=0.7)
    axes[1, 1].add_patch(
        plt.Circle((0, 0), 1.0, fill=False, color="black", linestyle="--")
    )
    axes[1, 1].axhline(0, color="black", linewidth=0.6)
    axes[1, 1].axvline(0, color="black", linewidth=0.6)
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].set_title("J eigenvalues (diagnostic only)")
    axes[1, 1].set_xlabel("real")
    axes[1, 1].set_ylabel("imaginary")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(label)
    figure.tight_layout()
    figure.savefig(out_dir / "spectrum.png", dpi=180)
    figure.savefig(out_dir / "spectrum.pdf")
    plt.close(figure)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(analyze(args.artifact, args.out_dir, args.label), indent=2))


if __name__ == "__main__":
    main()
