from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _distribution(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().double().flatten()
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=flat.dtype
        ),
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "minimum": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q05": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "q99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        "l1_norm": float(flat.abs().sum()),
        "l2_norm": float(torch.linalg.vector_norm(flat)),
        "maximum_absolute_value": float(flat.abs().max()),
    }


def _spectrum(value: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    singular = torch.linalg.svdvals(value.detach().double())
    normalized = singular / singular.sum().clamp_min(1e-30)
    entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum()
    maximum = singular[0]
    minimum = singular[-1]
    return singular, {
        "spectral_norm": float(maximum),
        "minimum_singular_value": float(minimum),
        "condition_number": (
            float(maximum / minimum) if minimum > 0 else math.inf
        ),
        "frobenius_norm": float(torch.linalg.matrix_norm(value.double(), "fro")),
        "stable_rank": float(singular.square().sum() / maximum.square()),
        "entropy_effective_rank": float(entropy.exp()),
        "rank_at_relative_threshold": {
            f"{threshold:g}": int((singular > maximum * threshold).sum())
            for threshold in (1e-2, 1e-3, 1e-4, 1e-6)
        },
        "top_singular_values": [float(item) for item in singular[:20]],
    }


def analyze_controller(artifact: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].detach().double()
    A = state["A"].detach().double()
    B = state["B"].detach().double()
    bias = state["bias"].detach().double()
    AB = A @ B
    J = torch.diag(diagonal) + AB
    ab_singular, ab_spectrum = _spectrum(AB)
    j_singular, j_spectrum = _spectrum(J)
    eigenvalues = torch.linalg.eigvals(J)
    eigenvalue_moduli = eigenvalues.abs()
    repeated_operator_norms: dict[str, float] = {}
    for exponent in (1, 2, 4, 8, 16, 32, 40, 50):
        powered = torch.linalg.matrix_power(J, exponent)
        repeated_operator_norms[str(exponent)] = float(
            torch.linalg.matrix_norm(powered, ord=2)
        )
    diagonal_delta = diagonal - 1.0
    summary: dict[str, Any] = {
        "artifact": str(artifact),
        "rank": int(payload["rank"]),
        "dimension": int(diagonal.numel()),
        "seed": int(payload["seed"]),
        "anchor_step": int(payload["anchor_step"]),
        "training": {
            "curriculum": payload.get("controller_curriculum"),
            "logical_max_length": payload.get("controller_logical_max_length"),
            "lr_schedule": payload.get("controller_lr_schedule", "legacy_cosine"),
            "warmup_updates": payload.get("controller_warmup_updates"),
            "stable_updates": payload.get("controller_stable_updates", 0),
            "final_lr_ratio": payload.get("controller_final_lr_ratio"),
            "learning_rate_multiplier": payload.get("learning_rate_multiplier"),
            "post_final_j": payload.get("controller_post_final_j", False),
            "training_budget": payload.get("training_budget"),
        },
        "diagonal_D": {
            "values": _distribution(diagonal),
            "delta_from_identity": _distribution(diagonal_delta),
            "fraction_below_one": float((diagonal < 1).double().mean()),
            "fraction_above_one": float((diagonal > 1).double().mean()),
            "fraction_abs_delta_above": {
                f"{threshold:g}": float(
                    (diagonal_delta.abs() > threshold).double().mean()
                )
                for threshold in (1e-4, 1e-3, 1e-2)
            },
        },
        "parameters": {
            "A": _distribution(A),
            "B": _distribution(B),
            "AB": _distribution(AB),
            "bias": _distribution(bias),
        },
        "AB_spectrum": ab_spectrum,
        "J_spectrum": {
            **j_spectrum,
            "spectral_radius": float(eigenvalue_moduli.max()),
            "minimum_eigenvalue_modulus": float(eigenvalue_moduli.min()),
            "fraction_eigenvalue_modulus_above_one": float(
                (eigenvalue_moduli > 1).double().mean()
            ),
            "maximum_absolute_imaginary_part": float(
                eigenvalues.imag.abs().max()
            ),
            "nonnormality_commutator_ratio": float(
                torch.linalg.matrix_norm(J.T @ J - J @ J.T, "fro")
                / torch.linalg.matrix_norm(J, "fro").square().clamp_min(1e-30)
            ),
            "repeated_operator_norms": repeated_operator_norms,
        },
    }
    arrays = {
        "diagonal": diagonal.numpy(),
        "diagonal_delta": diagonal_delta.numpy(),
        "A": A.numpy(),
        "B": B.numpy(),
        "AB": AB.numpy(),
        "J": J.numpy(),
        "AB_singular": ab_singular.numpy(),
        "J_singular": j_singular.numpy(),
        "eigenvalues": eigenvalues.numpy(),
    }
    return summary, arrays


def _write_tables(out_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    with (out_dir / "singular_spectra.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "AB_singular_value", "J_singular_value"))
        maximum = max(len(arrays["AB_singular"]), len(arrays["J_singular"]))
        for index in range(maximum):
            writer.writerow(
                (
                    index + 1,
                    arrays["AB_singular"][index]
                    if index < len(arrays["AB_singular"])
                    else "",
                    arrays["J_singular"][index]
                    if index < len(arrays["J_singular"])
                    else "",
                )
            )
    with (out_dir / "J_eigenvalues.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "real", "imaginary", "modulus"))
        for index, value in enumerate(arrays["eigenvalues"], start=1):
            writer.writerow((index, value.real, value.imag, abs(value)))


def _plot(out_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].hist(arrays["diagonal"], bins=32, color="#4472C4")
    axes[0, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("D diagonal values")
    axes[0, 0].set_xlabel("D[i]")

    axes[0, 1].hist(arrays["A"].ravel(), bins=60, alpha=0.7, label="A")
    axes[0, 1].hist(arrays["B"].ravel(), bins=60, alpha=0.7, label="B")
    axes[0, 1].set_title("LoRA factor parameters")
    axes[0, 1].legend()

    axes[0, 2].hist(arrays["AB"].ravel(), bins=60, color="#70AD47")
    axes[0, 2].set_title("AB matrix entries")
    axes[0, 2].set_xlabel("(AB)[i,j]")

    axes[1, 0].semilogy(
        np.arange(1, len(arrays["AB_singular"]) + 1),
        np.maximum(arrays["AB_singular"], 1e-16),
        label="AB",
    )
    axes[1, 0].semilogy(
        np.arange(1, len(arrays["J_singular"]) + 1),
        np.maximum(arrays["J_singular"], 1e-16),
        label="J",
    )
    axes[1, 0].set_title("Singular-value spectra")
    axes[1, 0].set_xlabel("index")
    axes[1, 0].legend()

    eigenvalues = arrays["eigenvalues"]
    theta = np.linspace(0, 2 * np.pi, 512)
    axes[1, 1].plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)
    axes[1, 1].scatter(eigenvalues.real, eigenvalues.imag, s=12, alpha=0.7)
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_title("J eigenvalues and unit circle")
    axes[1, 1].set_xlabel("real")
    axes[1, 1].set_ylabel("imaginary")

    diagonal = np.diag(arrays["J"])
    off_diagonal = arrays["J"] - np.diag(diagonal)
    axes[1, 2].hist(off_diagonal.ravel(), bins=60, alpha=0.75)
    axes[1, 2].set_title("J off-diagonal entries")
    axes[1, 2].set_xlabel("J[i,j], i != j")

    figure.tight_layout()
    figure.savefig(out_dir / "controller_matrix_analysis.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, arrays = analyze_controller(args.controller)
    with (args.out_dir / "matrix_analysis.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    _write_tables(args.out_dir, arrays)
    _plot(args.out_dir, arrays)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
