from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze D, AB, and D+AB for a diagonal low-rank J."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _complex_points(values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = values.detach().cpu()
    return values.real.numpy(), values.imag.numpy()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    item = payload["modules"][args.label]
    state = item["state_dict"]
    left = state["A"].double()
    right = state["B"].double()
    diagonal = state["diagonal_scale"].double()
    correction = left @ right
    weight = torch.diag(diagonal) + correction
    # The nonzero eigenvalues of AB equal those of BA.  Using the small
    # rank-by-rank matrix avoids turning numerical roundoff into fake roots.
    correction_nonzero_eigenvalues = torch.linalg.eigvals(right @ left)
    weight_eigenvalues = torch.linalg.eigvals(weight)
    correction_singular_values = torch.linalg.svdvals(correction)
    commutator = torch.diag(diagonal) @ correction - correction @ torch.diag(
        diagonal
    )
    determinant_correction = torch.eye(
        item["rank"], dtype=torch.float64
    ) + right @ torch.diag(1.0 / diagonal) @ left

    summary = {
        "source_artifact": str(args.artifact),
        "label": args.label,
        "dimension": int(item["dimension"]),
        "correction_rank": int(item["rank"]),
        "exact_zero_eigenvalue_multiplicity_of_AB": int(
            item["dimension"] - item["rank"]
        ),
        "D": {
            "mean": float(diagonal.mean()),
            "std": float(diagonal.std(unbiased=False)),
            "min": float(diagonal.min()),
            "max": float(diagonal.max()),
            "below_one": int(diagonal.lt(1).sum()),
            "above_one": int(diagonal.gt(1).sum()),
            "logabsdet": float(torch.log(diagonal.abs()).sum()),
        },
        "AB": {
            "nonzero_eigenvalue_abs_min": float(
                correction_nonzero_eigenvalues.abs().min()
            ),
            "nonzero_eigenvalue_abs_median": float(
                correction_nonzero_eigenvalues.abs().median()
            ),
            "spectral_radius": float(
                correction_nonzero_eigenvalues.abs().max()
            ),
            "complex_nonzero_eigenvalues": int(
                correction_nonzero_eigenvalues.imag.abs().gt(1e-8).sum()
            ),
            "nonzero_eigenvalue_abs_angle_median": float(
                correction_nonzero_eigenvalues.angle().abs().median()
            ),
            "trace": float(torch.trace(correction)),
            "singular_value_max": float(correction_singular_values[0]),
            "singular_value_min_nonzero": float(
                correction_singular_values[item["rank"] - 1]
            ),
            "frobenius_norm": float(correction.norm()),
            "nonnormality_ratio": float(
                (
                    correction.T @ correction
                    - correction @ correction.T
                ).norm()
                / correction.norm().square()
            ),
        },
        "J": {
            "eigenvalue_abs_min": float(weight_eigenvalues.abs().min()),
            "eigenvalue_abs_median": float(weight_eigenvalues.abs().median()),
            "spectral_radius": float(weight_eigenvalues.abs().max()),
            "eigenvalues_within_0p1_of_zero": int(
                weight_eigenvalues.abs().lt(0.1).sum()
            ),
            "eigenvalues_within_0p1_of_one": int(
                (weight_eigenvalues - 1).abs().lt(0.1).sum()
            ),
            "eigenvalue_abs_angle_median": float(
                weight_eigenvalues.angle().abs().median()
            ),
            "logabsdet": float(torch.linalg.slogdet(weight)[1]),
        },
        "D_AB_commutator_relative_norm": float(
            commutator.norm()
            / (torch.diag(diagonal).norm() * correction.norm())
        ),
        "matrix_determinant_lemma_logabs_correction": float(
            torch.linalg.slogdet(determinant_correction)[1]
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spectrum_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    axis = axes[0, 0]
    axis.scatter(diagonal.numpy(), np.zeros(len(diagonal)), s=15, alpha=0.65)
    axis.axvline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_title("$W_0=D$: 256 real eigenvalues")
    axis.set_xlabel("real")
    axis.set_ylabel("imaginary")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    real, imaginary = _complex_points(correction_nonzero_eigenvalues)
    axis.scatter(real, imaginary, s=28, alpha=0.8, label="48 nonzero eigs")
    axis.scatter(
        [0.0],
        [0.0],
        marker="x",
        s=80,
        color="black",
        label="zero (multiplicity 208)",
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set_title("$AB$: exact low-rank spectrum")
    axis.set_xlabel("real")
    axis.set_ylabel("imaginary")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    real, imaginary = _complex_points(weight_eigenvalues)
    axis.scatter(real, imaginary, s=15, alpha=0.65)
    unit_circle = plt.Circle(
        (0.0, 0.0), 1.0, fill=False, color="black", linestyle="--", linewidth=1
    )
    axis.add_patch(unit_circle)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.axvline(0.0, color="black", linewidth=0.7)
    axis.set_title("$J=D+AB$: eigenvalues")
    axis.set_xlabel("real")
    axis.set_ylabel("imaginary")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    axis.semilogy(
        np.arange(1, len(correction_singular_values) + 1),
        correction_singular_values.numpy(),
        marker=".",
        markersize=3,
        linewidth=1,
    )
    axis.axvline(item["rank"], color="black", linestyle="--", linewidth=1)
    axis.set_title("$AB$: singular values (nonnormality check)")
    axis.set_xlabel("index")
    axis.set_ylabel("singular value")
    axis.grid(alpha=0.25)

    figure.suptitle(
        "Diagonal + low-rank rejuvenator spectrum\n"
        f"{args.label}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(args.out_dir / "spectrum.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
