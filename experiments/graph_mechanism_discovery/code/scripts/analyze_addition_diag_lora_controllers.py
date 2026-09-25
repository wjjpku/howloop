#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_controller(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("controllers must use LABEL=PATH")
    return label, Path(path)


def quantiles(value: np.ndarray) -> dict[str, float]:
    return {
        "min": float(value.min()),
        "q01": float(np.quantile(value, 0.01)),
        "q05": float(np.quantile(value, 0.05)),
        "median": float(np.median(value)),
        "q95": float(np.quantile(value, 0.95)),
        "q99": float(np.quantile(value, 0.99)),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "rms": float(np.sqrt(np.mean(value**2))),
    }


def effective_rank(singular: np.ndarray) -> float:
    total = float(singular.sum())
    if total <= 0:
        return 0.0
    probability = singular / total
    entropy = -float(np.sum(probability * np.log(probability + 1e-30)))
    return float(np.exp(entropy))


def stable_rank(singular: np.ndarray) -> float:
    if not len(singular) or singular[0] <= 0:
        return 0.0
    return float(np.sum(singular**2) / singular[0] ** 2)


def analyze(label: str, path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].double().numpy()
    left = state["A"].double().numpy()
    right = state["B"].double().numpy()
    bias = state["bias"].double().numpy()
    dimension = diagonal.shape[0]
    low_rank = left @ right
    weight = np.diag(diagonal) + low_rank
    delta = weight - np.eye(dimension)
    low_rank_singular = np.linalg.svd(low_rank, compute_uv=False)
    delta_singular = np.linalg.svd(delta, compute_uv=False)
    weight_singular = np.linalg.svd(weight, compute_uv=False)
    eigenvalues = np.linalg.eigvals(weight)
    low_rank_energy = low_rank_singular**2
    delta_energy = delta_singular**2
    row = {
        "label": label,
        "path": str(path),
        "snapshot_update": (
            int(payload["snapshot_update"])
            if payload.get("snapshot_update") is not None
            else None
        ),
        "snapshot_total_updates": (
            int(payload["snapshot_total_updates"])
            if payload.get("snapshot_total_updates") is not None
            else None
        ),
        # Early controller artifacts predate the explicit metadata field but
        # use the same diagonal/A/B/bias state schema.
        "parameterization": payload.get(
            "controller_parameterization", "diagonal_low_rank"
        ),
        "rank": int(payload["rank"]),
        "dimension": dimension,
        "trainable_parameter_count": int(
            diagonal.size + left.size + right.size + bias.size
        ),
        "row_vector_convention": "J(h)=h@diag(D)+(h@A)@B+b; W=diag(D)+A@B",
        "diagonal": quantiles(diagonal),
        "diagonal_minus_one": quantiles(diagonal - 1.0),
        "A": quantiles(left),
        "B": quantiles(right),
        "bias": quantiles(bias),
        "AB_frobenius_norm": float(np.linalg.norm(low_rank)),
        "diagonal_delta_frobenius_norm": float(np.linalg.norm(diagonal - 1.0)),
        "delta_weight_frobenius_norm": float(np.linalg.norm(delta)),
        "AB_operator_norm": float(low_rank_singular[0]),
        "delta_weight_operator_norm": float(delta_singular[0]),
        "AB_effective_rank": effective_rank(low_rank_singular),
        "AB_stable_rank": stable_rank(low_rank_singular),
        "delta_effective_rank": effective_rank(delta_singular),
        "delta_stable_rank": stable_rank(delta_singular),
        "AB_top8_energy_fraction": float(
            low_rank_energy[:8].sum() / max(low_rank_energy.sum(), 1e-30)
        ),
        "delta_top8_energy_fraction": float(
            delta_energy[:8].sum() / max(delta_energy.sum(), 1e-30)
        ),
        "J_singular_min": float(weight_singular[-1]),
        "J_singular_max": float(weight_singular[0]),
        "J_condition_number": float(
            weight_singular[0] / max(weight_singular[-1], 1e-30)
        ),
        "J_spectral_radius": float(np.abs(eigenvalues).max()),
        "J_min_eigenvalue_modulus": float(np.abs(eigenvalues).min()),
        "J_eigenvalues_modulus_gt_1p01": int(np.sum(np.abs(eigenvalues) > 1.01)),
        "J_eigenvalues_modulus_lt_0p99": int(np.sum(np.abs(eigenvalues) < 0.99)),
        "singular_values": {
            "AB": low_rank_singular.tolist(),
            "delta_weight": delta_singular.tolist(),
            "J": weight_singular.tolist(),
        },
        "eigenvalues": {
            "real": eigenvalues.real.tolist(),
            "imag": eigenvalues.imag.tolist(),
        },
    }
    arrays = {
        "diagonal": diagonal,
        "A": left.ravel(),
        "B": right.ravel(),
        "bias": bias,
        "AB_singular": low_rank_singular,
        "delta_singular": delta_singular,
        "J_singular": weight_singular,
        "eigenvalues": eigenvalues,
    }
    return row, arrays


def plot(
    rows: list[dict[str, Any]],
    arrays: list[dict[str, np.ndarray]],
    path: Path,
    *,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(18, 4.2 * len(rows)), constrained_layout=True, squeeze=False
    )
    for row_index, (row, values) in enumerate(zip(rows, arrays, strict=True)):
        axis = axes[row_index, 0]
        axis.hist(values["diagonal"] - 1.0, bins=40, color="tab:blue", alpha=0.85)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"{row['label']}: D - 1", xlabel="diagonal residual")

        axis = axes[row_index, 1]
        axis.hist(values["A"], bins=50, alpha=0.65, label="A")
        axis.hist(values["B"], bins=50, alpha=0.65, label="B")
        axis.set(title="low-rank factor entries", xlabel="parameter value")
        axis.legend()

        axis = axes[row_index, 2]
        axis.semilogy(values["AB_singular"] + 1e-15, label="AB")
        axis.semilogy(values["delta_singular"] + 1e-15, label="J-I")
        axis.semilogy(values["J_singular"], label="J")
        axis.set(title="singular spectra", xlabel="index", ylabel="singular value")
        axis.legend()

        axis = axes[row_index, 3]
        eigenvalues = values["eigenvalues"]
        axis.scatter(eigenvalues.real, eigenvalues.imag, s=12, alpha=0.7)
        theta = np.linspace(0, 2 * np.pi, 400)
        axis.plot(np.cos(theta), np.sin(theta), color="black", linestyle="--", linewidth=1)
        axis.set(title="J eigenvalues", xlabel="real", ylabel="imaginary")
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle(title)
    figure.savefig(path, dpi=190)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", action="append", type=parse_controller, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Addition Diag+LoRA controllers (row-vector convention)",
    )
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    arrays: list[dict[str, np.ndarray]] = []
    for label, path in args.controller:
        row, values = analyze(label, path)
        rows.append(row)
        arrays.append(values)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "controller_analysis.json").write_text(
        json.dumps({"status": "complete", "controllers": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    plot(
        rows,
        arrays,
        args.out_dir / "controller_parameter_spectra.png",
        title=args.title,
    )
    print(json.dumps({"status": "complete", "controllers": rows}, indent=2))


if __name__ == "__main__":
    main()
