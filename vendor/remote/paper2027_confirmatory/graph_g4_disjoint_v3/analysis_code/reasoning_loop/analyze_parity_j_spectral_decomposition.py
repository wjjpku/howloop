"""Decompose one or more shared affine controller J checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


POWER_STEPS = (1, 2, 4, 8, 16, 20, 32, 40, 50, 75, 99, 100, 150, 200)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        action="append",
        required=True,
        help="Controller as LABEL=PATH; repeat for multiple anchors.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--figure-title",
        default="Shared affine J: spectral decomposition",
    )
    parser.add_argument(
        "--plot-name",
        default="j_spectral_decomposition.png",
    )
    parser.add_argument(
        "--claim-subject",
        default="the corresponding hidden states",
        help="Subject used in the evidence-boundary note.",
    )
    return parser.parse_args(argv)


def parse_controllers(values: Sequence[str]) -> list[tuple[str, Path]]:
    output: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("--controller must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        output.append((label, Path(raw_path)))
    return output


def effective_rank(singular: np.ndarray) -> float:
    probability = singular / max(float(singular.sum()), 1e-30)
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-30))))
    return float(np.exp(entropy))


def stable_rank(singular: np.ndarray) -> float:
    return float(np.sum(singular**2) / max(float(singular[0] ** 2), 1e-30))


def normalized_nonnormality(matrix: np.ndarray) -> float:
    numerator = np.linalg.norm(matrix.T @ matrix - matrix @ matrix.T, "fro")
    denominator = max(float(np.linalg.norm(matrix, "fro") ** 2), 1e-30)
    return float(numerator / denominator)


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.T @ right, "fro") ** 2 / left.shape[1])


def load_controller(label: str, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].detach().double().numpy()
    A = state["A"].detach().double().numpy()
    B = state["B"].detach().double().numpy()
    bias = state["bias"].detach().double().numpy()
    AB = A @ B
    weight = np.diag(diagonal) + AB
    delta = weight - np.eye(weight.shape[0])
    u, singular, vt = np.linalg.svd(weight, full_matrices=True)
    delta_u, delta_singular, delta_vt = np.linalg.svd(delta, full_matrices=True)
    ab_u, ab_singular, ab_vt = np.linalg.svd(AB, full_matrices=True)
    eigenvalues, eigenvectors = np.linalg.eig(weight)
    ab_nonzero_eigenvalues = np.linalg.eigvals(B @ A)
    return {
        "label": label,
        "path": path,
        "payload": payload,
        "diagonal": diagonal,
        "A": A,
        "B": B,
        "bias": bias,
        "AB": AB,
        "weight": weight,
        "delta": delta,
        "u": u,
        "singular": singular,
        "vt": vt,
        "delta_u": delta_u,
        "delta_singular": delta_singular,
        "delta_vt": delta_vt,
        "ab_u": ab_u,
        "ab_singular": ab_singular,
        "ab_vt": ab_vt,
        "eigenvalues": eigenvalues,
        "ab_nonzero_eigenvalues": ab_nonzero_eigenvalues,
        "eigenvector_condition": float(np.linalg.cond(eigenvectors)),
    }


def power_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    weight = item["weight"]
    bias = item["bias"]
    spectral_radius = float(np.abs(item["eigenvalues"]).max())
    output: list[dict[str, Any]] = []
    powered = np.eye(weight.shape[0])
    composed_bias = np.zeros_like(bias)
    for step in range(1, max(POWER_STEPS) + 1):
        powered = powered @ weight
        composed_bias = composed_bias @ weight + bias
        if step not in POWER_STEPS:
            continue
        singular = np.linalg.svd(powered, compute_uv=False)
        output.append(
            {
                "label": item["label"],
                "power": step,
                "operator_norm": float(singular[0]),
                "minimum_singular_value": float(singular[-1]),
                "condition_number": float(singular[0] / singular[-1]),
                "spectral_radius_power": float(spectral_radius**step),
                "transient_gain_over_spectral_radius": float(
                    singular[0] / max(spectral_radius**step, 1e-30)
                ),
                "composed_bias_l2": float(np.linalg.norm(composed_bias)),
            }
        )
    return output


def summarize(item: dict[str, Any]) -> dict[str, Any]:
    diagonal = item["diagonal"]
    diagonal_delta = diagonal - 1.0
    singular = item["singular"]
    delta_singular = item["delta_singular"]
    ab_singular = item["ab_singular"]
    eigenvalues = item["eigenvalues"]
    power_lookup = {row["power"]: row for row in power_rows(item)}
    dimension = len(diagonal)
    return {
        "label": item["label"],
        "artifact": str(item["path"]),
        "anchor_step": int(item["payload"]["anchor_step"]),
        "dimension": dimension,
        "factor_rank": int(item["payload"]["rank"]),
        "affine_note": (
            "J(h)=hW+b. The spectrum belongs to W; in homogeneous row "
            "coordinates the affine matrix adds one eigenvalue at 1."
        ),
        "D": {
            "mean": float(diagonal.mean()),
            "std": float(diagonal.std()),
            "minimum": float(diagonal.min()),
            "maximum": float(diagonal.max()),
            "maximum_abs_delta_from_identity": float(np.abs(diagonal_delta).max()),
            "delta_l2": float(np.linalg.norm(diagonal_delta)),
        },
        "bias": {
            "l2": float(np.linalg.norm(item["bias"])),
            "maximum_abs": float(np.abs(item["bias"]).max()),
            "composed_l2_power_20": power_lookup[20]["composed_bias_l2"],
            "composed_l2_power_40": power_lookup[40]["composed_bias_l2"],
            "composed_l2_power_99": power_lookup[99]["composed_bias_l2"],
            "composed_l2_power_200": power_lookup[200]["composed_bias_l2"],
        },
        "AB": {
            "frobenius_norm": float(np.linalg.norm(item["AB"], "fro")),
            "spectral_norm": float(ab_singular[0]),
            "smallest_allowed_singular": float(ab_singular[47]),
            "stable_rank": stable_rank(ab_singular),
            "entropy_effective_rank": effective_rank(ab_singular),
            "normalized_nonnormality": normalized_nonnormality(item["AB"]),
            "nonzero_eigenvalue_spectral_radius": float(
                np.abs(item["ab_nonzero_eigenvalues"]).max()
            ),
            "singular_to_eigen_radius_ratio": float(
                ab_singular[0]
                / max(float(np.abs(item["ab_nonzero_eigenvalues"]).max()), 1e-30)
            ),
            "complex_nonzero_eigenvalue_count": int(
                np.sum(np.abs(item["ab_nonzero_eigenvalues"].imag) > 1e-8)
            ),
            "top_singular_values": [float(value) for value in ab_singular[:16]],
        },
        "delta_W_equals_J_minus_I": {
            "frobenius_norm": float(np.linalg.norm(item["delta"], "fro")),
            "spectral_norm": float(delta_singular[0]),
            "stable_rank": stable_rank(delta_singular),
            "entropy_effective_rank": effective_rank(delta_singular),
            "normalized_nonnormality": normalized_nonnormality(item["delta"]),
            "rank_above_1e_minus_3": int(np.sum(delta_singular > 1e-3)),
            "rank_above_1e_minus_4": int(np.sum(delta_singular > 1e-4)),
            "top_singular_values": [float(value) for value in delta_singular[:16]],
        },
        "W_singular_spectrum": {
            "maximum": float(singular[0]),
            "q95": float(np.quantile(singular, 0.95)),
            "median": float(np.median(singular)),
            "q05": float(np.quantile(singular, 0.05)),
            "minimum": float(singular[-1]),
            "condition_number": float(singular[0] / singular[-1]),
            "count_above_1_plus_1e_minus_3": int(np.sum(singular > 1.001)),
            "count_below_1_minus_1e_minus_3": int(np.sum(singular < 0.999)),
            "count_near_identity_within_1e_minus_3": int(
                np.sum(np.abs(singular - 1.0) <= 1e-3)
            ),
        },
        "W_eigen_spectrum": {
            "spectral_radius": float(np.abs(eigenvalues).max()),
            "minimum_modulus": float(np.abs(eigenvalues).min()),
            "median_modulus": float(np.median(np.abs(eigenvalues))),
            "count_modulus_above_1_plus_1e_minus_4": int(
                np.sum(np.abs(eigenvalues) > 1.0001)
            ),
            "count_modulus_below_1_minus_1e_minus_4": int(
                np.sum(np.abs(eigenvalues) < 0.9999)
            ),
            "maximum_abs_imaginary": float(np.abs(eigenvalues.imag).max()),
            "log_abs_determinant": float(
                torch.linalg.slogdet(torch.from_numpy(item["weight"]))[1]
            ),
            "normalized_nonnormality": normalized_nonnormality(item["weight"]),
            "eigenvector_condition_number": item["eigenvector_condition"],
        },
        "powers": {
            str(power): {
                key: value
                for key, value in power_lookup[power].items()
                if key not in {"label", "power"}
            }
            for power in (20, 40, 75, 99, 100, 200)
        },
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(items: Sequence[dict[str, Any]], summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item, summary in zip(items, summaries, strict=True):
        output.append(
            {
                "label": item["label"],
                "anchor_step": summary["anchor_step"],
                "D_max_abs_delta": summary["D"]["maximum_abs_delta_from_identity"],
                "bias_l2": summary["bias"]["l2"],
                "AB_frobenius": summary["AB"]["frobenius_norm"],
                "AB_spectral_norm": summary["AB"]["spectral_norm"],
                "AB_stable_rank": summary["AB"]["stable_rank"],
                "delta_W_spectral_norm": summary["delta_W_equals_J_minus_I"]["spectral_norm"],
                "W_singular_max": summary["W_singular_spectrum"]["maximum"],
                "W_singular_min": summary["W_singular_spectrum"]["minimum"],
                "W_condition": summary["W_singular_spectrum"]["condition_number"],
                "W_spectral_radius": summary["W_eigen_spectrum"]["spectral_radius"],
                "W_nonnormality": summary["W_eigen_spectrum"]["normalized_nonnormality"],
                "W_power_40_norm": summary["powers"]["40"]["operator_norm"],
                "W_power_99_norm": summary["powers"]["99"]["operator_norm"],
                "bias_power_99_l2": summary["bias"]["composed_l2_power_99"],
            }
        )
    return output


def pairwise_rows(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            delta_cosine = float(
                np.vdot(left["delta"].ravel(), right["delta"].ravel()).real
                / max(
                    np.linalg.norm(left["delta"]) * np.linalg.norm(right["delta"]),
                    1e-30,
                )
            )
            for rank in (1, 2, 4, 8, 16, 32, 48):
                output.append(
                    {
                        "left": left["label"],
                        "right": right["label"],
                        "rank": rank,
                        "delta_matrix_cosine": delta_cosine,
                        "AB_input_left_singular_overlap": subspace_overlap(
                            left["ab_u"][:, :rank], right["ab_u"][:, :rank]
                        ),
                        "AB_output_right_singular_overlap": subspace_overlap(
                            left["ab_vt"].T[:, :rank], right["ab_vt"].T[:, :rank]
                        ),
                        "J_top_input_singular_overlap": subspace_overlap(
                            left["u"][:, :rank], right["u"][:, :rank]
                        ),
                        "J_bottom_input_singular_overlap": subspace_overlap(
                            left["u"][:, -rank:], right["u"][:, -rank:]
                        ),
                    }
                )
    return output


def mode_rows(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        for family, singular, u, vt in (
            ("W", item["singular"], item["u"], item["vt"]),
            ("delta_W", item["delta_singular"], item["delta_u"], item["delta_vt"]),
            ("AB", item["ab_singular"], item["ab_u"], item["ab_vt"]),
        ):
            for index in list(range(16)) + list(range(len(singular) - 8, len(singular))):
                input_direction = u[:, index]
                output_direction = vt.T[:, index]
                output.append(
                    {
                        "label": item["label"],
                        "family": family,
                        "descending_index": index + 1,
                        "singular_value": float(singular[index]),
                        "input_output_cosine": float(input_direction @ output_direction),
                        "bias_output_direction_projection": float(
                            item["bias"] @ output_direction
                        ),
                    }
                )
    return output


def plot_summary(
    out_dir: Path,
    items: Sequence[dict[str, Any]],
    power: Sequence[dict[str, Any]],
    *,
    figure_title: str,
    plot_name: str,
) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(items)))
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for color, item in zip(colors, items, strict=True):
        label = item["label"]
        axes[0, 0].plot(item["singular"], color=color, label=label)
        axes[0, 1].semilogy(np.maximum(item["delta_singular"], 1e-12), color=color, label=label)
        axes[0, 2].scatter(
            item["eigenvalues"].real,
            item["eigenvalues"].imag,
            s=9,
            alpha=0.55,
            color=color,
            label=label,
        )
        axes[1, 0].hist(
            item["diagonal"] - 1.0,
            bins=28,
            histtype="step",
            linewidth=1.3,
            color=color,
            label=label,
        )
        selected = [row for row in power if row["label"] == label]
        axes[1, 1].plot(
            [row["power"] for row in selected],
            [row["operator_norm"] for row in selected],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
        axes[1, 2].plot(
            [row["power"] for row in selected],
            [row["composed_bias_l2"] for row in selected],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )

    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Singular values of W")
    axes[0, 0].set_xlabel("descending index")
    axes[0, 0].set_ylabel("singular value")
    axes[0, 1].set_title("Singular values of ΔW = W-I")
    axes[0, 1].set_xlabel("descending index")
    axes[0, 2].set_title("Eigenvalues of W (zoom around 1)")
    axes[0, 2].set_xlabel("real")
    axes[0, 2].set_ylabel("imaginary")
    axes[0, 2].axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 2].axhline(0.0, color="black", linewidth=0.6)
    axes[0, 2].set_aspect("equal", adjustable="datalim")
    axes[1, 0].set_title("D-I diagonal entries")
    axes[1, 0].set_xlabel("D[i]-1")
    axes[1, 1].set_title("Worst-case repeated linear gain ||W^n||₂")
    axes[1, 1].set_xlabel("n")
    axes[1, 1].set_ylabel("operator norm")
    axes[1, 1].set_yscale("log")
    axes[1, 2].set_title("Accumulated affine bias norm")
    axes[1, 2].set_xlabel("n")
    axes[1, 2].set_ylabel("||b(I+W+...+W^(n-1))||₂")
    axes[1, 2].set_yscale("log")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(figure_title)
    figure.tight_layout()
    figure.savefig(out_dir / plot_name, dpi=190)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = [load_controller(label, path) for label, path in parse_controllers(args.controller)]
    summaries = [summarize(item) for item in items]
    powers = [row for item in items for row in power_rows(item)]
    write_csv(args.out_dir / "anchor_comparison.csv", comparison_rows(items, summaries))
    write_csv(args.out_dir / "power_dynamics.csv", powers)
    write_csv(args.out_dir / "pairwise_subspace_overlap.csv", pairwise_rows(items))
    write_csv(args.out_dir / "singular_mode_summary.csv", mode_rows(items))
    for item in items:
        np.savez_compressed(
            args.out_dir / f"{item['label']}_spectral_arrays.npz",
            W=item["weight"],
            bias=item["bias"],
            eigenvalues=item["eigenvalues"],
            U=item["u"],
            singular_values=item["singular"],
            Vt=item["vt"],
            delta_U=item["delta_u"],
            delta_singular_values=item["delta_singular"],
            delta_Vt=item["delta_vt"],
            AB_U=item["ab_u"],
            AB_singular_values=item["ab_singular"],
            AB_Vt=item["ab_vt"],
        )
    (args.out_dir / "spectral_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "claim_boundary": (
                    "This is an ambient operator decomposition. Singular/eigen modes "
                    "become circuit evidence only after projection and intervention on "
                    f"{args.claim_subject}."
                ),
                "controllers": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_summary(
        args.out_dir,
        items,
        powers,
        figure_title=args.figure_title,
        plot_name=args.plot_name,
    )
    print(json.dumps({"status": "complete", "controllers": len(items)}, indent=2))


if __name__ == "__main__":
    main()
