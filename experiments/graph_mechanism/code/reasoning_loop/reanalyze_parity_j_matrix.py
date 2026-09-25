"""Reader-first matrix reanalysis for the trained parity affine controllers.

The controller uses row states: J(h) = h W + b, with W = diag(D) + A B.
This script deliberately separates:

1. the SVD of Delta = W-I (one-step read/write geometry),
2. the eigendecomposition of W (J-only repeated dynamics), and
3. affine bias geometry (which is not contained in the spectrum of W).

It does not run the frozen Transformer and therefore makes no causal circuit claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from scipy.linalg import eig


RANKS = (1, 2, 4, 8, 16, 32, 48)
POWERS = (1, 2, 4, 8, 16, 20, 32, 40, 64, 75, 99, 100, 150, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        action="append",
        required=True,
        help="Controller as LABEL=CHECKPOINT; repeat for each anchor.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_controller(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--controller must use LABEL=CHECKPOINT")
    label, path = value.split("=", 1)
    return label, Path(path)


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Mean squared projection between equal-dimensional orthonormal bases."""
    return float(np.linalg.norm(left.T.conj() @ right, "fro") ** 2 / left.shape[1])


def capture(vector: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of vector squared norm captured by an orthonormal basis."""
    return float(
        np.linalg.norm(basis.T.conj() @ vector) ** 2
        / max(np.linalg.norm(vector) ** 2, 1e-30)
    )


def load_item(label: str, path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].detach().double().numpy()
    a = state["A"].detach().double().numpy()
    b_factor = state["B"].detach().double().numpy()
    bias = state["bias"].detach().double().numpy()
    ab = a @ b_factor
    delta = np.diag(diagonal - 1.0) + ab
    weight = np.eye(len(diagonal)) + delta

    u, singular, vt = np.linalg.svd(delta)
    weight_singular = np.linalg.svd(weight, compute_uv=False)
    v = vt.T
    eigenvalues, left_eigenvectors, right_eigenvectors = eig(
        weight, left=True, right=True
    )
    eigen_condition = np.array(
        [
            1.0
            / max(
                abs(np.vdot(left_eigenvectors[:, index], right_eigenvectors[:, index])),
                1e-300,
            )
            for index in range(len(eigenvalues))
        ]
    )
    symmetric = (delta + delta.T) / 2.0
    skew = (delta - delta.T) / 2.0

    return {
        "label": label,
        "path": path,
        "payload": payload,
        "diagonal": diagonal,
        "a": a,
        "b_factor": b_factor,
        "bias": bias,
        "ab": ab,
        "delta": delta,
        "weight": weight,
        "weight_singular": weight_singular,
        "u": u,
        "singular": singular,
        "v": v,
        "eigenvalues": eigenvalues,
        "left_eigenvectors": left_eigenvectors,
        "right_eigenvectors": right_eigenvectors,
        "eigen_condition": eigen_condition,
        "eigenbasis_condition": float(np.linalg.cond(right_eigenvectors)),
        "symmetric": symmetric,
        "skew": skew,
    }


def power_analysis(item: dict[str, object]) -> list[dict[str, object]]:
    weight = item["weight"]
    bias = item["bias"]
    assert isinstance(weight, np.ndarray) and isinstance(bias, np.ndarray)
    eigenvalues = item["eigenvalues"]
    assert isinstance(eigenvalues, np.ndarray)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    powered = np.eye(weight.shape[0])
    accumulated_bias = np.zeros_like(bias)
    output: list[dict[str, object]] = []
    for step in range(1, max(POWERS) + 1):
        powered = powered @ weight
        accumulated_bias = accumulated_bias @ weight + bias
        if step not in POWERS:
            continue
        power_u, power_singular, power_vt = np.linalg.svd(powered)
        output.append(
            {
                "label": item["label"],
                "power": step,
                "operator_norm": float(power_singular[0]),
                "minimum_singular": float(power_singular[-1]),
                "condition": float(power_singular[0] / power_singular[-1]),
                "spectral_radius_power": spectral_radius**step,
                "transient_ratio": float(
                    power_singular[0] / max(spectral_radius**step, 1e-30)
                ),
                "accumulated_bias_norm": float(np.linalg.norm(accumulated_bias)),
                "worst_input_in_delta_top4": capture(
                    power_u[:, 0], item["u"][:, :4]
                ),
                "worst_output_in_delta_top4": capture(
                    power_vt.T[:, 0], item["v"][:, :4]
                ),
            }
        )
    return output


def rank_rows(item: dict[str, object]) -> list[dict[str, object]]:
    singular = item["singular"]
    u = item["u"]
    v = item["v"]
    bias = item["bias"]
    assert all(isinstance(value, np.ndarray) for value in (singular, u, v, bias))
    total_energy = float(np.sum(singular**2))
    return [
        {
            "label": item["label"],
            "rank": rank,
            "delta_energy": float(np.sum(singular[:rank] ** 2) / total_energy),
            "input_output_subspace_overlap": overlap(u[:, :rank], v[:, :rank]),
            "random_subspace_overlap": rank / len(singular),
            "bias_energy_in_input_space": capture(bias, u[:, :rank]),
            "bias_energy_in_output_space": capture(bias, v[:, :rank]),
        }
        for rank in RANKS
    ]


def summary(item: dict[str, object], powers: list[dict[str, object]]) -> dict[str, object]:
    diagonal = item["diagonal"]
    ab = item["ab"]
    delta = item["delta"]
    weight = item["weight"]
    singular = item["singular"]
    u = item["u"]
    v = item["v"]
    bias = item["bias"]
    eigenvalues = item["eigenvalues"]
    eigen_condition = item["eigen_condition"]
    symmetric = item["symmetric"]
    skew = item["skew"]
    assert all(
        isinstance(value, np.ndarray)
        for value in (
            diagonal,
            ab,
            delta,
            weight,
            singular,
            u,
            v,
            bias,
            eigenvalues,
            eigen_condition,
            symmetric,
            skew,
        )
    )
    symmetric_eigenvalues = np.linalg.eigvalsh(symmetric)
    delta_eigenvalues = eigenvalues - 1.0
    angle_index = int(np.argmax(np.abs(np.angle(eigenvalues))))
    largest_angle = float(abs(np.angle(eigenvalues[angle_index])))
    power_lookup = {int(row["power"]): row for row in powers}
    total_energy = float(np.sum(singular**2))
    mode_rows = []
    for index in range(8):
        mode_rows.append(
            {
                "index": index + 1,
                "singular_value": float(singular[index]),
                "delta_energy": float(singular[index] ** 2 / total_energy),
                "input_output_cosine": float(u[:, index] @ v[:, index]),
                "bias_output_projection": float(bias @ v[:, index]),
                "bias_output_energy": float((bias @ v[:, index]) ** 2 / np.sum(bias**2)),
                "bias_input_energy": float((bias @ u[:, index]) ** 2 / np.sum(bias**2)),
            }
        )

    return {
        "label": item["label"],
        "checkpoint": str(item["path"]),
        "anchor_step": int(item["payload"]["anchor_step"]),
        "dimension": len(diagonal),
        "factor_rank": int(item["payload"]["rank"]),
        "training_loss": item["payload"]["loss"],
        "state_loss_weight": float(item["payload"]["state_loss_weight"]),
        "logical_lengths": item["payload"]["controller_sampled_logical_lengths"],
        "decomposition": {
            "delta_frobenius": float(np.linalg.norm(delta, "fro")),
            "ab_frobenius": float(np.linalg.norm(ab, "fro")),
            "diagonal_delta_frobenius": float(np.linalg.norm(diagonal - 1.0)),
            "diagonal_delta_to_ab_ratio": float(
                np.linalg.norm(diagonal - 1.0) / np.linalg.norm(ab, "fro")
            ),
            "diagonal_max_abs_delta": float(np.max(np.abs(diagonal - 1.0))),
            "delta_spectral_norm": float(singular[0]),
            "delta_eigen_radius": float(np.max(np.abs(delta_eigenvalues))),
            "singular_to_eigen_radius_ratio": float(
                singular[0] / np.max(np.abs(delta_eigenvalues))
            ),
            "top_rank_energy": {
                str(rank): float(np.sum(singular[:rank] ** 2) / total_energy)
                for rank in RANKS
            },
        },
        "read_write_geometry": {
            "subspace_overlap": {
                str(rank): overlap(u[:, :rank], v[:, :rank]) for rank in RANKS
            },
            "random_overlap": {str(rank): rank / len(diagonal) for rank in RANKS},
            "bias_energy_in_input_space": {
                str(rank): capture(bias, u[:, :rank]) for rank in RANKS
            },
            "bias_energy_in_output_space": {
                str(rank): capture(bias, v[:, :rank]) for rank in RANKS
            },
            "top_modes": mode_rows,
        },
        "symmetric_skew": {
            "symmetric_frobenius": float(np.linalg.norm(symmetric, "fro")),
            "skew_frobenius": float(np.linalg.norm(skew, "fro")),
            "symmetric_energy_fraction": float(
                np.linalg.norm(symmetric, "fro") ** 2 / np.linalg.norm(delta, "fro") ** 2
            ),
            "skew_energy_fraction": float(
                np.linalg.norm(skew, "fro") ** 2 / np.linalg.norm(delta, "fro") ** 2
            ),
            "most_contracting_first_order_rate": float(symmetric_eigenvalues[0]),
            "most_expanding_first_order_rate": float(symmetric_eigenvalues[-1]),
        },
        "eigendecomposition": {
            "weight_spectral_radius": float(np.max(np.abs(eigenvalues))),
            "weight_minimum_eigen_modulus": float(np.min(np.abs(eigenvalues))),
            "largest_abs_angle_radians": largest_angle,
            "implied_period_j_only": float(2.0 * np.pi / max(largest_angle, 1e-30)),
            "distance_of_nearest_eigenvalue_to_minus_one": float(
                np.min(np.abs(eigenvalues + 1.0))
            ),
            "eigenbasis_condition": item["eigenbasis_condition"],
            "individual_eigen_condition_median": float(np.median(eigen_condition)),
            "individual_eigen_condition_q95": float(np.quantile(eigen_condition, 0.95)),
            "individual_eigen_condition_maximum": float(np.max(eigen_condition)),
        },
        "affine_bias": {
            "bias_norm": float(np.linalg.norm(bias)),
            "accumulated_bias_norm_40": power_lookup[40]["accumulated_bias_norm"],
            "accumulated_bias_norm_99": power_lookup[99]["accumulated_bias_norm"],
            "accumulated_bias_norm_200": power_lookup[200]["accumulated_bias_norm"],
        },
        "j_only_powers": {
            str(power): {
                key: value
                for key, value in power_lookup[power].items()
                if key not in {"label", "power"}
            }
            for power in (20, 40, 99, 200)
        },
    }


def pair_rows(items: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            left_delta = left["delta"]
            right_delta = right["delta"]
            assert isinstance(left_delta, np.ndarray) and isinstance(right_delta, np.ndarray)
            cosine = float(
                np.vdot(left_delta.ravel(), right_delta.ravel()).real
                / (np.linalg.norm(left_delta) * np.linalg.norm(right_delta))
            )
            for rank in RANKS:
                rows.append(
                    {
                        "left": left["label"],
                        "right": right["label"],
                        "rank": rank,
                        "delta_cosine": cosine,
                        "input_subspace_overlap": overlap(
                            left["u"][:, :rank], right["u"][:, :rank]
                        ),
                        "output_subspace_overlap": overlap(
                            left["v"][:, :rank], right["v"][:, :rank]
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_anchor1(item: dict[str, object], powers: list[dict[str, object]], out_dir: Path) -> None:
    singular = item["singular"]
    u = item["u"]
    v = item["v"]
    bias = item["bias"]
    assert all(isinstance(value, np.ndarray) for value in (singular, u, v, bias))
    ranks = np.arange(1, 49)
    energy = np.cumsum(singular[:48] ** 2) / np.sum(singular**2)
    uv_overlap = np.array([overlap(u[:, :rank], v[:, :rank]) for rank in ranks])
    input_capture = np.array([capture(bias, u[:, :rank]) for rank in ranks])
    output_capture = np.array([capture(bias, v[:, :rank]) for rank in ranks])

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    axes[0, 0].semilogy(np.arange(1, 65), singular[:64], marker="o", markersize=2.5)
    axes[0, 0].axvline(48, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("SVD of ΔW: singular values")
    axes[0, 0].set_xlabel("mode rank")
    axes[0, 0].set_ylabel("singular value")

    axes[0, 1].plot(ranks, energy, label="ΔW energy retained")
    axes[0, 1].axhline(0.85, color="grey", linestyle="--", linewidth=0.8)
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_title("Low-rank concentration")
    axes[0, 1].set_xlabel("kept modes")
    axes[0, 1].set_ylabel("fraction")

    axes[1, 0].plot(ranks, uv_overlap, label="observed input/output overlap")
    axes[1, 0].plot(ranks, ranks / len(singular), linestyle="--", label="random baseline r/256")
    axes[1, 0].set_title("Read space versus write space")
    axes[1, 0].set_xlabel("subspace rank")
    axes[1, 0].set_ylabel("mean squared overlap")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(ranks, output_capture, label="bias in write space V")
    axes[1, 1].plot(ranks, input_capture, label="bias in read space U")
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].set_title("Where the affine bias points")
    axes[1, 1].set_xlabel("subspace rank")
    axes[1, 1].set_ylabel("bias squared-norm captured")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.suptitle("Parity J (anchor 1): one-step read/write geometry")
    figure.tight_layout()
    figure.savefig(out_dir / "anchor1_read_write_geometry.png", dpi=190)
    plt.close(figure)

    eigenvalues = item["eigenvalues"]
    eigen_condition = item["eigen_condition"]
    delta = item["delta"]
    symmetric = item["symmetric"]
    skew = item["skew"]
    assert all(
        isinstance(value, np.ndarray)
        for value in (eigenvalues, eigen_condition, delta, symmetric, skew)
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    scatter = axes[0, 0].scatter(
        eigenvalues.real,
        eigenvalues.imag,
        c=np.log10(eigen_condition),
        s=14,
        cmap="viridis",
        alpha=0.8,
    )
    axes[0, 0].axvline(1.0, color="black", linestyle="--", linewidth=0.7)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_title("Eigenvalues of W; color = log10 sensitivity")
    axes[0, 0].set_xlabel("real")
    axes[0, 0].set_ylabel("imaginary")
    figure.colorbar(scatter, ax=axes[0, 0], fraction=0.046)

    steps = [int(row["power"]) for row in powers]
    axes[0, 1].plot(steps, [row["operator_norm"] for row in powers], marker="o", label="||W^n||2")
    axes[0, 1].plot(steps, [row["spectral_radius_power"] for row in powers], marker="o", label="rho(W)^n")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Nonnormal transient under J alone")
    axes[0, 1].set_xlabel("n")
    axes[0, 1].set_ylabel("gain")
    axes[0, 1].legend(fontsize=8)

    image = axes[1, 0].imshow(delta, cmap="coolwarm", aspect="auto")
    axes[1, 0].set_title("ΔW in model coordinates")
    axes[1, 0].set_xlabel("written coordinate")
    axes[1, 0].set_ylabel("read coordinate")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].bar(
        ["symmetric\n(stretch/compress)", "skew\n(redirect)"],
        [
            np.linalg.norm(symmetric, "fro") ** 2 / np.linalg.norm(delta, "fro") ** 2,
            np.linalg.norm(skew, "fro") ** 2 / np.linalg.norm(delta, "fro") ** 2,
        ],
        color=["#4C78A8", "#F58518"],
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("fraction of ΔW squared Frobenius norm")
    axes[1, 1].set_title("ΔW is neither pure scaling nor pure rotation")
    axes[0, 1].grid(alpha=0.25)
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle("Parity J (anchor 1): eigen dynamics and nonnormality")
    figure.tight_layout()
    figure.savefig(out_dir / "anchor1_eigen_dynamics.png", dpi=190)
    plt.close(figure)


def plot_anchors(items: list[dict[str, object]], out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for item in items:
        singular = item["singular"]
        u = item["u"]
        v = item["v"]
        bias = item["bias"]
        assert all(isinstance(value, np.ndarray) for value in (singular, u, v, bias))
        ranks = np.arange(1, 49)
        axes[0].plot(ranks, np.cumsum(singular[:48] ** 2) / np.sum(singular**2), label=item["label"])
        axes[1].plot(ranks, [overlap(u[:, :rank], v[:, :rank]) for rank in ranks], label=item["label"])
        axes[2].plot(ranks, [capture(bias, v[:, :rank]) for rank in ranks], label=item["label"])
    axes[0].set_title("ΔW cumulative energy")
    axes[1].set_title("read/write overlap")
    axes[2].set_title("bias energy in write space")
    axes[1].plot(np.arange(1, 49), np.arange(1, 49) / 256, "k--", linewidth=0.8, label="random")
    for axis in axes:
        axis.set_xlabel("rank")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("fraction")
    axes[0].legend(fontsize=8)
    figure.suptitle("Parity J: robustness across anchor controls")
    figure.tight_layout()
    figure.savefig(out_dir / "anchor_robustness.png", dpi=190)
    plt.close(figure)


def plot_w_delta_d_anchor1(item: dict[str, object], out_dir: Path) -> None:
    weight_singular = item["weight_singular"]
    delta_singular = item["singular"]
    diagonal = item["diagonal"]
    assert all(
        isinstance(value, np.ndarray)
        for value in (weight_singular, delta_singular, diagonal)
    )
    dimension = len(diagonal)
    indices = np.arange(1, dimension + 1)
    delta_energy = np.cumsum(delta_singular**2) / np.sum(delta_singular**2)
    diagonal_delta = diagonal - 1.0

    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))

    axes[0, 0].plot(indices, weight_singular, linewidth=1.5)
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].axhspan(0.999, 1.001, color="grey", alpha=0.14)
    axes[0, 0].set_ylim(0.975, 1.023)
    axes[0, 0].set_title("Singular values of W")
    axes[0, 0].set_xlabel("descending rank")
    axes[0, 0].set_ylabel("singular value")

    axes[0, 1].plot(indices, weight_singular - 1.0, linewidth=1.5)
    axes[0, 1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Same spectrum, centered at identity")
    axes[0, 1].set_xlabel("descending rank")
    axes[0, 1].set_ylabel("singular value minus 1")

    axes[0, 2].semilogy(indices, np.maximum(delta_singular, 1e-12), linewidth=1.5)
    axes[0, 2].axvline(4, color="#E45756", linestyle="--", linewidth=1.0, label="top 4")
    axes[0, 2].axvline(48, color="black", linestyle="--", linewidth=1.0, label="factor rank 48")
    axes[0, 2].set_title("Singular values of delta W = W-I")
    axes[0, 2].set_xlabel("descending rank")
    axes[0, 2].set_ylabel("singular value, log scale")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(indices, delta_energy, linewidth=1.5)
    axes[1, 0].scatter([4, 48], delta_energy[[3, 47]], color=["#E45756", "black"], zorder=3)
    axes[1, 0].axhline(0.85, color="grey", linestyle="--", linewidth=0.8)
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].set_title("Cumulative squared energy of delta W")
    axes[1, 0].set_xlabel("kept singular modes")
    axes[1, 0].set_ylabel("energy retained")

    axes[1, 1].plot(indices, diagonal_delta, linewidth=0.8, alpha=0.85)
    axes[1, 1].scatter(indices, diagonal_delta, s=5, alpha=0.65)
    axes[1, 1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("D minus identity by hidden coordinate")
    axes[1, 1].set_xlabel("hidden coordinate")
    axes[1, 1].set_ylabel("D[i] - 1")

    axes[1, 2].hist(diagonal_delta, bins=30, color="#4C78A8", alpha=0.85)
    axes[1, 2].axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 2].set_title("Distribution of D[i] - 1")
    axes[1, 2].set_xlabel("D[i] - 1")
    axes[1, 2].set_ylabel("coordinate count")
    axes[1, 2].xaxis.set_major_locator(MaxNLocator(5))
    axes[1, 2].ticklabel_format(axis="x", style="sci", scilimits=(-4, -4))

    for axis in axes.ravel():
        axis.grid(alpha=0.22)
    figure.suptitle("Parity J anchor 1: W spectrum, delta-W spectrum, and D")
    figure.tight_layout()
    figure.savefig(out_dir / "anchor1_w_delta_w_d.png", dpi=200)
    plt.close(figure)


def plot_w_delta_d_anchors(items: list[dict[str, object]], out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(items)))
    for color, item in zip(colors, items, strict=True):
        weight_singular = item["weight_singular"]
        delta_singular = item["singular"]
        diagonal = item["diagonal"]
        assert all(
            isinstance(value, np.ndarray)
            for value in (weight_singular, delta_singular, diagonal)
        )
        label = str(item["label"])
        axes[0].plot(weight_singular - 1.0, color=color, label=label)
        axes[1].semilogy(np.maximum(delta_singular, 1e-12), color=color, label=label)
        axes[2].plot(diagonal - 1.0, color=color, alpha=0.8, label=label)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("W singular values minus 1")
    axes[0].set_xlabel("descending rank")
    axes[0].set_ylabel("deviation from identity")
    axes[1].axvline(47, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("delta-W singular values")
    axes[1].set_xlabel("descending rank")
    axes[1].set_ylabel("singular value, log scale")
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("D[i] minus 1")
    axes[2].set_xlabel("hidden coordinate")
    axes[2].set_ylabel("diagonal change")
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    figure.suptitle("Parity J anchor controls")
    figure.tight_layout()
    figure.savefig(out_dir / "anchors_w_delta_w_d_comparison.png", dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = [load_item(*parse_controller(value)) for value in args.controller]
    all_power_rows: list[dict[str, object]] = []
    all_rank_rows: list[dict[str, object]] = []
    summaries = []
    for item in items:
        powers = power_analysis(item)
        all_power_rows.extend(powers)
        all_rank_rows.extend(rank_rows(item))
        summaries.append(summary(item, powers))
    pairs = pair_rows(items)
    write_csv(args.out_dir / "rank_geometry.csv", all_rank_rows)
    write_csv(args.out_dir / "j_only_power_dynamics.csv", all_power_rows)
    write_csv(args.out_dir / "anchor_pair_geometry.csv", pairs)
    (args.out_dir / "matrix_reanalysis.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "evidence_boundary": (
                    "Exact ambient matrix analysis only. A direction becomes a parity "
                    "circuit component only after hidden-state projection and causal "
                    "keep/delete/rotation interventions in the frozen model."
                ),
                "controllers": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    anchor1_index = next(
        (index for index, item in enumerate(items) if item["label"] == "anchor1"), 0
    )
    anchor1 = items[anchor1_index]
    anchor1_powers = [
        row for row in all_power_rows if row["label"] == anchor1["label"]
    ]
    plot_anchor1(anchor1, anchor1_powers, args.out_dir)
    plot_anchors(items, args.out_dir)
    plot_w_delta_d_anchor1(anchor1, args.out_dir)
    plot_w_delta_d_anchors(items, args.out_dir)
    print(json.dumps({"status": "complete", "controllers": len(items)}, indent=2))


if __name__ == "__main__":
    main()
