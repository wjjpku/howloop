"""Visualize the Addition and Sum-Reverse affine J controllers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm


TASKS = ("Addition", "SumReverse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def symmetric_limit(arrays: list[np.ndarray], percentile: float | None = None) -> float:
    values = np.concatenate([np.abs(array).ravel() for array in arrays])
    if percentile is None:
        limit = float(values.max())
    else:
        limit = float(np.percentile(values, percentile))
    return max(limit, 1e-12)


def heatmap(
    axis: plt.Axes,
    array: np.ndarray,
    *,
    limit: float,
    title: str,
    aspect: str = "auto",
) -> matplotlib.image.AxesImage:
    image = axis.imshow(
        array,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect=aspect,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    return image


def load_data(result_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    controller_names = {
        "Addition": "addition.pt",
        "SumReverse": "sum_reverse.pt",
    }
    output: dict[str, dict[str, np.ndarray]] = {}
    for task in TASKS:
        spectral = np.load(result_dir / f"{task}_spectral_arrays.npz")
        payload = torch.load(
            result_dir / "controllers" / controller_names[task],
            map_location="cpu",
            weights_only=False,
        )
        state = payload["controller_state_dict"]
        output[task] = {
            **{key: spectral[key] for key in spectral.files},
            "D_delta": state["diagonal"].detach().double().numpy() - 1.0,
            "A": state["A"].detach().double().numpy(),
            "B": state["B"].detach().double().numpy(),
        }
    return output


def plot_parameters(result_dir: Path, data: dict[str, dict[str, np.ndarray]]) -> None:
    columns = (
        ("D_delta", lambda x: x.reshape(16, 16), "D−I (16×16 display)"),
        ("A", lambda x: x, "A: read factor (256×48)"),
        ("B", lambda x: x, "B: write factor (48×256)"),
        ("W", lambda x: x - np.eye(x.shape[0]), "AB + (D−I) = ΔW"),
        ("bias", lambda x: x.reshape(16, 16), "bias b (16×16 display)"),
    )
    limits = {
        key: symmetric_limit([transform(data[task][key]) for task in TASKS], 99.5)
        for key, transform, _ in columns
    }
    figure, axes = plt.subplots(2, len(columns), figsize=(18, 8.2))
    for row, task in enumerate(TASKS):
        for column, (key, transform, title) in enumerate(columns):
            image = heatmap(
                axes[row, column],
                transform(data[task][key]),
                limit=limits[key],
                title=title,
            )
            if column == 0:
                axes[row, column].set_ylabel(task, fontsize=12, fontweight="bold")
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    figure.suptitle(
        "Addition and Sum-Reverse J parameters\n"
        "D and b are reshaped only for display; neighboring pixels have no spatial meaning",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(result_dir / "addition_sumreverse_j_parameters.png", dpi=210)
    plt.close(figure)


def plot_effective_operators(
    result_dir: Path, data: dict[str, dict[str, np.ndarray]]
) -> None:
    processed: dict[str, dict[str, np.ndarray | float]] = {}
    for task in TASKS:
        delta = data[task]["W"] - np.eye(data[task]["W"].shape[0])
        singular = data[task]["delta_singular_values"]
        top4 = (
            data[task]["delta_U"][:, :4]
            @ np.diag(singular[:4])
            @ data[task]["delta_Vt"][:4, :]
        )
        norm = float(np.linalg.norm(delta, "fro"))
        processed[task] = {
            "W": data[task]["W"],
            "delta": delta,
            "delta_unit": delta / norm,
            "top4_unit": top4 / norm,
            "tail_unit": (delta - top4) / norm,
            "top4_energy": float(np.sum(singular[:4] ** 2) / np.sum(singular**2)),
        }

    w_limit = symmetric_limit([processed[task]["W"] for task in TASKS])
    delta_limit = symmetric_limit([processed[task]["delta"] for task in TASKS], 99.8)
    unit_limit = symmetric_limit(
        [
            processed[task][key]
            for task in TASKS
            for key in ("delta_unit", "top4_unit", "tail_unit")
        ],
        99.8,
    )
    figure, axes = plt.subplots(2, 5, figsize=(19, 8.2))
    for row, task in enumerate(TASKS):
        entries = (
            ("W", w_limit, "W = I + ΔW"),
            ("delta", delta_limit, "ΔW, shared absolute scale"),
            ("delta_unit", unit_limit, "ΔW / ||ΔW||F"),
            (
                "top4_unit",
                unit_limit,
                f"top-4 reconstruction ({processed[task]['top4_energy']:.1%} energy)",
            ),
            ("tail_unit", unit_limit, "remaining modes after top 4"),
        )
        for column, (key, limit, title) in enumerate(entries):
            image = heatmap(
                axes[row, column],
                processed[task][key],
                limit=limit,
                title=title,
                aspect="equal",
            )
            if column == 0:
                axes[row, column].set_ylabel(task, fontsize=12, fontweight="bold")
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
    figure.suptitle(
        "Effective linear operator of J: absolute magnitude and directional pattern",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(result_dir / "addition_sumreverse_j_operator_heatmaps.png", dpi=210)
    plt.close(figure)


def plot_svd_read_write(
    result_dir: Path, data: dict[str, dict[str, np.ndarray]]
) -> None:
    direction_limit = symmetric_limit(
        [
            data[task][key][:, :8] if key == "delta_U" else data[task][key][:8, :]
            for task in TASKS
            for key in ("delta_U", "delta_Vt")
        ],
        99.5,
    )
    figure, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    for row, task in enumerate(TASKS):
        item = data[task]
        singular = item["delta_singular_values"][:48]
        energy = np.cumsum(singular**2) / np.sum(singular**2)
        x = np.arange(1, 49)
        axes[row, 0].plot(x, singular, marker="o", markersize=2.5, color="#26547c")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_xlabel("mode r")
        axes[row, 0].set_ylabel("singular value", color="#26547c")
        axes[row, 0].tick_params(axis="y", labelcolor="#26547c")
        axes[row, 0].grid(alpha=0.25)
        cumulative_axis = axes[row, 0].twinx()
        cumulative_axis.plot(x, energy, color="#ef476f", linewidth=1.8)
        cumulative_axis.set_ylim(0.0, 1.03)
        cumulative_axis.set_ylabel("cumulative squared energy", color="#ef476f")
        cumulative_axis.tick_params(axis="y", labelcolor="#ef476f")
        axes[row, 0].set_title("Strength of the 48 ΔW modes")

        read_image = heatmap(
            axes[row, 1],
            item["delta_U"][:, :8].T,
            limit=direction_limit,
            title="top-8 read directions u_r",
        )
        axes[row, 1].set_yticks(range(8), labels=range(1, 9))
        axes[row, 1].set_ylabel("mode r")
        axes[row, 1].set_xlabel("hidden dimension")
        figure.colorbar(read_image, ax=axes[row, 1], fraction=0.046, pad=0.03)

        write_image = heatmap(
            axes[row, 2],
            item["delta_Vt"][:8, :],
            limit=direction_limit,
            title="top-8 write directions v_r",
        )
        axes[row, 2].set_yticks(range(8), labels=range(1, 9))
        axes[row, 2].set_ylabel("mode r")
        axes[row, 2].set_xlabel("hidden dimension")
        figure.colorbar(write_image, ax=axes[row, 2], fraction=0.046, pad=0.03)

        cosines = np.sum(
            item["delta_U"][:, :16] * item["delta_Vt"][:16, :].T,
            axis=0,
        )
        colors = np.where(cosines >= 0.0, "#ef476f", "#26547c")
        axes[row, 3].bar(np.arange(1, 17), cosines, color=colors)
        axes[row, 3].axhline(0.0, color="black", linewidth=0.8)
        axes[row, 3].set_ylim(-0.32, 0.32)
        axes[row, 3].set_xlabel("mode r")
        axes[row, 3].set_ylabel("cos(u_r, v_r)")
        axes[row, 3].set_title("read/write direction alignment")
        axes[row, 3].grid(axis="y", alpha=0.25)

        axes[row, 0].text(
            -0.24,
            0.5,
            task,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    figure.suptitle(
        "Delta W = U Sigma V^T: J reads h.u_r and writes sigma_r(h.u_r)v_r",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(result_dir / "addition_sumreverse_j_svd_read_write.png", dpi=210)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    data = load_data(args.result_dir)
    plot_parameters(args.result_dir, data)
    plot_effective_operators(args.result_dir, data)
    plot_svd_read_write(args.result_dir, data)
    print(
        "\n".join(
            [
                str(args.result_dir / "addition_sumreverse_j_parameters.png"),
                str(args.result_dir / "addition_sumreverse_j_operator_heatmaps.png"),
                str(args.result_dir / "addition_sumreverse_j_svd_read_write.png"),
            ]
        )
    )


if __name__ == "__main__":
    main()
