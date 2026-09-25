from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PANEL_SEEDS = (0, 1, 6, 8)


def load_top1_matrices(summary_path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    matrices = {
        seed: np.asarray(payload["seeds"][str(seed)]["top1_match_fraction"])
        for seed in PANEL_SEEDS
    }
    for seed, matrix in matrices.items():
        if matrix.shape != (17, 10):
            raise ValueError(f"seed {seed} has unexpected matrix shape {matrix.shape}")
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError(f"seed {seed} top-1 rows do not sum to one")
    return matrices


def plot_four_seed_row(matrices: dict[int, np.ndarray], output_path: Path) -> None:
    figure = plt.figure(figsize=(9.25, 2.85))
    grid = figure.add_gridspec(
        1,
        4,
        left=0.06,
        right=0.925,
        bottom=0.19,
        top=0.82,
        wspace=0.18,
    )
    axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    image = None
    for axis, seed in zip(axes, PANEL_SEEDS, strict=True):
        values = matrices[seed]
        image = axis.imshow(
            values,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        maxima = values.argmax(axis=1)
        axis.plot(
            maxima,
            np.arange(values.shape[0]),
            "o",
            ms=2.0,
            color="white",
            mec="black",
            mew=0.22,
        )
        axis.axhline(8.5, color="#ef4444", linestyle="--", linewidth=0.8)
        axis.set_title(f"seed {seed}", fontsize=9, pad=2)
        axis.set_xticks(range(10))
        axis.set_xticklabels([rf"$f^{position}$" for position in range(10)], fontsize=6.5)
        axis.set_yticks(range(0, 17, 2))
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel("path target", fontsize=7.5, labelpad=2)
    axes[0].set_ylabel("loop boundary", fontsize=8)

    assert image is not None
    colorbar_axis = figure.add_axes((0.94, 0.21, 0.012, 0.58))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("top-1 output fraction", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.text(
        0.06,
        0.955,
        "Intermediate readout accuracy on newly sampled random 10-cycle graphs",
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrices = load_top1_matrices(args.summary)
    for extension in ("pdf", "png"):
        plot_four_seed_row(
            matrices,
            args.out_dir / f"seed0_1_6_8_f0_f9_top1_accuracy_row.{extension}",
        )


if __name__ == "__main__":
    main()
