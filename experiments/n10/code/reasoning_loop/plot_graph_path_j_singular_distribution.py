"""Visualize the full and low-tail singular spectra of the seven graph-path J maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


AGES = tuple(range(2, 9))
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_spectra(path: Path) -> dict[int, np.ndarray]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    state = artifact["state_dict"]
    diagonal = torch.diag(state["shared_diagonal_scale"].double())
    shared = state["shared_A"].double() @ state["shared_B"].double()
    spectra: dict[int, np.ndarray] = {}
    for age in AGES:
        stage = state[f"stage_A.{age}"].double() @ state[f"stage_B.{age}"].double()
        spectra[age] = torch.linalg.svdvals(diagonal + shared + stage).numpy()
    return spectra


def write_tables(spectra: dict[int, np.ndarray], out_dir: Path) -> None:
    with (out_dir / "singular_values_by_j.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["j", "source_age", "rank_descending", "singular_value"],
        )
        writer.writeheader()
        for age, singular in spectra.items():
            for rank, value in enumerate(singular, start=1):
                writer.writerow(
                    {
                        "j": f"J{age}",
                        "source_age": age,
                        "rank_descending": rank,
                        "singular_value": float(value),
                    }
                )

    with (out_dir / "singular_threshold_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["j", "source_age", "tau", "count_below_tau"],
        )
        writer.writeheader()
        for age, singular in spectra.items():
            for tau in TAUS:
                writer.writerow(
                    {
                        "j": f"J{age}",
                        "source_age": age,
                        "tau": tau,
                        "count_below_tau": int(np.count_nonzero(singular < tau)),
                    }
                )


def plot(spectra: dict[int, np.ndarray], path: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(AGES)))
    figure = plt.figure(figsize=(13.2, 8.6), dpi=220)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.08, 1.0), hspace=0.3, wspace=0.22)
    full_axis = figure.add_subplot(grid[0, :])
    tail_axis = figure.add_subplot(grid[1, 0])
    count_axis = figure.add_subplot(grid[1, 1])

    for color, age in zip(colors, AGES):
        singular = spectra[age]
        full_axis.plot(
            np.arange(1, singular.size + 1),
            singular,
            color=color,
            linewidth=1.45,
            label=rf"$J_{{{age}}}: H_{age}\to H_{{{age-1}}}$",
        )
        tail = np.sort(singular)[:24]
        tail_axis.plot(
            np.arange(1, tail.size + 1),
            tail,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=2.8,
        )

    full_axis.set_yscale("log")
    full_axis.set(
        title="Full singular spectra of the seven $J$ maps",
        xlabel="Singular-value rank (largest to smallest)",
        ylabel="Singular value (log scale)",
        xlim=(1, 256),
    )
    full_axis.grid(alpha=0.2, which="both")
    full_axis.legend(ncol=4, fontsize=8.7, loc="lower left")

    tau_colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(TAUS)))
    for tau, color in zip(TAUS, tau_colors):
        tail_axis.axhline(tau, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        tail_axis.text(24.35, tau, rf"$\tau={tau:g}$", color=color, va="center", fontsize=8)
    tail_axis.set(
        title="Low-spectrum tail",
        xlabel="Tail rank (1 = smallest)",
        ylabel="Singular value",
        xlim=(1, 26.4),
        ylim=(-0.01, 0.86),
        xticks=[1, 4, 8, 12, 16, 20, 24],
    )
    tail_axis.grid(alpha=0.2)

    counts = np.asarray(
        [[np.count_nonzero(spectra[age] < tau) for tau in TAUS] for age in AGES]
    )
    image = count_axis.imshow(counts, cmap="Blues", aspect="auto", vmin=0, vmax=16)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count_axis.text(
                column,
                row,
                str(int(counts[row, column])),
                ha="center",
                va="center",
                color="white" if counts[row, column] >= 9 else "#111827",
                fontsize=10,
                fontweight="bold",
            )
    count_axis.set(
        title=r"Number of singular values below each $\tau$",
        xlabel="Singular-value floor",
        ylabel="$J$ map",
        xticks=np.arange(len(TAUS)),
        xticklabels=[rf"$\tau={tau:g}$" for tau in TAUS],
        yticks=np.arange(len(AGES)),
        yticklabels=[rf"$J_{age}$" for age in AGES],
    )
    figure.colorbar(image, ax=count_axis, fraction=0.046, pad=0.04, label="count")

    figure.suptitle("Singular-value distribution and adaptive floor coverage", fontsize=16)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spectra = load_spectra(args.bank_artifact)
    write_tables(spectra, args.out_dir)
    plot(spectra, args.out_dir / "singular_distribution.png")


if __name__ == "__main__":
    main()
