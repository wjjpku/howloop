#!/usr/bin/env python3
"""Plot the strict closed-loop outcomes for the component-supervised D8L8 model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def condition_curve(
    rows: list[dict[str, str]],
    condition: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["condition"] == condition),
        key=lambda row: int(row["cycle"]),
    )
    return (
        np.asarray([int(row["cycle"]) for row in selected]),
        np.asarray([float(row[metric]) for row in selected]),
    )


def plot_closed_loop(results: Path) -> None:
    focus_dir = (
        results
        / "remote_artifacts"
        / "unit_j_component_best7k_reset3_focus"
    )
    primary_dir = (
        results
        / "remote_artifacts"
        / "unit_j_component_best7k_primary"
    )
    focus = read_rows(focus_dir / "closed_loop.csv")
    primary = read_rows(primary_dir / "closed_loop.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    curves = [
        ("focused learned affine J", focus, "task_reset3", "#0072B2", 2.6),
        ("mixed-objective learned J", primary, "task_reset3", "#E69F00", 1.8),
        ("exact young h2 interface", focus, "exact_H2_every", "#009E73", 2.0),
        ("no rejuvenation", focus, "no_control", "#555555", 1.5),
        ("shuffled learned output", focus, "shuffled_final_reset3", "#CC79A7", 1.5),
    ]
    for axis, x_limit in zip(axes, [(1, 200), (1, 40)]):
        for label, rows, condition, color, width in curves:
            x, y = condition_curve(rows, condition, "accuracy_nonendpoint")
            axis.plot(x, y, label=label, color=color, linewidth=width)
        axis.axhline(0.9, color="black", linestyle=":", linewidth=1.2)
        axis.set_xlim(*x_limit)
        axis.set_ylim(-0.02, 1.03)
        axis.set_xlabel("additional two-hop F cycles")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("strict accuracy, endpoint-return cases excluded")
    axes[0].set_title("200-cycle view")
    axes[1].set_title("first 40 cycles")
    axes[1].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Component-supervised Pre-Norm D8L8: closed-loop rejuvenation")
    fig.tight_layout()
    fig.savefig(results / "focused_affine_J_nonendpoint_accuracy.png", dpi=180)
    plt.close(fig)


def plot_spectrum(results: Path) -> None:
    maps_path = (
        results
        / "remote_artifacts"
        / "unit_j_component_best7k_reset3_focus"
        / "unit_j_maps.pt"
    )
    payload = torch.load(maps_path, map_location="cpu", weights_only=False)
    weight_tensor = payload["maps"]["task"]["weight"].double()
    weight = weight_tensor.numpy()
    eigenvalues = np.linalg.eigvals(weight)
    _, log_abs_det_tensor = torch.linalg.slogdet(weight_tensor)
    log_abs_det = float(log_abs_det_tensor)
    median_modulus = float(np.median(np.abs(eigenvalues)))
    spectral_radius = float(np.max(np.abs(eigenvalues)))

    bandwidth = 0.035
    density_at_eigenvalues = np.asarray(
        [
            np.exp(
                -(np.abs(eigenvalues - candidate) ** 2)
                / (2.0 * bandwidth**2)
            ).sum()
            for candidate in eigenvalues
        ]
    )
    peak = eigenvalues[int(density_at_eigenvalues.argmax())]

    fig, axis = plt.subplots(figsize=(6.4, 6.1))
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    axis.plot(np.cos(theta), np.sin(theta), "--", color="#9AA5B1", linewidth=1.2)
    axis.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        s=19,
        alpha=0.68,
        color="#2F6FDB",
    )
    axis.scatter([peak.real], [peak.imag], marker="x", s=90, color="red")
    axis.axhline(0.0, color="#C7CDD4", linewidth=0.8)
    axis.axvline(0.0, color="#C7CDD4", linewidth=0.8)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.15, 1.15)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel(r"Re($\lambda$)")
    axis.set_ylabel(r"Im($\lambda$)")
    axis.set_title(
        "Focused full-interface affine J (linear part)\n"
        f"peak≈{peak.real:.3f}{peak.imag:+.3f}i, "
        f"median |λ|={median_modulus:.3f}, "
        f"ρ={spectral_radius:.3f}, log|det|={log_abs_det:.1f}"
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results / "focused_affine_J_complex_spectrum.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "results/graph_path_prenorm_component_global_J_20260731"
        ),
    )
    args = parser.parse_args()
    plot_closed_loop(args.results)
    plot_spectrum(args.results)


if __name__ == "__main__":
    main()
