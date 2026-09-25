from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_eigenvalues(
    *,
    artifact: Path,
    output_png: Path,
    output_svg: Path | None,
    variant: str,
) -> None:
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    matrix = payload[variant]["matrix"].detach().cpu().double()
    eigenvalues = torch.linalg.eigvals(matrix).numpy()
    real = eigenvalues.real
    imaginary = eigenvalues.imag
    modulus = np.abs(eigenvalues)
    spectral_radius = float(modulus.max())

    figure, axis = plt.subplots(figsize=(8.2, 8.2), dpi=220)
    angle = np.linspace(0.0, 2.0 * np.pi, 1000)
    axis.plot(
        np.cos(angle),
        np.sin(angle),
        color="#6b7280",
        linewidth=1.4,
        linestyle="--",
        label="Unit circle",
        zorder=1,
    )
    axis.plot(
        spectral_radius * np.cos(angle),
        spectral_radius * np.sin(angle),
        color="#dc2626",
        linewidth=1.1,
        linestyle=":",
        label=rf"Spectral radius = {spectral_radius:.4f}",
        zorder=1,
    )
    scatter = axis.scatter(
        real,
        imaginary,
        c=modulus,
        cmap="viridis",
        s=30,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, zorder=0)
    axis.axvline(0.0, color="#9ca3af", linewidth=0.8, zorder=0)
    axis.scatter(
        [0.0],
        [0.0],
        marker="+",
        s=75,
        linewidths=1.3,
        color="#111827",
        zorder=4,
    )
    limit = max(1.04, spectral_radius * 1.08)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"Real part, $\operatorname{Re}(\lambda)$", fontsize=12)
    axis.set_ylabel(r"Imaginary part, $\operatorname{Im}(\lambda)$", fontsize=12)
    axis.set_title(
        "Global linear rejuvenator J: eigenvalues in the complex plane\n"
        rf"$d=256$, all $|\lambda|<0.99$, "
        rf"$\rho(J)={spectral_radius:.4f}$",
        fontsize=13,
        pad=13,
    )
    axis.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.8)
    axis.legend(loc="upper right", frameon=True, fontsize=9)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Modulus $|\lambda|$", fontsize=11)
    figure.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, bbox_inches="tight")
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot global rejuvenator eigenvalues in the complex plane."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path)
    parser.add_argument(
        "--variant",
        choices=("linear", "affine"),
        default="linear",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plot_eigenvalues(
        artifact=args.artifact,
        output_png=args.output_png,
        output_svg=args.output_svg,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
