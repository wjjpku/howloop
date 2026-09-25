"""Plot Parity diagonal-band metrics while retaining absolute n and t axes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metric", default="exact_match")
    parser.add_argument("--maximum", type=int, default=500)
    parser.add_argument("--half-width", type=int, default=10)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                key = (str(raw["variant"]), int(raw["length"]), int(raw["loop"]))
                if key in seen:
                    raise ValueError(f"duplicate cell: {key}")
                seen.add(key)
                rows.append(
                    {
                        "variant": key[0],
                        "length": key[1],
                        "loop": key[2],
                        "exact_match": float(raw["exact_match"]),
                        "parity_token_accuracy": float(raw["parity_token_accuracy"]),
                    }
                )
    return rows


def full_matrix(
    rows: list[dict[str, Any]], variant: str, metric: str, maximum: int
) -> np.ndarray:
    matrix = np.full((maximum, maximum), np.nan, dtype=np.float64)
    for row in rows:
        if row["variant"] != variant:
            continue
        n = int(row["length"])
        t = int(row["loop"])
        if 1 <= n <= maximum and 1 <= t <= maximum:
            matrix[n - 1, t - 1] = float(row[metric])
    return matrix


def colormap(name: str) -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    return cmap


def draw_panel(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    n_start: int,
    n_stop: int,
    t_start: int,
    t_stop: int,
    title: str,
    cmap: matplotlib.colors.Colormap,
    vmin: float,
    vmax: float,
) -> matplotlib.image.AxesImage:
    selected = matrix[n_start - 1 : n_stop, t_start - 1 : t_stop]
    image = axis.imshow(
        selected,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        extent=(t_start - 0.5, t_stop + 0.5, n_start - 0.5, n_stop + 0.5),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    diagonal_start = max(n_start, t_start)
    diagonal_stop = min(n_stop, t_stop)
    axis.plot(
        [diagonal_start, diagonal_stop],
        [diagonal_start, diagonal_stop],
        color="white" if cmap.name != "coolwarm" else "black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
        label="t=n",
    )
    axis.set_xlim(t_start - 0.5, t_stop + 0.5)
    axis.set_ylim(n_start - 0.5, n_stop + 0.5)
    axis.set_xlabel("executed recurrent loops t")
    axis.set_ylabel("input length n")
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8, framealpha=0.8)
    return image


def plot_overview(
    matrices: dict[str, np.ndarray], out_dir: Path, maximum: int
) -> None:
    cmap = colormap("viridis")
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), sharex=True, sharey=True)
    for axis, variant in zip(axes, ("raw", "J"), strict=True):
        image = draw_panel(
            axis,
            matrices[variant],
            n_start=1,
            n_stop=maximum,
            t_start=1,
            t_stop=maximum,
            title=variant,
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="strict exact match")
    figure.suptitle("Parity near the registered diagonal (absolute n and t axes)")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_nt_diagonal_overview_raw_J.png", dpi=240)
    plt.close(figure)


def plot_zoom_range(
    matrices: dict[str, np.ndarray],
    out_dir: Path,
    *,
    n_start: int,
    n_stop: int,
    maximum: int,
    half_width: int,
) -> None:
    t_start = max(1, n_start - half_width)
    t_stop = min(maximum, n_stop + half_width)
    cmap = colormap("viridis")
    figure, axes = plt.subplots(1, 2, figsize=(14.8, 6.6), sharex=True, sharey=True)
    for axis, variant in zip(axes, ("raw", "J"), strict=True):
        image = draw_panel(
            axis,
            matrices[variant],
            n_start=n_start,
            n_stop=n_stop,
            t_start=t_start,
            t_stop=t_stop,
            title=f"{variant}: n={n_start}-{n_stop}",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="strict exact match")
    figure.suptitle(f"Parity diagonal zoom: n={n_start}-{n_stop}, t={t_start}-{t_stop}")
    figure.tight_layout()
    figure.savefig(
        out_dir / f"parity_nt_diagonal_zoom_n{n_start:03d}_{n_stop:03d}_raw_J.png",
        dpi=240,
    )
    plt.close(figure)


def plot_zoom_grid(
    matrices: dict[str, np.ndarray], out_dir: Path, maximum: int, half_width: int
) -> None:
    ranges = [(start, min(start + 99, maximum)) for start in range(1, maximum + 1, 100)]
    cmap = colormap("viridis")
    figure, axes = plt.subplots(len(ranges), 2, figsize=(14.5, 4.3 * len(ranges)), squeeze=False)
    for row_index, (n_start, n_stop) in enumerate(ranges):
        t_start = max(1, n_start - half_width)
        t_stop = min(maximum, n_stop + half_width)
        for column_index, variant in enumerate(("raw", "J")):
            image = draw_panel(
                axes[row_index, column_index],
                matrices[variant],
                n_start=n_start,
                n_stop=n_stop,
                t_start=t_start,
                t_stop=t_stop,
                title=f"{variant}: n={n_start}-{n_stop}, t={t_start}-{t_stop}",
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            figure.colorbar(image, ax=axes[row_index, column_index], fraction=0.035, pad=0.015)
    figure.suptitle("Parity t approximately n: five absolute-coordinate zooms", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(out_dir / "parity_nt_diagonal_five_ranges_raw_J.png", dpi=220)
    plt.close(figure)


def plot_difference(
    matrices: dict[str, np.ndarray], out_dir: Path, maximum: int
) -> None:
    difference = matrices["J"] - matrices["raw"]
    cmap = colormap("coolwarm")
    figure, axis = plt.subplots(figsize=(8.2, 7.4))
    image = draw_panel(
        axis,
        difference,
        n_start=1,
        n_stop=maximum,
        t_start=1,
        t_stop=maximum,
        title="J - raw",
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="accuracy difference")
    figure.suptitle("Parity controller effect near t=n (absolute coordinates)")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_nt_diagonal_overview_J_minus_raw.png", dpi=240)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrices = {
        variant: full_matrix(rows, variant, args.metric, args.maximum)
        for variant in ("raw", "J")
    }
    plot_overview(matrices, args.out_dir, args.maximum)
    plot_difference(matrices, args.out_dir, args.maximum)
    plot_zoom_grid(matrices, args.out_dir, args.maximum, args.half_width)
    for n_start in range(1, args.maximum + 1, 100):
        plot_zoom_range(
            matrices,
            args.out_dir,
            n_start=n_start,
            n_stop=min(n_start + 99, args.maximum),
            maximum=args.maximum,
            half_width=args.half_width,
        )


if __name__ == "__main__":
    main()
