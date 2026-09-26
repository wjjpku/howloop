"""Compare the original easy-range Parity J with boundary-trained controllers.

The script only consumes saved evaluation artifacts.  It writes publication-
style PNG/PDF figures plus compact source CSV files used by the figures.
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
import pandas as pd
import torch
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


DEFAULT_ROOT = Path(
    "/data/paperexperiment/Documents/github/LooPlus/results/"
    "parity_input_once_audit_20260811"
)
SEED_RANGES = {0: (80, 200), 1: (64, 128), 2: (48, 100)}
COLORS = {
    "raw": "#1f2937",
    "old": "#d97706",
    "new": "#0f9d8a",
    "rank128": "#2563eb",
    "dense": "#c026d3",
}
PHASE_COLUMN_COMPRESSION = 0.10
PHASE_COLUMN_MAX_OFFSET = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT / "boundary_rank_sweep_summary",
    )
    return parser.parse_args()


def save_figure(figure: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(out_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def read_csv_if_present(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() and path.stat().st_size else None


def geometric_median(values: pd.Series) -> float:
    positive = np.maximum(values.to_numpy(dtype=float), 1e-15)
    return float(np.exp(np.median(np.log(positive))))


def plot_training_signal(root: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 7.6), sharex="col")
    for seed in range(3):
        old_path = root / f"controllers/seed{seed}/extension20to40/training.csv"
        new_path = root / f"boundary_rank_sweep_v1/seed{seed}/boundary_rank48/training.csv"
        datasets = {
            "Old easy-range J": (pd.read_csv(old_path), COLORS["old"]),
            "New boundary J": (pd.read_csv(new_path), COLORS["new"]),
        }
        for label, (frame, color) in datasets.items():
            progress = frame["global_update"] / frame["global_update"].max()
            ce = np.maximum(frame["task_ce"].to_numpy(float), 1e-15)
            grad = np.maximum(
                frame["mean_preclip_gradient_norm"].to_numpy(float), 1e-15
            )
            axes[0, seed].plot(progress, ce, color=color, linewidth=1.6, label=label)
            axes[1, seed].plot(progress, grad, color=color, linewidth=1.35, label=label)
            rows.append(
                {
                    "seed": seed,
                    "controller": label,
                    "first_exact_match": float(frame["mean_final_exact_match"].iloc[0]),
                    "median_task_ce": geometric_median(frame["task_ce"]),
                    "median_preclip_gradient_norm": geometric_median(
                        frame["mean_preclip_gradient_norm"]
                    ),
                    "maximum_global_update": int(frame["global_update"].max()),
                }
            )

        for row, ylabel in enumerate(("Observed task CE", "Pre-clip gradient norm")):
            axis = axes[row, seed]
            axis.set_yscale("log")
            axis.grid(alpha=0.22, linewidth=0.6)
            axis.set_ylabel(ylabel if seed == 0 else "")
            axis.set_title(f"seed{seed}" if row == 0 else "")
            if row == 1:
                axis.set_xlabel("Fraction of training budget")
        axes[0, seed].legend(frameon=False, fontsize=9, loc="best")

    figure.suptitle(
        "Old easy-range training was saturated; boundary training supplies a live signal",
        fontsize=15,
        y=1.01,
    )
    figure.text(
        0.5,
        -0.015,
        "CE objectives use their recorded training temperatures; gradients are shown as the direct no-signal diagnostic.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "parity_j_training_signal_old_vs_boundary")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "training_signal_summary.csv", index=False)
    return summary


def collect_far_horizon(root: Path) -> pd.DataFrame:
    old = pd.read_csv(root / "paper_figure_recheck_summary/far_horizon_multiseed.csv")
    old = old[old["condition"].isin(["raw", "extension J"])].copy()
    old["controller"] = old["condition"].map(
        {"raw": "raw (old evaluation)", "extension J": "old easy-range J"}
    )
    old = old.rename(columns={"exact_match": "accuracy"})[
        ["seed", "controller", "length", "accuracy"]
    ]
    frames = [old]
    for seed in range(3):
        path = (
            root
            / f"paper_figure_recheck/seed{seed}/extension20to40/long_horizon/horizon.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[frame["length"] > 200][
            ["length", "variant", "exact_match"]
        ].copy()
        frame["seed"] = seed
        frame["controller"] = frame["variant"].map(
            {"raw": "raw (old evaluation)", "full": "old easy-range J"}
        )
        frame = frame.rename(columns={"exact_match": "accuracy"})
        frames.append(frame[["seed", "controller", "length", "accuracy"]])
    for seed in range(3):
        for label, pretty in (
            ("boundary_rank48", "new boundary J (rank 48)"),
            ("boundary_rank128", "new boundary J (rank 128)"),
            ("boundary_dense", "new boundary J (dense)"),
        ):
            path = (
                root
                / f"boundary_rank_sweep_v1/evaluations/seed{seed}/{label}/far_horizon/horizon.csv"
            )
            frame = read_csv_if_present(path)
            if frame is None:
                continue
            selected = frame[["length", "variant", "exact_match"]].copy()
            selected["seed"] = seed
            selected["controller"] = selected["variant"].map(
                {"raw": "raw (new evaluation)", "full": pretty}
            )
            selected = selected.rename(columns={"exact_match": "accuracy"})
            frames.append(selected[["seed", "controller", "length", "accuracy"]])
    return pd.concat(frames, ignore_index=True)


def plot_far_horizon(root: Path, out_dir: Path) -> pd.DataFrame:
    data = collect_far_horizon(root)
    data.to_csv(out_dir / "far_horizon_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 4.6), sharey=True)
    requested = (
        ("raw (new evaluation)", "raw", "raw", "-"),
        ("raw (old evaluation)", "raw (old eval)", "raw", ":"),
        ("old easy-range J", "old easy-range J", "old", "--"),
        ("new boundary J (rank 48)", "new boundary J (rank 48)", "new", "-"),
        ("new boundary J (rank 128)", "rank 128", "rank128", "-"),
        ("new boundary J (dense)", "dense", "dense", "-"),
    )
    for seed, axis in enumerate(axes):
        selected_seed = data[data["seed"] == seed]
        seen_raw_new = "raw (new evaluation)" in set(selected_seed["controller"])
        for internal, label, color_key, linestyle in requested:
            if internal == "raw (old evaluation)" and seen_raw_new:
                continue
            series = selected_seed[selected_seed["controller"] == internal].sort_values(
                "length"
            )
            if series.empty:
                continue
            axis.plot(
                series["length"],
                series["accuracy"],
                marker="o",
                markersize=3.4,
                linewidth=2.0 if "rank 48" in internal else 1.45,
                linestyle=linestyle,
                color=COLORS[color_key],
                label=label,
            )
        minimum, maximum = SEED_RANGES[seed]
        axis.axvspan(minimum, maximum, color=COLORS["new"], alpha=0.08)
        axis.axvline(20, color="#9ca3af", linewidth=0.9, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlim(9, 1050)
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"seed{seed}: boundary train n={minimum}–{maximum}")
        axis.set_xlabel("Input length n (registered t=n)")
        axis.grid(alpha=0.22, linewidth=0.6)
        if seed == 0:
            axis.set_ylabel("Exact-match accuracy")
        axis.legend(frameon=False, fontsize=8, loc="lower left")
    figure.suptitle(
        "Registered Parity accuracy: old easy-range J vs boundary-trained J",
        fontsize=15,
        y=1.02,
    )
    figure.text(
        0.5,
        -0.02,
        "Old curve: 512 examples through n=200 and 32 examples above n=200; new curve: 256 examples per shown length.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "parity_j_registered_accuracy_old_vs_boundary")
    return data


def heatmap_matrix(path: Path, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame[frame["variant"] == variant]
    lengths = np.arange(1, 501)
    offsets = np.arange(-10, 11)
    matrix = np.full((len(lengths), len(offsets)), np.nan, dtype=float)
    by_length = {int(value): index for index, value in enumerate(lengths)}
    by_offset = {int(value): index for index, value in enumerate(offsets)}
    for row in frame.itertuples(index=False):
        length = int(row.length)
        offset = int(row.relative_loop)
        if length in by_length and offset in by_offset:
            matrix[by_length[length], by_offset[offset]] = float(
                row.parity_token_accuracy
            )
    return matrix, lengths, offsets


def plot_heatmaps(root: Path, out_dir: Path) -> None:
    available_seeds: list[int] = []
    records: dict[tuple[int, str], np.ndarray] = {}
    for seed in range(3):
        old_path = (
            root
            / f"paper_figure_recheck/seed{seed}/extension20to40/diagonal_band/diagonal_band_metrics.csv"
        )
        new_path = (
            root
            / f"boundary_rank_sweep_v1/evaluations/seed{seed}/boundary_rank48/diagonal_band/diagonal_band_metrics.csv"
        )
        if not new_path.is_file():
            continue
        records[(seed, "raw")], _, _ = heatmap_matrix(new_path, "raw")
        records[(seed, "old")], _, _ = heatmap_matrix(old_path, "J")
        records[(seed, "new")], _, _ = heatmap_matrix(new_path, "J")
        available_seeds.append(seed)
    if not available_seeds:
        return

    figure, axes = plt.subplots(
        len(available_seeds),
        3,
        figsize=(12.8, 3.45 * len(available_seeds)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for row_index, seed in enumerate(available_seeds):
        minimum, maximum = SEED_RANGES[seed]
        for column_index, (kind, title) in enumerate(
            (("raw", "raw"), ("old", "old easy-range J"), ("new", "new boundary J, rank 48"))
        ):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                records[(seed, kind)],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(-10.5, 10.5, 0.5, 500.5),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.axvline(0, color="white", alpha=0.65, linewidth=0.9)
            if kind == "new":
                axis.axhline(minimum, color="white", alpha=0.8, linewidth=0.9)
                axis.axhline(maximum, color="white", alpha=0.8, linewidth=0.9)
            axis.set_title(f"seed{seed} · {title}", fontsize=11)
            axis.set_xlabel("Relative depth d=t−n")
            if column_index == 0:
                axis.set_ylabel("Input length n")
    assert image is not None
    top = 0.88 if len(available_seeds) == 1 else 0.94
    bottom = 0.1 if len(available_seeds) == 1 else 0.07
    figure.subplots_adjust(
        left=0.07,
        right=0.89,
        bottom=bottom,
        top=top,
        wspace=0.13,
        hspace=0.24,
    )
    colorbar_axis = figure.add_axes([0.91, bottom, 0.015, top - bottom])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Parity-token accuracy")
    figure.suptitle(
        "Parity phase bands near the registered line: raw vs old J vs boundary-trained J",
        fontsize=15,
        y=0.985,
    )
    save_figure(figure, out_dir, "parity_j_heatmaps_raw_old_boundary")


def phase_column_transform(t: float, n: float) -> tuple[float, float]:
    """Map a square (t, n) cell into the paper's rhombus phase coordinates."""

    return t - n, PHASE_COLUMN_COMPRESSION * (t + n)


def phase_column_cell(t: int, n: int) -> list[tuple[float, float]]:
    return [
        phase_column_transform(t - 0.5, n - 0.5),
        phase_column_transform(t + 0.5, n - 0.5),
        phase_column_transform(t + 0.5, n + 0.5),
        phase_column_transform(t - 0.5, n + 0.5),
    ]


def draw_phase_basis(axis: plt.Axes) -> None:
    origin = np.asarray([0.0, -1.65])
    for vector, label, alignment in (
        (np.asarray([3.6, 3.6 * PHASE_COLUMN_COMPRESSION]), "$t$", "left"),
        (np.asarray([-3.6, 3.6 * PHASE_COLUMN_COMPRESSION]), "$n$", "right"),
    ):
        axis.annotate(
            "",
            xy=origin + vector,
            xytext=origin,
            arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "#222222"},
            annotation_clip=False,
        )
        axis.text(
            *(origin + 1.10 * vector),
            label,
            ha=alignment,
            va="center",
            fontsize=7,
            color="#222222",
        )
    axis.plot(
        [origin[0], origin[0]],
        [origin[1], origin[1] + 1.8],
        color="#222222",
        lw=0.55,
        alpha=0.55,
        clip_on=False,
    )


def draw_phase_training_boundary(
    axis: plt.Axes,
    length: int,
    *,
    linestyle: str,
    linewidth: float,
    alpha: float,
) -> None:
    offsets = np.asarray(
        [-PHASE_COLUMN_MAX_OFFSET - 0.5, PHASE_COLUMN_MAX_OFFSET + 0.5]
    )
    # At fixed n, t=n+(t-n), so v=c(t+n)=c(2n+(t-n)).
    vertical = PHASE_COLUMN_COMPRESSION * (2.0 * length + offsets)
    axis.plot(
        offsets,
        vertical,
        color="white",
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        zorder=4,
    )


def draw_phase_column(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    variant: str,
    title: str,
    controller_training_range: tuple[int, int] | None,
) -> PolyCollection:
    selected = frame[
        (frame["variant"] == variant)
        & (frame["relative_loop"].abs() <= PHASE_COLUMN_MAX_OFFSET)
        & (frame["length"].between(1, 500))
    ]
    polygons: list[list[tuple[float, float]]] = []
    values: list[float] = []
    for row in selected.itertuples(index=False):
        n = int(row.length)
        t = n + int(row.relative_loop)
        if t < 1:
            continue
        polygons.append(phase_column_cell(t, n))
        values.append(float(row.exact_match))

    collection = PolyCollection(
        polygons,
        array=np.asarray(values),
        cmap="viridis",
        norm=Normalize(0.0, 1.0),
        edgecolors="none",
        antialiased=False,
        rasterized=True,
    )
    axis.add_collection(collection)
    axis.axvline(0, color="white", lw=0.7, ls=(0, (1.2, 2.2)), alpha=0.9)
    if controller_training_range is None:
        draw_phase_training_boundary(
            axis, 20, linestyle=":", linewidth=0.65, alpha=0.82
        )
    else:
        for boundary in controller_training_range:
            draw_phase_training_boundary(
                axis, boundary, linestyle="-", linewidth=0.72, alpha=0.88
            )
        if 20 not in controller_training_range:
            draw_phase_training_boundary(
                axis, 20, linestyle=":", linewidth=0.55, alpha=0.65
            )
    draw_phase_basis(axis)

    axis.set_xlim(-PHASE_COLUMN_MAX_OFFSET - 1.0, PHASE_COLUMN_MAX_OFFSET + 1.0)
    axis.set_ylim(
        -2.35,
        PHASE_COLUMN_COMPRESSION * (2 * 500 + PHASE_COLUMN_MAX_OFFSET + 2),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=9, pad=4)
    axis.set_xticks([-10, 0, 10])
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)
    return collection


def plot_phase_columns(root: Path, out_dir: Path) -> None:
    """Compare raw with the best fully validated common J across three seeds."""

    figure, axes = plt.subplots(
        1,
        6,
        figsize=(11.8, 7.15),
        sharey=True,
        squeeze=False,
    )
    collection: PolyCollection | None = None
    group_axes: list[list[plt.Axes]] = []
    for seed in range(3):
        new_path = (
            root
            / f"boundary_rank_sweep_v1/evaluations/seed{seed}/boundary_rank48/diagonal_band/diagonal_band_metrics.csv"
        )
        if not new_path.is_file():
            raise FileNotFoundError(f"Missing phase-column data for seed{seed}")
        new_frame = pd.read_csv(new_path)
        panels = (
            (new_frame, "raw", "raw", None),
            (new_frame, "J", "best validated J", SEED_RANGES[seed]),
        )
        current_group: list[plt.Axes] = []
        for local_column, (frame, variant, title, training_range) in enumerate(panels):
            axis = axes[0, seed * 2 + local_column]
            collection = draw_phase_column(
                axis,
                frame,
                variant=variant,
                title=title,
                controller_training_range=training_range,
            )
            current_group.append(axis)
        group_axes.append(current_group)

    length_ticks = [1, 100, 200, 300, 400, 500]
    axes[0, 0].set_yticks(
        [2 * PHASE_COLUMN_COMPRESSION * length for length in length_ticks]
    )
    axes[0, 0].set_yticklabels([str(length) for length in length_ticks])
    axes[0, 0].set_ylabel(r"position along $t=n$ (input length $n$)")
    for axis in axes[0, 1:]:
        axis.tick_params(labelleft=False)

    figure.subplots_adjust(
        left=0.055,
        right=0.925,
        bottom=0.13,
        top=0.86,
        wspace=0.08,
    )
    for seed, current_group in enumerate(group_axes):
        left = current_group[0].get_position().x0
        right = current_group[-1].get_position().x1
        figure.text(
            0.5 * (left + right),
            0.905,
            f"seed{seed}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        if seed < 2:
            separator = 0.5 * (
                current_group[-1].get_position().x1
                + group_axes[seed + 1][0].get_position().x0
            )
            figure.add_artist(
                plt.Line2D(
                    [separator, separator],
                    [0.13, 0.91],
                    transform=figure.transFigure,
                    color="#d1d5db",
                    linewidth=0.7,
                )
            )

    figure.suptitle(
        "Parity phase columns near the registered line: raw vs best validated boundary J",
        fontsize=15,
        y=0.985,
    )
    figure.supxlabel(r"relative call offset $t-n$", y=0.07, fontsize=10)
    figure.text(
        0.49,
        0.025,
        "Each rhombus is one (n,t) evaluation cell; dotted line: backbone training limit n=20; solid lines: J training band.",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    assert collection is not None
    colorbar_axis = figure.add_axes([0.94, 0.13, 0.014, 0.73])
    colorbar = figure.colorbar(collection, cax=colorbar_axis)
    colorbar.set_label("Strict exact-match accuracy")
    colorbar.ax.tick_params(labelsize=7)
    save_figure(figure, out_dir, "parity_j_phase_columns_raw_best_boundary")


def collect_capacity(root: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(3):
        for label in ("boundary_rank48", "boundary_rank128", "boundary_dense"):
            path = root / f"boundary_rank_sweep_v1/seed{seed}/{label}/selection.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            rows.append(
                {
                    "seed": seed,
                    "controller": label,
                    "selected_update": payload["selected_update"],
                    "boundary_mean_accuracy": payload["boundary_mean_accuracy"],
                    "boundary_mean_ce": payload["boundary_mean_ce"],
                    "minimum_retention_delta": payload["minimum_retention_delta"],
                    "retention_pass": payload["retention_pass"],
                    "training_min_length": payload["logical_training_range"][0],
                    "training_max_length": payload["logical_training_range"][1],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "controller_capacity_summary.csv", index=False)
    return frame


def controller_matrix(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["controller_state_dict"]
    diagonal = state["diagonal"].detach().float().cpu().numpy()
    left = state["A"].detach().float().cpu().numpy()
    right = state["B"].detach().float().cpu().numpy()
    # Row-vector convention: J(h)=hD+(hA)B+b, hence W=D+AB.
    return np.diag(diagonal) + left @ right


def plot_controller_spectra(root: Path, out_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 7.6), sharex=True)
    for seed in range(3):
        paths = {
            "old easy-range J": (
                root / f"controllers/seed{seed}/extension20to40/controller.pt",
                COLORS["old"],
            ),
            "new boundary J": (
                root
                / f"boundary_rank_sweep_v1/seed{seed}/boundary_rank48/best_controller.pt",
                COLORS["new"],
            ),
        }
        for label, (path, color) in paths.items():
            matrix = controller_matrix(path)
            delta = matrix - np.eye(matrix.shape[0])
            singular = np.linalg.svd(matrix, compute_uv=False)
            delta_singular = np.linalg.svd(delta, compute_uv=False)
            index = np.arange(1, len(singular) + 1)
            axes[0, seed].plot(index, singular, color=color, linewidth=1.7, label=label)
            axes[1, seed].plot(
                index,
                np.maximum(delta_singular, 1e-12),
                color=color,
                linewidth=1.7,
                label=label,
            )
            for order, value in enumerate(singular, start=1):
                records.append(
                    {
                        "seed": seed,
                        "controller": label,
                        "matrix": "W",
                        "singular_order": order,
                        "singular_value": float(value),
                    }
                )
            for order, value in enumerate(delta_singular, start=1):
                records.append(
                    {
                        "seed": seed,
                        "controller": label,
                        "matrix": "delta_W",
                        "singular_order": order,
                        "singular_value": float(value),
                    }
                )
        axes[0, seed].axhline(1.0, color="#6b7280", linestyle=":", linewidth=0.9)
        axes[0, seed].set_title(f"seed{seed}")
        axes[0, seed].set_ylabel("Singular value of W" if seed == 0 else "")
        axes[1, seed].set_ylabel("Singular value of ΔW" if seed == 0 else "")
        axes[1, seed].set_xlabel("Singular-value order")
        axes[1, seed].set_yscale("log")
        for row in range(2):
            axes[row, seed].grid(alpha=0.22, linewidth=0.6)
        axes[0, seed].legend(frameon=False, fontsize=9, loc="best")
    figure.suptitle(
        "Controller spectrum changes when J is trained where the backbone actually fails",
        fontsize=15,
        y=1.01,
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "parity_j_spectrum_old_vs_boundary")
    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "controller_spectrum_old_vs_boundary.csv", index=False)
    return frame


def ridge_drift(path: Path, variant: str) -> tuple[float, float, float]:
    payload = json.loads(path.read_text())
    record = payload["metrics"]["exact_match"][variant]
    drift = float(record["phase_drift"]["loops_per_100_tokens"])
    ci = record["bootstrap"]["slope_ci95"]
    return drift, 100.0 * float(ci[0]), 100.0 * float(ci[1])


def plot_ridge_drift(root: Path, out_dir: Path) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    for seed in range(3):
        old_path = (
            root
            / f"paper_figure_recheck/seed{seed}/extension20to40/diagonal_band/"
            "ridge_slope/parity_ridge_slope_analysis.json"
        )
        supported = (
            root
            / f"boundary_rank_sweep_v1/evaluations/seed{seed}/boundary_rank48/"
            "ridge_supported/parity_ridge_slope_analysis.json"
        )
        ood = (
            root
            / f"boundary_rank_sweep_v1/evaluations/seed{seed}/boundary_rank48/"
            "ridge_ood/parity_ridge_slope_analysis.json"
        )
        if not (supported.is_file() and ood.is_file()):
            continue
        for condition, path, variant, fit_range in (
            ("raw · new OOD range", ood, "raw", "strict OOD"),
            ("old easy-range J · n=11–490", old_path, "J", "old full range"),
            ("new boundary J · train band", supported, "J", "J support"),
            ("new boundary J · strict OOD", ood, "J", "strict OOD"),
        ):
            drift, lower, upper = ridge_drift(path, variant)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "fit_range": fit_range,
                    "loops_per_100_tokens": drift,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "ridge_drift_old_vs_boundary.csv", index=False)
    conditions = [
        "raw · new OOD range",
        "old easy-range J · n=11–490",
        "new boundary J · train band",
        "new boundary J · strict OOD",
    ]
    colors = [COLORS["raw"], COLORS["old"], "#5eead4", COLORS["new"]]
    available_seeds = sorted(int(value) for value in frame["seed"].unique())
    figure, axes = plt.subplots(
        1,
        len(available_seeds),
        figsize=(4.95 * len(available_seeds), 4.5),
        sharey=True,
        squeeze=False,
    )
    for column, seed in enumerate(available_seeds):
        axis = axes[0, column]
        selected = frame[frame["seed"] == seed].set_index("condition").loc[conditions]
        values = selected["loops_per_100_tokens"].to_numpy(float)
        lower = values - selected["ci95_lower"].to_numpy(float)
        upper = selected["ci95_upper"].to_numpy(float) - values
        x = np.arange(len(conditions))
        axis.bar(x, values, color=colors, width=0.72)
        axis.errorbar(
            x,
            values,
            yerr=np.vstack((lower, upper)),
            fmt="none",
            ecolor="#111827",
            elinewidth=1.0,
            capsize=2.5,
        )
        axis.axhline(0, color="#111827", linewidth=0.9)
        axis.set_xticks(x, ["raw\nOOD", "old J\n11–490", "new J\ntrain band", "new J\nOOD"])
        axis.set_title(f"seed{seed}")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        if column == 0:
            axis.set_ylabel("Phase-ridge drift (calls per 100 tokens)")
    figure.suptitle(
        "Boundary training can flatten the registered phase locally without locking it forever",
        fontsize=15,
        y=1.02,
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "parity_j_ridge_drift_old_vs_boundary")
    return frame


def write_manifest(out_dir: Path, generated: list[Path]) -> None:
    payload = {
        "status": "complete",
        "figures": [str(path) for path in generated if path.is_file()],
        "claim_boundary": (
            "Boundary-band repair tests controller learnability and finite phase "
            "registration. It does not by itself establish permanent OOD phase locking "
            "or identify the backbone's full Parity circuit."
        ),
    }
    (out_dir / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_signal(args.root, args.out_dir)
    plot_far_horizon(args.root, args.out_dir)
    plot_heatmaps(args.root, args.out_dir)
    plot_phase_columns(args.root, args.out_dir)
    collect_capacity(args.root, args.out_dir)
    plot_controller_spectra(args.root, args.out_dir)
    plot_ridge_drift(args.root, args.out_dir)
    generated = sorted(args.out_dir.glob("*.png")) + sorted(args.out_dir.glob("*.pdf"))
    write_manifest(args.out_dir, generated)


if __name__ == "__main__":
    main()
