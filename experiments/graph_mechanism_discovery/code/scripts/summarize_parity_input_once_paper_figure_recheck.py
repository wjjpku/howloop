#!/usr/bin/env python3
"""Create cross-seed figures for the input-once Parity paper-figure audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


RAW = "#111827"
STRICT = "#2563eb"
EXTENSION = "#0f9d58"
ORANGE = "#e68a00"
RED = "#d62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(figure: plt.Figure, out_dir: Path, stem: str) -> None:
    figure.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def heatmap_rows(root: Path, seed: int, label: str) -> list[dict[str, str]]:
    return read_csv(
        root / f"seed{seed}" / label / "loop_depth_heatmap" / "loop_depth_metrics.csv"
    )


def select_heatmap(
    rows: list[dict[str, str]], variant: str, metric: str
) -> tuple[np.ndarray, list[int], list[int]]:
    selected = [row for row in rows if row["variant"] == variant]
    lengths = sorted({int(row["length"]) for row in selected})
    loops = sorted({int(row["loop"]) for row in selected})
    length_index = {value: index for index, value in enumerate(lengths)}
    loop_index = {value: index for index, value in enumerate(loops)}
    matrix = np.full((len(lengths), len(loops)), np.nan)
    for row in selected:
        matrix[length_index[int(row["length"])], loop_index[int(row["loop"])]] = float(
            row[metric]
        )
    return matrix, lengths, loops


def plot_heatmaps(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(14.5, 11.0), sharex=True, sharey=True)
    image = None
    columns = (
        ("strict_id10to20", "raw", "raw backbone"),
        ("strict_id10to20", "J", r"strict-ID $J$ (trained 10--20)"),
        ("extension20to40", "J", r"extension $J$ (trained 20--40)"),
    )
    for seed in range(3):
        cache: dict[str, list[dict[str, str]]] = {}
        for column, (label, variant, title) in enumerate(columns):
            cache.setdefault(label, heatmap_rows(root, seed, label))
            matrix, lengths, loops = select_heatmap(cache[label], variant, "exact_match")
            axis = axes[seed, column]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(loops[0] - 0.5, loops[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            diagonal_max = min(lengths[-1], loops[-1])
            axis.plot([1, diagonal_max], [1, diagonal_max], "w--", lw=0.8)
            axis.axhline(20.5, color="white", ls=":", lw=0.9)
            if seed == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"seed {seed}\ninput length $n$")
            if seed == 2:
                axis.set_xlabel("executed calls $t$")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    colorbar.set_label("strict exact-match accuracy")
    figure.suptitle(
        "Input-once Parity: loop-depth maps (dotted: train boundary; dashed: $t=n$)",
        fontsize=14,
    )
    save(figure, out_dir, "parity_input_once_heatmaps_multiseed")


def plot_centered_heatmaps(root: Path, out_dir: Path) -> None:
    """Show the same data in registered coordinates d=t-n."""
    figure, axes = plt.subplots(3, 3, figsize=(12.2, 10.2), sharex=True, sharey=True)
    columns = (
        ("strict_id10to20", "raw", "raw backbone"),
        ("strict_id10to20", "J", "strict-ID J"),
        ("extension20to40", "J", "extension J"),
    )
    offsets = list(range(-10, 13))
    image = None
    for seed in range(3):
        cache: dict[str, list[dict[str, str]]] = {}
        for column, (label, variant, title) in enumerate(columns):
            cache.setdefault(label, heatmap_rows(root, seed, label))
            selected = [row for row in cache[label] if row["variant"] == variant]
            lengths = sorted({int(row["length"]) for row in selected})
            matrix = np.full((len(lengths), len(offsets)), np.nan)
            lookup = {
                (int(row["length"]), int(row["relative_loop"])): float(row["exact_match"])
                for row in selected
            }
            for length_index, length in enumerate(lengths):
                for offset_index, offset in enumerate(offsets):
                    matrix[length_index, offset_index] = lookup.get((length, offset), np.nan)
            axis = axes[seed, column]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(offsets[0] - 0.5, offsets[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.axvline(0, color="white", ls="--", lw=0.9)
            axis.axhline(20.5, color="white", ls=":", lw=0.9)
            if seed == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"seed {seed}\ninput length $n$")
            if seed == 2:
                axis.set_xlabel("relative calls $d=t-n$")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    colorbar.set_label("strict exact-match accuracy")
    figure.suptitle("Input-once Parity: registered local phase bands through length 100")
    save(figure, out_dir, "parity_input_once_centered_heatmaps_multiseed")


def variant_curve(path: Path, variant: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(path) if row["variant"] == variant]
    rows.sort(key=lambda row: int(row["length"]))
    return (
        np.asarray([int(row["length"]) for row in rows]),
        np.asarray([float(row["exact_match"]) for row in rows]),
    )


def plot_far_horizon(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), sharex=True, sharey=True)
    summary: list[dict[str, Any]] = []
    for seed, axis in enumerate(axes):
        strict_path = root / f"seed{seed}/strict_id10to20/far_horizon/horizon.csv"
        extension_path = root / f"seed{seed}/extension20to40/far_horizon/horizon.csv"
        for path, variant, label, color in (
            (strict_path, "raw", "raw", RAW),
            (strict_path, "full", "strict-ID J", STRICT),
            (extension_path, "full", "extension J", EXTENSION),
        ):
            x, y = variant_curve(path, variant)
            axis.plot(x, y, marker="o", ms=3.2, lw=1.6, label=label, color=color)
            for length, accuracy in zip(x, y, strict=True):
                summary.append(
                    {
                        "seed": seed,
                        "condition": label,
                        "length": int(length),
                        "exact_match": float(accuracy),
                    }
                )
        axis.axvline(20.5, color="#6b7280", ls=":", lw=0.8)
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("length $n$ and registered calls $t=n$")
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("strict exact-match accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity: registered-readout length generalization")
    save(figure, out_dir, "parity_input_once_far_horizon_multiseed")
    write_csv(out_dir / "far_horizon_multiseed.csv", summary)


def plot_four_phase(root: Path, out_dir: Path) -> None:
    metrics = (
        ("polar_rotation_period_calls", "fitted local period (calls)"),
        ("transition_r2", "2D transition $R^2$"),
        ("shared_phase_plane_energy_fraction", "energy in shared 2D plane"),
        ("readout_direction_in_phase_plane_energy_fraction", "readout energy in 2D plane"),
        ("four_step_alignment_cosine", "four-step alignment cosine"),
        ("two_step_antialignment_cosine", "two-step anti-alignment cosine"),
    )
    rows: list[dict[str, Any]] = []
    conditions = (
        ("strict_id10to20", "raw", "raw"),
        ("strict_id10to20", "controlled", "strict-ID J"),
        ("extension20to40", "controlled", "extension J"),
    )
    for seed in range(3):
        payloads: dict[str, dict[str, Any]] = {}
        for label, variant, condition in conditions:
            payloads.setdefault(
                label,
                json.loads(
                    (root / f"seed{seed}/{label}/four_phase/summary.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            record = payloads[label]["dynamics"][variant]["heldout_evaluation"]
            phase_origin = next(
                item
                for item in payloads[label]["phase_origin_locking"]
                if item["variant"] == variant
            )
            row: dict[str, Any] = {
                "seed": seed,
                "condition": condition,
                "phase_origin_slope_per_length": phase_origin[
                    "unwrapped_angle_slope_per_length"
                ],
            }
            for key, _ in metrics:
                row[key] = record[key]
            rows.append(row)

    figure, axes = plt.subplots(
        2, 3, figsize=(13.6, 8.4), constrained_layout=True
    )
    colors = {"raw": RAW, "strict-ID J": STRICT, "extension J": EXTENSION}
    offsets = {"raw": -0.18, "strict-ID J": 0.0, "extension J": 0.18}
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        for condition in colors:
            selected = [row for row in rows if row["condition"] == condition]
            axis.scatter(
                np.asarray([row["seed"] for row in selected]) + offsets[condition],
                [row[metric] for row in selected],
                s=42,
                color=colors[condition],
                label=condition,
                zorder=3,
            )
        if metric == "polar_rotation_period_calls":
            axis.axhline(4.0, color="#6b7280", ls="--", lw=0.8)
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("backbone seed")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity: hidden four-phase diagnostics")
    save(figure, out_dir, "parity_input_once_four_phase_multiseed")
    write_csv(out_dir / "four_phase_multiseed.csv", rows)


def local_stat(x: np.ndarray, values: np.ndarray, half_width: float, kind: str) -> np.ndarray:
    result = []
    for center in x:
        selected = values[np.abs(x - center) <= half_width]
        result.append(selected.mean() if kind == "mean" else selected.min())
    return np.asarray(result)


def plot_long_horizon(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.8), sharex=True, sharey=True)
    summaries: list[dict[str, Any]] = []
    for seed, axis in enumerate(axes):
        path = root / f"seed{seed}/extension20to40/long_horizon/horizon.csv"
        for variant, label, color in (("raw", "raw", RAW), ("full", "extension J", EXTENSION)):
            x, y = variant_curve(path, variant)
            trend = local_stat(x, y, 50.0, "mean")
            axis.plot(x, trend, color=color, lw=1.8, label=label)
            axis.plot(x, local_stat(x, y, 25.0, "min"), color=color, lw=0.9, ls=":")
            record: dict[str, Any] = {
                "seed": seed,
                "condition": label,
                "exact_match_at_100": float(y[np.flatnonzero(x == 100)[0]]),
                "exact_match_at_1000": float(y[np.flatnonzero(x == 1000)[0]]),
            }
            for start, end in ((100, 300), (310, 600), (620, 1000)):
                chosen = y[(x >= start) & (x <= end)]
                record[f"mean_exact_match_{start}_{end}"] = float(chosen.mean())
            for threshold in (0.9, 0.75, 0.5):
                hits = np.flatnonzero(trend >= threshold)
                record[f"last_length_local_mean_ge_{threshold}"] = (
                    int(x[hits[-1]]) if hits.size else None
                )
            summaries.append(record)
        axis.axvline(20.5, color="#6b7280", ls=":", lw=0.8)
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("length $n$ and calls $t=n$")
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("local exact-match trend / floor")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity: dense long-horizon registered readout")
    save(figure, out_dir, "parity_input_once_long_horizon_multiseed")
    write_csv(out_dir / "long_horizon_multiseed_summary.csv", summaries)


def plot_diagonal_bands(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), sharex=True, sharey=True)
    image = None
    for seed in range(3):
        rows = read_csv(
            root
            / f"seed{seed}/extension20to40/diagonal_band/diagonal_band_metrics.csv"
        )
        lengths = sorted({int(row["length"]) for row in rows})
        offsets = sorted({int(row["relative_loop"]) for row in rows})
        for row_index, variant in enumerate(("raw", "J")):
            matrix = np.full((len(lengths), len(offsets)), np.nan)
            for row in rows:
                if row["variant"] != variant:
                    continue
                matrix[lengths.index(int(row["length"])), offsets.index(int(row["relative_loop"]))] = float(
                    row["exact_match"]
                )
            axis = axes[row_index, seed]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(offsets[0] - 0.5, offsets[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.axvline(0, color="white", ls="--", lw=0.8)
            axis.axhline(20.5, color="white", ls=":", lw=0.8)
            if row_index == 0:
                axis.set_title(f"seed {seed}")
            if seed == 0:
                axis.set_ylabel(("raw" if row_index == 0 else "extension J") + "\nlength $n$")
            if row_index == 1:
                axis.set_xlabel("relative calls $d=t-n$")
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
    colorbar.set_label("strict exact-match accuracy")
    figure.suptitle("Input-once Parity: diagonal phase bands through length 500")
    save(figure, out_dir, "parity_input_once_diagonal_bands_multiseed")


def plot_ridge_slopes(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.9), sharex=True, sharey=True)
    summaries: list[dict[str, Any]] = []
    for seed, axis in enumerate(axes):
        ridge_dir = root / f"seed{seed}/extension20to40/diagonal_band/ridge_slope"
        rows = [
            row
            for row in read_csv(ridge_dir / "parity_ridge_phase_by_length.csv")
            if row["metric"] == "exact_match"
        ]
        payload = json.loads(
            (ridge_dir / "parity_ridge_slope_analysis.json").read_text(
                encoding="utf-8"
            )
        )["metrics"]["exact_match"]
        for variant, label, color in (("raw", "raw", RAW), ("J", "extension J", EXTENSION)):
            selected = sorted(
                (row for row in rows if row["variant"] == variant),
                key=lambda row: float(row["length_n"]),
            )
            x = np.asarray([float(row["length_n"]) for row in selected])
            y = np.asarray(
                [float(row["unwrapped_offset_t_minus_n"]) for row in selected]
            )
            equation = payload[variant]["fitted_equation_t_from_n"]
            drift = float(equation["t_per_n_slope"]) - 1.0
            intercept = float(equation["intercept"])
            axis.scatter(x, y, s=3.0, alpha=0.22, color=color)
            axis.plot(
                x,
                drift * x + intercept,
                lw=1.7,
                color=color,
                label=f"{label}: {100 * drift:+.2f} calls/100 tokens",
            )
            summaries.append(
                {
                    "seed": seed,
                    "condition": label,
                    "t_per_n_slope": float(equation["t_per_n_slope"]),
                    "phase_drift_calls_per_100_tokens": 100.0 * drift,
                    "intercept": intercept,
                    "fit_r2": float(payload[variant]["fit_quality"]["r_squared"]),
                }
            )
        axis.axhline(0, color="#6b7280", ls="--", lw=0.8)
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("input length $n$")
        axis.grid(axis="y", alpha=0.18)
        axis.legend(frameon=False, fontsize=7.5)
    axes[0].set_ylabel("unwrapped correct-ridge offset $t-n$")
    figure.suptitle("Input-once Parity: phase-ridge drift through length 500")
    save(figure, out_dir, "parity_input_once_ridge_slopes_multiseed")
    write_csv(out_dir / "ridge_slopes_multiseed.csv", summaries)


def plot_phase_summary_per_seed(root: Path, out_dir: Path) -> None:
    """Rebuild the paper's local-surface plus long diagonal-band composition."""
    for seed in range(3):
        local_rows = heatmap_rows(root, seed, "extension20to40")
        band_rows = read_csv(
            root
            / f"seed{seed}/extension20to40/diagonal_band/diagonal_band_metrics.csv"
        )
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(11.2, 7.7),
            gridspec_kw={"height_ratios": (1.0, 1.15)},
        )
        image = None
        for column, variant in enumerate(("raw", "J")):
            matrix, lengths, loops = select_heatmap(local_rows, variant, "exact_match")
            axis = axes[0, column]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(loops[0] - 0.5, loops[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            diagonal_max = min(lengths[-1], loops[-1])
            axis.plot([1, diagonal_max], [1, diagonal_max], "w--", lw=0.8)
            axis.axhline(20.5, color="white", ls=":", lw=0.8)
            axis.set_title(("A  Raw recurrence" if column == 0 else "B  Extension-J recurrence"), loc="left", fontweight="bold")
            axis.set_xlabel("executed calls $t$")
            axis.set_ylabel("input length $n$")

            band_lengths = sorted({int(row["length"]) for row in band_rows})
            offsets = sorted({int(row["relative_loop"]) for row in band_rows})
            band = np.full((len(band_lengths), len(offsets)), np.nan)
            for row in band_rows:
                if row["variant"] != variant:
                    continue
                band[
                    band_lengths.index(int(row["length"])),
                    offsets.index(int(row["relative_loop"])),
                ] = float(row["exact_match"])
            axis = axes[1, column]
            image = axis.imshow(
                band,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(offsets[0] - 0.5, offsets[-1] + 0.5, band_lengths[0] - 0.5, band_lengths[-1] + 0.5),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.axvline(0, color="white", ls="--", lw=0.8)
            axis.axhline(20.5, color="white", ls=":", lw=0.8)
            axis.set_title(("C  Raw phase band" if column == 0 else "D  Extension-J phase band"), loc="left", fontweight="bold")
            axis.set_xlabel("relative calls $d=t-n$")
            axis.set_ylabel("input length $n$")
        assert image is not None
        colorbar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
        colorbar.set_label("strict exact-match accuracy")
        figure.suptitle(f"Input-once Parity phase summary — backbone seed {seed}")
        save(figure, out_dir, f"parity_input_once_phase_summary_seed{seed}")


def plot_circuit(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), sharey="row")
    summaries: list[dict[str, Any]] = []
    for seed in range(3):
        circuit = root / f"seed{seed}/raw_circuit"
        transfer = [
            row
            for row in read_csv(circuit / "source_answer_transfer.csv")
            if int(row["length"]) == 20 and int(row["bit_position"]) == 0
        ]
        for site, label, color in (
            ("source_token", "patch source", ORANGE),
            ("answer_token", "patch answer", STRICT),
        ):
            selected = sorted(
                (row for row in transfer if row["site"] == site),
                key=lambda row: int(row["loop"]),
            )
            axes[0, seed].plot(
                [int(row["loop"]) for row in selected],
                [float(row["recovery"]) for row in selected],
                marker="o",
                ms=2.5,
                lw=1.3,
                color=color,
                label=label,
            )
        skip = [
            row
            for row in read_csv(circuit / "mlp_skip.csv")
            if int(row["skipped_loop"]) > 0
        ]
        source_by_loop = {
            int(row["loop"]): float(row["recovery"])
            for row in transfer
            if row["site"] == "source_token"
        }
        answer_by_loop = {
            int(row["loop"]): float(row["recovery"])
            for row in transfer
            if row["site"] == "answer_token"
        }
        crossover = min(
            loop
            for loop in sorted(source_by_loop)
            if answer_by_loop[loop] >= source_by_loop[loop]
        )
        read_t = [float(row["accuracy"]) for row in skip if int(row["eval_loop"]) == 20]
        read_t1 = [float(row["accuracy"]) for row in skip if int(row["eval_loop"]) == 21]
        summaries.append(
            {
                "seed": seed,
                "source_answer_recovery_crossover_call": crossover,
                "mean_accuracy_after_skip_read_T": float(np.mean(read_t)),
                "mean_accuracy_after_skip_read_T_plus_1": float(np.mean(read_t1)),
                "fraction_skips_perfect_at_T_plus_1": float(np.mean(np.asarray(read_t1) == 1.0)),
            }
        )
        for eval_loop, label, color in (
            (20, "read at T", RED),
            (21, "read at T+1", EXTENSION),
        ):
            selected = sorted(
                (row for row in skip if int(row["eval_loop"]) == eval_loop),
                key=lambda row: int(row["skipped_loop"]),
            )
            axes[1, seed].plot(
                [int(row["skipped_loop"]) for row in selected],
                [float(row["accuracy"]) for row in selected],
                marker="o",
                ms=2.5,
                lw=1.3,
                color=color,
                label=label,
            )
        axes[0, seed].set_title(f"seed {seed}")
        axes[0, seed].set_xlabel("patch call")
        axes[1, seed].set_xlabel("skipped MLP call")
        axes[0, seed].grid(axis="y", alpha=0.18)
        axes[1, seed].grid(axis="y", alpha=0.18)
    axes[0, 0].set_ylabel("normalized causal recovery")
    axes[1, 0].set_ylabel("parity-token accuracy")
    axes[0, -1].legend(frameon=False, fontsize=8)
    axes[1, -1].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity: source-to-answer transfer and MLP clock")
    save(figure, out_dir, "parity_input_once_circuit_multiseed")
    write_csv(out_dir / "circuit_multiseed_summary.csv", summaries)


def plot_head_ablation(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.8), sharex=True, sharey=True)
    summaries: list[dict[str, Any]] = []
    for seed, axis in enumerate(axes):
        rows = [
            row
            for row in read_csv(
                root / f"seed{seed}/raw_circuit/attention_head_ablation.csv"
            )
            if int(row["ablated_head"]) >= 0
        ]
        for eval_loop, label, color in ((20, "read at T", RED), (21, "read at T+1", EXTENSION)):
            selected = sorted(
                (row for row in rows if int(row["eval_loop"]) == eval_loop),
                key=lambda row: int(row["ablated_head"]),
            )
            axis.plot(
                [int(row["ablated_head"]) for row in selected],
                [float(row["accuracy"]) for row in selected],
                marker="o",
                ms=2.0,
                lw=1.0,
                color=color,
                label=label,
            )
            weakest = min(selected, key=lambda row: float(row["accuracy"]))
            summaries.append(
                {
                    "seed": seed,
                    "eval_loop": eval_loop,
                    "weakest_head": int(weakest["ablated_head"]),
                    "minimum_accuracy": float(weakest["accuracy"]),
                }
            )
        axis.set_title(f"seed {seed}")
        axis.set_xlabel("head ablated at every call")
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("parity-token accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity: cumulative single-head ablation")
    save(figure, out_dir, "parity_input_once_head_ablation_multiseed")
    write_csv(out_dir / "head_ablation_multiseed_summary.csv", summaries)


def plot_head26_replications(root: Path, out_dir: Path) -> None:
    paths = [root / "seed2/raw_circuit/attention_head_ablation.csv"]
    paths.extend(
        sorted(
            (root / "seed2/raw_circuit_replication").glob(
                "analysis_seed*/attention_head_ablation.csv"
            )
        )
    )
    summaries: list[dict[str, Any]] = []
    for path in paths:
        rows = read_csv(path)
        manifest_path = path.parent / "manifest.json"
        analysis_seed = (
            json.loads(manifest_path.read_text(encoding="utf-8"))["analysis_seed"]
            if manifest_path.exists()
            else path.parent.name
        )
        for eval_loop in (20, 21):
            baseline = next(
                row
                for row in rows
                if int(row["ablated_head"]) == -1
                and int(row["eval_loop"]) == eval_loop
            )
            head26 = next(
                row
                for row in rows
                if int(row["ablated_head"]) == 26
                and int(row["eval_loop"]) == eval_loop
            )
            summaries.append(
                {
                    "analysis_seed": analysis_seed,
                    "eval_loop": eval_loop,
                    "baseline_accuracy": float(baseline["accuracy"]),
                    "head26_ablated_accuracy": float(head26["accuracy"]),
                    "accuracy_change": float(head26["accuracy"])
                    - float(baseline["accuracy"]),
                }
            )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for axis, eval_loop in zip(axes, (20, 21), strict=True):
        selected = [row for row in summaries if row["eval_loop"] == eval_loop]
        x = np.arange(len(selected))
        width = 0.36
        axis.bar(
            x - width / 2,
            [row["baseline_accuracy"] for row in selected],
            width,
            color=RAW,
            label="no ablation",
        )
        axis.bar(
            x + width / 2,
            [row["head26_ablated_accuracy"] for row in selected],
            width,
            color=RED,
            label="ablate head 26 every call",
        )
        axis.set_xticks(x, [f"data {index}" for index in range(len(selected))])
        axis.set_title("read at T" if eval_loop == 20 else "read at T+1")
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("parity-token accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity seed 2: head-26 replication across data seeds")
    save(figure, out_dir, "parity_input_once_head26_replications")
    write_csv(out_dir / "head26_replications.csv", summaries)


def plot_j_svd(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), sharey=True)
    summaries: list[dict[str, Any]] = []
    labels = ("raw_no_J", "full_J", "top4", "delete_top4")
    display = ("raw", "full J", "top-4", "delete top-4")
    for seed, axis in enumerate(axes):
        rows = [
            row
            for row in read_csv(
                root / f"seed{seed}/extension20to40/j_svd/variant_summary.csv"
            )
            if int(row["length"]) == 100
        ]
        indexed = {row["variant"]: row for row in rows}
        random_rows = [
            row for row in rows if row["variant"].startswith("random_output_top4")
        ]
        x = np.arange(5)
        values_t = [float(indexed[label]["target_exact_match"]) for label in labels]
        values_t1 = [
            float(indexed[label]["target_plus_one_exact_match"]) for label in labels
        ]
        values_t.append(float(np.mean([float(row["target_exact_match"]) for row in random_rows])))
        values_t1.append(
            float(np.mean([float(row["target_plus_one_exact_match"]) for row in random_rows]))
        )
        for name, value_t, value_t1 in zip(
            (*display, "random write"), values_t, values_t1, strict=True
        ):
            summaries.append(
                {
                    "seed": seed,
                    "condition": name,
                    "length": 100,
                    "accuracy_T": value_t,
                    "accuracy_T_plus_1": value_t1,
                }
            )
        axis.bar(x - 0.18, values_t, 0.36, color=STRICT, label="T")
        axis.bar(x + 0.18, values_t1, 0.36, color=ORANGE, label="T+1")
        axis.set_xticks(x, (*display, "random write"), rotation=24, ha="right")
        axis.set_title(f"seed {seed}")
        axis.set_ylim(0.0, 1.03)
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("strict exact-match accuracy at length 100")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity: causal SVD-component interventions")
    save(figure, out_dir, "parity_input_once_j_svd_multiseed")
    write_csv(out_dir / "j_svd_multiseed_summary.csv", summaries)


def plot_j_hidden_effect(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), sharex=True, sharey=True)
    ranks = (1, 2, 4, 8, 16, 32, 48)
    summaries: list[dict[str, Any]] = []
    for seed, axis in enumerate(axes):
        rows = [
            row
            for row in read_csv(
                root / f"seed{seed}/extension20to40/j_hidden_effect/step_metrics.csv"
            )
            if row["scope"] == "answer_tokens"
            and int(row["output_step"]) == int(row["length"])
        ]
        rows.sort(key=lambda row: int(row["length"]))
        for index, row in enumerate(rows):
            length = int(row["length"])
            energy = [float(row[f"top{rank}_weight_effect_energy"]) for rank in ranks]
            axis.plot(
                ranks,
                energy,
                marker="o",
                ms=3.0,
                lw=1.2,
                alpha=0.45 + 0.12 * index,
                label=f"L{length}",
            )
            summaries.append(
                {
                    "seed": seed,
                    "length": length,
                    "correction_to_source": float(row["correction_to_source"]),
                    "top1_on_state_energy": energy[0],
                    "top2_on_state_energy": energy[1],
                    "top4_on_state_energy": energy[2],
                    "top8_on_state_energy": energy[3],
                    "post_F_gain_over_correction": float(row["post_F_gain_over_correction"]),
                }
            )
        axis.axvline(4, color="#6b7280", ls="--", lw=0.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ranks, [str(rank) for rank in ranks])
        axis.set_ylim(0.0, 1.03)
        axis.set_title(f"seed {seed}")
        axis.set_xlabel(r"rank retained from SVD of $J-I$")
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("fraction of correction energy on answer states")
    axes[-1].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle("Input-once Parity: SVD modes measured on real hidden states")
    save(figure, out_dir, "parity_input_once_j_on_state_svd_energy")
    write_csv(out_dir / "j_on_state_svd_energy_multiseed.csv", summaries)


def plot_controller_spectra(root: Path, out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    rows: list[dict[str, Any]] = []
    for label, title, color in (
        ("strict_id10to20", "strict-ID J", STRICT),
        ("extension20to40", "extension J", EXTENSION),
    ):
        for seed in range(3):
            payload = torch.load(
                root.parent / "controllers" / f"seed{seed}" / label / "controller.pt",
                map_location="cpu",
                weights_only=False,
            )
            state = payload["controller_state_dict"]
            diagonal = state["diagonal"].float()
            a = state["A"].float()
            b_factor = state["B"].float()
            matrix = torch.diag(diagonal) + a @ b_factor
            delta = matrix - torch.eye(matrix.shape[0])
            singular = torch.linalg.svdvals(delta).numpy()
            legend_label = title if seed == 2 else None
            axes[0].semilogy(
                np.arange(1, len(singular) + 1),
                singular,
                color=color,
                alpha=0.35 + 0.25 * seed,
                lw=1.1,
                label=legend_label,
            )
            axes[1].hist(
                torch.linalg.svdvals(matrix).numpy(),
                bins=32,
                histtype="step",
                color=color,
                alpha=0.35 + 0.25 * seed,
                lw=1.1,
                label=legend_label,
            )
            rows.append(
                {
                    "seed": seed,
                    "condition": title,
                    "delta_frobenius": float(torch.linalg.matrix_norm(delta).item()),
                    "delta_operator": float(torch.linalg.matrix_norm(delta, 2).item()),
                    "bias_norm": float(state["bias"].float().norm().item()),
                    "matrix_singular_min": float(torch.linalg.svdvals(matrix).min().item()),
                    "matrix_singular_max": float(torch.linalg.svdvals(matrix).max().item()),
                }
            )
    axes[0].set(xlabel="rank", ylabel=r"singular value of $J-I$", title=r"Correction spectrum $J-I$")
    axes[1].set(xlabel=r"singular value of $J$", ylabel="count", title=r"Near-identity spectrum of $J$")
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Input-once Parity controllers across three backbones")
    save(figure, out_dir, "parity_input_once_controller_spectra")
    write_csv(out_dir / "controller_spectrum_summary.csv", rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    plot_heatmaps(root, out_dir)
    plot_centered_heatmaps(root, out_dir)
    plot_far_horizon(root, out_dir)
    plot_four_phase(root, out_dir)
    plot_long_horizon(root, out_dir)
    plot_diagonal_bands(root, out_dir)
    plot_ridge_slopes(root, out_dir)
    plot_phase_summary_per_seed(root, out_dir)
    plot_circuit(root, out_dir)
    plot_head_ablation(root, out_dir)
    plot_head26_replications(root, out_dir)
    plot_j_svd(root, out_dir)
    plot_j_hidden_effect(root, out_dir)
    plot_controller_spectra(root, out_dir)


if __name__ == "__main__":
    main()
