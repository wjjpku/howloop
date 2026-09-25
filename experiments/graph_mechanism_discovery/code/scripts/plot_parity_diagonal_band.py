"""Merge Parity diagonal-band shards and make focused raw-versus-J figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("exact_match", "strict exact match"),
    ("parity_token_accuracy", "parity-token accuracy"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--half-width", type=int, default=10)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = dict(raw)
                for key in ("length", "target_loop", "loop", "relative_loop", "examples"):
                    row[key] = int(row[key])
                for key in (
                    "exact_match",
                    "parity_token_accuracy",
                    "correct_parity_probability",
                    "mean_parity_margin",
                ):
                    row[key] = float(row[key])
                key = (str(row["variant"]), int(row["length"]), int(row["loop"]))
                if key in lookup:
                    raise ValueError(f"duplicate metric cell: {key}")
                lookup[key] = row
    return sorted(lookup.values(), key=lambda row: (row["variant"], row["length"], row["loop"]))


def matrix_for(
    rows: list[dict[str, Any]], variant: str, metric: str, lengths: list[int], deltas: list[int]
) -> np.ndarray:
    lookup = {
        (int(row["length"]), int(row["relative_loop"])): float(row[metric])
        for row in rows
        if row["variant"] == variant
    }
    return np.asarray(
        [[lookup.get((length, delta), np.nan) for delta in deltas] for length in lengths],
        dtype=np.float64,
    )


def plot_band_heatmaps(
    rows: list[dict[str, Any]], lengths: list[int], deltas: list[int], out_dir: Path
) -> None:
    for metric, label in METRICS:
        figure, axes = plt.subplots(1, 2, figsize=(13.8, 8.0), sharex=True, sharey=True)
        for axis, variant in zip(axes, ("raw", "J"), strict=True):
            matrix = matrix_for(rows, variant, metric, lengths, deltas)
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(deltas[0] - 0.5, deltas[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
                vmin=0,
                vmax=1,
                cmap="viridis",
            )
            axis.axvline(0, color="white", linestyle="--", linewidth=1.1)
            axis.axhline(20.5, color="white", linewidth=0.8, alpha=0.8)
            axis.set_title(variant)
            axis.set_xlabel(r"relative loop $\Delta=t-n$")
            axis.set_ylabel("input length n")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        figure.suptitle(f"Parity near t=n: {label}")
        figure.tight_layout()
        figure.savefig(out_dir / f"parity_diagonal_band_{metric}_raw_J.png", dpi=220)
        plt.close(figure)


def plot_differences(
    rows: list[dict[str, Any]], lengths: list[int], deltas: list[int], out_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 8.0), sharex=True, sharey=True)
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        difference = matrix_for(rows, "J", metric, lengths, deltas) - matrix_for(
            rows, "raw", metric, lengths, deltas
        )
        image = axis.imshow(
            difference,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(deltas[0] - 0.5, deltas[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=1.0)
        axis.axhline(20.5, color="black", linewidth=0.8, alpha=0.65)
        axis.set_title(f"J - raw: {label}")
        axis.set_xlabel(r"relative loop $\Delta=t-n$")
        axis.set_ylabel("input length n")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    figure.suptitle("Parity controller effect near the registered readout")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_diagonal_band_J_minus_raw.png", dpi=220)
    plt.close(figure)


def plot_centerline(
    rows: list[dict[str, Any]], lengths: list[int], out_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        for variant, color in (("raw", "#2563eb"), ("J", "#dc2626")):
            lookup = {
                int(row["length"]): float(row[metric])
                for row in rows
                if row["variant"] == variant and int(row["relative_loop"]) == 0
            }
            axis.plot(lengths, [lookup.get(length, np.nan) for length in lengths], label=variant, color=color, linewidth=1.2)
        axis.axvline(20.5, color="black", linestyle=":", linewidth=1.0, label="train-length boundary")
        axis.set_ylabel(label)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
        axis.legend()
    axes[-1].set_xlabel("input length n, evaluated at t=n")
    figure.suptitle("Parity main diagonal out to n=500")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_t_equals_n_centerline_raw_J.png", dpi=220)
    plt.close(figure)


def length_bins(maximum_length: int) -> list[tuple[int, int]]:
    candidates = ((1, 20), (21, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500))
    return [(start, min(stop, maximum_length)) for start, stop in candidates if start <= maximum_length]


def plot_binned_profiles(
    rows: list[dict[str, Any]], deltas: list[int], maximum_length: int, out_dir: Path
) -> None:
    bins = length_bins(maximum_length)
    figure, axes = plt.subplots(len(bins), 2, figsize=(13.5, 2.55 * len(bins)), sharex=True, sharey=True, squeeze=False)
    for row_index, (start, stop) in enumerate(bins):
        for column_index, (metric, label) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            for variant, color in (("raw", "#2563eb"), ("J", "#dc2626")):
                values = []
                for delta in deltas:
                    selected = [
                        float(row[metric])
                        for row in rows
                        if row["variant"] == variant
                        and start <= int(row["length"]) <= stop
                        and int(row["relative_loop"]) == delta
                    ]
                    values.append(float(np.mean(selected)) if selected else np.nan)
                axis.plot(deltas, values, marker="o", markersize=2.5, linewidth=1.2, label=variant, color=color)
            axis.axvline(0, color="black", linestyle=":", linewidth=0.9)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
            axis.set_title(f"n={start}-{stop}: {label}")
            axis.set_ylabel("mean accuracy")
            axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel(r"relative loop $\Delta=t-n$")
    axes[-1, 1].set_xlabel(r"relative loop $\Delta=t-n$")
    figure.suptitle("Mean diagonal profile by length regime")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_diagonal_profiles_by_length_bin.png", dpi=220)
    plt.close(figure)


def plot_selected_slices(
    rows: list[dict[str, Any]], lengths: list[int], deltas: list[int], out_dir: Path
) -> None:
    requested = (20, 40, 60, 100, 150, 200, 300, 400, 500)
    selected_lengths = sorted({min(lengths, key=lambda value: abs(value - target)) for target in requested})
    figure, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    for axis, length in zip(axes.flat, selected_lengths, strict=False):
        for variant, color, linestyle in (("raw", "#2563eb", "-"), ("J", "#dc2626", "--")):
            lookup = {
                int(row["relative_loop"]): float(row["exact_match"])
                for row in rows
                if row["variant"] == variant and int(row["length"]) == length
            }
            axis.plot(deltas, [lookup.get(delta, np.nan) for delta in deltas], marker="o", markersize=2.8, linewidth=1.2, linestyle=linestyle, color=color, label=variant)
        axis.axvline(0, color="black", linestyle=":", linewidth=0.8)
        axis.set_title(f"n={length}")
        axis.grid(alpha=0.2)
        axis.set_ylim(-0.02, 1.02)
    for axis in axes[-1]:
        axis.set_xlabel(r"$\Delta=t-n$")
    for axis in axes[:, 0]:
        axis.set_ylabel("strict exact match")
    axes.flat[0].legend()
    figure.suptitle("Local strict-accuracy slices at selected lengths")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_selected_length_diagonal_slices.png", dpi=220)
    plt.close(figure)


def peak_by_length(
    rows: list[dict[str, Any]], variant: str, lengths: list[int], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_values: list[float] = []
    best_values: list[float] = []
    best_offsets: list[int] = []
    for length in lengths:
        candidates = [
            row
            for row in rows
            if row["variant"] == variant and int(row["length"]) == length
        ]
        target = next(row for row in candidates if int(row["relative_loop"]) == 0)
        best = max(
            candidates,
            key=lambda row: (
                float(row[metric]),
                -abs(int(row["relative_loop"])),
            ),
        )
        target_values.append(float(target[metric]))
        best_values.append(float(best[metric]))
        best_offsets.append(int(best["relative_loop"]))
    return (
        np.asarray(target_values),
        np.asarray(best_values),
        np.asarray(best_offsets),
    )


def rolling_mean(values: np.ndarray, window: int = 11) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def plot_registered_vs_local_best(
    rows: list[dict[str, Any]], lengths: list[int], out_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, sharey=True)
    for axis, variant in zip(axes, ("raw", "J"), strict=True):
        target, best, _ = peak_by_length(rows, variant, lengths, "exact_match")
        axis.plot(lengths, rolling_mean(target), color="#dc2626", linewidth=1.5, label="t=n (11-length mean)")
        axis.plot(lengths, rolling_mean(best), color="#111827", linewidth=1.5, label="best within |t-n|<=10 (11-length mean)")
        axis.fill_between(lengths, rolling_mean(target), rolling_mean(best), color="#f59e0b", alpha=0.22)
        axis.axvline(20.5, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(variant)
        axis.set_ylabel("strict exact match")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[-1].set_xlabel("input length n")
    figure.suptitle("Parity capability versus registered-time alignment")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_registered_vs_local_best.png", dpi=220)
    plt.close(figure)


def plot_peak_phase(
    rows: list[dict[str, Any]], lengths: list[int], out_dir: Path
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    for variant, color, marker in (("raw", "#2563eb", "o"), ("J", "#dc2626", "x")):
        _, _, offsets = peak_by_length(rows, variant, lengths, "exact_match")
        axes[0].scatter(lengths, offsets, s=8, alpha=0.65, color=color, marker=marker, label=variant)
        axes[1].scatter(lengths, np.mod(offsets, 4), s=8, alpha=0.65, color=color, marker=marker, label=variant)
    axes[0].axhline(0, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel(r"best offset $\Delta^*$")
    axes[0].set_yticks(range(-10, 11, 2))
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].set_yticks((0, 1, 2, 3))
    axes[1].set_ylabel(r"phase class $\Delta^*\;\mathrm{mod}\;4$")
    axes[1].set_xlabel("input length n")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.suptitle("Parity local-peak phase: raw drift versus J locking")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_local_peak_phase_by_length.png", dpi=220)
    plt.close(figure)


def summarize(rows: list[dict[str, Any]], lengths: list[int], deltas: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "status": "complete",
        "length_range": [min(lengths), max(lengths)],
        "evaluated_length_count": len(lengths),
        "relative_loop_range": [min(deltas), max(deltas)],
        "variants": {},
    }
    for variant in ("raw", "J"):
        variant_output: dict[str, Any] = {}
        for metric, _ in METRICS:
            target_values, best_values, best_offsets_array = peak_by_length(
                rows, variant, lengths, metric
            )
            best_offsets = best_offsets_array.tolist()
            variant_output[metric] = {
                "mean_at_t_equals_n": float(np.mean(target_values)),
                "minimum_at_t_equals_n": float(np.min(target_values)),
                "mean_local_best_within_band": float(np.mean(best_values)),
                "best_offset_counts": {str(key): value for key, value in sorted(Counter(best_offsets).items())},
                "best_offset_modulo_4_counts": {
                    str(key): value
                    for key, value in sorted(
                        Counter(int(offset) % 4 for offset in best_offsets).items()
                    )
                },
                "t_equals_n_is_best_fraction": float(np.mean(np.asarray(best_offsets) == 0)),
            }
        output["variants"][variant] = variant_output
    return output


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    lengths = sorted({int(row["length"]) for row in rows})
    deltas = list(range(-args.half_width, args.half_width + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_band_heatmaps(rows, lengths, deltas, args.out_dir)
    plot_differences(rows, lengths, deltas, args.out_dir)
    plot_centerline(rows, lengths, args.out_dir)
    plot_binned_profiles(rows, deltas, max(lengths), args.out_dir)
    plot_selected_slices(rows, lengths, deltas, args.out_dir)
    plot_registered_vs_local_best(rows, lengths, args.out_dir)
    plot_peak_phase(rows, lengths, args.out_dir)
    analysis = summarize(rows, lengths, deltas)
    analysis["metrics_files"] = [str(path) for path in args.metrics]
    (args.out_dir / "diagonal_band_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
