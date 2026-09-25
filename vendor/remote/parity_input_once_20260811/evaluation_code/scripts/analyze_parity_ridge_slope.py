"""Estimate and visualize Parity phase-ridge slopes near the registered t=n line.

The accuracy surface has an approximately four-loop periodic carrier, so an
ordinary argmax regression confounds the same ridge with its +/-4-loop copies.
This script extracts the carrier phase for each input length with a Fourier
coefficient at period four, unwraps that phase across length, and fits

    t = (1 + s) n + b.

It also reports the equivalent slope in the heatmap coordinates (x=t, y=n),
uses a moving-block residual bootstrap for uncertainty, and overlays the fitted
ridge family on the original absolute-coordinate heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np


METRICS = (
    "exact_match",
    "correct_parity_probability",
    "mean_parity_margin",
)
VARIANTS = ("raw", "J")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-length", type=int, default=11)
    parser.add_argument("--maximum-length", type=int, default=490)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-block-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> dict[tuple[str, int], dict[int, dict[str, float]]]:
    values: dict[tuple[str, int], dict[int, dict[str, float]]] = defaultdict(dict)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                variant = row["variant"]
                length = int(row["length"])
                offset = int(row["relative_loop"])
                key = (variant, length)
                if offset in values[key]:
                    raise ValueError(f"duplicate cell: variant={variant}, n={length}, d={offset}")
                values[key][offset] = {metric: float(row[metric]) for metric in METRICS}
    return values


def extract_phase(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    variant: str,
    metric: str,
    minimum_length: int,
    maximum_length: int,
    period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths: list[int] = []
    coefficients: list[complex] = []
    expected_offsets: np.ndarray | None = None
    for length in range(minimum_length, maximum_length + 1):
        cells = values[(variant, length)]
        offsets = np.asarray(sorted(cells), dtype=np.float64)
        if expected_offsets is None:
            expected_offsets = offsets
        elif not np.array_equal(offsets, expected_offsets):
            raise ValueError(f"inconsistent offset support at variant={variant}, n={length}")
        response = np.asarray([cells[int(offset)][metric] for offset in offsets])
        response = response - response.mean()
        scale = response.std()
        if scale > 0:
            response = response / scale
        coefficient = np.sum(response * np.exp(1j * 2 * np.pi * offsets / period))
        lengths.append(length)
        coefficients.append(complex(coefficient))

    lengths_array = np.asarray(lengths, dtype=np.float64)
    coefficients_array = np.asarray(coefficients, dtype=np.complex128)
    unwrapped_offset = np.unwrap(np.angle(coefficients_array)) * period / (2 * np.pi)
    amplitude = np.abs(coefficients_array)
    return lengths_array, unwrapped_offset, amplitude


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = slope * x + intercept
    residual = y - prediction
    total = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / total if total > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "prediction": prediction,
        "residual": residual,
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def block_bootstrap_slope(
    x: np.ndarray,
    fit: dict[str, Any],
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    prediction = np.asarray(fit["prediction"])
    residual = np.asarray(fit["residual"])
    count = len(x)
    if block_length > count:
        raise ValueError("bootstrap block length exceeds the number of fitted lengths")
    starts = np.arange(count - block_length + 1)
    blocks_needed = int(np.ceil(count / block_length))
    slopes = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sampled_residual = np.concatenate(
            [residual[start : start + block_length] for start in sampled_starts]
        )[:count]
        slopes[index] = linear_fit(x, prediction + sampled_residual)["slope"]
    lower, median, upper = np.quantile(slopes, [0.025, 0.5, 0.975])
    return {
        "samples": samples,
        "block_length": block_length,
        "slope_ci95": [float(lower), float(upper)],
        "slope_bootstrap_median": float(median),
        "slope_bootstrap_std": float(slopes.std(ddof=1)),
    }


def scan_period(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    variant: str,
    metric: str,
    minimum_length: int,
    maximum_length: int,
) -> dict[str, float]:
    offsets = np.asarray(sorted(values[(variant, minimum_length)]), dtype=np.float64)
    response = np.asarray(
        [
            [values[(variant, length)][int(offset)][metric] for offset in offsets]
            for length in range(minimum_length, maximum_length + 1)
        ],
        dtype=np.float64,
    )
    response = response - response.mean(axis=1, keepdims=True)
    response = response / np.maximum(response.std(axis=1, keepdims=True), 1e-12)
    periods = np.linspace(3.5, 4.5, 2001)
    basis = np.exp(1j * 2 * np.pi * offsets[:, None] / periods[None, :])
    scores = np.abs(response @ basis).mean(axis=0)
    best_index = int(np.argmax(scores))
    four_index = int(np.argmin(np.abs(periods - 4.0)))
    return {
        "best_period": float(periods[best_index]),
        "mean_amplitude_at_best": float(scores[best_index]),
        "mean_amplitude_at_period_4": float(scores[four_index]),
        "relative_gain_over_period_4": float(scores[best_index] / scores[four_index] - 1),
    }


def analyze_variant_metric(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    variant: str,
    metric: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    lengths, offset, amplitude = extract_phase(
        values,
        variant,
        metric,
        args.minimum_length,
        args.maximum_length,
        args.period,
    )
    fit = linear_fit(lengths, offset)
    bootstrap = block_bootstrap_slope(
        lengths,
        fit,
        args.bootstrap_samples,
        args.bootstrap_block_length,
        args.seed + (0 if variant == "raw" else 1),
    )
    offset_slope = fit["slope"]
    forward_slope = 1.0 + offset_slope
    forward_intercept = fit["intercept"]
    plot_slope = 1.0 / forward_slope
    plot_intercept = -forward_intercept / forward_slope
    offset_ci = bootstrap["slope_ci95"]
    forward_ci = [1.0 + offset_ci[0], 1.0 + offset_ci[1]]
    plot_ci = [1.0 / forward_ci[1], 1.0 / forward_ci[0]]

    split_edges = np.linspace(
        args.minimum_length, args.maximum_length + 1, 4, dtype=int
    )
    split_fits: list[dict[str, Any]] = []
    for start, stop_exclusive in zip(split_edges[:-1], split_edges[1:], strict=True):
        mask = (lengths >= start) & (lengths < stop_exclusive)
        split = linear_fit(lengths[mask], offset[mask])
        split_fits.append(
            {
                "length_range": [int(start), int(stop_exclusive - 1)],
                "offset_slope": split["slope"],
                "t_per_n_slope": 1.0 + split["slope"],
                "r_squared": split["r_squared"],
            }
        )

    result = {
        "variant": variant,
        "metric": metric,
        "fitted_equation_t_from_n": {
            "t_per_n_slope": forward_slope,
            "intercept": forward_intercept,
            "slope_ci95": forward_ci,
        },
        "equivalent_heatmap_equation_n_from_t": {
            "n_per_t_slope": plot_slope,
            "intercept": plot_intercept,
            "slope_ci95": plot_ci,
        },
        "phase_drift": {
            "loops_per_input_token": offset_slope,
            "loops_per_100_tokens": 100.0 * offset_slope,
            "predicted_offset_at_n500": 500.0 * offset_slope + forward_intercept,
            "input_tokens_per_four_loop_wrap": 4.0 / offset_slope,
            "observed_unwrapped_offset_first": float(offset[0]),
            "observed_unwrapped_offset_last": float(offset[-1]),
        },
        "fit_quality": {
            "r_squared": fit["r_squared"],
            "rmse_loops": fit["rmse"],
            "mae_loops": fit["mae"],
            "mean_fourier_amplitude": float(amplitude.mean()),
        },
        "bootstrap": bootstrap,
        "split_fits": split_fits,
        "period_scan": scan_period(
            values,
            variant,
            metric,
            args.minimum_length,
            args.maximum_length,
        ),
    }
    return result, lengths, offset, amplitude


def plot_phase_drift(
    phase: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    results: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharey=False)
    colors = {"raw": "#d95f02", "J": "#1b9e77"}
    for axis, variant in zip(axes, VARIANTS, strict=True):
        lengths, offset, _ = phase[(variant, "exact_match")]
        record = results["metrics"]["exact_match"][variant]
        slope = record["phase_drift"]["loops_per_input_token"]
        intercept = record["fitted_equation_t_from_n"]["intercept"]
        axis.scatter(lengths, offset, s=7, alpha=0.45, color=colors[variant], label="Fourier ridge phase")
        axis.plot(lengths, slope * lengths + intercept, color="black", linewidth=2, label="linear fit")
        axis.axhline(0, color="gray", linestyle="--", linewidth=1, label="registered t=n phase")
        axis.set_xlabel("input length n")
        axis.set_ylabel("unwrapped ridge offset t-n (loops)")
        axis.set_title(
            f"{variant}: {100*slope:.3f} loop drift / 100 tokens\n"
            f"RMSE={record['fit_quality']['rmse_loops']:.3f} loop"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle("Parity carrier-phase drift: raw accumulates; J nearly locks the phase")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_ridge_phase_drift.png", dpi=240)
    plt.close(figure)


def plot_absolute_ridge(
    phase: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    results: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), sharex=True, sharey=True)
    colors = {"raw": "#d95f02", "J": "#1b9e77"}
    for axis, variant in zip(axes, VARIANTS, strict=True):
        lengths, offset, amplitude = phase[(variant, "exact_match")]
        inferred_t = lengths + offset
        record = results["metrics"]["exact_match"][variant]
        slope = record["fitted_equation_t_from_n"]["t_per_n_slope"]
        intercept = record["fitted_equation_t_from_n"]["intercept"]
        predicted_t = slope * lengths + intercept
        axis.scatter(
            inferred_t,
            lengths,
            c=amplitude,
            cmap="viridis",
            s=8,
            alpha=0.65,
            label="inferred ridge",
        )
        axis.plot(predicted_t, lengths, color=colors[variant], linewidth=2.5, label="fitted ridge")
        axis.plot([0, 510], [0, 510], color="black", linestyle="--", linewidth=1.2, label="n=t")
        axis.set_xlim(0, 510)
        axis.set_ylim(0, 510)
        axis.set_aspect("equal")
        axis.set_xlabel("executed recurrent loops t")
        axis.set_ylabel("input length n")
        heatmap = record["equivalent_heatmap_equation_n_from_t"]
        axis.set_title(
            f"{variant}: n={heatmap['n_per_t_slope']:.6f} t"
            f"{heatmap['intercept']:+.3f}"
        )
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(alpha=0.15)
    figure.suptitle("Absolute-coordinate fit of the central Parity phase ridge")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_ridge_slope_absolute_nt.png", dpi=240)
    plt.close(figure)


def make_absolute_matrix(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    variant: str,
    metric: str,
    maximum: int,
) -> np.ndarray:
    matrix = np.full((maximum, maximum), np.nan, dtype=np.float64)
    for length in range(1, maximum + 1):
        for offset, record in values.get((variant, length), {}).items():
            loop = length + offset
            if 1 <= loop <= maximum:
                matrix[length - 1, loop - 1] = record[metric]
    return matrix


def evaluate_phase_schedule(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    variant: str,
    metric: str,
    lengths: np.ndarray,
    offset_slope: float,
    offset_intercept: float,
    period: float,
) -> dict[str, Any]:
    scheduled: list[float] = []
    registered: list[float] = []
    oracle: list[float] = []
    selected_offsets: list[int] = []
    for length_value in lengths:
        length = int(length_value)
        unwrapped = offset_slope * length + offset_intercept
        wrapped = unwrapped - period * np.rint(unwrapped / period)
        selected_offset = int(np.rint(wrapped))
        cells = values[(variant, length)]
        if selected_offset not in cells:
            raise ValueError(
                f"scheduled offset {selected_offset} unavailable at variant={variant}, n={length}"
            )
        selected_offsets.append(selected_offset)
        scheduled.append(cells[selected_offset][metric])
        registered.append(cells[0][metric])
        oracle.append(max(record[metric] for record in cells.values()))
    registered_mean = float(np.mean(registered))
    scheduled_mean = float(np.mean(scheduled))
    oracle_mean = float(np.mean(oracle))
    oracle_gap = oracle_mean - registered_mean
    return {
        "length_range": [int(lengths.min()), int(lengths.max())],
        "scheduled_mean": scheduled_mean,
        "registered_t_equals_n_mean": registered_mean,
        "local_oracle_mean": oracle_mean,
        "oracle_gap": oracle_gap,
        "fraction_of_oracle_gap_recovered": (
            float((scheduled_mean - registered_mean) / oracle_gap)
            if oracle_gap >= 0.01
            else None
        ),
        "selected_offset_counts": {
            str(offset): int(selected_offsets.count(offset))
            for offset in sorted(set(selected_offsets))
        },
    }


def schedule_validation(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    phase: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    results: dict[str, Any],
    period: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {"full_fit": {}, "half_split_cross_validation": {}, "coefficient_swap_control": {}}
    for variant in VARIANTS:
        lengths, offsets, _ = phase[(variant, "exact_match")]
        record = results["metrics"]["exact_match"][variant]
        slope = record["phase_drift"]["loops_per_input_token"]
        intercept = record["fitted_equation_t_from_n"]["intercept"]
        output["full_fit"][variant] = evaluate_phase_schedule(
            values, variant, "exact_match", lengths, slope, intercept, period
        )

        midpoint = len(lengths) // 2
        split_records: list[dict[str, Any]] = []
        for train_indices, test_indices in (
            (np.arange(0, midpoint), np.arange(midpoint, len(lengths))),
            (np.arange(midpoint, len(lengths)), np.arange(0, midpoint)),
        ):
            trained = linear_fit(lengths[train_indices], offsets[train_indices])
            evaluated = evaluate_phase_schedule(
                values,
                variant,
                "exact_match",
                lengths[test_indices],
                trained["slope"],
                trained["intercept"],
                period,
            )
            evaluated["train_length_range"] = [
                int(lengths[train_indices].min()),
                int(lengths[train_indices].max()),
            ]
            evaluated["trained_t_per_n_slope"] = 1.0 + trained["slope"]
            split_records.append(evaluated)
        output["half_split_cross_validation"][variant] = split_records

    lengths = phase[("raw", "exact_match")][0]
    for source_variant in VARIANTS:
        source = results["metrics"]["exact_match"][source_variant]
        slope = source["phase_drift"]["loops_per_input_token"]
        intercept = source["fitted_equation_t_from_n"]["intercept"]
        for target_variant in VARIANTS:
            key = f"{source_variant}_coefficient_on_{target_variant}_surface"
            output["coefficient_swap_control"][key] = evaluate_phase_schedule(
                values,
                target_variant,
                "exact_match",
                lengths,
                slope,
                intercept,
                period,
            )
    return output


def plot_schedule_validation(results: dict[str, Any], out_dir: Path) -> None:
    validation = results["schedule_validation"]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    colors = ("#777777", "#1b9e77", "#e6ab02")
    for axis, variant in zip(axes, VARIANTS, strict=True):
        full = validation["full_fit"][variant]
        cross = validation["half_split_cross_validation"][variant]
        cross_scheduled = np.mean([record["scheduled_mean"] for record in cross])
        cross_registered = np.mean([record["registered_t_equals_n_mean"] for record in cross])
        cross_oracle = np.mean([record["local_oracle_mean"] for record in cross])
        labels = ("t=n", "fitted phase\n(half-split CV)", "local oracle")
        values = (cross_registered, cross_scheduled, cross_oracle)
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_ylim(0, 1.04)
        axis.set_ylabel("strict exact match")
        axis.set_title(
            f"{variant}\nfull-fit scheduled={full['scheduled_mean']:.3f}"
        )
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Predictive check: a slope learned on one half selects the bright phase on the other")
    figure.tight_layout()
    figure.savefig(out_dir / "parity_ridge_schedule_cross_validation.png", dpi=240)
    plt.close(figure)


def plot_overlay(
    values: dict[tuple[str, int], dict[int, dict[str, float]]],
    results: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
) -> None:
    maximum = 500
    matrices = {
        variant: make_absolute_matrix(values, variant, "exact_match", maximum)
        for variant in VARIANTS
    }
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    ranges = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 500)]
    figure, axes = plt.subplots(len(ranges), 2, figsize=(14.5, 4.1 * len(ranges)), squeeze=False)
    for row_index, (start, stop) in enumerate(ranges):
        t_start = max(1, start - 10)
        t_stop = min(maximum, stop + 10)
        for column_index, variant in enumerate(VARIANTS):
            axis = axes[row_index, column_index]
            selected = matrices[variant][start - 1 : stop, t_start - 1 : t_stop]
            axis.imshow(
                selected,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
                extent=(t_start - 0.5, t_stop + 0.5, start - 0.5, stop + 0.5),
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            n_values = np.linspace(start, stop, 400)
            record = results["metrics"]["exact_match"][variant]
            slope = record["fitted_equation_t_from_n"]["t_per_n_slope"]
            intercept = record["fitted_equation_t_from_n"]["intercept"]
            for copy_index in range(-3, 4):
                fitted_t = slope * n_values + intercept + 4 * copy_index
                line = axis.plot(
                    fitted_t,
                    n_values,
                    color="#ff3355" if copy_index == 0 else "white",
                    linewidth=1.7 if copy_index == 0 else 0.8,
                    linestyle="-" if copy_index == 0 else ":",
                    alpha=0.95 if copy_index == 0 else 0.7,
                )[0]
                if copy_index == 0:
                    line.set_path_effects(
                        [path_effects.Stroke(linewidth=3.2, foreground="white"), path_effects.Normal()]
                    )
            axis.plot(n_values, n_values, color="black", linestyle="--", linewidth=1.2)
            axis.set_xlim(t_start - 0.5, t_stop + 0.5)
            axis.set_ylim(start - 0.5, stop + 0.5)
            axis.set_xlabel("executed recurrent loops t")
            axis.set_ylabel("input length n")
            axis.set_title(f"{variant}: n={start}-{stop}")
    figure.suptitle(
        "Measured Parity heatmaps with fitted period-4 ridge family\n"
        "pink: central fitted ridge; white dotted: period-4 copies; black dashed: t=n",
        y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(out_dir / "parity_ridge_fit_overlay_five_ranges_raw_J.png", dpi=220)
    plt.close(figure)


def write_phase_csv(
    phase: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    with (out_dir / "parity_ridge_phase_by_length.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "metric", "length_n", "unwrapped_offset_t_minus_n", "fourier_amplitude"))
        for metric in METRICS:
            for variant in VARIANTS:
                lengths, offsets, amplitudes = phase[(variant, metric)]
                for length, offset, amplitude in zip(lengths, offsets, amplitudes, strict=True):
                    writer.writerow((variant, metric, int(length), float(offset), float(amplitude)))


def write_metric_csv(results: dict[str, Any], out_dir: Path) -> None:
    with (out_dir / "parity_ridge_slope_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "variant",
                "metric",
                "t_per_n_slope",
                "t_per_n_ci95_low",
                "t_per_n_ci95_high",
                "n_per_t_slope",
                "n_per_t_ci95_low",
                "n_per_t_ci95_high",
                "phase_loops_per_100_tokens",
                "phase_wrap_tokens",
                "r_squared",
                "rmse_loops",
            )
        )
        for metric in METRICS:
            for variant in VARIANTS:
                record = results["metrics"][metric][variant]
                forward = record["fitted_equation_t_from_n"]
                heatmap = record["equivalent_heatmap_equation_n_from_t"]
                writer.writerow(
                    (
                        variant,
                        metric,
                        forward["t_per_n_slope"],
                        *forward["slope_ci95"],
                        heatmap["n_per_t_slope"],
                        *heatmap["slope_ci95"],
                        record["phase_drift"]["loops_per_100_tokens"],
                        record["phase_drift"]["input_tokens_per_four_loop_wrap"],
                        record["fit_quality"]["r_squared"],
                        record["fit_quality"]["rmse_loops"],
                    )
                )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    values = load_rows(args.metrics)
    phase: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    results: dict[str, Any] = {
        "method": {
            "carrier_period": args.period,
            "phase_estimator": "row-centered normalized Fourier coefficient, then phase unwrap",
            "fit_range": [args.minimum_length, args.maximum_length],
            "bootstrap": "moving-block residual bootstrap",
            "metrics_files": [str(path) for path in args.metrics],
        },
        "metrics": {},
    }
    for metric in METRICS:
        results["metrics"][metric] = {}
        for variant in VARIANTS:
            record, lengths, offset, amplitude = analyze_variant_metric(
                values, variant, metric, args
            )
            results["metrics"][metric][variant] = record
            phase[(variant, metric)] = (lengths, offset, amplitude)

    raw_slope = results["metrics"]["exact_match"]["raw"]["phase_drift"][
        "loops_per_input_token"
    ]
    j_slope = results["metrics"]["exact_match"]["J"]["phase_drift"][
        "loops_per_input_token"
    ]
    results["exact_match_comparison"] = {
        "raw_to_J_phase_slope_reduction_fraction": 1.0 - j_slope / raw_slope,
        "raw_to_J_phase_slope_reduction_factor": raw_slope / j_slope,
    }

    results["schedule_validation"] = schedule_validation(
        values, phase, results, args.period
    )

    with (args.out_dir / "parity_ridge_slope_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_phase_csv(phase, args.out_dir)
    write_metric_csv(results, args.out_dir)
    plot_phase_drift(phase, results, args.out_dir)
    plot_absolute_ridge(phase, results, args.out_dir)
    plot_overlay(values, results, args.out_dir)
    plot_schedule_validation(results, args.out_dir)


if __name__ == "__main__":
    main()
