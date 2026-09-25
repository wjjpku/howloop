#!/usr/bin/env python3
"""Plot the dense parity horizon scan with high-sample sparse anchors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_variant_csv(path: Path) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {"raw": {}, "full": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant not in values:
                continue
            values[variant][int(row["length"])] = float(row["exact_match"])
    return values


def merge_variant_csvs(paths: list[Path]) -> dict[str, dict[int, float]]:
    merged: dict[str, dict[int, float]] = {"raw": {}, "full": {}}
    for path in paths:
        current = read_variant_csv(path)
        for variant in merged:
            overlap = set(merged[variant]).intersection(current[variant])
            if overlap:
                raise ValueError(f"duplicate {variant} lengths in anchors: {sorted(overlap)}")
            merged[variant].update(current[variant])
    return merged


def aligned(values: dict[str, dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = sorted(set(values["raw"]).intersection(values["full"]))
    if set(values["raw"]) != set(values["full"]):
        raise ValueError("raw and full length grids differ")
    return (
        np.asarray(lengths, dtype=float),
        np.asarray([values["raw"][length] for length in lengths]),
        np.asarray([values["full"][length] for length in lengths]),
    )


def local_minimum(lengths: np.ndarray, values: np.ndarray, half_width: float) -> np.ndarray:
    return np.asarray(
        [values[np.abs(lengths - length) <= half_width].min() for length in lengths]
    )


def local_mean(lengths: np.ndarray, values: np.ndarray, half_width: float) -> np.ndarray:
    return np.asarray(
        [values[np.abs(lengths - length) <= half_width].mean() for length in lengths]
    )


def first_below(lengths: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    hits = lengths[values < threshold]
    return None if hits.size == 0 else int(hits[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-half-width", type=float, default=25.0)
    parser.add_argument("--trend-half-width", type=float, default=50.0)
    args = parser.parse_args()

    dense = read_variant_csv(args.dense)
    anchors = merge_variant_csvs(args.anchor)
    x, raw, full = aligned(dense)
    xa, rawa, fulla = aligned(anchors)
    delta = full - raw
    delta_anchor = fulla - rawa
    raw_floor = local_minimum(x, raw, args.local_half_width)
    full_floor = local_minimum(x, full, args.local_half_width)
    raw_trend = local_mean(x, raw, args.trend_half_width)
    full_trend = local_mean(x, full, args.trend_half_width)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 240,
        }
    )
    raw_color = "#cc5a3d"
    j_color = "#147d82"
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.25, 1.45], "hspace": 0.14},
    )

    ax = axes[0]
    ax.plot(x, raw, color=raw_color, lw=1.25, marker="o", ms=2.8, label="Raw, dense n=32")
    ax.plot(x, full, color=j_color, lw=1.5, marker="o", ms=2.8, label="J, dense n=32")
    ax.scatter(
        xa,
        rawa,
        s=49,
        facecolors="white",
        edgecolors=raw_color,
        linewidths=1.45,
        zorder=5,
        label="Raw anchors, n=128",
    )
    ax.scatter(
        xa,
        fulla,
        s=49,
        facecolors=j_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label="J anchors, n=128",
    )
    ax.axhline(0.5, color="#777777", lw=0.9, ls="--", alpha=0.75, label="Chance EM = 0.5")
    ax.set_ylim(-0.035, 1.045)
    ax.set_ylabel("Accuracy (strict target EM)")
    ax.set_title("Current parity-model boundary: accuracy vs recurrence loops")
    ax.legend(ncol=3, loc="lower right", frameon=True, fontsize=8.6)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.65)

    ax = axes[1]
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.fill_between(x, 0, delta, where=delta >= 0, color=j_color, alpha=0.18, interpolate=True)
    ax.fill_between(x, 0, delta, where=delta < 0, color=raw_color, alpha=0.18, interpolate=True)
    ax.plot(x, delta, color="#343a40", lw=1.15, marker="o", ms=2.5, label="Dense J - raw")
    ax.scatter(
        xa,
        delta_anchor,
        s=43,
        color="#111111",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        label="n=128 anchors",
    )
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Accuracy gain (J - raw)")
    ax.legend(loc="lower right", ncol=2, fontsize=8.6)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.65)

    ax = axes[2]
    trend_label = rf"mean within $\pm${args.trend_half_width:g} loops"
    floor_label = rf"minimum within $\pm${args.local_half_width:g} loops"
    ax.plot(x, raw_trend, color=raw_color, lw=1.75, label=f"Raw {trend_label}")
    ax.plot(x, full_trend, color=j_color, lw=2.0, label=f"J {trend_label}")
    ax.plot(x, raw_floor, color=raw_color, lw=1.0, ls=":", alpha=0.72, label=f"Raw {floor_label}")
    ax.plot(x, full_floor, color=j_color, lw=1.15, ls=":", alpha=0.78, label=f"J {floor_label}")
    thresholds = (0.95, 0.90, 0.75, 0.50)
    boundary_labels: list[str] = []
    boundary_summary: dict[str, int | str | None] = {}
    for threshold in thresholds:
        ax.axhline(threshold, color="#9a9a9a", lw=0.65, ls="--", alpha=0.42)
        hits = np.flatnonzero(full_trend >= threshold)
        key = f"j_trend_last_at_or_above_{threshold:.2f}"
        if hits.size == 0:
            boundary_summary[key] = None
            boundary_labels.append(f"{threshold:.0%}: never")
        elif hits[-1] == len(x) - 1:
            boundary_summary[key] = f">={int(x[-1])}"
            boundary_labels.append(f"{threshold:.0%}: >={int(x[-1])}")
        else:
            boundary = int(x[hits[-1]])
            boundary_summary[key] = boundary
            boundary_labels.append(f"{threshold:.0%}: L{boundary}")
            ax.axvline(boundary, color=j_color, lw=0.7, ls="--", alpha=0.32)
    ax.text(
        0.012,
        0.035,
        "J trend last attainment  |  " + "   ".join(boundary_labels),
        transform=ax.transAxes,
        fontsize=8.1,
        color="#244f52",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#b9d2d3", "alpha": 0.9},
    )
    ax.set_ylim(-0.035, 1.045)
    ax.set_ylabel("Local trend / floor")
    ax.set_xlabel("Logical length / recurrence loops")
    ax.legend(loc="upper right", ncol=2, fontsize=8.2)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.65)

    for ax in axes:
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].set_xticks(np.arange(100, 1001, 100))
    fig.text(
        0.01,
        0.008,
        "One released64 backbone/controller seed. J trained on logical lengths 20–40; anchor=1; no post-final J. "
        "Here logical length equals recurrence loops, T(n)=n. Dense estimates move in increments of 1/32; "
        "circles are independent n=128 evaluations.",
        fontsize=8.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.078, right=0.986, top=0.953, bottom=0.08)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    bin_summary: dict[str, dict[str, float | int]] = {}
    for lower in range(100, 1000, 100):
        upper = lower + 100
        mask = (x >= lower) & (x < upper if upper < 1000 else x <= upper)
        if not mask.any():
            continue
        bin_summary[f"{lower}-{upper}"] = {
            "points": int(mask.sum()),
            "raw_mean": float(raw[mask].mean()),
            "raw_min": float(raw[mask].min()),
            "j_mean": float(full[mask].mean()),
            "j_min": float(full[mask].min()),
            "mean_gain": float(delta[mask].mean()),
        }

    summary = {
        "dense_points": int(x.size),
        "dense_examples_per_point": 32,
        "anchor_points": int(xa.size),
        "anchor_examples_per_point": 128,
        "j_better_points": int((delta > 0).sum()),
        "j_tied_points": int((delta == 0).sum()),
        "j_worse_points": int((delta < 0).sum()),
        "mean_raw_em": float(raw.mean()),
        "mean_j_em": float(full.mean()),
        "mean_j_minus_raw": float(delta.mean()),
        "first_j_below_095": first_below(x, full, 0.95),
        "first_j_below_090": first_below(x, full, 0.90),
        "first_j_below_075": first_below(x, full, 0.75),
        "first_j_below_050": first_below(x, full, 0.50),
        "max_dense_gain": {"length": int(x[np.argmax(delta)]), "value": float(delta.max())},
        "min_dense_gain": {"length": int(x[np.argmin(delta)]), "value": float(delta.min())},
        "local_lower_envelope_half_width": args.local_half_width,
        "local_trend_half_width": args.trend_half_width,
        "trend_boundaries": boundary_summary,
        "hundred_loop_bins": bin_summary,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
