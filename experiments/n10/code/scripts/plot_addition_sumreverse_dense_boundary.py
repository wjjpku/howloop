#!/usr/bin/env python3
"""Plot a dense strict-EM horizon boundary for Addition or Sum-Reverse."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_parity_dense_horizon_20260803 import (
    aligned,
    local_mean,
    local_minimum,
    read_variant_csv,
)


TASK_LABELS = {
    "addition": "Addition",
    "sum_reverse": "Sum-Reverse",
}
TARGET_RULES = {
    "addition": "T(n)=n+1",
    "sum_reverse": "T(n)=n",
}


def read_target_anchors(path: Path) -> tuple[dict[str, dict[int, float]], int]:
    values: dict[str, dict[int, float]] = {"raw": {}, "full": {}}
    examples: set[int] = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant not in values or int(row["step"]) != int(row["target_step"]):
                continue
            length = int(row["length"])
            if length in values[variant]:
                raise ValueError(f"duplicate target anchor for {variant} length {length}")
            values[variant][length] = float(row["exact_match"])
            examples.add(int(row["examples"]))
    if len(examples) != 1:
        raise ValueError(f"anchor example counts differ: {sorted(examples)}")
    return values, examples.pop()


def first_below(lengths: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    hits = lengths[values < threshold]
    return None if hits.size == 0 else int(hits[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASK_LABELS), required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--anchor-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trend-half-width", type=float, default=4.0)
    parser.add_argument("--floor-half-width", type=float, default=2.0)
    args = parser.parse_args()

    dense = read_variant_csv(args.dense)
    anchors, anchor_examples = read_target_anchors(args.anchor_trajectory)
    x, raw, full = aligned(dense)
    xa, rawa, fulla = aligned(anchors)
    step_offset = 1 if args.task == "addition" else 0
    loop_x = x + step_offset
    loop_xa = xa + step_offset
    delta = full - raw
    delta_anchor = fulla - rawa
    raw_trend = local_mean(x, raw, args.trend_half_width)
    full_trend = local_mean(x, full, args.trend_half_width)
    raw_floor = local_minimum(x, raw, args.floor_half_width)
    full_floor = local_minimum(x, full, args.floor_half_width)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
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
        figsize=(10.8, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.25, 1.5], "hspace": 0.14},
    )

    ax = axes[0]
    ax.plot(loop_x, raw, color=raw_color, lw=1.25, marker="o", ms=3.0, label="Raw, dense n=64")
    ax.plot(loop_x, full, color=j_color, lw=1.5, marker="o", ms=3.0, label="J, dense n=64")
    ax.scatter(
        loop_xa,
        rawa,
        s=52,
        facecolors="white",
        edgecolors=raw_color,
        linewidths=1.45,
        zorder=5,
        label=f"Raw anchors, n={anchor_examples}",
    )
    ax.scatter(
        loop_xa,
        fulla,
        s=52,
        facecolors=j_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        label=f"J anchors, n={anchor_examples}",
    )
    ax.axhline(0.5, color="#777777", lw=0.9, ls="--", alpha=0.75, label="50% strict EM")
    ax.set_ylim(-0.035, 1.045)
    ax.set_ylabel("Accuracy (strict answer-sequence EM)")
    ax.set_title(f"Current {TASK_LABELS[args.task]} model boundary: accuracy vs recurrence loops")
    ax.legend(ncol=3, loc="lower left", frameon=True, fontsize=8.5)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.65)

    ax = axes[1]
    ax.axhline(0.0, color="#555555", lw=0.9)
    ax.fill_between(loop_x, 0, delta, where=delta >= 0, color=j_color, alpha=0.18, interpolate=True)
    ax.fill_between(loop_x, 0, delta, where=delta < 0, color=raw_color, alpha=0.18, interpolate=True)
    ax.plot(loop_x, delta, color="#343a40", lw=1.15, marker="o", ms=2.6, label="Dense J - raw")
    ax.scatter(
        loop_xa,
        delta_anchor,
        s=43,
        color="#111111",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
        label=f"n={anchor_examples} anchors",
    )
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Accuracy gain (J - raw)")
    ax.legend(loc="lower right", ncol=2, fontsize=8.6)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.65)

    ax = axes[2]
    trend_label = rf"mean within $\pm${args.trend_half_width:g} lengths"
    floor_label = rf"minimum within $\pm${args.floor_half_width:g} lengths"
    ax.plot(loop_x, raw_trend, color=raw_color, lw=1.75, label=f"Raw {trend_label}")
    ax.plot(loop_x, full_trend, color=j_color, lw=2.0, label=f"J {trend_label}")
    ax.plot(loop_x, raw_floor, color=raw_color, lw=1.0, ls=":", alpha=0.72, label=f"Raw {floor_label}")
    ax.plot(loop_x, full_floor, color=j_color, lw=1.15, ls=":", alpha=0.78, label=f"J {floor_label}")
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
            maximum_loops = int(loop_x[-1])
            boundary_summary[key] = f">={maximum_loops} loops"
            boundary_labels.append(f"{threshold:.0%}: >=T{maximum_loops}")
        else:
            boundary = int(x[hits[-1]])
            target_loops = boundary + step_offset
            boundary_summary[key] = {
                "logical_length": boundary,
                "target_loops": target_loops,
            }
            boundary_labels.append(f"{threshold:.0%}: T={target_loops}")
            ax.axvline(target_loops, color=j_color, lw=0.75, ls="--", alpha=0.32)
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
    ax.set_xlabel("Target recurrence loops T(n)")
    ax.legend(loc="upper right", ncol=2, fontsize=8.2)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.65)

    for ax in axes:
        ax.axvspan(20 + step_offset, 40 + step_offset, color=j_color, alpha=0.045, zorder=-10)
        ax.axvline(19 + step_offset, color="#444444", lw=0.8, ls=":", alpha=0.65)
        ax.set_xlim(float(loop_x.min()), float(loop_x.max()))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[2].set_xticks(np.arange(20, int(loop_x.max()) + 1, 10))
    fig.text(
        0.01,
        0.008,
        f"One official backbone/controller seed. Backbone trained on n=1–19; J trained on n=20–40 "
        f"(shaded); anchor=1; post-final J enabled; target rule {TARGET_RULES[args.task]}. "
        f"Dense estimates move in increments of 1/64; circles are independent n={anchor_examples} evaluations.",
        fontsize=8.0,
        color="#444444",
    )
    fig.subplots_adjust(left=0.091, right=0.986, top=0.953, bottom=0.083)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    bin_summary: dict[str, dict[str, float | int]] = {}
    for lower in range(20, 100, 10):
        upper = lower + 10
        mask = (x >= lower) & (x < upper if upper < 100 else x <= upper)
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
    first_below_summary: dict[str, dict[str, int] | None] = {}
    for threshold in (0.95, 0.90, 0.75, 0.50):
        logical_length = first_below(x, full, threshold)
        first_below_summary[f"{threshold:.2f}"] = (
            None
            if logical_length is None
            else {
                "logical_length": logical_length,
                "target_loops": logical_length + step_offset,
            }
        )
    summary = {
        "task": args.task,
        "target_loop_rule": TARGET_RULES[args.task],
        "dense_points": int(x.size),
        "dense_examples_per_point": 64,
        "anchor_points": int(xa.size),
        "anchor_examples_per_point": anchor_examples,
        "j_better_points": int((delta > 0).sum()),
        "j_tied_points": int((delta == 0).sum()),
        "j_worse_points": int((delta < 0).sum()),
        "first_j_below_threshold": first_below_summary,
        "max_dense_gain": {"length": int(x[np.argmax(delta)]), "value": float(delta.max())},
        "min_dense_gain": {"length": int(x[np.argmin(delta)]), "value": float(delta.min())},
        "local_trend_half_width": args.trend_half_width,
        "local_floor_half_width": args.floor_half_width,
        "trend_boundaries": boundary_summary,
        "ten_length_bins": bin_summary,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
