#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def indexed_metric(
    rows: list[dict[str, str]], metric: str
) -> dict[tuple[int, str], tuple[float, int]]:
    return {
        (int(row["length"]), row["variant"]): (
            float(row[metric]),
            int(row["examples"]),
        )
        for row in rows
        if row["variant"] in {"raw", "full"}
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    transition = indexed_metric(
        read_rows(args.transition_csv), "supervised_digit_exact_match"
    )
    context = indexed_metric(
        read_rows(args.context_csv), "supervised_digit_exact_match"
    )
    comparison = (
        indexed_metric(
            read_rows(args.comparison_transition_csv),
            "supervised_digit_exact_match",
        )
        if args.comparison_transition_csv is not None
        else None
    )
    transition_lengths = sorted({length for length, _ in transition})
    if transition_lengths != list(range(min(transition_lengths), max(transition_lengths) + 1)):
        raise ValueError("transition lengths must be contiguous")
    context_length = min(transition_lengths) - 1
    if any((context_length, variant) not in context for variant in ("raw", "full")):
        raise ValueError("context CSV must contain the length immediately before transition")

    lengths = [context_length, *transition_lengths]
    values: dict[str, list[float]] = {}
    for variant in ("raw", "full"):
        values[variant] = [
            context[(context_length, variant)][0],
            *[transition[(length, variant)][0] for length in transition_lengths],
        ]
    if comparison is not None:
        for length in transition_lengths:
            if comparison[(length, "raw")] != transition[(length, "raw")]:
                raise ValueError(
                    "comparison must use the same raw examples and evaluation seed"
                )
        values["comparison"] = [
            context[(context_length, "full")][0],
            *[comparison[(length, "full")][0] for length in transition_lengths],
        ]

    deltas = np.asarray(
        [
            transition[(length, "full")][0]
            - transition[(length, "raw")][0]
            for length in transition_lengths
        ]
    )
    # Conservative visual uncertainty: treats the raw and J proportions as
    # independent. They were evaluated on matched generators, so this is not a
    # paired significance test.
    standard_errors = []
    for length in transition_lengths:
        raw, raw_n = transition[(length, "raw")]
        full, full_n = transition[(length, "full")]
        standard_errors.append(
            math.sqrt(raw * (1 - raw) / raw_n + full * (1 - full) / full_n)
        )
    error95 = 1.96 * np.asarray(standard_errors)
    comparison_deltas: np.ndarray | None = None
    comparison_error95: np.ndarray | None = None
    if comparison is not None:
        comparison_deltas = np.asarray(
            [
                comparison[(length, "full")][0]
                - comparison[(length, "raw")][0]
                for length in transition_lengths
            ]
        )
        comparison_standard_errors = []
        for length in transition_lengths:
            raw, raw_n = comparison[(length, "raw")]
            full, full_n = comparison[(length, "full")]
            comparison_standard_errors.append(
                math.sqrt(raw * (1 - raw) / raw_n + full * (1 - full) / full_n)
            )
        comparison_error95 = 1.96 * np.asarray(comparison_standard_errors)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {
        "raw": "#6c757d",
        "full": "#d62728",
        "comparison": "#f58518",
    }
    labels = {
        "raw": "raw",
        "full": "J (5,376 updates)",
        "comparison": "J (14,336 updates)",
    }
    plotted_variants = ("raw", "full") + (
        ("comparison",) if comparison is not None else ()
    )
    for variant in plotted_variants:
        axes[0].plot(
            lengths,
            values[variant],
            marker="o",
            linewidth=2.2,
            color=colors[variant],
            label=labels[variant],
        )
    for threshold, style in ((0.90, "--"), (0.50, ":")):
        axes[0].axhline(
            threshold,
            color="#4c78a8",
            linestyle=style,
            linewidth=1.2,
            label=f"EM={threshold:.2f}",
        )
    axes[0].axvline(10.5, color="#4c78a8", linestyle="--", linewidth=1.1)
    axes[0].set(
        title="Out-of-horizon transition",
        xlabel="logical addition length m",
        ylabel="supervised m-digit exact match",
        xticks=lengths,
        ylim=(-0.02, 1.03),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    x = np.asarray(transition_lengths, dtype=float)
    width = 0.36 if comparison_deltas is not None else 0.72
    short_x = x - width / 2 if comparison_deltas is not None else x
    axes[1].bar(
        short_x,
        deltas,
        yerr=error95,
        width=width,
        capsize=4,
        color="#d62728",
        alpha=0.9,
        label=f"5,376: mean={deltas.mean():+.4f}",
    )
    if comparison_deltas is not None and comparison_error95 is not None:
        axes[1].bar(
            x + width / 2,
            comparison_deltas,
            yerr=comparison_error95,
            width=width,
            capsize=4,
            color="#f58518",
            alpha=0.9,
            label=f"14,336: mean={comparison_deltas.mean():+.4f}",
        )
    axes[1].axhline(0, color="black", linewidth=1)
    for position, delta in zip(short_x, deltas, strict=True):
        if abs(delta) < 0.005:
            continue
        axes[1].text(
            position,
            delta + (0.004 if delta >= 0 else -0.004),
            f"{delta:+.3f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8.5,
        )
    if comparison_deltas is not None:
        for position, delta in zip(x + width / 2, comparison_deltas, strict=True):
            if abs(delta) < 0.005:
                continue
            axes[1].text(
                position,
                delta + (0.004 if delta >= 0 else -0.004),
                f"{delta:+.3f}",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                fontsize=8.0,
            )
    axes[1].set(
        title="J gain at each transition length",
        xlabel="logical addition length m",
        ylabel="J EM − raw EM",
        xticks=x,
        xticklabels=[str(length) for length in transition_lengths],
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")
    figure.suptitle(
        "Addition LSB→MSB, causal, NoPE; T(m)=m\n"
        "m=12..17 uses 8,192 examples per length; error bars are conservative unpaired 95% intervals",
        fontsize=13,
    )
    figure.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "addition_tn_transition_evidence.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)

    frontier_values = {
        variant: dict(zip(lengths, values[variant], strict=True))
        for variant in plotted_variants
    }
    summary = {
        "status": "complete",
        "figure": str(output),
        "transition_lengths": transition_lengths,
        "examples_per_transition_length": {
            str(length): transition[(length, "raw")][1]
            for length in transition_lengths
        },
        "raw_transition_mean": float(
            np.mean([transition[(length, "raw")][0] for length in transition_lengths])
        ),
        "full_transition_mean": float(
            np.mean([transition[(length, "full")][0] for length in transition_lengths])
        ),
        "mean_gain": float(deltas.mean()),
        "comparison_full_transition_mean": (
            float(
                np.mean(
                    [
                        comparison[(length, "full")][0]
                        for length in transition_lengths
                    ]
                )
            )
            if comparison is not None
            else None
        ),
        "comparison_mean_gain": (
            float(comparison_deltas.mean())
            if comparison_deltas is not None
            else None
        ),
        "frontiers": {
            variant: {
                str(threshold): max(
                    length
                    for length, value in frontier_values[variant].items()
                    if value >= threshold
                )
                for threshold in (0.9, 0.5)
            }
            for variant in plotted_variants
        },
        "uncertainty_note": (
            "Error bars use independent-binomial variance although raw and J use "
            "matched generators; they are visual diagnostics, not a paired test."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-csv", type=Path, required=True)
    parser.add_argument("--comparison-transition-csv", type=Path)
    parser.add_argument("--context-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
