#!/usr/bin/env python3
"""Render the released64 parity J-horizon aggregate with seed error bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    ROOT
    / "results/paper_length_telomere_20260731"
    / "released64_j_horizon_aggregate/summary.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/paper_length_telomere_20260731/figures"
    / "released64_j_horizon_aggregate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric_rows(summary: dict) -> dict[tuple[str, int], dict]:
    return {
        (str(row["arm"]), int(row["length"])): row
        for row in summary["aggregate"]
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = metric_rows(summary)
    lengths = [int(length) for length in summary["evaluation_lengths"]]
    backbone_seeds = [int(seed) for seed in summary["backbone_seeds"]]
    examples = int(summary["examples_per_length_minimum"])

    display_arms = [
        ("raw", "Raw", "#4A5568", "o"),
        ("J_1-20", r"$J_{1\mathrm{--}20}$", "#2B6CB0", "s"),
        ("J_1-40", r"$J_{1\mathrm{--}40}$", "#C05621", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)

    horizon_payload = summary["horizons"]
    for arm, label, color, marker in display_arms:
        means = [rows[(arm, length)]["target_step_exact_match_mean"] for length in lengths]
        stds = [rows[(arm, length)]["target_step_exact_match_std"] for length in lengths]
        horizon_90 = horizon_payload["0.90"]["mean_curve"][arm]
        horizon_95 = horizon_payload["0.95"]["mean_curve"][arm]
        axes[0].errorbar(
            lengths,
            means,
            yerr=stds,
            label=f"{label}  ($\\tau_{{.90}}={horizon_90}$, "
            f"$\\tau_{{.95}}={horizon_95}$)",
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=5.3,
            capsize=2.5,
        )
    axes[0].axhline(0.90, color="#718096", linestyle="--", linewidth=1.0)
    axes[0].axhline(0.95, color="#A0AEC0", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Logical length $n$")
    axes[0].set_ylabel(r"Strict exact match at loop $n$")
    axes[0].set_ylim(-0.025, 1.035)
    axes[0].set_xticks(lengths, [str(length) for length in lengths], rotation=35)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8.1, loc="lower left")
    axes[0].set_title("A. Reliable computation horizon")

    control_arms = [
        ("raw", "Raw", "#4A5568"),
        ("J_1-20", "J20", "#2B6CB0"),
        ("J_1-20/no_AB", "J20\nno AB", "#90CDF4"),
        ("J_1-20/identity_D", "J20\nD = I", "#63B3ED"),
        ("J_1-40", "J40", "#C05621"),
        ("J_1-40/no_AB", "J40\nno AB", "#FBD38D"),
        ("J_1-40/identity_D", "J40\nD = I", "#ED8936"),
        ("J_1-40/executor_off", "Executor\noff", "#CBD5E0"),
    ]
    labels = [label for _, label, _ in control_arms]
    means = [rows[(arm, 100)]["target_step_exact_match_mean"] for arm, _, _ in control_arms]
    stds = [rows[(arm, 100)]["target_step_exact_match_std"] for arm, _, _ in control_arms]
    colors = [color for _, _, color in control_arms]
    bars = axes[1].bar(
        range(len(labels)),
        means,
        yerr=stds,
        color=colors,
        width=0.78,
        capsize=2.5,
    )
    axes[1].set_xticks(range(len(labels)), labels, fontsize=7.7)
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("Strict exact match at L100")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_title("B. Low-rank and frozen-executor controls")
    for bar, mean in zip(bars, means, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )

    fig.suptitle(
        "Fan et al. released64 parity baseline — "
        f"{len(backbone_seeds)} backbone seed(s), N={examples} per length",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.018,
        "Backbone train length 1–20; J uses answer-only CE; all endpoints use the registered loop=n.",
        ha="center",
        fontsize=9,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
