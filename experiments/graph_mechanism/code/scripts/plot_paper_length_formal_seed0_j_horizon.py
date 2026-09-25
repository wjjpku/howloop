#!/usr/bin/env python3
"""Render the paired GPU seed-0 J-horizon screen and component controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT
    / "results/paper_length_telomere_20260731/formal_seed0_j_horizon"
)
FIGURE_ROOT = ROOT / "results/paper_length_telomere_20260731/figures"
J20_SUMMARY = RESULT_ROOT / "audits/logical1to20/summary.json"
J40_SUMMARY = RESULT_ROOT / "audits/logical1to40/summary.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def target_em(summary: dict, mode: str, length: int) -> float:
    return float(
        summary["curves"][mode][str(length)]["target_step_exact_match"]
    )


def reliable_prefix(lengths: list[int], values: list[float], q: float) -> int | None:
    horizon = None
    for length, value in zip(lengths, values, strict=True):
        if value < q:
            break
        horizon = length
    return horizon


def main() -> None:
    j20 = load(J20_SUMMARY)
    j40 = load(J40_SUMMARY)
    lengths = [20, 30, 40, 50, 60, 75, 84, 100, 120, 150, 200]
    curves = {
        "Raw": [target_em(j20, "raw", length) for length in lengths],
        r"$J_{1\mathrm{--}20}$": [
            target_em(j20, "full", length) for length in lengths
        ],
        r"$J_{1\mathrm{--}40}$": [
            target_em(j40, "full", length) for length in lengths
        ],
    }

    colors = ["#4A5568", "#2B6CB0", "#C05621"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.35), constrained_layout=True)

    for (label, values), color, marker in zip(
        curves.items(), colors, markers, strict=True
    ):
        horizon_90 = reliable_prefix(lengths, values, 0.90)
        horizon_95 = reliable_prefix(lengths, values, 0.95)
        axes[0].plot(
            lengths,
            values,
            label=f"{label}  ($\\tau_{{.90}}={horizon_90}$, "
            f"$\\tau_{{.95}}={horizon_95}$)",
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=5.3,
        )
    axes[0].axhline(0.90, color="#718096", linestyle="--", linewidth=1.0)
    axes[0].axhline(0.95, color="#A0AEC0", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Logical length $n$")
    axes[0].set_ylabel(r"Strict exact match at loop $n$")
    axes[0].set_ylim(-0.025, 1.035)
    axes[0].set_xticks(lengths, [str(length) for length in lengths], rotation=35)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8.2, loc="lower left")
    axes[0].set_title("A. Finite right-shift of the reliable horizon")

    labels = [
        "Raw",
        "J20",
        "J20\nno AB",
        "J20\nD = I",
        "J40",
        "J40\nno AB",
        "J40\nD = I",
        "Executor\noff",
    ]
    values = [
        target_em(j20, "raw", 100),
        target_em(j20, "full", 100),
        target_em(j20, "no_AB", 100),
        target_em(j20, "identity_D", 100),
        target_em(j40, "full", 100),
        target_em(j40, "no_AB", 100),
        target_em(j40, "identity_D", 100),
        target_em(j40, "full_executor_off", 100),
    ]
    bar_colors = [
        colors[0],
        colors[1],
        "#90CDF4",
        "#63B3ED",
        colors[2],
        "#FBD38D",
        "#ED8936",
        "#CBD5E0",
    ]
    bars = axes[1].bar(range(len(labels)), values, color=bar_colors, width=0.78)
    axes[1].set_xticks(range(len(labels)), labels, fontsize=7.7)
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("Strict exact match at L100")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_title("B. Low-rank and frozen-executor controls")
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.024,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
        )

    fig.suptitle(
        "Released64 parity, seed-0 final checkpoint — paired GPU screen (N=512)",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.018,
        "Backbone train length 1–20; J uses answer-only CE; all endpoints use the registered loop=n.",
        ha="center",
        fontsize=9,
    )
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_ROOT / "formal_seed0_j_horizon_screen.png"
    pdf_path = FIGURE_ROOT / "formal_seed0_j_horizon_screen.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
