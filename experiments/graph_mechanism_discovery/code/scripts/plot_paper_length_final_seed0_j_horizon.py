#!/usr/bin/env python3
"""Render the paired seed-0 final-backbone J horizon pilot figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "results/paper_length_telomere_20260731/local_pilots"
OUTPUT_ROOT = ROOT / "results/paper_length_telomere_20260731/figures"
J20_SUMMARY = (
    PILOT_ROOT
    / "final_seed0_audit_j_logical1to20_fixed256_n128/summary.json"
)
J40_SUMMARY = (
    PILOT_ROOT
    / "final_seed0_audit_j_logical1to40_fixed256_n128/summary.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def target_em(summary: dict, mode: str, length: int) -> float:
    return float(
        summary["curves"][mode][str(length)]["target_step_exact_match"]
    )


def target_nll(summary: dict, mode: str, length: int) -> float:
    return float(
        summary["curves"][mode][str(length)]["target_step_answer_nll"]
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
    lengths = [20, 40, 50, 60, 75, 84, 100]
    curves = {
        "Raw": [target_em(j20, "raw", length) for length in lengths],
        r"$J_{1\mathrm{--}20}$": [
            target_em(j20, "full", length) for length in lengths
        ],
        r"$J_{1\mathrm{--}40}$": [
            target_em(j40, "full", length) for length in lengths
        ],
    }
    nll = {
        "Raw": [target_nll(j20, "raw", length) for length in lengths],
        r"$J_{1\mathrm{--}20}$": [
            target_nll(j20, "full", length) for length in lengths
        ],
        r"$J_{1\mathrm{--}40}$": [
            target_nll(j40, "full", length) for length in lengths
        ],
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "final_seed0_j_horizon_pilot.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "length",
                "raw_target_em",
                "j1_20_target_em",
                "j1_40_target_em",
                "raw_target_nll",
                "j1_20_target_nll",
                "j1_40_target_nll",
            ]
        )
        for index, length in enumerate(lengths):
            writer.writerow(
                [
                    length,
                    curves["Raw"][index],
                    curves[r"$J_{1\mathrm{--}20}$"][index],
                    curves[r"$J_{1\mathrm{--}40}$"][index],
                    nll["Raw"][index],
                    nll[r"$J_{1\mathrm{--}20}$"][index],
                    nll[r"$J_{1\mathrm{--}40}$"][index],
                ]
            )

    colors = ["#4A5568", "#2B6CB0", "#C05621"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.15), constrained_layout=True)

    for (label, values), color, marker in zip(
        curves.items(), colors, markers, strict=True
    ):
        horizon = reliable_prefix(lengths, values, 0.90)
        axes[0].plot(
            lengths,
            values,
            label=f"{label}  ($\\tau_{{.90}}={horizon}$)",
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=5.5,
        )
    axes[0].axhline(0.90, color="#718096", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Logical length $n$")
    axes[0].set_ylabel(r"Strict target EM at loop $n$")
    axes[0].set_ylim(-0.025, 1.035)
    axes[0].set_xticks(lengths)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    axes[0].set_title("A. Out-of-training-horizon execution")

    labels = [
        "Raw",
        "J20",
        "J20\nno AB",
        "J20\nD = I",
        "J20\nexecutor off",
        "J40",
        "J40\nexecutor off",
    ]
    values = [
        target_em(j20, "raw", 100),
        target_em(j20, "full", 100),
        target_em(j20, "no_AB", 100),
        target_em(j20, "identity_D", 100),
        target_em(j20, "full_executor_off", 100),
        target_em(j40, "full", 100),
        target_em(j40, "full_executor_off", 100),
    ]
    bar_colors = [
        colors[0],
        colors[1],
        "#90CDF4",
        "#63B3ED",
        "#CBD5E0",
        colors[2],
        "#FBD38D",
    ]
    bars = axes[1].bar(range(len(labels)), values, color=bar_colors, width=0.78)
    axes[1].set_xticks(range(len(labels)), labels, fontsize=8)
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("Strict target EM at L100")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].set_title("B. Component and executor controls")
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=0,
        )

    fig.suptitle(
        "Released64 parity, seed-0 final checkpoint — paired local pilot (N=128)",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.015,
        "Backbone train length 1–20; J uses pure final CE; every endpoint is loop=n.",
        ha="center",
        fontsize=9,
    )
    png_path = OUTPUT_ROOT / "final_seed0_j_horizon_pilot.png"
    pdf_path = OUTPUT_ROOT / "final_seed0_j_horizon_pilot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
