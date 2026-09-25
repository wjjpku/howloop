#!/usr/bin/env python3
"""Plot seed-0 paper-baseline gains and the J insertion-anchor sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = {
    "Parity": {
        "label": "parity_adaptive_step_released64_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
        "anchor": 1,
    },
    "Copy": {
        "label": "copy_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
        "anchor": 1,
    },
    "Addition": {
        "label": "addition_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
        "anchor": 1,
    },
    "Sum-Reverse": {
        "label": "sum_reverse_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
        "anchor": 1,
    },
}

DIAGNOSTIC_LENGTHS = {
    "Parity": 100,
    "Copy": 40,
    "Addition": 40,
    "Sum-Reverse": 40,
}

SWEEPS = {
    "Parity at L100": {
        "length": 100,
        "runs": {
            1: "parity_adaptive_step_released64_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
            10: "parity_adaptive_step_released64_seed0_rank48_identitywarmup_logical20to40_anchor10_seed211001",
            15: "parity_adaptive_step_released64_seed0_rank48_identitywarmup_logical20to40_anchor15_seed211001",
            19: "parity_adaptive_step_released64_seed0_rank48_identitywarmup_logical20to40_anchor19_seed211001",
        },
        "color": "#2B6CB0",
    },
    "Addition at L50": {
        "length": 50,
        "runs": {
            1: "addition_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
            10: "addition_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor10_seed211001",
            15: "addition_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor15_seed211001",
            19: "addition_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor19_seed211001",
        },
        "color": "#C05621",
    },
    "Sum-Reverse at L40": {
        "length": 40,
        "runs": {
            1: "sum_reverse_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor1_seed211001",
            10: "sum_reverse_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor10_seed211001",
            15: "sum_reverse_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor15_seed211001",
            18: "sum_reverse_adaptive_step_official_seed0_rank48_identitywarmup_logical20to40_anchor18_seed211001",
        },
        "color": "#2F855A",
    },
}


def target_curve(summary: dict, variant: str) -> tuple[list[int], list[float]]:
    lengths = [int(value) for value in summary["lengths"]]
    values = [
        float(summary["curves"][variant][str(length)]["target_step_exact_match"])
        for length in lengths
    ]
    return lengths, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        task: json.loads(
            (args.summary_root / config["label"] / "summary.json").read_text()
        )
        for task, config in RUNS.items()
    }

    figure = plt.figure(figsize=(12.6, 7.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    raw_color = "#718096"
    task_colors = ["#2B6CB0", "#805AD5", "#C05621", "#2F855A"]
    for index, ((task, config), color) in enumerate(
        zip(RUNS.items(), task_colors, strict=True)
    ):
        axis = figure.add_subplot(grid[0, index])
        summary = summaries[task]
        lengths, raw = target_curve(summary, "raw")
        _, full = target_curve(summary, "full")
        axis.plot(
            lengths,
            raw,
            "o--",
            color=raw_color,
            linewidth=1.7,
            markersize=4.5,
            label="Raw",
        )
        axis.plot(
            lengths,
            full,
            "o-",
            color=color,
            linewidth=2.2,
            markersize=4.5,
            label=f"J, anchor {config['anchor']}",
        )
        axis.axvspan(20, 40, color="#ECC94B", alpha=0.11, linewidth=0)
        axis.set_title(task)
        axis.set_xlabel("Logical length")
        axis.set_ylim(-0.035, 1.035)
        axis.grid(alpha=0.2)
        if index == 0:
            axis.set_ylabel("Strict target-step EM")
            axis.legend(frameon=False, fontsize=8, loc="lower left")

    anchor_axis = figure.add_subplot(grid[1, :])
    for label, values in SWEEPS.items():
        anchors = list(values["runs"])
        sweep_summaries = [
            json.loads(
                (args.summary_root / run_label / "summary.json").read_text()
            )
            for run_label in values["runs"].values()
        ]
        length = values["length"]
        em = [
            float(summary["curves"]["full"][str(length)]["target_step_exact_match"])
            for summary in sweep_summaries
        ]
        raw = float(
            sweep_summaries[0]["curves"]["raw"][str(length)][
                "target_step_exact_match"
            ]
        )
        color = values["color"]
        anchor_axis.plot(
            anchors,
            em,
            "o-",
            color=color,
            linewidth=2.2,
            markersize=6,
            label=label,
        )
        anchor_axis.axhline(
            raw, color=color, linestyle=":", linewidth=1.1, alpha=0.7
        )
        for anchor, score in zip(anchors, em, strict=True):
            anchor_axis.annotate(
                f"{score:.3f}",
                (anchor, score),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
    anchor_axis.set_xticks([1, 5, 10, 15, 18, 19])
    anchor_axis.set_xlabel("Anchor K (J first applied before loop K+1)")
    anchor_axis.set_ylabel("Strict target-step EM")
    anchor_axis.set_ylim(-0.045, 1.09)
    anchor_axis.grid(alpha=0.2)
    anchor_axis.legend(frameon=False, ncol=3, loc="lower left")
    anchor_axis.set_title(
        "Insertion timing is task-dependent: late is sufficient for phase correction; early is better for distributed execution"
    )

    figure.suptitle(
        "Identity-initialized diagonal + rank-48 J on the paper baselines (seed 0)",
        fontsize=13,
    )
    figure.text(
        0.5,
        -0.015,
        "Frozen backbone; J trained only on logical lengths 20-40 with final answer CE. "
        "Yellow region marks the J-training horizon; all endpoints use 512 examples.",
        ha="center",
        fontsize=9,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    png = args.output_root / "identity_anchor_paper_baselines_seed0.png"
    pdf = args.output_root / "identity_anchor_paper_baselines_seed0.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)

    control_variants = [
        ("Raw", "raw", "#718096"),
        ("Full J", "full", "#2B6CB0"),
        ("D = I", "identity_D", "#63B3ED"),
        ("No AB", "no_AB", "#F6AD55"),
        ("Executor off", "full_executor_off", "#CBD5E0"),
    ]
    task_names = list(RUNS)
    x_positions = list(range(len(task_names)))
    width = 0.16
    control_figure, control_axis = plt.subplots(
        figsize=(10.8, 4.7), constrained_layout=True
    )
    for variant_index, (display_name, variant, color) in enumerate(
        control_variants
    ):
        offset = (variant_index - 2) * width
        values = []
        for task in task_names:
            length = DIAGNOSTIC_LENGTHS[task]
            values.append(
                float(
                    summaries[task]["curves"][variant][str(length)][
                        "target_step_exact_match"
                    ]
                )
            )
        bars = control_axis.bar(
            [position + offset for position in x_positions],
            values,
            width=width,
            color=color,
            label=display_name,
        )
        for bar, value in zip(bars, values, strict=True):
            if 0.015 <= value < 0.9:
                control_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                )
    control_axis.set_xticks(
        x_positions,
        [
            f"{task}\nL{DIAGNOSTIC_LENGTHS[task]}"
            for task in task_names
        ],
    )
    control_axis.set_ylim(0, 1.13)
    control_axis.set_ylabel("Strict target-step EM")
    control_axis.grid(axis="y", alpha=0.2)
    control_axis.legend(frameon=False, ncol=5, loc="upper center")
    control_axis.set_title(
        "Component controls: the low-rank update and frozen recurrent executor are both required"
    )
    control_figure.text(
        0.5,
        -0.01,
        "Anchor 1; 512 examples per task-length. D = I retains AB+b; No AB retains D+b.",
        ha="center",
        fontsize=9,
    )
    control_png = args.output_root / "identity_anchor_component_controls_seed0.png"
    control_pdf = args.output_root / "identity_anchor_component_controls_seed0.pdf"
    control_figure.savefig(control_png, dpi=300, bbox_inches="tight")
    control_figure.savefig(control_pdf, bbox_inches="tight")
    print(control_png)
    print(control_pdf)


if __name__ == "__main__":
    main()
