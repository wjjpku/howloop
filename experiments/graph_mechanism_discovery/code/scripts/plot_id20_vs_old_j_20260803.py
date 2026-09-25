#!/usr/bin/env python3
"""Compare backbone-solved-range J training with transition-range training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TASK_LABELS = {"addition": "Addition", "sum_reverse": "Sum-Reverse"}


def read_rows(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    values: dict[str, dict[int, dict[str, Any]]] = {"raw": {}, "full": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant not in values:
                continue
            length = int(row["length"])
            values[variant][length] = {
                **row,
                "length": length,
                "examples": int(row["examples"]),
                "exact_successes": int(row["exact_successes"]),
                "exact_match": float(row["exact_match"]),
            }
    return values


def audit_matched_raw(
    old: dict[str, dict[int, dict[str, Any]]],
    new: dict[str, dict[int, dict[str, Any]]],
) -> list[int]:
    overlap = sorted(set(old["raw"]).intersection(new["raw"]))
    mismatches = [
        length
        for length in overlap
        if (
            old["raw"][length]["examples"],
            old["raw"][length]["exact_successes"],
        )
        != (
            new["raw"][length]["examples"],
            new["raw"][length]["exact_successes"],
        )
    ]
    if mismatches:
        raise ValueError(f"raw matched-seed audit failed at lengths {mismatches}")
    return overlap


def aggregate(
    rows: dict[int, dict[str, Any]], minimum: int, maximum: int
) -> dict[str, int | float] | None:
    selected = [rows[length] for length in sorted(rows) if minimum <= length <= maximum]
    if not selected:
        return None
    successes = sum(row["exact_successes"] for row in selected)
    examples = sum(row["examples"] for row in selected)
    return {
        "length_points": len(selected),
        "exact_successes": successes,
        "examples": examples,
        "micro_exact_match": successes / examples,
    }


def first_below(
    rows: dict[int, dict[str, Any]], *, minimum: int, threshold: float
) -> int | None:
    for length in sorted(rows):
        if length >= minimum and rows[length]["exact_match"] < threshold:
            return length
    return None


def arrays(rows: dict[int, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(sorted(rows), dtype=float)
    values = np.asarray([rows[int(length)]["exact_match"] for length in lengths])
    return lengths, values


def main() -> None:
    parser = argparse.ArgumentParser()
    for task in TASK_LABELS:
        parser.add_argument(f"--{task.replace('_', '-')}-old", type=Path, required=True)
        parser.add_argument(f"--{task.replace('_', '-')}-new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "addition": (args.addition_old, args.addition_new),
        "sum_reverse": (args.sum_reverse_old, args.sum_reverse_new),
    }
    data: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "evaluation": "strict answer-sequence exact match",
        "dense_examples_per_length": 64,
        "training_ranges": {
            "transition_range_j": [20, 40],
            "backbone_solved_range_j": [1, 20],
        },
        "interpretation_scope": (
            "one official seed-0 backbone and one controller seed per curve; "
            "not a universal impossibility claim"
        ),
        "tasks": {},
    }
    for task, (old_path, new_path) in paths.items():
        old, new = read_rows(old_path), read_rows(new_path)
        overlap = audit_matched_raw(old, new)
        data[task] = (old, new)
        task_summary: dict[str, Any] = {
            "matched_raw_overlap": [min(overlap), max(overlap)],
            "matched_raw_points": len(overlap),
            "matched_raw_audit": "pass",
            "ranges": {},
            "first_below_threshold_from_n20": {},
        }
        for lower, upper in ((1, 20), (21, 30), (31, 40), (41, 50), (51, 100)):
            key = f"{lower}-{upper}"
            task_summary["ranges"][key] = {
                "raw": aggregate(new["raw"], lower, upper),
                "transition_range_j_20_40": aggregate(
                    old["full"], lower, upper
                ),
                "backbone_solved_range_j_1_20_50k": aggregate(
                    new["full"], lower, upper
                ),
            }
        for threshold in (0.95, 0.90, 0.75, 0.50):
            task_summary["first_below_threshold_from_n20"][f"{threshold:.2f}"] = {
                "raw": first_below(new["raw"], minimum=20, threshold=threshold),
                "transition_range_j_20_40": first_below(
                    old["full"], minimum=20, threshold=threshold
                ),
                "backbone_solved_range_j_1_20_50k": first_below(
                    new["full"], minimum=20, threshold=threshold
                ),
            }
        task_summary["selected_lengths"] = {
            str(length): {
                "raw": new["raw"][length]["exact_match"],
                "transition_range_j_20_40": old["full"][length]["exact_match"],
                "backbone_solved_range_j_1_20_50k": new["full"][length][
                    "exact_match"
                ],
            }
            for length in (20, 25, 30, 35, 40, 45, 50, 60, 75, 100)
        }
        summary["tasks"][task] = task_summary

    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 240})
    colors = {"raw": "#b44b35", "old": "#6f69a8", "new": "#087e8b"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), sharex="col")
    for column, task in enumerate(("addition", "sum_reverse")):
        old, new = data[task]
        x_raw, y_raw = arrays(new["raw"])
        x_old, y_old = arrays(old["full"])
        x_new, y_new = arrays(new["full"])
        ax = axes[0, column]
        ax.axvspan(
            1,
            20,
            color=colors["new"],
            alpha=0.065,
            label="Backbone-solved regime / J train range n=1–20",
        )
        ax.plot(x_raw, y_raw, color=colors["raw"], lw=1.2, ls="--", label="Raw backbone")
        ax.plot(
            x_old,
            y_old,
            color=colors["old"],
            lw=1.45,
            label="J trained on transition range n=20–40",
        )
        ax.plot(
            x_new,
            y_new,
            color=colors["new"],
            lw=1.65,
            label="J trained on solved range n=1–20 (50k updates)",
        )
        ax.axhline(0.5, color="#777", lw=0.8, ls=":")
        ax.set_title(TASK_LABELS[task])
        ax.set_ylim(-0.035, 1.035)
        ax.grid(axis="y", color="#ddd", lw=0.7)
        if column == 0:
            ax.set_ylabel("Strict sequence EM")
        ax.legend(fontsize=8.2, loc="lower left")

        common = sorted(set(old["full"]).intersection(new["full"]))
        x = np.asarray(common)
        raw = np.asarray([new["raw"][length]["exact_match"] for length in common])
        old_delta = np.asarray([old["full"][length]["exact_match"] for length in common]) - raw
        new_delta = np.asarray([new["full"][length]["exact_match"] for length in common]) - raw
        ax = axes[1, column]
        ax.axvspan(1, 20, color=colors["new"], alpha=0.065)
        ax.axhline(0, color="#555", lw=0.8)
        ax.plot(
            x,
            old_delta,
            color=colors["old"],
            lw=1.35,
            label="Transition-range J − raw",
        )
        ax.plot(
            x,
            new_delta,
            color=colors["new"],
            lw=1.55,
            label="Solved-range J − raw",
        )
        ax.set_ylim(-1.035, 1.035)
        ax.set_xlim(1, 100)
        ax.set_xlabel("Logical length n")
        ax.grid(axis="y", color="#ddd", lw=0.7)
        ax.legend(fontsize=8.3, loc="lower right")
        if column == 0:
            ax.set_ylabel("EM gain")
    fig.suptitle(
        "Does final-CE training restricted to the backbone-solved regime "
        "extend the length horizon?"
    )
    fig.text(
        0.01,
        0.008,
        "Official seed-0 frozen backbones; rank-48 diagonal+low-rank J; identity initialization; "
        "anchor=1; post-final J; 64 matched examples/length. One controller seed per curve: "
        "a within-setting comparison, not a universal impossibility claim.",
        fontsize=8.2,
        color="#444",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.09, hspace=0.16, wspace=0.11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
