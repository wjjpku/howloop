#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series(
    rows: Sequence[dict[str, str]], variant: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["variant"] == variant),
        key=lambda row: int(row["length"]),
    )
    return (
        np.asarray([int(row["length"]) for row in selected]),
        np.asarray([float(row[metric]) for row in selected]),
    )


def interval_summary(
    rows: Sequence[dict[str, str]], lo: int, hi: int
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for variant in ("raw", "full"):
        selected = [
            row
            for row in rows
            if row["variant"] == variant
            and lo <= int(row["length"]) <= hi
        ]
        result[variant] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in (
                "supervised_digit_exact_match",
                "final_carry_accuracy",
                "full_arithmetic_exact_match",
                "supervised_digit_bit_accuracy",
            )
        }
    result["j_minus_raw"] = {
        metric: result["full"][metric] - result["raw"][metric]
        for metric in result["raw"]
    }
    return result


def plot(
    *,
    old_rows: Sequence[dict[str, str]],
    new_rows: Sequence[dict[str, str]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    systems = (
        ("old: fixed n=10, T(m)=m+1", old_rows, "--", 0.75),
        ("new: variable m≤10, T(m)=m", new_rows, "-", 1.0),
    )
    colors = {"raw": "#6c757d", "full": "#d62728"}
    for label, rows, linestyle, alpha in systems:
        for variant in ("raw", "full"):
            lengths, digit_em = series(
                rows, variant, "supervised_digit_exact_match"
            )
            _, carry = series(rows, variant, "final_carry_accuracy")
            _, full_em = series(rows, variant, "full_arithmetic_exact_match")
            axes[0, 0].plot(
                lengths,
                digit_em,
                color=colors[variant],
                linestyle=linestyle,
                linewidth=2,
                alpha=alpha,
                label=f"{label} / {variant}",
            )
            axes[0, 1].plot(
                lengths,
                carry,
                color=colors[variant],
                linestyle=linestyle,
                linewidth=2,
                alpha=alpha,
                label=f"{label} / {variant}",
            )
            axes[1, 0].plot(
                lengths,
                full_em,
                color=colors[variant],
                linestyle=linestyle,
                linewidth=2,
                alpha=alpha,
                label=f"{label} / {variant}",
            )
        lengths, raw_digit = series(
            rows, "raw", "supervised_digit_exact_match"
        )
        _, j_digit = series(rows, "full", "supervised_digit_exact_match")
        axes[1, 1].plot(
            lengths,
            j_digit - raw_digit,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=alpha,
            label=label,
        )

    panels = (
        (axes[0, 0], "Supervised m-digit exact match", "m-digit EM"),
        (axes[0, 1], "Final carry (excluded from new CE)", "carry accuracy"),
        (axes[1, 0], "Full arithmetic exact match", "m digits + carry EM"),
        (axes[1, 1], "J gain over raw on m-digit EM", "J - raw"),
    )
    maximum_length = max(
        max(int(row["length"]) for row in old_rows),
        max(int(row["length"]) for row in new_rows),
    )
    for axis, title, ylabel in panels:
        axis.axvspan(0.5, 10.5, color="#4c78a8", alpha=0.07)
        axis.axvline(10.5, color="#4c78a8", linestyle=":", linewidth=1.2)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set(
            title=title,
            xlabel="logical addition length m",
            ylabel=ylabel,
            xlim=(1, maximum_length),
        )
        if axis is not axes[1, 1]:
            axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "Addition LSB→MSB, causal, NoPE: common-metric comparison\n"
        "solid=new variable-length T(m)=m; dashed=old fixed-n T(m)=m+1",
        fontsize=15,
    )
    figure.savefig(output, dpi=190)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_rows = read_csv(args.old_endpoint_csv)
    new_rows = read_csv(args.new_endpoint_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "addition_tn_rule_common_metric_comparison.png"
    plot(old_rows=old_rows, new_rows=new_rows, output=output)
    summary = {
        "status": "complete",
        "old_endpoint_csv": str(args.old_endpoint_csv),
        "new_endpoint_csv": str(args.new_endpoint_csv),
        "intervals": {
            "ID_1_10": {
                "old": interval_summary(old_rows, 1, 10),
                "new": interval_summary(new_rows, 1, 10),
            },
            "OOD_11_20": {
                "old": interval_summary(old_rows, 11, 20),
                "new": interval_summary(new_rows, 11, 20),
            },
            "OOD_21_30": {
                "old": interval_summary(old_rows, 21, 30),
                "new": interval_summary(new_rows, 21, 30),
            },
        },
        "figure": str(output),
        "claim_boundary": (
            "This is a common behavioral metric comparison, not a T-only "
            "ablation: the new system also changes the backbone length "
            "distribution and excludes final carry from CE."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-endpoint-csv", type=Path, required=True)
    parser.add_argument("--new-endpoint-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
