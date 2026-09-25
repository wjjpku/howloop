"""Create matched before/after tables and plots for long H1-start F/J schedules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=dynamic-eval-dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_plot(
    rows: list[dict[str, Any]], schedules: list[str], metric: str,
    title: str, ylabel: str, path: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(16, 13), dpi=180, squeeze=False)
    for axis, schedule in zip(axes.flat, schedules, strict=True):
        selected = [row for row in rows if row["schedule"] == schedule]
        for label in dict.fromkeys(row["bank"] for row in selected):
            current = sorted(
                [row for row in selected if row["bank"] == label],
                key=lambda row: row["forward_count"],
            )
            axis.plot(
                [row["forward_count"] for row in current],
                [row[metric] for row in current],
                marker="o", markersize=2.3, linewidth=1.1, label=label,
            )
        axis.axvline(8, color="gray", linestyle=":", linewidth=0.8)
        axis.set_title(schedule)
        axis.set_xlabel("physical F count from raw input")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sources = {}
    for item in args.run:
        label, raw = item.split("=", 1)
        run_dir = Path(raw)
        sources[label] = str(run_dir)
        for row in read_csv(run_dir / "dynamic_j_forward_summary.csv"):
            rows.append(
                {
                    "bank": label,
                    **{
                        key: (
                            int(value)
                            if key in {"forward_count", "logical_phase", "j_count", "observations"}
                            else float(value)
                            if key.endswith("_mean") or key.endswith("_sem")
                            else value
                        )
                        for key, value in row.items()
                    },
                }
            )
    schedules = list(dict.fromkeys(row["schedule"] for row in rows))
    if len(schedules) != 9:
        raise ValueError(f"expected nine standard schedules, found {schedules}")
    comparison: list[dict[str, Any]] = []
    key_states = (8, 9, 10, 12, 16, 24, 32, 40)
    for label in dict.fromkeys(row["bank"] for row in rows):
        for schedule in schedules:
            selected = [
                row for row in rows
                if row["bank"] == label and row["schedule"] == schedule
            ]
            late = [row for row in selected if row["forward_count"] >= 9]
            item: dict[str, Any] = {
                "bank": label,
                "schedule": schedule,
                "late_accuracy_mean_H9_to_H40": float(
                    np.mean([row["accuracy_mean"] for row in late])
                ),
                "late_accuracy_min_H9_to_H40": float(
                    np.min([row["accuracy_mean"] for row in late])
                ),
                "last_state_accuracy_at_least_0_95": max(
                    (
                        row["forward_count"] for row in selected
                        if row["accuracy_mean"] >= 0.95
                    ),
                    default=0,
                ),
                "state_rms_range": float(
                    max(row["state_rms_mean"] for row in selected)
                    - min(row["state_rms_mean"] for row in selected)
                ),
                "answer_norm_range": float(
                    max(row["answer_norm_mean"] for row in selected)
                    - min(row["answer_norm_mean"] for row in selected)
                ),
            }
            for state in key_states:
                match = [row for row in selected if row["forward_count"] == state]
                item[f"H{state}_accuracy"] = match[0]["accuracy_mean"] if match else ""
                item[f"H{state}_state_rms"] = match[0]["state_rms_mean"] if match else ""
            comparison.append(item)
    write_csv(args.out_dir / "long_schedule_comparison.csv", comparison)
    metric_plot(
        rows, schedules, "accuracy_mean", "Long-schedule functional accuracy",
        "diagnostic graph accuracy", args.out_dir / "long_schedule_accuracy.png",
    )
    metric_plot(
        rows, schedules, "state_rms_mean", "Long-schedule hidden-state RMS",
        "hidden RMS", args.out_dir / "long_schedule_hidden_rms.png",
    )
    metric_plot(
        rows, schedules, "answer_norm_mean", "Long-schedule answer-token norm",
        "answer-token L2 norm", args.out_dir / "long_schedule_answer_norm.png",
    )
    result = {
        "status": "complete",
        "sources": sources,
        "schedules": schedules,
        "comparison": comparison,
        "claim_boundary": (
            "Readouts away from the backbone's supervised H8 boundary are diagnostic; "
            "high endpoint accuracy does not imply all intermediate readouts are calibrated."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
