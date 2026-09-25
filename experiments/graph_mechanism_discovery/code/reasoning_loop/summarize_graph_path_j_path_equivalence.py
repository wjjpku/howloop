"""Aggregate matched before/after path-equivalence training and evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="name=/path/to/run")
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


def aggregate_eval(run: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["family"], int(row["back_count"]))].append(row)
    output: list[dict[str, Any]] = []
    for (label, family, back_count), values in groups.items():
        accuracies = [
            float(value[side])
            for value in values
            for side in ("left_accuracy", "right_accuracy")
        ]
        agreements = [float(value["prediction_agreement"]) for value in values]
        output.append(
            {
                "run": run,
                "label": label,
                "family": family,
                "back_count": back_count,
                "paths": len(accuracies),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_min": float(np.min(accuracies)),
                "accuracy_sem": float(np.std(accuracies, ddof=1) / np.sqrt(len(accuracies)))
                if len(accuracies) > 1
                else 0.0,
                "prediction_agreement_mean": float(np.mean(agreements)),
            }
        )
    return output


def aggregate_training(run: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        before = np.asarray([float(row[f"J{index}_grad_before"]) for index in range(1, 8)])
        after = np.asarray([float(row[f"J{index}_grad_after"]) for index in range(1, 8)])
        positive_before = before[before > 0]
        positive_after = after[after > 0]
        output.append(
            {
                "run": run,
                "stage": row["stage"],
                "stage_step": int(row["stage_step"]),
                "back_count": int(row["back_count"]),
                "loss": float(row["loss"]),
                "accuracy": float(row["accuracy"]),
                "learning_rate": float(row["learning_rate"]),
                "global_grad_norm": float(row["global_grad_norm"]),
                "call_spread_global": int(row["call_spread_global"]),
                "gradient_ratio_before": float(positive_before.max() / positive_before.min())
                if len(positive_before)
                else 1.0,
                "gradient_ratio_after": float(positive_after.max() / positive_after.min())
                if len(positive_after)
                else 1.0,
            }
        )
    return output


def plot_eval(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=180)
    for run in dict.fromkeys(row["run"] for row in rows):
        selected = [
            row for row in rows
            if row["run"] == run
            and row["family"] == "equivalent_words"
            and row["label"].startswith("after_")
        ]
        if not selected:
            continue
        last_label = selected[-1]["label"]
        selected = sorted(
            [row for row in selected if row["label"] == last_label],
            key=lambda row: row["back_count"],
        )
        axes[0].plot(
            [row["back_count"] for row in selected],
            [row["accuracy_mean"] for row in selected],
            "o-", label=f"{run}:{last_label}",
        )
        axes[1].plot(
            [row["back_count"] for row in selected],
            [row["prediction_agreement_mean"] for row in selected],
            "o-", label=f"{run}:{last_label}",
        )
    before = [
        row for row in rows
        if row["family"] == "equivalent_words" and row["label"] == "before"
    ]
    if before:
        reference_run = before[0]["run"]
        selected = sorted(
            [row for row in before if row["run"] == reference_run],
            key=lambda row: row["back_count"],
        )
        axes[0].plot(
            [row["back_count"] for row in selected],
            [row["accuracy_mean"] for row in selected],
            "o--", color="black", label="parent bank",
        )
        axes[1].plot(
            [row["back_count"] for row in selected],
            [row["prediction_agreement_mean"] for row in selected],
            "o--", color="black", label="parent bank",
        )
    for axis, title in zip(
        axes,
        ("held-out equivalent-path accuracy", "prediction agreement across equivalent paths"),
        strict=True,
    ):
        axis.axhline(0.95, color="gray", linestyle=":", linewidth=0.9)
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("number of J calls")
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_training(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    offset = 0
    for run in dict.fromkeys(row["run"] for row in rows):
        selected = [row for row in rows if row["run"] == run]
        x = np.arange(offset, offset + len(selected))
        axes[0, 0].plot(x, [row["loss"] for row in selected], alpha=0.7, label=run)
        axes[0, 1].plot(x, [row["accuracy"] for row in selected], alpha=0.7, label=run)
        axes[1, 0].plot(x, [row["call_spread_global"] for row in selected], alpha=0.7, label=run)
        axes[1, 1].plot(
            x, [row["gradient_ratio_before"] for row in selected], alpha=0.35,
            label=f"{run}:before",
        )
        axes[1, 1].plot(
            x, [row["gradient_ratio_after"] for row in selected], alpha=0.8,
            label=f"{run}:after",
        )
        offset += len(selected)
    axes[0, 0].set_yscale("symlog", linthresh=1e-4)
    axes[0, 0].set_title("training CE")
    axes[0, 1].set_title("training accuracy")
    axes[0, 1].set_ylim(0, 1.03)
    axes[1, 0].set_title("cumulative J-call max-min")
    axes[1, 1].set_title("stage-gradient max/min")
    axes[1, 1].set_yscale("log")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for item in args.run:
        name, raw = item.split("=", 1)
        run_dir = Path(raw)
        eval_rows.extend(aggregate_eval(name, read_csv(run_dir / "fixed_evaluation.csv")))
        training_rows.extend(aggregate_training(name, read_csv(run_dir / "training.csv")))
        manifests[name] = json.loads((run_dir / "run_manifest.json").read_text())
    write_csv(args.out_dir / "fixed_evaluation_aggregate.csv", eval_rows)
    write_csv(args.out_dir / "training_diagnostics.csv", training_rows)
    plot_eval(eval_rows, args.out_dir / "before_after_composition.png")
    plot_training(training_rows, args.out_dir / "training_diagnostics.png")
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "runs": manifests,
                "evaluation_rows": len(eval_rows),
                "training_rows": len(training_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
