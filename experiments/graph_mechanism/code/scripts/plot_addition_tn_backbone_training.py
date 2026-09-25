#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.history_csv)
    validation = [row for row in rows if row.get("validation_exact_match")]
    if not validation:
        raise ValueError("history contains no validation rows")
    steps = np.asarray([int(row["step"]) for row in validation])
    em = np.asarray([float(row["validation_exact_match"]) for row in validation])
    loss = np.asarray([float(row["validation_loss"]) for row in validation])
    learning_rate = np.asarray([float(row["learning_rate"]) for row in validation])
    best_index = int(np.argmax(em))

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0].plot(steps, em, marker="o", markersize=3.2, linewidth=1.8)
    axes[0].axhline(
        args.stability_threshold,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"stability gate={args.stability_threshold:.3f}",
    )
    axes[0].scatter(
        [steps[best_index]],
        [em[best_index]],
        color="#d62728",
        s=45,
        zorder=3,
        label=f"best={em[best_index]:.4f} @ {steps[best_index]:,}",
    )
    axes[0].set(
        title="Held-out m=10 exact match during variable-length backbone training",
        xlabel="optimizer step",
        ylabel="supervised 10-digit EM",
        ylim=(-0.02, 1.02),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].plot(steps, loss, color="#4c78a8", linewidth=1.8, label="validation CE")
    axes[1].set(
        title="Validation CE and scheduled learning rate",
        xlabel="optimizer step",
        ylabel="validation CE",
    )
    axes[1].grid(alpha=0.25)
    lr_axis = axes[1].twinx()
    lr_axis.plot(
        steps,
        learning_rate,
        color="#f58518",
        linestyle="--",
        linewidth=1.5,
        label="learning rate",
    )
    lr_axis.set_ylabel("learning rate")
    handles, labels = axes[1].get_legend_handles_labels()
    lr_handles, lr_labels = lr_axis.get_legend_handles_labels()
    axes[1].legend(handles + lr_handles, labels + lr_labels, loc="upper right")
    figure.suptitle(
        "Addition LSB→MSB, causal, NoPE; one random m per batch; T(m)=m",
        fontsize=14,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "backbone_training_validation_curve.png"
    figure.savefig(output, dpi=190)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)

    summary = {
        "status": "complete",
        "history_csv": str(args.history_csv),
        "validation_points": len(validation),
        "validation_step_range": [int(steps.min()), int(steps.max())],
        "best_validation_exact_match": float(em[best_index]),
        "best_validation_step": int(steps[best_index]),
        "final_validation_exact_match": float(em[-1]),
        "stability_threshold": args.stability_threshold,
        "figure": str(output),
        "note": "The final backbone gate additionally evaluates every ID length m=1..10 at two held-out checkpoints.",
    }
    (args.out_dir / "training_curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--stability-threshold", type=float, default=0.995)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
