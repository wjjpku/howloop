"""Aggregate training curves for the program-path Group RMSNorm experiments."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSpec:
    label: str
    step_offset: int
    history_path: Path


def parse_run_spec(text: str) -> RunSpec:
    name, separator, raw_path = text.rpartition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("run must be LABEL[@STEP_OFFSET]=HISTORY.csv")
    label, offset_separator, raw_offset = name.rpartition("@")
    if offset_separator:
        try:
            step_offset = int(raw_offset)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("STEP_OFFSET must be an integer") from exc
    else:
        label = name
        step_offset = 0
    return RunSpec(label=label, step_offset=step_offset, history_path=Path(raw_path))


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def final_loop_accuracy_key(rows: list[dict[str, float]]) -> str:
    keys = [key for key in rows[0] if key.startswith("eval_acc_loop_")]
    if not keys:
        raise KeyError("history is missing eval_acc_loop_* columns")
    return max(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "G0": "#2563eb",
        "G1": "#d97706",
        "G4": "#dc2626",
        "G16": "#7c3aed",
    }
    summaries: dict[str, dict[str, float | str]] = {}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for spec in args.run:
        rows = read_history(spec.history_path)
        steps = [row["step"] + spec.step_offset for row in rows]
        accuracy_key = final_loop_accuracy_key(rows)
        accuracy = [row[accuracy_key] for row in rows]
        loss = [row["train_final_loss"] for row in rows]
        condition = spec.label.split()[0]
        style = "--" if spec.step_offset else "-"
        axes[0].plot(steps, accuracy, label=spec.label, color=colors.get(condition), linestyle=style)
        axes[1].plot(steps, loss, label=spec.label, color=colors.get(condition), linestyle=style)
        best_index = max(range(len(rows)), key=lambda idx: accuracy[idx])
        summaries[spec.label] = {
            "history": str(spec.history_path),
            "step_offset": spec.step_offset,
            "final_accuracy": accuracy[-1],
            "best_accuracy": accuracy[best_index],
            "best_step": steps[best_index],
        }

    axes[0].set_ylabel("mixed-depth final-loop accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("training update")
    axes[1].set_ylabel("final-only train loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Program-path training dynamics")
    fig.tight_layout()
    fig.savefig(args.out_dir / "combined_training_dynamics.png", dpi=180)
    plt.close(fig)

    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
