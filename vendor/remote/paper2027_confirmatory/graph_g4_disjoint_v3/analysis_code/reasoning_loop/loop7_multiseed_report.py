from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reasoning_loop.checkpoint_sweep_overloop import evaluate_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed


def parse_labeled_path(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, path = text.split("=", 1)
        return label, Path(path)
    path = Path(text)
    match = re.search(r"seed(\d+)", path.name)
    label = f"seed{match.group(1)}" if match else path.name
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify D6/L6 seeds by loop7 target-lock vs successor-like overloop behavior."
    )
    parser.add_argument("--checkpoint-dir", action="append", default=[], help="LABEL=/path/to/run_dir")
    parser.add_argument("--root", type=Path, help="Root containing graphpath_*_seed*/ run dirs")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="checkpoint_step_*.pt")
    parser.add_argument("--max-eval-loop", type=int, default=10)
    parser.add_argument("--path-positions", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026070702)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--successor-threshold", type=float, default=0.5)
    return parser.parse_args()


def discover_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    dirs = [parse_labeled_path(item) for item in args.checkpoint_dir]
    if args.root is not None:
        for path in sorted(args.root.glob("graphpath_*_seed*")):
            if path.is_dir():
                match = re.search(r"seed(\d+)", path.name)
                label = f"seed{match.group(1)}" if match else path.name
                dirs.append((label, path))
    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for label, path in dirs:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, path))
    return unique


def row_for_step(items: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    for item in items:
        if int(item["step"]) == step:
            return item
    return None


def loop7_metrics(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "loop6_target_f6": item["target_acc_by_loop"][5],
        "loop7_target_f6": item["target_acc_by_loop"][6],
        "loop7_rolling_f7": item["rolling_acc_to_f_loop_by_loop"][6],
        "loop7_best_position": item["best_position_by_loop"][6],
        "loop7_best_acc": item["best_acc_by_loop"][6],
        "loop7_margin": item["mean_top1_margin_by_loop"][6],
    }


def save_plots(out_dir: Path, summaries_by_label: dict[str, list[dict[str, Any]]], classification_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, items in sorted(summaries_by_label.items()):
        items = sorted(items, key=lambda item: item["step"])
        steps = [item["step"] for item in items]
        f7 = [item["rolling_acc_to_f_loop_by_loop"][6] for item in items]
        ax.plot(steps, f7, marker="o", linewidth=1.4, alpha=0.75, label=label)
    ax.axhline(0.125, color="gray", linestyle=":", linewidth=1, label="random 1/8")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="successor threshold")
    ax.set_xlabel("train step")
    ax.set_ylabel("loop7 accuracy to f7")
    ax.set_ylim(0, 1.03)
    ax.set_title("D6/L6 multi-seed loop7 -> f7 trajectory")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "loop7_f7_trajectories_by_seed.png", dpi=180)
    plt.close(fig)

    labels = [row["label"] for row in classification_rows]
    values = [float(row["max_loop7_f7"]) for row in classification_rows]
    colors = ["#d95f02" if row["classification"] == "successor_like" else "#1b9e77" for row in classification_rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.125, color="gray", linestyle=":", linewidth=1, label="random 1/8")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="successor threshold")
    ax.set_ylabel("max loop7 accuracy to f7 over checkpoints")
    ax.set_title("D6/L6 multi-seed successor-like branch score")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "loop7_f7_max_by_seed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for row in classification_rows:
        x = float(row["step3500_loop7_f7"])
        y = float(row["step3500_loop7_f6"])
        color = "#d95f02" if row["classification"] == "successor_like" else "#1b9e77"
        ax.scatter(x, y, s=70, color=color)
        ax.text(x + 0.01, y, row["label"], fontsize=8)
    ax.axvline(0.125, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.125, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("step3500 loop7 -> f7")
    ax.set_ylabel("step3500 loop7 -> target f6")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_title("D6/L6 step3500 branch map")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "step3500_loop7_branch_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    dirs = discover_dirs(args)
    if not dirs:
        raise ValueError("No checkpoint dirs found")

    summaries_by_label: dict[str, list[dict[str, Any]]] = {}
    all_loop_rows: list[dict[str, Any]] = []
    all_path_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []

    for label, ckpt_dir in dirs:
        checkpoints = sorted(ckpt_dir.glob(args.glob))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints matching {args.glob!r} in {ckpt_dir}")
        summaries: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            summary, loop_rows, path_rows = evaluate_checkpoint(
                checkpoint,
                label=label,
                device=device,
                max_eval_loop=args.max_eval_loop,
                path_positions=args.path_positions,
                batch_size=args.batch_size,
                batches=args.batches,
            )
            summaries.append(summary)
            all_loop_rows.extend(loop_rows)
            all_path_rows.extend(path_rows)
            print(f"done {label} step={summary['step']}", flush=True)
        summaries = sorted(summaries, key=lambda item: item["step"])
        summaries_by_label[label] = summaries

        max_item = max(summaries, key=lambda item: item["rolling_acc_to_f_loop_by_loop"][6])
        step3500 = row_for_step(summaries, 3500) or min(summaries, key=lambda item: abs(int(item["step"]) - 3500))
        final_item = summaries[-1]
        max_metrics = loop7_metrics(max_item)
        step3500_metrics = loop7_metrics(step3500)
        final_metrics = loop7_metrics(final_item)
        classification = (
            "successor_like"
            if max_metrics["loop7_rolling_f7"] >= args.successor_threshold
            and int(max_metrics["loop7_best_position"]) == 7
            else "target_lock"
        )
        classification_rows.append(
            {
                "label": label,
                "classification": classification,
                "max_loop7_f7": max_metrics["loop7_rolling_f7"],
                "max_loop7_f7_step": max_item["step"],
                "max_step_loop7_f6": max_metrics["loop7_target_f6"],
                "max_step_best_position": max_metrics["loop7_best_position"],
                "step3500_loop7_f6": step3500_metrics["loop7_target_f6"],
                "step3500_loop7_f7": step3500_metrics["loop7_rolling_f7"],
                "step3500_best_position": step3500_metrics["loop7_best_position"],
                "final_step": final_item["step"],
                "final_loop7_f6": final_metrics["loop7_target_f6"],
                "final_loop7_f7": final_metrics["loop7_rolling_f7"],
                "final_best_position": final_metrics["loop7_best_position"],
            }
        )

    classification_rows = sorted(
        classification_rows,
        key=lambda row: int(re.search(r"\d+", row["label"]).group(0)) if re.search(r"\d+", row["label"]) else row["label"],
    )
    with (args.out_dir / "loop7_multiseed_classification.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(classification_rows[0].keys()))
        writer.writeheader()
        writer.writerows(classification_rows)
    with (args.out_dir / "loop7_multiseed_loop_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_loop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_loop_rows)
    with (args.out_dir / "loop7_multiseed_path_position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_path_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_path_rows)

    report = {
        "max_eval_loop": args.max_eval_loop,
        "path_positions": args.path_positions,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "successor_threshold": args.successor_threshold,
        "classification": classification_rows,
        "runs": summaries_by_label,
    }
    (args.out_dir / "loop7_multiseed_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plots(args.out_dir, summaries_by_label, classification_rows)
    print(json.dumps({"classification": classification_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
