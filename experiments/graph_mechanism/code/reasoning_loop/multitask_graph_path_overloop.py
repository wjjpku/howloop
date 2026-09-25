from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.multitask_graph_path import (
    LoopedMultiTaskTransformer,
    MultiTaskGraphConfig,
    TokenScheme,
    evaluate_task,
    make_token_scheme,
)


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"run spec must be NAME=/path/to/final.pt, got {text!r}")
    name, path = text.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overloop analysis for multitask graph-path checkpoints.")
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/final.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-eval-loop", type=int, default=12)
    parser.add_argument("--path-positions", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026070719)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def save_heatmap(arr: np.ndarray, *, path: Path, title: str, label: str) -> None:
    loops, positions = arr.shape
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(positions), [str(i) for i in range(1, positions + 1)])
    ax.set_yticks(range(loops), [str(i) for i in range(1, loops + 1)])
    ax.set_xlabel("actual path position k")
    ax.set_ylabel("readout loop t")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_namespace_mass(metrics: dict[str, Any], *, path: Path, title: str) -> None:
    xs = np.arange(1, len(metrics["rolling_acc_by_loop"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, values in metrics["namespace_mass_by_loop"].items():
        ax.plot(xs, values, marker="o", label=key)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("probability mass")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_task_curves(metrics: dict[str, Any], *, trained_loops: int, path: Path, title: str) -> None:
    xs = np.arange(1, len(metrics["rolling_acc_by_loop"]) + 1)
    trained_idx = trained_loops - 1
    trained_final_acc = [row[trained_idx] for row in metrics["acc_to_pos"]]
    main_values = metrics.get("query_acc_by_loop", metrics["rolling_acc_by_loop"])
    main_label = "acc to input query" if "query_acc_by_loop" in metrics else "acc to f^loop"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, main_values, marker="o", label=main_label)
    ax.plot(xs, trained_final_acc, marker="o", label=f"acc to trained final f^{trained_loops}")
    ax.plot(xs, metrics["best_acc_by_loop"], marker="o", label="best path-position acc")
    ax.axvline(trained_loops, color="gray", linestyle="--", label="trained loops")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def analyze_run(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    max_eval_loop: int,
    path_positions: int,
    batch_size: int,
    batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = MultiTaskGraphConfig(**ckpt["config"])
    scheme = make_token_scheme(cfg)
    model = LoopedMultiTaskTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    task_metrics: dict[str, Any] = {}
    loop_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    namespace_rows: list[dict[str, Any]] = []
    for task_label, task in [("task_a", "A"), ("task_b", "B")]:
        metrics = evaluate_task(
            model,
            cfg,
            scheme,
            task,  # type: ignore[arg-type]
            device=device,
            batch_size=batch_size,
            batches=batches,
            max_loops=max_eval_loop,
            path_positions=path_positions,
            amp_enabled=device.type == "cuda",
        )
        task_metrics[task_label] = metrics
        task_dir = run_dir / task_label
        task_dir.mkdir(parents=True, exist_ok=True)
        save_heatmap(
            np.array(metrics["acc_to_pos"], dtype=np.float32),
            path=task_dir / "overloop_acc_by_path_position.png",
            title=f"{name} {task_label}: accuracy to f^k",
            label="accuracy",
        )
        save_heatmap(
            np.array(metrics["mean_prob_to_pos"], dtype=np.float32),
            path=task_dir / "overloop_prob_by_path_position.png",
            title=f"{name} {task_label}: probability to f^k",
            label="mean probability",
        )
        if "query_depth_acc" in metrics:
            save_heatmap(
                np.array(metrics["query_depth_acc"], dtype=np.float32).T,
                path=task_dir / "overloop_acc_by_input_query_depth.png",
                title=f"{name} {task_label}: accuracy to input query depth",
                label="accuracy",
            )
            save_heatmap(
                np.array(metrics["query_depth_prob"], dtype=np.float32).T,
                path=task_dir / "overloop_prob_by_input_query_depth.png",
                title=f"{name} {task_label}: probability to input query depth",
                label="mean probability",
            )
        plot_task_curves(
            metrics,
            trained_loops=cfg.max_loops,
            path=task_dir / "overloop_accuracy_curves.png",
            title=f"{name} {task_label}: continuation vs trained final",
        )
        plot_namespace_mass(
            metrics,
            path=task_dir / "namespace_probability_mass.png",
            title=f"{name} {task_label}: output probability mass by token namespace",
        )
        for loop_idx in range(max_eval_loop):
            loop_rows.append(
                {
                    "model": name,
                    "overlap_mode": cfg.overlap_mode,
                    "task": task_label,
                    "loop": loop_idx + 1,
                    "rolling_acc": metrics["rolling_acc_by_loop"][loop_idx],
                    "query_acc": metrics.get("query_acc_by_loop", [""] * max_eval_loop)[loop_idx],
                    "best_position": metrics["best_position_by_loop"][loop_idx],
                    "best_acc": metrics["best_acc_by_loop"][loop_idx],
                    "mean_entropy": metrics["mean_entropy_by_loop"][loop_idx],
                }
            )
            for key, values in metrics["namespace_mass_by_loop"].items():
                namespace_rows.append(
                    {
                        "model": name,
                        "overlap_mode": cfg.overlap_mode,
                        "task": task_label,
                        "loop": loop_idx + 1,
                        "namespace": key,
                        "prob_mass": values[loop_idx],
                    }
                )
            for pos_idx in range(path_positions):
                path_rows.append(
                    {
                        "model": name,
                        "overlap_mode": cfg.overlap_mode,
                        "task": task_label,
                        "loop": loop_idx + 1,
                        "actual_path_position": pos_idx + 1,
                        "acc": metrics["acc_to_pos"][loop_idx][pos_idx],
                        "mean_prob": metrics["mean_prob_to_pos"][loop_idx][pos_idx],
                    }
                )
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "config": asdict(cfg),
        "token_scheme": asdict(scheme),
        "parameter_count": ckpt.get("parameter_count"),
        "tasks": task_metrics,
    }
    return summary, loop_rows, path_rows, namespace_rows


def save_combined_plots(out_dir: Path, summaries: dict[str, Any], max_eval_loop: int, path_positions: int) -> None:
    xs = np.arange(1, max_eval_loop + 1)
    for task_label in ["task_a", "task_b"]:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for name, summary in summaries.items():
            metrics = summary["tasks"][task_label]
            main_values = metrics.get("query_acc_by_loop", metrics["rolling_acc_by_loop"])
            axes[0].plot(xs, main_values, marker="o", label=name)
            axes[1].plot(xs, metrics["best_position_by_loop"], marker="o", label=name)
        if any("query_acc_by_loop" in summary["tasks"][task_label] for summary in summaries.values()):
            axes[0].set_title(f"{task_label}: accuracy to input query target")
        else:
            axes[0].set_title(f"{task_label}: accuracy to rolling path position")
        axes[0].set_ylabel("accuracy")
        axes[0].set_ylim(0, 1.02)
        axes[0].axhline(0.5, color="black", linestyle=":", linewidth=1)
        axes[0].grid(alpha=0.2)
        axes[1].plot(xs, np.minimum(xs, path_positions), linestyle=":", color="gray", label="k=t")
        axes[1].set_title(f"{task_label}: best matched path position")
        axes[1].set_xlabel("readout loop t")
        axes[1].set_ylabel("best k")
        axes[1].set_ylim(0.5, path_positions + 0.5)
        axes[1].grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"combined_{task_label}_overloop.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    run_specs = [parse_run_spec(text) for text in args.run]
    summaries: dict[str, Any] = {}
    loop_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    namespace_rows: list[dict[str, Any]] = []
    for name, checkpoint in run_specs:
        summary, run_loop_rows, run_path_rows, run_namespace_rows = analyze_run(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            max_eval_loop=args.max_eval_loop,
            path_positions=args.path_positions,
            batch_size=args.batch_size,
            batches=args.batches,
        )
        summaries[name] = summary
        loop_rows.extend(run_loop_rows)
        path_rows.extend(run_path_rows)
        namespace_rows.extend(run_namespace_rows)
        print("done", name, flush=True)
    for filename, rows in [
        ("multitask_overloop_loop_metrics.csv", loop_rows),
        ("multitask_overloop_path_metrics.csv", path_rows),
        ("multitask_overloop_namespace_metrics.csv", namespace_rows),
    ]:
        with (args.out_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "max_eval_loop": args.max_eval_loop,
        "path_positions": args.path_positions,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "models": summaries,
    }
    (args.out_dir / "multitask_overloop_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_combined_plots(args.out_dir, summaries, args.max_eval_loop, args.path_positions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
