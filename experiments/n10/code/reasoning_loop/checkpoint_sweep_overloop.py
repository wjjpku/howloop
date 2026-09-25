from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_loop import GraphPathConfig, LoopedGraphPathTransformer, pick_device, set_seed
from reasoning_loop.overloop_compare import make_fixed_query_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track graph-path overloop behavior across training checkpoints."
    )
    parser.add_argument("--checkpoint-dir", action="append", required=True, help="LABEL=/path/to/run_dir")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="checkpoint_step_*.pt")
    parser.add_argument("--max-eval-loop", type=int, default=10)
    parser.add_argument("--path-positions", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026070617)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def parse_labeled_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"expected LABEL=/path, got {text!r}")
    label, path = text.split("=", 1)
    if not label:
        raise ValueError("label cannot be empty")
    return label, Path(path)


def checkpoint_step(path: Path, ckpt: dict[str, Any]) -> int:
    if "step" in ckpt:
        return int(ckpt["step"])
    match = re.search(r"step_(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer step from {path}")
    return int(match.group(1))


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    *,
    label: str,
    device: torch.device,
    max_eval_loop: int,
    path_positions: int,
    batch_size: int,
    batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = GraphPathConfig(**ckpt["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    query_depth = cfg.max_depth
    loops = max_eval_loop
    positions = path_positions
    count = 0
    correct_to_pos = torch.zeros(loops, positions, device=device)
    target_correct = torch.zeros(loops, device=device)
    rolling_correct = torch.zeros(loops, device=device)
    target_prob_sum = torch.zeros(loops, device=device)
    rolling_prob_sum = torch.zeros(loops, device=device)
    entropy_sum = torch.zeros(loops, device=device)
    margin_sum = torch.zeros(loops, device=device)

    for _ in range(batches):
        tokens, targets_by_pos = make_fixed_query_batch(
            cfg, query_depth, batch_size, positions, device
        )
        logits_by_loop = model.forward_all(tokens, max_loops=loops)["logits_by_loop"]
        probs_by_loop = logits_by_loop.softmax(dim=-1)
        pred = logits_by_loop.argmax(dim=-1)
        target = targets_by_pos[:, query_depth - 1]
        count += tokens.shape[0]

        for loop_idx in range(loops):
            logits = logits_by_loop[:, loop_idx, :]
            probs = probs_by_loop[:, loop_idx, :]
            loop_pred = pred[:, loop_idx]
            target_prob_sum[loop_idx] += probs.gather(1, target[:, None]).squeeze(1).sum()
            target_correct[loop_idx] += loop_pred.eq(target).float().sum()

            rolling_idx = min(loop_idx, positions - 1)
            rolling_target = targets_by_pos[:, rolling_idx]
            rolling_prob_sum[loop_idx] += probs.gather(1, rolling_target[:, None]).squeeze(1).sum()
            rolling_correct[loop_idx] += loop_pred.eq(rolling_target).float().sum()

            top2 = logits.topk(2, dim=-1).values
            margin_sum[loop_idx] += (top2[:, 0] - top2[:, 1]).sum()
            entropy_sum[loop_idx] += (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum()

            for pos_idx in range(positions):
                pos_target = targets_by_pos[:, pos_idx]
                correct_to_pos[loop_idx, pos_idx] += loop_pred.eq(pos_target).float().sum()

    acc_to_pos = correct_to_pos / count
    best_acc, best_pos = acc_to_pos.max(dim=1)
    target_acc = target_correct / count
    rolling_acc = rolling_correct / count
    target_prob = target_prob_sum / count
    rolling_prob = rolling_prob_sum / count
    entropy = entropy_sum / count
    margin = margin_sum / count
    step = checkpoint_step(checkpoint, ckpt)

    summary = {
        "label": label,
        "checkpoint": str(checkpoint),
        "step": step,
        "config": asdict(cfg),
        "query_depth": query_depth,
        "trained_loops": cfg.max_loops,
        "examples": count,
        "target_acc_by_loop": [float(x) for x in target_acc.detach().cpu()],
        "rolling_acc_to_f_loop_by_loop": [float(x) for x in rolling_acc.detach().cpu()],
        "target_mean_prob_by_loop": [float(x) for x in target_prob.detach().cpu()],
        "rolling_mean_prob_by_loop": [float(x) for x in rolling_prob.detach().cpu()],
        "best_position_by_loop": [int(x) + 1 for x in best_pos.detach().cpu().tolist()],
        "best_acc_by_loop": [float(x) for x in best_acc.detach().cpu()],
        "mean_entropy_by_loop": [float(x) for x in entropy.detach().cpu()],
        "mean_top1_margin_by_loop": [float(x) for x in margin.detach().cpu()],
    }

    loop_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for loop_idx in range(loops):
        loop_rows.append(
            {
                "label": label,
                "step": step,
                "loop": loop_idx + 1,
                "query_depth": query_depth,
                "target_acc": float(target_acc[loop_idx].cpu()),
                "rolling_acc_to_f_loop": float(rolling_acc[loop_idx].cpu()),
                "target_mean_prob": float(target_prob[loop_idx].cpu()),
                "rolling_mean_prob": float(rolling_prob[loop_idx].cpu()),
                "best_position": int(best_pos[loop_idx].cpu()) + 1,
                "best_acc": float(best_acc[loop_idx].cpu()),
                "mean_entropy": float(entropy[loop_idx].cpu()),
                "mean_top1_margin": float(margin[loop_idx].cpu()),
            }
        )
        for pos_idx in range(positions):
            path_rows.append(
                {
                    "label": label,
                    "step": step,
                    "loop": loop_idx + 1,
                    "actual_path_position": pos_idx + 1,
                    "acc": float(acc_to_pos[loop_idx, pos_idx].cpu()),
                }
            )
    return summary, loop_rows, path_rows


def save_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: Path,
    label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(col_labels) + 3), max(5, 0.28 * len(row_labels) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("checkpoint")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plots(out_dir: Path, summaries_by_label: dict[str, list[dict[str, Any]]], max_eval_loop: int) -> None:
    xs_loop = [str(i) for i in range(1, max_eval_loop + 1)]
    for label, summaries in summaries_by_label.items():
        summaries = sorted(summaries, key=lambda item: item["step"])
        steps = [item["step"] for item in summaries]
        row_labels = [str(step) for step in steps]
        target_matrix = np.array([item["target_acc_by_loop"] for item in summaries], dtype=np.float32)
        rolling_matrix = np.array([item["rolling_acc_to_f_loop_by_loop"] for item in summaries], dtype=np.float32)
        best_pos_matrix = np.array([item["best_position_by_loop"] for item in summaries], dtype=np.float32)
        margin_matrix = np.array([item["mean_top1_margin_by_loop"] for item in summaries], dtype=np.float32)

        safe = label.replace("/", "_")
        save_heatmap(
            target_matrix,
            row_labels=row_labels,
            col_labels=xs_loop,
            title=f"{label}: target accuracy by checkpoint and loop",
            path=out_dir / f"{safe}_target_acc_heatmap.png",
            label="target accuracy",
            vmin=0,
            vmax=1,
        )
        save_heatmap(
            rolling_matrix,
            row_labels=row_labels,
            col_labels=xs_loop,
            title=f"{label}: rolling f^loop accuracy by checkpoint and loop",
            path=out_dir / f"{safe}_rolling_acc_heatmap.png",
            label="rolling accuracy",
            vmin=0,
            vmax=1,
        )
        save_heatmap(
            best_pos_matrix,
            row_labels=row_labels,
            col_labels=xs_loop,
            title=f"{label}: best matched path position by checkpoint and loop",
            path=out_dir / f"{safe}_best_position_heatmap.png",
            label="best path position",
            vmin=1,
            vmax=max_eval_loop,
            cmap="magma",
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(steps, target_matrix[:, 5], marker="o", label="loop6 acc to target f6")
        ax.plot(steps, target_matrix[:, 6], marker="o", label="loop7 acc to target f6")
        ax.plot(steps, rolling_matrix[:, 6], marker="o", label="loop7 acc to f7")
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("train step")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.03)
        ax.set_title(f"{label}: does loop7 keep computing f7 or lock target?")
        ax.grid(alpha=0.2)
        ax_margin = ax.twinx()
        ax_margin.plot(
            steps,
            margin_matrix[:, 6],
            marker="x",
            color="gray",
            alpha=0.65,
            label="loop7 top1 margin",
        )
        ax_margin.set_ylabel("top1 margin")
        lines, labels = ax.get_legend_handles_labels()
        margin_lines, margin_labels = ax_margin.get_legend_handles_labels()
        ax.legend(lines + margin_lines, labels + margin_labels, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe}_loop7_transition.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, summaries in summaries_by_label.items():
        summaries = sorted(summaries, key=lambda item: item["step"])
        steps = [item["step"] for item in summaries]
        rolling_loop7 = [item["rolling_acc_to_f_loop_by_loop"][6] for item in summaries]
        target_loop7 = [item["target_acc_by_loop"][6] for item in summaries]
        ax.plot(steps, rolling_loop7, marker="o", label=f"{label}: loop7 to f7")
        ax.plot(steps, target_loop7, marker="x", label=f"{label}: loop7 to target f6")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("train step")
    ax.set_ylabel("accuracy")
    ax.set_title("Loop7 transition comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_loop7_transition.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    summaries_by_label: dict[str, list[dict[str, Any]]] = {}
    all_loop_rows: list[dict[str, Any]] = []
    all_path_rows: list[dict[str, Any]] = []

    for label, ckpt_dir in [parse_labeled_path(item) for item in args.checkpoint_dir]:
        checkpoints = sorted(ckpt_dir.glob(args.glob))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints matching {args.glob!r} in {ckpt_dir}")
        summaries_by_label[label] = []
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
            summaries_by_label[label].append(summary)
            all_loop_rows.extend(loop_rows)
            all_path_rows.extend(path_rows)
            print(f"done {label} step={summary['step']}", flush=True)

    with (args.out_dir / "checkpoint_sweep_loop_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_loop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_loop_rows)
    with (args.out_dir / "checkpoint_sweep_path_position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_path_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_path_rows)
    report = {
        "max_eval_loop": args.max_eval_loop,
        "path_positions": args.path_positions,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "runs": summaries_by_label,
    }
    (args.out_dir / "checkpoint_sweep_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_plots(args.out_dir, summaries_by_label, args.max_eval_loop)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
