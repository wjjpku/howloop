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

from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"run spec must be NAME=/path/to/best.pt, got {text!r}")
    name, path = text.split("=", 1)
    if not name:
        raise ValueError("run spec name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare overloop behavior for graph-path looped-transformer checkpoints."
    )
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/best.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-eval-loop", type=int, default=16)
    parser.add_argument("--path-positions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026070607)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def make_fixed_query_batch(
    cfg: GraphPathConfig,
    query_depth: int,
    batch_size: int,
    path_positions: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.rand(batch_size, cfg.node_count, device=device)
    successors = noise.argsort(dim=-1)
    src = torch.arange(cfg.node_count, device=device).view(1, cfg.node_count).expand(batch_size, -1)

    edge_triplets = torch.empty(batch_size, cfg.node_count, 3, dtype=torch.long, device=device)
    edge_triplets[:, :, 0] = cfg.edge_token
    edge_triplets[:, :, 1] = src
    edge_triplets[:, :, 2] = successors

    start = torch.randint(0, cfg.node_count, (batch_size,), dtype=torch.long, device=device)
    tokens = torch.empty(batch_size, cfg.seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = cfg.bos_token
    tokens[:, 1 : 1 + 3 * cfg.node_count] = edge_triplets.reshape(batch_size, 3 * cfg.node_count)
    tokens[:, -4] = cfg.query_token
    tokens[:, -3] = start
    tokens[:, -2] = cfg.depth_token_base + query_depth - 1
    tokens[:, -1] = cfg.answer_token

    targets = torch.empty(batch_size, path_positions, dtype=torch.long, device=device)
    current = start
    for step in range(path_positions):
        current = successors.gather(1, current.view(-1, 1)).squeeze(1)
        targets[:, step] = current
    return tokens, targets


def save_heatmap(
    arr: np.ndarray,
    *,
    title: str,
    path: Path,
    label: str,
    trained_loops: int,
    query_depth: int,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    loops, positions = arr.shape
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("actual path position k")
    ax.set_ylabel("readout loop t")
    ax.set_xticks(range(positions), [str(i) for i in range(1, positions + 1)])
    ax.set_yticks(range(loops), [str(i) for i in range(1, loops + 1)])
    ax.axhline(trained_loops - 0.5, color="white", linestyle="--", linewidth=1)
    ax.axvline(query_depth - 0.5, color="white", linestyle="--", linewidth=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    non_alias_correct_to_pos = torch.zeros(loops, positions, device=device)
    non_alias_count_to_pos = torch.zeros(loops, positions, device=device)
    logit_sum_to_pos = torch.zeros(loops, positions, device=device)
    prob_sum_to_pos = torch.zeros(loops, positions, device=device)
    target_logit_sum = torch.zeros(loops, device=device)
    target_prob_sum = torch.zeros(loops, device=device)
    rolling_logit_sum = torch.zeros(loops, device=device)
    rolling_prob_sum = torch.zeros(loops, device=device)
    rolling_correct = torch.zeros(loops, device=device)
    target_correct = torch.zeros(loops, device=device)
    entropy_sum = torch.zeros(loops, device=device)
    margin_sum = torch.zeros(loops, device=device)

    with torch.no_grad():
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
                target_logit = logits.gather(1, target[:, None]).squeeze(1)
                target_prob = probs.gather(1, target[:, None]).squeeze(1)
                target_logit_sum[loop_idx] += target_logit.sum()
                target_prob_sum[loop_idx] += target_prob.sum()
                target_correct[loop_idx] += loop_pred.eq(target).float().sum()

                top2 = logits.topk(2, dim=-1).values
                margin_sum[loop_idx] += (top2[:, 0] - top2[:, 1]).sum()
                entropy_sum[loop_idx] += (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum()

                rolling_idx = min(loop_idx, positions - 1)
                rolling_target = targets_by_pos[:, rolling_idx]
                rolling_logit_sum[loop_idx] += logits.gather(1, rolling_target[:, None]).squeeze(1).sum()
                rolling_prob_sum[loop_idx] += probs.gather(1, rolling_target[:, None]).squeeze(1).sum()
                rolling_correct[loop_idx] += loop_pred.eq(rolling_target).float().sum()

                for pos_idx in range(positions):
                    pos_target = targets_by_pos[:, pos_idx]
                    correct_to_pos[loop_idx, pos_idx] += loop_pred.eq(pos_target).float().sum()
                    logit_sum_to_pos[loop_idx, pos_idx] += logits.gather(1, pos_target[:, None]).squeeze(1).sum()
                    prob_sum_to_pos[loop_idx, pos_idx] += probs.gather(1, pos_target[:, None]).squeeze(1).sum()
                    alias_mask = pos_target.ne(target)
                    non_alias_count_to_pos[loop_idx, pos_idx] += alias_mask.float().sum()
                    non_alias_correct_to_pos[loop_idx, pos_idx] += (
                        loop_pred[alias_mask].eq(pos_target[alias_mask]).float().sum()
                    )

    acc_to_pos = correct_to_pos / count
    non_alias_acc_to_pos = non_alias_correct_to_pos / non_alias_count_to_pos.clamp_min(1)
    mean_logit_to_pos = logit_sum_to_pos / count
    mean_prob_to_pos = prob_sum_to_pos / count
    best_acc, best_pos = acc_to_pos.max(dim=1)
    non_alias_best_acc, non_alias_best_pos = non_alias_acc_to_pos.max(dim=1)
    target_acc = target_correct / count
    rolling_acc = rolling_correct / count
    target_prob = target_prob_sum / count
    target_logit = target_logit_sum / count
    rolling_prob = rolling_prob_sum / count
    rolling_logit = rolling_logit_sum / count
    entropy = entropy_sum / count
    margin = margin_sum / count

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(
        mean_prob_to_pos.detach().cpu().numpy(),
        title=f"{name}: mean probability assigned to f^k(s)",
        path=run_dir / "overloop_mean_prob_by_path_position.png",
        label="mean probability",
        trained_loops=cfg.max_loops,
        query_depth=query_depth,
        vmin=0,
        vmax=1,
    )
    save_heatmap(
        mean_logit_to_pos.detach().cpu().numpy(),
        title=f"{name}: mean logit assigned to f^k(s)",
        path=run_dir / "overloop_mean_logit_by_path_position.png",
        label="mean logit",
        trained_loops=cfg.max_loops,
        query_depth=query_depth,
        cmap="coolwarm",
    )
    save_heatmap(
        acc_to_pos.detach().cpu().numpy(),
        title=f"{name}: accuracy to f^k(s)",
        path=run_dir / "overloop_acc_by_path_position.png",
        label="accuracy",
        trained_loops=cfg.max_loops,
        query_depth=query_depth,
        vmin=0,
        vmax=1,
    )

    xs = np.arange(1, loops + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, target_acc.detach().cpu().numpy(), marker="o", label=f"acc to target f^{query_depth}")
    ax.plot(xs, rolling_acc.detach().cpu().numpy(), marker="o", label="acc to f^loop")
    ax.plot(xs, best_acc.detach().cpu().numpy(), marker="o", label="best path-position acc")
    ax.axvline(cfg.max_loops, color="gray", linestyle="--", label="trained loop count")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{name}: overloop target vs rolling-depth accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "overloop_accuracy_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, best_pos.detach().cpu().numpy() + 1, marker="o", label="raw best k")
    ax.plot(xs, non_alias_best_pos.detach().cpu().numpy() + 1, marker="x", label="non-alias best k")
    ax.plot(xs, np.minimum(xs, positions), linestyle=":", color="gray", label="k=t reference")
    ax.axhline(query_depth, color="black", linestyle="--", linewidth=1, label="query target k")
    ax.axvline(cfg.max_loops, color="gray", linestyle="--", linewidth=1, label="trained loop count")
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("best matched path position k")
    ax.set_ylim(0.5, positions + 0.5)
    ax.set_title(f"{name}: best-matched f^k(s) under overloop")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "overloop_best_position_curve.png", dpi=180)
    plt.close(fig)

    loop_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for loop_idx in range(loops):
        loop_rows.append(
            {
                "model": name,
                "trained_depth": cfg.max_depth,
                "trained_loops": cfg.max_loops,
                "query_depth": query_depth,
                "loop": loop_idx + 1,
                "target_position": query_depth,
                "target_acc": float(target_acc[loop_idx].cpu()),
                "target_mean_logit": float(target_logit[loop_idx].cpu()),
                "target_mean_prob": float(target_prob[loop_idx].cpu()),
                "rolling_position": loop_idx + 1,
                "rolling_acc_to_f_loop": float(rolling_acc[loop_idx].cpu()),
                "rolling_mean_logit": float(rolling_logit[loop_idx].cpu()),
                "rolling_mean_prob": float(rolling_prob[loop_idx].cpu()),
                "best_position": int(best_pos[loop_idx].cpu()) + 1,
                "best_acc": float(best_acc[loop_idx].cpu()),
                "non_alias_best_position": int(non_alias_best_pos[loop_idx].cpu()) + 1,
                "non_alias_best_acc": float(non_alias_best_acc[loop_idx].cpu()),
                "mean_entropy": float(entropy[loop_idx].cpu()),
                "mean_top1_margin": float(margin[loop_idx].cpu()),
            }
        )
        for pos_idx in range(positions):
            path_rows.append(
                {
                    "model": name,
                    "trained_depth": cfg.max_depth,
                    "trained_loops": cfg.max_loops,
                    "query_depth": query_depth,
                    "loop": loop_idx + 1,
                    "actual_path_position": pos_idx + 1,
                    "acc": float(acc_to_pos[loop_idx, pos_idx].cpu()),
                    "non_alias_acc_excluding_pos_eq_target": float(
                        non_alias_acc_to_pos[loop_idx, pos_idx].cpu()
                    ),
                    "mean_logit": float(mean_logit_to_pos[loop_idx, pos_idx].cpu()),
                    "mean_prob": float(mean_prob_to_pos[loop_idx, pos_idx].cpu()),
                }
            )

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt.get("step"),
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
        "non_alias_best_position_by_loop": [
            int(x) + 1 for x in non_alias_best_pos.detach().cpu().tolist()
        ],
        "non_alias_best_acc_by_loop": [float(x) for x in non_alias_best_acc.detach().cpu()],
        "mean_entropy_by_loop": [float(x) for x in entropy.detach().cpu()],
        "mean_top1_margin_by_loop": [float(x) for x in margin.detach().cpu()],
    }
    return summary, loop_rows, path_rows


def save_combined_plots(out_dir: Path, summaries: dict[str, Any], max_eval_loop: int, path_positions: int) -> None:
    xs = np.arange(1, max_eval_loop + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, metric, ylabel, title in [
        (axes[0], "target_acc_by_loop", "acc", "Accuracy to max-query target"),
        (axes[1], "rolling_acc_to_f_loop_by_loop", "acc", "Accuracy to rolling depth f^loop(s)"),
        (axes[2], "target_mean_prob_by_loop", "mean prob", "Mean probability assigned to target"),
    ]:
        for name, summary in summaries.items():
            ax.plot(xs, summary[metric], marker="o", label=name)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("readout loop t")
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_overloop_target_vs_rolling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, summary in summaries.items():
        ax.plot(xs, summary["best_position_by_loop"], marker="o", label=name)
    ax.plot(xs, xs, linestyle=":", color="gray", label="k=t reference")
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("best matched path position k")
    ax.set_ylim(0.5, path_positions + 0.5)
    ax.set_title("Best matched path position under overloop")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_overloop_best_position.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    run_specs = [parse_run_spec(text) for text in args.run]
    summaries: dict[str, Any] = {}
    all_loop_rows: list[dict[str, Any]] = []
    all_path_rows: list[dict[str, Any]] = []
    for name, checkpoint in run_specs:
        summary, loop_rows, path_rows = analyze_run(
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
        all_loop_rows.extend(loop_rows)
        all_path_rows.extend(path_rows)
        print("done", name, flush=True)

    with (args.out_dir / "overloop_loop_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_loop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_loop_rows)
    with (args.out_dir / "overloop_path_position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_path_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_path_rows)

    report = {
        "max_eval_loop": args.max_eval_loop,
        "path_positions": args.path_positions,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "models": summaries,
    }
    (args.out_dir / "overloop_compare_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_combined_plots(args.out_dir, summaries, args.max_eval_loop, args.path_positions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
