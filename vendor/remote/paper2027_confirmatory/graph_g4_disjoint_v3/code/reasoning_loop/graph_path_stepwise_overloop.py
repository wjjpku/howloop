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

from reasoning_loop.graph_path_loop import LoopedGraphPathTransformer, pick_device, set_seed
from reasoning_loop.graph_path_stepwise import StepwiseGraphPathConfig, make_stepwise_batch


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"run spec must be NAME=/path/to/best.pt, got {text!r}")
    name, path = text.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze overloop behavior for no-depth stepwise graph-path checkpoints."
    )
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/best.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-eval-loop", type=int, default=16)
    parser.add_argument("--path-positions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026070711)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def save_heatmap(
    arr: np.ndarray,
    *,
    title: str,
    path: Path,
    label: str,
    trained_loops: int,
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
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = StepwiseGraphPathConfig(**ckpt["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loops = max_eval_loop
    positions = path_positions
    count = 0
    correct_to_pos = torch.zeros(loops, positions, device=device)
    prob_to_pos = torch.zeros(loops, positions, device=device)
    logit_to_pos = torch.zeros(loops, positions, device=device)
    entropy_sum = torch.zeros(loops, device=device)
    margin_sum = torch.zeros(loops, device=device)

    for _ in range(batches):
        tokens, targets_by_pos, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=positions
        )
        logits_by_loop = model.forward_all(tokens, max_loops=loops)["logits_by_loop"]
        probs_by_loop = logits_by_loop.softmax(dim=-1)
        pred = logits_by_loop.argmax(dim=-1)
        count += tokens.shape[0]
        for loop_idx in range(loops):
            logits = logits_by_loop[:, loop_idx, :]
            probs = probs_by_loop[:, loop_idx, :]
            top2 = logits.topk(2, dim=-1).values
            margin_sum[loop_idx] += (top2[:, 0] - top2[:, 1]).sum()
            entropy_sum[loop_idx] += (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum()
            for pos_idx in range(positions):
                target = targets_by_pos[:, pos_idx]
                correct_to_pos[loop_idx, pos_idx] += pred[:, loop_idx].eq(target).float().sum()
                prob_to_pos[loop_idx, pos_idx] += probs.gather(1, target[:, None]).squeeze(1).sum()
                logit_to_pos[loop_idx, pos_idx] += logits.gather(1, target[:, None]).squeeze(1).sum()

    acc_to_pos = correct_to_pos / count
    mean_prob_to_pos = prob_to_pos / count
    mean_logit_to_pos = logit_to_pos / count
    best_acc, best_pos = acc_to_pos.max(dim=1)
    rolling_idx = torch.arange(loops, device=device).clamp(max=positions - 1)
    rolling_acc = acc_to_pos[torch.arange(loops, device=device), rolling_idx]
    rolling_prob = mean_prob_to_pos[torch.arange(loops, device=device), rolling_idx]
    trained_final_idx = min(cfg.max_loops, positions) - 1
    trained_final_acc = acc_to_pos[:, trained_final_idx]
    trained_final_prob = mean_prob_to_pos[:, trained_final_idx]

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(
        acc_to_pos.detach().cpu().numpy(),
        title=f"{name}: accuracy to f^k(start)",
        path=run_dir / "overloop_acc_by_path_position.png",
        label="accuracy",
        trained_loops=cfg.max_loops,
        vmin=0,
        vmax=1,
    )
    save_heatmap(
        mean_prob_to_pos.detach().cpu().numpy(),
        title=f"{name}: mean probability assigned to f^k(start)",
        path=run_dir / "overloop_prob_by_path_position.png",
        label="mean probability",
        trained_loops=cfg.max_loops,
        vmin=0,
        vmax=1,
    )
    save_heatmap(
        mean_logit_to_pos.detach().cpu().numpy(),
        title=f"{name}: mean logit assigned to f^k(start)",
        path=run_dir / "overloop_logit_by_path_position.png",
        label="mean logit",
        trained_loops=cfg.max_loops,
        cmap="coolwarm",
    )

    xs = np.arange(1, loops + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, rolling_acc.detach().cpu().numpy(), marker="o", label="acc to f^loop")
    ax.plot(xs, trained_final_acc.detach().cpu().numpy(), marker="o", label=f"acc to trained final f^{cfg.max_loops}")
    ax.plot(xs, best_acc.detach().cpu().numpy(), marker="o", label="best path-position acc")
    ax.axvline(cfg.max_loops, color="gray", linestyle="--", label="trained loop count")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{name}: overloop continuation vs trained final")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "overloop_accuracy_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, best_pos.detach().cpu().numpy() + 1, marker="o", label="best matched k")
    ax.plot(xs, np.minimum(xs, positions), linestyle=":", color="gray", label="k=t reference")
    ax.axhline(cfg.max_loops, color="black", linestyle="--", linewidth=1, label="trained final k")
    ax.axvline(cfg.max_loops, color="gray", linestyle="--", linewidth=1, label="trained loops")
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("best matched path position k")
    ax.set_ylim(0.5, positions + 0.5)
    ax.set_title(f"{name}: best path position under overloop")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "overloop_best_position_curve.png", dpi=180)
    plt.close(fig)

    loop_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for loop_idx in range(loops):
        loop_rows.append(
            {
                "model": name,
                "loss_mode": ckpt.get("loss_mode", ""),
                "trained_loops": cfg.max_loops,
                "loop": loop_idx + 1,
                "rolling_position": min(loop_idx + 1, positions),
                "rolling_acc": float(rolling_acc[loop_idx].cpu()),
                "rolling_mean_prob": float(rolling_prob[loop_idx].cpu()),
                "trained_final_position": cfg.max_loops,
                "trained_final_acc": float(trained_final_acc[loop_idx].cpu()),
                "trained_final_mean_prob": float(trained_final_prob[loop_idx].cpu()),
                "best_position": int(best_pos[loop_idx].cpu()) + 1,
                "best_acc": float(best_acc[loop_idx].cpu()),
                "mean_entropy": float((entropy_sum[loop_idx] / count).cpu()),
                "mean_top1_margin": float((margin_sum[loop_idx] / count).cpu()),
            }
        )
        for pos_idx in range(positions):
            path_rows.append(
                {
                    "model": name,
                    "loss_mode": ckpt.get("loss_mode", ""),
                    "trained_loops": cfg.max_loops,
                    "loop": loop_idx + 1,
                    "actual_path_position": pos_idx + 1,
                    "acc": float(acc_to_pos[loop_idx, pos_idx].cpu()),
                    "mean_prob": float(mean_prob_to_pos[loop_idx, pos_idx].cpu()),
                    "mean_logit": float(mean_logit_to_pos[loop_idx, pos_idx].cpu()),
                }
            )

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "config": asdict(cfg),
        "loss_mode": ckpt.get("loss_mode", ""),
        "trained_loops": cfg.max_loops,
        "examples": count,
        "rolling_acc_by_loop": [float(x) for x in rolling_acc.detach().cpu()],
        "rolling_mean_prob_by_loop": [float(x) for x in rolling_prob.detach().cpu()],
        "trained_final_acc_by_loop": [float(x) for x in trained_final_acc.detach().cpu()],
        "trained_final_mean_prob_by_loop": [float(x) for x in trained_final_prob.detach().cpu()],
        "best_position_by_loop": [int(x) + 1 for x in best_pos.detach().cpu().tolist()],
        "best_acc_by_loop": [float(x) for x in best_acc.detach().cpu()],
        "mean_entropy_by_loop": [float(x) for x in (entropy_sum / count).detach().cpu()],
        "mean_top1_margin_by_loop": [float(x) for x in (margin_sum / count).detach().cpu()],
    }
    return summary, loop_rows, path_rows


def save_combined_plots(out_dir: Path, summaries: dict[str, Any], max_eval_loop: int, path_positions: int) -> None:
    xs = np.arange(1, max_eval_loop + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ax, metric, ylabel, title in [
        (axes[0], "rolling_acc_by_loop", "acc", "Accuracy to rolling path position f^loop"),
        (axes[1], "trained_final_acc_by_loop", "acc", "Accuracy to trained final path position"),
        (axes[2], "rolling_mean_prob_by_loop", "mean prob", "Mean probability assigned to f^loop"),
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
    fig.savefig(out_dir / "combined_overloop_continuation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, summary in summaries.items():
        ax.plot(xs, summary["best_position_by_loop"], marker="o", label=name)
    ax.plot(xs, np.minimum(xs, path_positions), linestyle=":", color="gray", label="k=t reference")
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

    with (args.out_dir / "stepwise_overloop_loop_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_loop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_loop_rows)
    with (args.out_dir / "stepwise_overloop_path_position_metrics.csv").open("w", newline="", encoding="utf-8") as f:
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
    (args.out_dir / "stepwise_overloop_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_combined_plots(args.out_dir, summaries, args.max_eval_loop, args.path_positions)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
