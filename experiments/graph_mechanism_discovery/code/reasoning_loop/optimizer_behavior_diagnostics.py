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

from reasoning_loop.graph_path_loop import GraphPathConfig, LoopedGraphPathTransformer, pick_device, set_seed
from reasoning_loop.overloop_compare import make_fixed_query_batch, parse_run_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostics that amplify AdamW-vs-Muon behavioral differences for L1x6 graph-path models."
    )
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/final.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-eval-loop", type=int, default=16)
    parser.add_argument("--path-positions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--noise-batches", type=int, default=32)
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=2026070721)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device) -> tuple[LoopedGraphPathTransformer, GraphPathConfig, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = GraphPathConfig(**ckpt["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def path_position_metrics(
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    *,
    device: torch.device,
    max_eval_loop: int,
    path_positions: int,
    batch_size: int,
    batches: int,
) -> dict[str, Any]:
    query_depth = cfg.max_depth
    correct = torch.zeros(max_eval_loop, path_positions, device=device)
    prob_sum = torch.zeros(max_eval_loop, path_positions, device=device)
    count = 0
    for _ in range(batches):
        tokens, targets_by_pos = make_fixed_query_batch(cfg, query_depth, batch_size, path_positions, device)
        logits_by_loop = model.forward_all(tokens, max_loops=max_eval_loop)["logits_by_loop"]
        probs_by_loop = logits_by_loop.softmax(dim=-1)
        pred = logits_by_loop.argmax(dim=-1)
        count += tokens.shape[0]
        for loop_idx in range(max_eval_loop):
            logits_probs = probs_by_loop[:, loop_idx, :]
            loop_pred = pred[:, loop_idx]
            for pos_idx in range(path_positions):
                target = targets_by_pos[:, pos_idx]
                correct[loop_idx, pos_idx] += loop_pred.eq(target).float().sum()
                prob_sum[loop_idx, pos_idx] += logits_probs.gather(1, target[:, None]).squeeze(1).sum()
    acc = correct / count
    prob = prob_sum / count
    target_idx = query_depth - 1
    rolling = torch.tensor(
        [float(acc[i, min(i, path_positions - 1)].detach().cpu()) for i in range(max_eval_loop)]
    )
    return {
        "acc_to_pos": acc.detach().cpu().tolist(),
        "prob_to_pos": prob.detach().cpu().tolist(),
        "target_acc_by_loop": acc[:, target_idx].detach().cpu().tolist(),
        "rolling_acc_by_loop": rolling.tolist(),
        "best_position_by_loop": (acc.argmax(dim=1) + 1).detach().cpu().tolist(),
        "best_acc_by_loop": acc.max(dim=1).values.detach().cpu().tolist(),
        "prefix_successor_score": float(acc[: query_depth - 1].diagonal().mean().detach().cpu()),
        "target_lock_score_loop7_16": float(acc[query_depth:max_eval_loop, target_idx].mean().detach().cpu()),
        "overloop_successor_score_loop7_16": float(
            torch.stack(
                [acc[i, min(i, path_positions - 1)] for i in range(query_depth, max_eval_loop)]
            ).mean().detach().cpu()
        ),
    }


@torch.no_grad()
def noise_basin_metrics(
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    *,
    device: torch.device,
    max_eval_loop: int,
    path_positions: int,
    batch_size: int,
    batches: int,
    noise_scales: list[float],
) -> list[dict[str, Any]]:
    query_depth = cfg.max_depth
    rows: list[dict[str, Any]] = []
    target_idx = query_depth - 1
    readout_loops = [query_depth, query_depth + 1, 8, 12, max_eval_loop]
    readout_loops = sorted({loop for loop in readout_loops if 1 <= loop <= max_eval_loop})
    for noise_scale in noise_scales:
        target_correct = {loop: torch.zeros((), device=device) for loop in readout_loops}
        target_prob = {loop: torch.zeros((), device=device) for loop in readout_loops}
        rolling_correct = {loop: torch.zeros((), device=device) for loop in readout_loops}
        count = 0
        for _ in range(batches):
            tokens, targets_by_pos = make_fixed_query_batch(cfg, query_depth, batch_size, path_positions, device)
            x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            for loop in range(1, max_eval_loop + 1):
                for block in model.blocks:
                    x = block(x)
                if loop == query_depth and noise_scale > 0:
                    answer_state = x[:, -1, :]
                    rms = answer_state.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(answer_state.dtype)
                    x[:, -1, :] = answer_state + torch.randn_like(answer_state) * rms * noise_scale
                if loop in target_correct:
                    final_state = model.ln_final(x[:, -1, :])
                    logits = model.unembed(final_state)[:, : cfg.node_count]
                    probs = logits.softmax(dim=-1)
                    pred = logits.argmax(dim=-1)
                    target = targets_by_pos[:, target_idx]
                    rolling_target = targets_by_pos[:, min(loop - 1, path_positions - 1)]
                    target_correct[loop] += pred.eq(target).float().sum()
                    target_prob[loop] += probs.gather(1, target[:, None]).squeeze(1).sum()
                    rolling_correct[loop] += pred.eq(rolling_target).float().sum()
            count += tokens.shape[0]
        for loop in readout_loops:
            rows.append(
                {
                    "noise_scale": noise_scale,
                    "readout_loop": loop,
                    "target_acc": float((target_correct[loop] / count).detach().cpu()),
                    "target_prob": float((target_prob[loop] / count).detach().cpu()),
                    "rolling_acc": float((rolling_correct[loop] / count).detach().cpu()),
                }
            )
    return rows


def save_path_heatmap(arr: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xlabel("path position k")
    ax.set_ylabel("readout loop t")
    ax.set_xticks(range(arr.shape[1]), [str(i) for i in range(1, arr.shape[1] + 1)])
    ax.set_yticks(range(arr.shape[0]), [str(i) for i in range(1, arr.shape[0] + 1)])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary_plots(out_dir: Path, summaries: dict[str, Any], noise_rows: list[dict[str, Any]]) -> None:
    names = list(summaries)
    scores = {
        "prefix successor\nloops 1-5": [summaries[name]["prefix_successor_score"] for name in names],
        "target lock\nloops 7-16": [summaries[name]["target_lock_score_loop7_16"] for name in names],
        "overloop successor\nloops 7-16": [summaries[name]["overloop_successor_score_loop7_16"] for name in names],
    }
    x = np.arange(len(scores))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, name in enumerate(names):
        ax.bar(x - 0.4 + width / 2 + idx * width, [values[idx] for values in scores.values()], width, label=name)
    ax.set_xticks(x, list(scores.keys()))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("accuracy score")
    ax.set_title("Behavior scores that separate optimizer-trained L1x6 models")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "optimizer_behavior_scores.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in names:
        xs = np.arange(1, len(summaries[name]["rolling_acc_by_loop"]) + 1)
        ax.plot(xs, summaries[name]["rolling_acc_by_loop"], marker="o", label=f"{name}: rolling f^loop")
    ax.axvline(6, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Max-query rollout: successor-like prefix behavior")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "optimizer_rolling_successor_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in names:
        xs = np.arange(1, len(summaries[name]["target_acc_by_loop"]) + 1)
        ax.plot(xs, summaries[name]["target_acc_by_loop"], marker="o", label=f"{name}: target f6")
    ax.axvline(6, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Max-query rollout: target attractor retention")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "optimizer_target_retention_curves.png", dpi=180)
    plt.close(fig)

    for loop in [7, 12, 16]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in names:
            xs = []
            ys = []
            for row in noise_rows:
                if row["model"] == name and row["readout_loop"] == loop:
                    xs.append(row["noise_scale"])
                    ys.append(row["target_acc"])
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("answer-state noise scale after loop6")
        ax.set_ylabel("target f6 accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Noise basin test: target retention at loop {loop}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"noise_basin_target_acc_loop{loop}.png", dpi=180)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    summaries: dict[str, Any] = {}
    noise_rows: list[dict[str, Any]] = []
    for name, checkpoint in [parse_run_spec(text) for text in args.run]:
        model, cfg, ckpt = load_model(checkpoint, device)
        path_metrics = path_position_metrics(
            model,
            cfg,
            device=device,
            max_eval_loop=args.max_eval_loop,
            path_positions=args.path_positions,
            batch_size=args.batch_size,
            batches=args.batches,
        )
        model_noise_rows = noise_basin_metrics(
            model,
            cfg,
            device=device,
            max_eval_loop=args.max_eval_loop,
            path_positions=args.path_positions,
            batch_size=args.batch_size,
            batches=args.noise_batches,
            noise_scales=args.noise_scales,
        )
        for row in model_noise_rows:
            row["model"] = name
        noise_rows.extend(model_noise_rows)
        summaries[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": ckpt.get("step"),
            "config": asdict(cfg),
            "parameter_count": ckpt.get("parameter_count"),
            **path_metrics,
        }
        save_path_heatmap(
            np.array(path_metrics["acc_to_pos"], dtype=np.float32),
            args.out_dir / f"{name}_path_position_acc_heatmap.png",
            f"{name}: accuracy to f^k under max query depth",
        )
        print("done", name, flush=True)
    report = {
        "max_eval_loop": args.max_eval_loop,
        "path_positions": args.path_positions,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "noise_batches": args.noise_batches,
        "noise_scales": args.noise_scales,
        "models": summaries,
        "noise_rows": noise_rows,
    }
    (args.out_dir / "optimizer_behavior_diagnostics_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(args.out_dir / "noise_basin_metrics.csv", noise_rows)
    score_rows = []
    for name, summary in summaries.items():
        score_rows.append(
            {
                "model": name,
                "prefix_successor_score": summary["prefix_successor_score"],
                "target_lock_score_loop7_16": summary["target_lock_score_loop7_16"],
                "overloop_successor_score_loop7_16": summary["overloop_successor_score_loop7_16"],
            }
        )
    write_csv(args.out_dir / "behavior_scores.csv", score_rows)
    save_summary_plots(args.out_dir, summaries, noise_rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
