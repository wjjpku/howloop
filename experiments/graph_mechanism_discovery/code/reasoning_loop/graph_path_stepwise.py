from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.graph_path_loop import (
    LoopedGraphPathTransformer,
    TransformerBlock,
    build_optimizer_param_groups,
    cosine_lr,
    count_parameters,
    pick_device,
    set_optimizer_base_lr,
    set_seed,
)
from reasoning_loop.graph_path_stepwise_standard import StandardMacroStepGraphPathTransformer


@dataclass
class StepwiseGraphPathConfig:
    node_count: int = 8
    max_depth: int = 6
    d_model: int = 256
    n_heads: int = 4
    d_mlp: int = 1024
    n_layers: int = 2
    max_loops: int = 6
    dropout: float = 0.0
    block_schedule: str = "all_blocks"

    @property
    def edge_token(self) -> int:
        return self.node_count

    @property
    def query_token(self) -> int:
        return self.node_count + 1

    @property
    def answer_token(self) -> int:
        return self.node_count + 2

    @property
    def bos_token(self) -> int:
        return self.node_count + 3

    @property
    def vocab_size(self) -> int:
        return self.node_count + 4

    @property
    def seq_len(self) -> int:
        return 1 + 3 * self.node_count + 3


def build_stepwise_model(
    cfg: StepwiseGraphPathConfig,
    *,
    architecture: str,
) -> nn.Module:
    if architecture == "looped":
        return LoopedGraphPathTransformer(cfg)
    if architecture == "standard":
        return StandardMacroStepGraphPathTransformer(cfg)
    raise ValueError(f"unknown architecture: {architecture}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train no-depth graph-path looped transformers with trajectory-level "
            "regularization."
        )
    )
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=["looped", "standard"],
        default="looped",
        help="looped reuses n_layers blocks; standard uses distinct blocks at every macro-step.",
    )
    parser.add_argument("--loops", type=int, nargs="+", default=[6])
    parser.add_argument(
        "--loss-mode",
        choices=["intermediate", "transition", "final"],
        default="intermediate",
        help=(
            "intermediate: loop t is supervised to f^t(start). transition: loop1 "
            "and loopL are anchored, intermediate loops are constrained by the "
            "graph successor transition. final: no trajectory regularizer, only "
            "the last loop is supervised to f^L(start)."
        ),
    )
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--first-anchor-weight", type=float, default=1.0)
    parser.add_argument("--final-anchor-weight", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-lr-scale", type=float, default=1.0)
    parser.add_argument("--attention-lr-scale", type=float, default=1.0)
    parser.add_argument("--mlp-lr-scale", type=float, default=1.0)
    parser.add_argument("--norm-lr-scale", type=float, default=1.0)
    parser.add_argument("--readout-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--block-lr-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional multiplier for each physical block. The number of values "
            "must equal --n-layers; multipliers compose with component scales."
        ),
    )
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initialization-seed",
        type=int,
        default=None,
        help="Explicit parameter initialization seed for paired sweeps.",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Reset RNG to this seed after initialization for paired data streams.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-eval-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-dir", type=Path, default=Path("results/graph_path_stepwise"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def make_stepwise_batch(
    cfg: StepwiseGraphPathConfig,
    batch_size: int,
    device: torch.device,
    *,
    path_positions: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = cfg.max_depth if path_positions is None else path_positions
    noise = torch.rand(batch_size, cfg.node_count, device=device)
    successors = noise.argsort(dim=-1)
    src = torch.arange(cfg.node_count, device=device).view(1, cfg.node_count).expand(batch_size, -1)

    edge_triplets = torch.empty(batch_size, cfg.node_count, 3, dtype=torch.long, device=device)
    edge_triplets[:, :, 0] = cfg.edge_token
    edge_triplets[:, :, 1] = src
    edge_triplets[:, :, 2] = successors

    start = torch.randint(0, cfg.node_count, (batch_size,), dtype=torch.long, device=device)
    targets_by_pos = torch.empty(batch_size, positions, dtype=torch.long, device=device)
    current = start
    for step in range(positions):
        current = successors.gather(1, current.view(-1, 1)).squeeze(1)
        targets_by_pos[:, step] = current

    tokens = torch.empty(batch_size, cfg.seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = cfg.bos_token
    tokens[:, 1 : 1 + 3 * cfg.node_count] = edge_triplets.reshape(batch_size, 3 * cfg.node_count)
    tokens[:, -3] = cfg.query_token
    tokens[:, -2] = start
    tokens[:, -1] = cfg.answer_token
    return tokens, targets_by_pos, successors, start


def ce_to_positions(logits_by_loop: torch.Tensor, targets_by_pos: torch.Tensor) -> torch.Tensor:
    batch, loops, node_count = logits_by_loop.shape
    if targets_by_pos.shape[1] < loops:
        raise ValueError("targets_by_pos must contain at least one target per loop")
    target = targets_by_pos[:, :loops].reshape(batch * loops)
    return F.cross_entropy(
        logits_by_loop.reshape(batch * loops, node_count),
        target,
        reduction="none",
    ).view(batch, loops)


def pushforward_successor(probs: torch.Tensor, successors: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(probs)
    out.scatter_add_(1, successors, probs)
    return out


def transition_consistency_loss(
    logits_by_loop: torch.Tensor,
    targets_by_pos: torch.Tensor,
    successors: torch.Tensor,
    *,
    transition_weight: float,
    first_anchor_weight: float,
    final_anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loop1_ce = F.cross_entropy(logits_by_loop[:, 0, :], targets_by_pos[:, 0])
    final_ce = F.cross_entropy(logits_by_loop[:, -1, :], targets_by_pos[:, logits_by_loop.shape[1] - 1])
    if logits_by_loop.shape[1] <= 1:
        transition_ce = logits_by_loop.new_zeros(())
    else:
        probs = logits_by_loop[:, :-1, :].softmax(dim=-1)
        shifted = torch.stack(
            [pushforward_successor(probs[:, idx, :], successors) for idx in range(probs.shape[1])],
            dim=1,
        ).detach()
        log_next = logits_by_loop[:, 1:, :].log_softmax(dim=-1)
        transition_ce = -(shifted * log_next).sum(dim=-1).mean()
    loss = (
        first_anchor_weight * loop1_ce
        + final_anchor_weight * final_ce
        + transition_weight * transition_ce
    )
    return loss, {
        "train_loop1_ce": loop1_ce.detach(),
        "train_final_ce": final_ce.detach(),
        "train_transition_ce": transition_ce.detach(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    cfg: StepwiseGraphPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    max_loops: int,
    path_positions: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    correct_to_pos = torch.zeros(max_loops, path_positions, device=device)
    prob_to_pos = torch.zeros(max_loops, path_positions, device=device)
    loss_to_pos = torch.zeros(max_loops, path_positions, device=device)
    entropy_sum = torch.zeros(max_loops, device=device)
    margin_sum = torch.zeros(max_loops, device=device)
    count = 0

    autocast_device = "cuda" if device.type == "cuda" else device.type
    for _ in range(batches):
        tokens, targets_by_pos, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=path_positions
        )
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]
        probs_by_loop = logits_by_loop.softmax(dim=-1)
        pred = logits_by_loop.argmax(dim=-1)
        count += tokens.shape[0]
        for loop_idx in range(max_loops):
            logits = logits_by_loop[:, loop_idx, :]
            probs = probs_by_loop[:, loop_idx, :]
            top2 = logits.topk(2, dim=-1).values
            margin_sum[loop_idx] += (top2[:, 0] - top2[:, 1]).sum()
            entropy_sum[loop_idx] += (-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)).sum()
            for pos_idx in range(path_positions):
                target = targets_by_pos[:, pos_idx]
                correct_to_pos[loop_idx, pos_idx] += pred[:, loop_idx].eq(target).float().sum()
                prob_to_pos[loop_idx, pos_idx] += probs.gather(1, target[:, None]).squeeze(1).sum()
                loss_to_pos[loop_idx, pos_idx] += F.cross_entropy(logits, target, reduction="sum")

    acc_to_pos = correct_to_pos / count
    mean_prob_to_pos = prob_to_pos / count
    mean_loss_to_pos = loss_to_pos / count
    best_acc, best_pos = acc_to_pos.max(dim=1)
    rolling_idx = torch.arange(max_loops, device=device).clamp(max=path_positions - 1)
    rolling_acc = acc_to_pos[torch.arange(max_loops, device=device), rolling_idx]
    rolling_loss = mean_loss_to_pos[torch.arange(max_loops, device=device), rolling_idx]
    train_positions = torch.arange(min(max_loops, cfg.max_depth), device=device)
    train_acc = acc_to_pos[train_positions, train_positions].mean() if train_positions.numel() else torch.zeros((), device=device)
    final_target_idx = min(cfg.max_depth, path_positions) - 1
    final_target_acc = acc_to_pos[max_loops - 1, final_target_idx]
    final_target_loss = mean_loss_to_pos[max_loops - 1, final_target_idx]

    return {
        "examples": count,
        "path_positions": path_positions,
        "acc_to_pos": acc_to_pos.detach().cpu().tolist(),
        "mean_prob_to_pos": mean_prob_to_pos.detach().cpu().tolist(),
        "mean_loss_to_pos": mean_loss_to_pos.detach().cpu().tolist(),
        "rolling_acc_by_loop": [float(x) for x in rolling_acc.detach().cpu()],
        "rolling_loss_by_loop": [float(x) for x in rolling_loss.detach().cpu()],
        "best_position_by_loop": [int(x) + 1 for x in best_pos.detach().cpu().tolist()],
        "best_acc_by_loop": [float(x) for x in best_acc.detach().cpu()],
        "trained_step_mean_acc": float(train_acc.detach().cpu()),
        "final_target_acc": float(final_target_acc.detach().cpu()),
        "final_target_loss": float(final_target_loss.detach().cpu()),
        "mean_entropy_by_loop": [float(x) for x in (entropy_sum / count).detach().cpu()],
        "mean_top1_margin_by_loop": [float(x) for x in (margin_sum / count).detach().cpu()],
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames = [
        "step",
        "lr",
        "train_loss",
        "train_intermediate_loss",
        "train_loop1_ce",
        "train_final_ce",
        "train_transition_ce",
        "elapsed_sec",
        "trained_step_mean_acc",
        "final_target_acc",
        "final_target_loss",
    ]
    max_loops = len(history[-1]["rolling_acc_by_loop"])
    for idx in range(max_loops):
        fieldnames.append(f"rolling_acc_loop_{idx + 1}")
        fieldnames.append(f"best_position_loop_{idx + 1}")
        fieldnames.append(f"best_acc_loop_{idx + 1}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            flat = {key: row.get(key, "") for key in fieldnames}
            for idx, value in enumerate(row["rolling_acc_by_loop"]):
                flat[f"rolling_acc_loop_{idx + 1}"] = value
            for idx, value in enumerate(row["best_position_by_loop"]):
                flat[f"best_position_loop_{idx + 1}"] = value
            for idx, value in enumerate(row["best_acc_by_loop"]):
                flat[f"best_acc_loop_{idx + 1}"] = value
            writer.writerow(flat)


def save_heatmap(arr: np.ndarray, *, path: Path, title: str, label: str, vmin: float, vmax: float) -> None:
    loops, positions = arr.shape
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(positions), [str(i) for i in range(1, positions + 1)])
    ax.set_yticks(range(loops), [str(i) for i in range(1, loops + 1)])
    ax.set_xlabel("actual path position k")
    ax.set_ylabel("readout loop t")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_run(run_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if history:
        steps = [row["step"] for row in history]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for idx in range(len(history[-1]["rolling_acc_by_loop"])):
            ax.plot(steps, [row["rolling_acc_by_loop"][idx] for row in history], label=f"loop {idx + 1}->f^{idx + 1}")
        ax.set_xlabel("train step")
        ax.set_ylabel("matched step accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title("No-depth graph path: loop t accuracy to f^t")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "matched_step_accuracy_over_training.png", dpi=180)
        plt.close(fig)

    final = summary["final_metrics"]
    save_heatmap(
        np.array(final["acc_to_pos"], dtype=np.float32),
        path=run_dir / "final_loop_by_path_accuracy_heatmap.png",
        title="Final loop x path-position accuracy",
        label="accuracy",
        vmin=0,
        vmax=1,
    )
    save_heatmap(
        np.array(final["mean_prob_to_pos"], dtype=np.float32),
        path=run_dir / "final_loop_by_path_probability_heatmap.png",
        title="Final mean probability assigned to path position",
        label="mean probability",
        vmin=0,
        vmax=1,
    )

    xs = np.arange(1, len(final["best_position_by_loop"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(xs, final["best_position_by_loop"], marker="o", label="best matched k")
    ax.plot(xs, xs, linestyle=":", color="gray", label="k=t")
    ax.set_xlabel("readout loop t")
    ax.set_ylabel("best matched path position k")
    ax.set_title("Best path position at training loops")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "final_best_position_curve.png", dpi=180)
    plt.close(fig)


def train_one(
    cfg: StepwiseGraphPathConfig,
    args: argparse.Namespace,
    *,
    max_loops: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    run_cfg = StepwiseGraphPathConfig(**{**asdict(cfg), "max_loops": max_loops})
    run_prefix = "stepgraph" if args.architecture == "looped" else "stepgraph_standard"
    run_dir = out_dir / (
        f"{run_prefix}_{args.loss_mode}_N{run_cfg.node_count}_D{run_cfg.max_depth}_"
        f"d{run_cfg.d_model}_B{run_cfg.n_layers}_L{max_loops}_seed{args.seed}"
    )
    if args.run_suffix:
        run_dir = run_dir.with_name(f"{run_dir.name}_{args.run_suffix}")
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} exists. Pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    architecture_offset = 0 if args.architecture == "looped" else 200_003
    initialization_seed = (
        args.seed
        + 1009 * max_loops
        + (17 if args.loss_mode == "transition" else 0)
        + architecture_offset
        if args.initialization_seed is None
        else args.initialization_seed
    )
    set_seed(initialization_seed)

    model = build_stepwise_model(run_cfg, architecture=args.architecture).to(device)
    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        ckpt_cfg = StepwiseGraphPathConfig(**ckpt["config"])
        if asdict(ckpt_cfg) != asdict(run_cfg):
            raise ValueError(
                f"checkpoint config {ckpt_cfg} does not match requested config {run_cfg}"
            )
        model.load_state_dict(ckpt["model"])
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    if args.data_seed is not None:
        set_seed(args.data_seed)
    block_lr_scales = args.block_lr_scales
    if block_lr_scales is not None and len(block_lr_scales) != run_cfg.n_layers:
        raise ValueError(
            "--block-lr-scales must provide exactly one value per physical block"
        )
    component_lr_scales = {
        "embedding": args.embedding_lr_scale,
        "attention": args.attention_lr_scale,
        "mlp": args.mlp_lr_scale,
        "norm": args.norm_lr_scale,
        "readout": args.readout_lr_scale,
    }
    optimizer_groups = build_optimizer_param_groups(
        model,
        base_lr=args.lr,
        component_lr_scales=component_lr_scales,
        block_lr_scales=block_lr_scales,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    param_count = count_parameters(model)
    autocast_device = "cuda" if device.type == "cuda" else device.type
    start_time = time.time()
    best_score = -1.0
    best_step = 0
    history: list[dict[str, Any]] = []

    metadata = {
        "config": asdict(run_cfg),
        "args": vars(args),
        "parameter_count": param_count,
        "device": str(device),
        "task": "graph_path_stepwise_no_depth",
        "architecture": args.architecture,
        "loss_mode": args.loss_mode,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "run_dir": str(run_dir),
        "initialization_seed": initialization_seed,
        "data_seed": args.data_seed,
        "optimizer_groups": [
            {
                "group_name": str(group["group_name"]),
                "lr_scale": float(group["lr_scale"]),
                "parameter_count": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer_groups
        ],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(step - 1, base_lr=args.lr, total_steps=args.steps, warmup_steps=args.warmup_steps)
        set_optimizer_base_lr(optimizer, lr)
        tokens, targets_by_pos, successors, _ = make_stepwise_batch(
            run_cfg, args.batch_size, device, path_positions=max(run_cfg.max_depth, max_loops)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]  # type: ignore[union-attr]
            per_loop_ce = ce_to_positions(logits_by_loop, targets_by_pos)
            intermediate_loss = per_loop_ce.mean()
            if args.loss_mode == "intermediate":
                loss = intermediate_loss
                pieces = {
                    "train_loop1_ce": per_loop_ce[:, 0].mean().detach(),
                    "train_final_ce": per_loop_ce[:, -1].mean().detach(),
                    "train_transition_ce": logits_by_loop.new_zeros(()).detach(),
                }
            elif args.loss_mode == "transition":
                loss, pieces = transition_consistency_loss(
                    logits_by_loop,
                    targets_by_pos,
                    successors,
                    transition_weight=args.transition_weight,
                    first_anchor_weight=args.first_anchor_weight,
                    final_anchor_weight=args.final_anchor_weight,
                )
            else:
                final_ce = per_loop_ce[:, -1].mean()
                loss = final_ce
                pieces = {
                    "train_loop1_ce": per_loop_ce[:, 0].mean().detach(),
                    "train_final_ce": final_ce.detach(),
                    "train_transition_ce": logits_by_loop.new_zeros(()).detach(),
                }
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            metrics = evaluate(
                model,
                run_cfg,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_loops=max_loops,
                path_positions=max(run_cfg.max_depth, max_loops),
                amp_enabled=args.amp,
            )
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "train_intermediate_loss": float(intermediate_loss.detach().cpu()),
                "train_loop1_ce": float(pieces["train_loop1_ce"].cpu()),
                "train_final_ce": float(pieces["train_final_ce"].cpu()),
                "train_transition_ce": float(pieces["train_transition_ce"].cpu()),
                "elapsed_sec": time.time() - start_time,
                **metrics,
            }
            history.append(row)
            write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            score = (
                metrics["final_target_acc"]
                if args.loss_mode == "final"
                else metrics["trained_step_mean_acc"]
            )
            if args.save_checkpoints and args.save_eval_checkpoints:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(run_cfg),
                        "step": step,
                        "metrics": metrics,
                        "parameter_count": param_count,
                        "task": "graph_path_stepwise_no_depth",
                        "architecture": args.architecture,
                        "loss_mode": args.loss_mode,
                        "initialization_seed": initialization_seed,
                        "data_seed": args.data_seed,
                    },
                    run_dir / f"checkpoint_step_{step:05d}.pt",
                )
            if score > best_score:
                best_score = score
                best_step = step
                if args.save_checkpoints:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": asdict(run_cfg),
                            "step": step,
                            "metrics": metrics,
                            "parameter_count": param_count,
                            "task": "graph_path_stepwise_no_depth",
                            "architecture": args.architecture,
                            "loss_mode": args.loss_mode,
                            "initialization_seed": initialization_seed,
                            "data_seed": args.data_seed,
                        },
                        run_dir / "best.pt",
                    )
            if step % args.print_every == 0 or step == 1 or step == args.steps:
                accs = " ".join(f"L{i+1}->f{i+1}:{acc:.3f}" for i, acc in enumerate(metrics["rolling_acc_by_loop"]))
                print(
                    f"[{args.loss_mode} L={max_loops} step={step:05d}] "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"step_mean={metrics['trained_step_mean_acc']:.3f} "
                    f"final={metrics['final_target_acc']:.3f} {accs}",
                    flush=True,
                )

    final_metrics = evaluate(
        model,
        run_cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 64),
        max_loops=max_loops,
        path_positions=max(run_cfg.max_depth, max_loops),
        amp_enabled=args.amp,
    )
    summary = {
        "max_loops": max_loops,
        "architecture": args.architecture,
        "loss_mode": args.loss_mode,
        "parameter_count": param_count,
        "initialization_seed": initialization_seed,
        "data_seed": args.data_seed,
        "best_step": best_step,
        "best_trained_step_mean_accuracy": best_score,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(run_cfg),
                "step": args.steps,
                "metrics": final_metrics,
                "parameter_count": param_count,
                "task": "graph_path_stepwise_no_depth",
                "architecture": args.architecture,
                "loss_mode": args.loss_mode,
                "initialization_seed": initialization_seed,
                "data_seed": args.data_seed,
            },
            run_dir / "final.pt",
        )
    plot_run(run_dir, history, summary)
    return summary


def main() -> None:
    args = parse_args()
    if max(args.loops) > args.max_depth:
        raise ValueError("--max-depth should be at least as large as the maximum trained loop count")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cfg = StepwiseGraphPathConfig(
        node_count=args.node_count,
        max_depth=args.max_depth,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=max(args.loops),
        dropout=args.dropout,
    )
    print(
        f"device={device} cfg={cfg} architecture={args.architecture} loss_mode={args.loss_mode}",
        flush=True,
    )
    summaries = [
        train_one(cfg, args, max_loops=max_loops, device=device, out_dir=args.out_dir)
        for max_loops in args.loops
    ]
    summary_name = (
        f"summary_{args.loss_mode}.json"
        if args.architecture == "looped"
        else f"summary_{args.architecture}_{args.loss_mode}.json"
    )
    (args.out_dir / summary_name).write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
