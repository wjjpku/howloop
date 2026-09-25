from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    ce_by_loop,
    cosine_lr,
    count_parameters,
    pick_device,
    set_seed,
)


@dataclass
class ProgramPathConfig:
    node_count: int = 8
    relation_count: int = 2
    max_depth: int = 6
    program_length: int = 12
    d_model: int = 256
    n_heads: int = 4
    d_mlp: int = 1024
    n_layers: int = 2
    max_loops: int = 6
    dropout: float = 0.0
    outer_norm_groups: int = 0
    residual_projection_groups: int = 0
    rms_norm_eps: float = 1e-6
    inner_norm_style: str = "pre_layernorm"
    readout_norm_style: str = "layernorm"
    block_style: str = "legacy"
    rope_theta: float = 1_000_000.0
    initializer_range: float = 0.02

    @property
    def relation_token_base(self) -> int:
        return self.node_count

    @property
    def map_token(self) -> int:
        return self.node_count + self.relation_count

    @property
    def query_token(self) -> int:
        return self.node_count + self.relation_count + 1

    @property
    def answer_token(self) -> int:
        return self.node_count + self.relation_count + 2

    @property
    def bos_token(self) -> int:
        return self.node_count + self.relation_count + 3

    @property
    def depth_token_base(self) -> int:
        return self.node_count + self.relation_count + 4

    @property
    def vocab_size(self) -> int:
        return self.node_count + self.relation_count + 4 + self.max_depth

    @property
    def seq_len(self) -> int:
        return 1 + 4 * self.relation_count * self.node_count + self.program_length + 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train looped transformers on a programmed multi-hop retrieval task."
    )
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--relation-count", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--program-length", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--loops", type=int, nargs="+", default=[6])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--outer-norm-groups", type=int, default=0)
    parser.add_argument("--residual-projection-groups", type=int, default=0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument(
        "--inner-norm-style",
        choices=("pre_layernorm", "pre_rmsnorm", "ouro_sandwich_rms"),
        default="pre_layernorm",
    )
    parser.add_argument(
        "--readout-norm-style",
        choices=("layernorm", "rmsnorm", "identity"),
        default="layernorm",
    )
    parser.add_argument(
        "--block-style",
        choices=("legacy", "tiny_ouro"),
        default="legacy",
    )
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--initializer-range", type=float, default=0.02)
    parser.add_argument("--aux-loss", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-eval-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--out-dir", type=Path, default=Path("results/program_path_loop"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class LoopedProgramPathTransformer(LoopedGraphPathTransformer):
    pass


def canonical_model_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def load_initial_model_state(model: nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload.get("model_state"))
    if state is None:
        raise KeyError("checkpoint must contain 'model' or 'model_state'")
    model.load_state_dict(canonical_model_state_dict(state), strict=True)


def make_program_path_batch(
    cfg: ProgramPathConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.program_length < cfg.max_depth:
        raise ValueError("program_length must be >= max_depth")

    noise = torch.rand(batch_size, cfg.relation_count, cfg.node_count, device=device)
    successors = noise.argsort(dim=-1)
    src = torch.arange(cfg.node_count, device=device).view(1, 1, cfg.node_count)
    src = src.expand(batch_size, cfg.relation_count, -1)
    rel = torch.arange(cfg.relation_count, device=device).view(1, cfg.relation_count, 1)
    rel = rel.expand(batch_size, -1, cfg.node_count)

    map_quads = torch.empty(
        batch_size,
        cfg.relation_count,
        cfg.node_count,
        4,
        dtype=torch.long,
        device=device,
    )
    map_quads[:, :, :, 0] = cfg.map_token
    map_quads[:, :, :, 1] = cfg.relation_token_base + rel
    map_quads[:, :, :, 2] = src
    map_quads[:, :, :, 3] = successors

    start = torch.randint(0, cfg.node_count, (batch_size,), dtype=torch.long, device=device)
    depth = torch.randint(1, cfg.max_depth + 1, (batch_size,), dtype=torch.long, device=device)
    program = torch.randint(
        0,
        cfg.relation_count,
        (batch_size, cfg.program_length),
        dtype=torch.long,
        device=device,
    )

    targets_by_depth = torch.empty(batch_size, cfg.program_length, dtype=torch.long, device=device)
    current = start
    batch_idx = torch.arange(batch_size, device=device)
    for step in range(cfg.program_length):
        step_rel = program[:, step]
        selected_successors = successors[batch_idx, step_rel, :]
        current = selected_successors.gather(1, current.view(-1, 1)).squeeze(1)
        targets_by_depth[:, step] = current
    target = targets_by_depth.gather(1, (depth - 1).view(-1, 1)).squeeze(1)

    tokens = torch.empty(batch_size, cfg.seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = cfg.bos_token
    maps_flat = map_quads.reshape(batch_size, 4 * cfg.relation_count * cfg.node_count)
    tokens[:, 1 : 1 + maps_flat.shape[1]] = maps_flat
    query_start = 1 + maps_flat.shape[1]
    tokens[:, query_start] = cfg.query_token
    tokens[:, query_start + 1] = start
    tokens[:, query_start + 2] = cfg.depth_token_base + depth - 1
    tokens[:, query_start + 3 : query_start + 3 + cfg.program_length] = (
        cfg.relation_token_base + program
    )
    tokens[:, -1] = cfg.answer_token
    return tokens, target, depth, targets_by_depth


@torch.no_grad()
def evaluate(
    model: LoopedProgramPathTransformer,
    cfg: ProgramPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    max_loops: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    model.eval()
    correct_by_loop = torch.zeros(max_loops, device=device)
    count_by_loop = torch.zeros(max_loops, device=device)
    correct_by_depth_loop = torch.zeros(cfg.max_depth, max_loops, device=device)
    count_by_depth_loop = torch.zeros(cfg.max_depth, max_loops, device=device)
    loss_sum_by_loop = torch.zeros(max_loops, device=device)
    earliest_correct_sum = torch.zeros((), device=device)
    earliest_correct_count = torch.zeros((), device=device)

    autocast_device = "cuda" if device.type == "cuda" else device.type
    for _ in range(batches):
        tokens, target, depth, _ = make_program_path_batch(cfg, batch_size, device)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]
            losses = ce_by_loop(logits_by_loop, target)
        pred = logits_by_loop.argmax(dim=-1)
        correct = pred.eq(target[:, None])
        correct_by_loop += correct.float().sum(dim=0)
        count_by_loop += correct.shape[0]
        loss_sum_by_loop += losses.float().sum(dim=0)
        any_correct = correct.any(dim=1)
        first_correct = correct.float().argmax(dim=1).float() + 1.0
        earliest_correct_sum += first_correct[any_correct].sum()
        earliest_correct_count += any_correct.float().sum()
        for depth_idx in range(cfg.max_depth):
            mask = depth.eq(depth_idx + 1)
            if mask.any():
                correct_by_depth_loop[depth_idx] += correct[mask].float().sum(dim=0)
                count_by_depth_loop[depth_idx] += mask.float().sum()

    loop_acc = correct_by_loop / count_by_loop.clamp_min(1)
    loop_loss = loss_sum_by_loop / count_by_loop.clamp_min(1)
    depth_loop_acc = correct_by_depth_loop / count_by_depth_loop.clamp_min(1)
    earliest = earliest_correct_sum / earliest_correct_count.clamp_min(1)
    return {
        "loop_accuracy": [float(x) for x in loop_acc.detach().cpu()],
        "loop_loss": [float(x) for x in loop_loss.detach().cpu()],
        "depth_loop_accuracy": depth_loop_acc.detach().cpu().tolist(),
        "earliest_correct_loop": float(earliest.detach().cpu()),
        "any_correct_fraction": float((earliest_correct_count / count_by_loop[0].clamp_min(1)).detach().cpu()),
    }


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    scalar_keys = [
        "step",
        "lr",
        "train_loss",
        "train_final_loss",
        "train_aux_loss",
        "elapsed_sec",
        "earliest_correct_loop",
        "any_correct_fraction",
    ]
    max_loop = len(history[-1]["loop_accuracy"])
    for idx in range(max_loop):
        scalar_keys.append(f"eval_acc_loop_{idx + 1}")
        scalar_keys.append(f"eval_loss_loop_{idx + 1}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in history:
            flat = {key: row.get(key, "") for key in scalar_keys}
            for idx, value in enumerate(row["loop_accuracy"]):
                flat[f"eval_acc_loop_{idx + 1}"] = value
            for idx, value in enumerate(row["loop_loss"]):
                flat[f"eval_loss_loop_{idx + 1}"] = value
            writer.writerow(flat)


def plot_run(run_dir: Path, history: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if not history:
        return
    steps = [row["step"] for row in history]
    max_loop = len(history[-1]["loop_accuracy"])
    plt.figure(figsize=(9, 5))
    for idx in range(max_loop):
        plt.plot(steps, [row["loop_accuracy"][idx] for row in history], label=f"loop {idx + 1}")
    plt.xlabel("train step")
    plt.ylabel("eval accuracy")
    plt.ylim(0, 1.02)
    plt.title(f"Program-path accuracy by internal loop, max_loops={max_loop}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(run_dir / "accuracy_by_loop_over_training.png", dpi=180)
    plt.close()

    depth_loop = np.array(summary["final_metrics"]["depth_loop_accuracy"], dtype=np.float32)
    plt.figure(figsize=(1.1 * max_loop + 3, 5))
    im = plt.imshow(depth_loop, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="accuracy")
    plt.xticks(range(max_loop), [str(i + 1) for i in range(max_loop)])
    plt.yticks(range(depth_loop.shape[0]), [str(i + 1) for i in range(depth_loop.shape[0])])
    plt.xlabel("internal loop used for readout")
    plt.ylabel("queried program depth")
    plt.title("Final depth x loop accuracy")
    for y in range(depth_loop.shape[0]):
        for x in range(depth_loop.shape[1]):
            value = depth_loop[y, x]
            color = "black" if value > 0.65 else "white"
            plt.text(x, y, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "final_depth_loop_accuracy_heatmap.png", dpi=180)
    plt.close()


def train_one_loop_count(
    cfg: ProgramPathConfig,
    args: argparse.Namespace,
    *,
    max_loops: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    if cfg.outer_norm_groups and cfg.residual_projection_groups:
        raise ValueError(
            "outer_norm_groups and residual_projection_groups cannot both be nonzero"
        )
    architecture_tag = ""
    if cfg.inner_norm_style != "pre_layernorm" or cfg.readout_norm_style != "layernorm":
        architecture_tag = (
            f"_inner-{cfg.inner_norm_style}_readout-{cfg.readout_norm_style}"
        )
    if cfg.block_style != "legacy":
        architecture_tag = f"_block-{cfg.block_style}{architecture_tag}"
    run_dir = out_dir / (
        f"programpath_N{cfg.node_count}_R{cfg.relation_count}_D{cfg.max_depth}_"
        f"P{cfg.program_length}_d{cfg.d_model}_B{cfg.n_layers}_L{max_loops}_"
        f"outerG{cfg.outer_norm_groups}_projG{cfg.residual_projection_groups}"
        f"{architecture_tag}_seed{args.seed}{args.run_suffix}"
    )
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"{run_dir} exists. Pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = ProgramPathConfig(**{**asdict(cfg), "max_loops": max_loops})
    set_seed(args.seed + 1009 * max_loops + 7919 * run_cfg.relation_count)
    model: nn.Module = LoopedProgramPathTransformer(run_cfg).to(device)
    if args.init_checkpoint is not None:
        load_initial_model_state(model, args.init_checkpoint)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    param_count = count_parameters(model)
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_step = 0
    start_time = time.time()
    autocast_device = "cuda" if device.type == "cuda" else device.type

    metadata = {
        "config": asdict(run_cfg),
        "args": vars(args),
        "parameter_count": param_count,
        "device": str(device),
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(step - 1, base_lr=args.lr, total_steps=args.steps, warmup_steps=args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tokens, target, _, _ = make_program_path_batch(run_cfg, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_by_loop = model.forward_all(tokens, max_loops=max_loops)["logits_by_loop"]  # type: ignore[union-attr]
            per_loop_loss = ce_by_loop(logits_by_loop, target)
            final_loss = per_loop_loss[:, -1].mean()
            aux_loss = per_loop_loss[:, :-1].mean() if max_loops > 1 else final_loss
            loss = final_loss + args.aux_loss * aux_loss
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            metrics = evaluate(
                model,  # type: ignore[arg-type]
                run_cfg,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                max_loops=max_loops,
                amp_enabled=args.amp,
            )
            final_acc = metrics["loop_accuracy"][-1]
            row = {
                "step": step,
                "lr": lr,
                "train_loss": float(loss.detach().cpu()),
                "train_final_loss": float(final_loss.detach().cpu()),
                "train_aux_loss": float(aux_loss.detach().cpu()),
                "elapsed_sec": time.time() - start_time,
                **metrics,
            }
            history.append(row)
            write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if args.save_checkpoints and args.save_eval_checkpoints:
                torch.save(
                    {
                        "model": canonical_model_state_dict(model.state_dict()),
                        "config": asdict(run_cfg),
                        "step": step,
                        "metrics": metrics,
                        "parameter_count": param_count,
                        "task": "program_path",
                    },
                    run_dir / f"checkpoint_step_{step:05d}.pt",
                )
            if final_acc > best_acc:
                best_acc = final_acc
                best_step = step
                if args.save_checkpoints:
                    torch.save(
                        {
                            "model": canonical_model_state_dict(model.state_dict()),
                            "config": asdict(run_cfg),
                            "step": step,
                            "metrics": metrics,
                            "parameter_count": param_count,
                            "task": "program_path",
                        },
                        run_dir / "best.pt",
                    )
            if step % args.print_every == 0 or step == 1 or step == args.steps:
                loop_acc = " ".join(f"L{i+1}:{acc:.3f}" for i, acc in enumerate(metrics["loop_accuracy"]))
                print(
                    f"[L={max_loops} step={step:05d}] loss={float(loss.detach().cpu()):.4f} "
                    f"final_acc={final_acc:.3f} {loop_acc}",
                    flush=True,
                )

    final_metrics = evaluate(
        model,  # type: ignore[arg-type]
        run_cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 32),
        max_loops=max_loops,
        amp_enabled=args.amp,
    )
    summary = {
        "max_loops": max_loops,
        "parameter_count": param_count,
        "best_step": best_step,
        "best_final_accuracy": best_acc,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_checkpoints:
        torch.save(
            {
                "model": canonical_model_state_dict(model.state_dict()),
                "config": asdict(run_cfg),
                "step": args.steps,
                "metrics": final_metrics,
                "parameter_count": param_count,
                "task": "program_path",
            },
            run_dir / "final.pt",
        )
    plot_run(run_dir, history, summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.program_length < args.max_depth:
        raise ValueError("--program-length must be >= --max-depth")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cfg = ProgramPathConfig(
        node_count=args.node_count,
        relation_count=args.relation_count,
        max_depth=args.max_depth,
        program_length=args.program_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=max(args.loops),
        dropout=args.dropout,
        outer_norm_groups=args.outer_norm_groups,
        residual_projection_groups=args.residual_projection_groups,
        rms_norm_eps=args.rms_norm_eps,
        inner_norm_style=args.inner_norm_style,
        readout_norm_style=args.readout_norm_style,
        block_style=args.block_style,
        rope_theta=args.rope_theta,
        initializer_range=args.initializer_range,
    )
    print(f"device={device} cfg={cfg}", flush=True)
    summaries = []
    for max_loops in args.loops:
        summaries.append(train_one_loop_count(cfg, args, max_loops=max_loops, device=device, out_dir=out_dir))
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    rows = []
    for summary in summaries:
        final_acc = summary["final_metrics"]["loop_accuracy"][-1]
        rows.append({"max_loops": summary["max_loops"], "final_accuracy": final_acc})
    with (out_dir / "loop_count_final_accuracy_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["max_loops", "final_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
