from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_loop import (
    cosine_lr,
    count_parameters,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_stepwise import (
    StepwiseGraphPathConfig,
    build_stepwise_model,
    make_stepwise_batch,
)


def fixed_target_loss(
    logits_by_loop: torch.Tensor,
    targets_by_pos: torch.Tensor,
    *,
    target_depth: int,
) -> torch.Tensor:
    if logits_by_loop.ndim != 3:
        raise ValueError("logits_by_loop must have shape [batch, loops, nodes]")
    if not 1 <= target_depth <= targets_by_pos.shape[1]:
        raise ValueError("target_depth is outside the available path positions")
    return F.cross_entropy(
        logits_by_loop[:, -1],
        targets_by_pos[:, target_depth - 1],
    )


@torch.no_grad()
def evaluate_fixed_target(
    model: torch.nn.Module,
    cfg: StepwiseGraphPathConfig,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    active_macro_steps: int,
    target_depth: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    if active_macro_steps < 1 or target_depth < 1:
        raise ValueError("active_macro_steps and target_depth must be positive")
    correct = torch.zeros(
        active_macro_steps,
        target_depth,
        device=device,
    )
    total = 0
    loss_sum = 0.0
    autocast_device = "cuda" if device.type == "cuda" else device.type
    model.eval()
    for _ in range(batches):
        tokens, targets_by_pos, _, _ = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=target_depth,
        )
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model.forward_all(  # type: ignore[attr-defined]
                tokens,
                max_loops=active_macro_steps,
            )["logits_by_loop"]
            loss = fixed_target_loss(
                logits,
                targets_by_pos,
                target_depth=target_depth,
            )
        prediction = logits.argmax(dim=-1)
        for readout in range(active_macro_steps):
            correct[readout] += (
                prediction[:, readout, None]
                .eq(targets_by_pos[:, :target_depth])
                .sum(dim=0)
            )
        total += batch_size
        loss_sum += float(loss.item()) * batch_size
    accuracy = (correct / total).cpu().tolist()
    return {
        "active_macro_steps": active_macro_steps,
        "target_depth": target_depth,
        "readout_position_accuracy": accuracy,
        "final_target_accuracy": float(accuracy[-1][target_depth - 1]),
        "final_loss": loss_sum / total,
        "examples": total,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train graph-path models with a fixed target depth and variable compute."
    )
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--target-depth", type=int, default=6)
    parser.add_argument("--active-macro-steps", type=int, default=1)
    parser.add_argument("--architecture", choices=["looped", "standard"], default="looped")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/resource_conditioned_reuse_20260716/graph_fixed_target"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.target_depth < 1 or args.active_macro_steps < 1:
        parser.error("target depth and active macro-steps must be positive")
    if args.steps < 1 or args.eval_every < 1 or args.eval_batches < 1:
        parser.error("training and evaluation counts must be positive")
    return args


def train_one(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = StepwiseGraphPathConfig(
        node_count=args.node_count,
        max_depth=max(args.target_depth, args.active_macro_steps),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        max_loops=args.active_macro_steps,
        dropout=args.dropout,
    )
    run_name = args.run_name or (
        f"fixed_D{args.target_depth}_R{args.active_macro_steps}_"
        f"{args.architecture}_d{args.d_model}_seed{args.seed}"
    )
    run_dir = args.out_dir / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_stepwise_model(cfg, architecture=args.architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    autocast_device = "cuda" if device.type == "cuda" else device.type
    for step in range(1, args.steps + 1):
        model.train()
        lr = cosine_lr(
            step - 1,
            base_lr=args.lr,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        tokens, targets_by_pos, _, _ = make_stepwise_batch(
            cfg,
            args.batch_size,
            device,
            path_positions=args.target_depth,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits = model.forward_all(  # type: ignore[attr-defined]
                tokens,
                max_loops=args.active_macro_steps,
            )["logits_by_loop"]
            loss = fixed_target_loss(
                logits,
                targets_by_pos,
                target_depth=args.target_depth,
            )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_fixed_target(
                model,
                cfg,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                active_macro_steps=args.active_macro_steps,
                target_depth=args.target_depth,
                amp_enabled=args.amp,
            )
            history.append(
                {
                    "step": step,
                    "lr": lr,
                    "train_loss": float(loss.detach().cpu()),
                    "final_target_accuracy": metrics["final_target_accuracy"],
                    "final_loss": metrics["final_loss"],
                }
            )
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
    final_metrics = evaluate_fixed_target(
        model,
        cfg,
        device=device,
        batch_size=args.eval_batch_size,
        batches=max(args.eval_batches, 4),
        active_macro_steps=args.active_macro_steps,
        target_depth=args.target_depth,
        amp_enabled=args.amp,
    )
    checkpoint = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "architecture": args.architecture,
        "target_depth": args.target_depth,
        "active_macro_steps": args.active_macro_steps,
        "seed": args.seed,
        "step": args.steps,
        "metrics": final_metrics,
    }
    torch.save(checkpoint, run_dir / "final.pt")
    summary = {
        "run_name": run_name,
        "config": asdict(cfg),
        "architecture": args.architecture,
        "target_depth": args.target_depth,
        "active_macro_steps": args.active_macro_steps,
        "seed": args.seed,
        "steps": args.steps,
        "parameter_count": count_parameters(model),
        "final_metrics": final_metrics,
        "run_dir": str(run_dir.resolve()),
        "device": str(device),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train_one(args, pick_device(args.device))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
