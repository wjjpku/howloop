from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_loop import count_parameters, pick_device, set_seed
from reasoning_loop.portable_component_train import (
    cosine_lr,
    load_training_checkpoint,
    save_training_checkpoint,
)
from reasoning_loop.sharded_evidence import (
    ShardedEvidenceConfig,
    ShardedEvidenceModel,
    make_batch,
    make_visibility_indices,
)


FORMAL_CONDITIONS = ("full", "shard", "shard_reset", "shard_shuffled")


@torch.no_grad()
def evaluate(
    model: ShardedEvidenceModel,
    cfg: ShardedEvidenceConfig,
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    if batches < 1 or batch_size < 1:
        raise ValueError("evaluation batch size and batches must be positive")
    model.eval()
    correct = torch.zeros(cfg.loops, device=device)
    total = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(batches):
        bits, labels = make_batch(
            batch_size=batch_size,
            evidence_count=cfg.evidence_count,
            device=device,
            generator=generator,
        )
        visibility = make_visibility_indices(
            condition,
            batch_size=batch_size,
            cfg=cfg,
            device=device,
            generator=generator,
        )
        logits, _ = model(
            bits,
            visibility,
            reset_between_loops=condition == "shard_reset",
        )
        correct += logits.argmax(dim=-1).eq(labels[:, None]).float().sum(dim=0)
        total += batch_size
    model.train()
    per_loop = (correct / total).detach().cpu().tolist()
    return {
        "per_loop_accuracy": per_loop,
        "final_accuracy": float(per_loop[-1]),
    }


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    condition: str,
    cfg: ShardedEvidenceConfig,
    args: argparse.Namespace,
    *,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    if condition not in FORMAL_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    set_seed(seed)
    run_name = args.run_name or f"{condition}_d{cfg.d_model}_L{cfg.loops}_seed{seed}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = ShardedEvidenceModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    start_step = 0
    history: list[dict[str, Any]] = []
    source_condition: str | None = None
    if args.resume is not None:
        source_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        source_condition = source_payload.get("mode")
        start_step, history = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expected_cfg=cfg,
            expected_seed=seed,
        )
    stop_step = args.steps if args.stop_step is None else args.stop_step
    if not 0 <= start_step < stop_step <= args.steps:
        raise ValueError("require 0 <= start_step < stop_step <= steps")
    if args.checkpoint_every < 1 or args.eval_every < 1:
        raise ValueError("checkpoint_every and eval_every must be positive")

    started = time.time()
    best_accuracy = max(
        (float(row["final_accuracy"]) for row in history), default=-1.0
    )
    best_step = max(
        (
            int(row["step"])
            for row in history
            if float(row["final_accuracy"]) == best_accuracy
        ),
        default=0,
    )
    for step in range(start_step + 1, stop_step + 1):
        lr = cosine_lr(step - 1, args.steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        bits, labels = make_batch(
            batch_size=args.batch_size,
            evidence_count=cfg.evidence_count,
            device=device,
        )
        visibility = make_visibility_indices(
            condition,
            batch_size=args.batch_size,
            cfg=cfg,
            device=device,
        )
        logits, _ = model(
            bits,
            visibility,
            reset_between_loops=condition == "shard_reset",
        )
        loss = F.cross_entropy(logits[:, -1], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_eval = step == start_step + 1 or step % args.eval_every == 0 or step == stop_step
        if should_eval:
            metrics = evaluate(
                model,
                cfg,
                condition=condition,
                device=device,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                seed=args.eval_seed + step,
            )
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": lr,
                "final_accuracy": metrics["final_accuracy"],
                "per_loop_accuracy": json.dumps(metrics["per_loop_accuracy"]),
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            if metrics["final_accuracy"] > best_accuracy:
                best_accuracy = metrics["final_accuracy"]
                best_step = step
                torch.save(model.state_dict(), run_dir / "best.pt")
            if step % args.print_every == 0 or step == stop_step:
                loops_text = ",".join(f"{value:.3f}" for value in metrics["per_loop_accuracy"])
                print(
                    f"[{condition} seed={seed} step={step:05d}] "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"final={metrics['final_accuracy']:.3f} loops={loops_text}",
                    flush=True,
                )

        if step % args.checkpoint_every == 0 or step == stop_step:
            save_training_checkpoint(
                run_dir / f"checkpoint_step_{step:07d}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                mode=condition,
                seed=seed,
                step=step,
                total_steps=args.steps,
                history=history,
            )

    final_metrics = evaluate(
        model,
        cfg,
        condition=condition,
        device=device,
        batch_size=args.eval_batch_size,
        batches=args.eval_batches,
        seed=args.eval_seed + 100_000,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "condition": condition,
            "seed": seed,
            "step": stop_step,
        },
        run_dir / "final.pt",
    )
    _write_history(run_dir / "history.csv", history)
    summary = {
        "condition": condition,
        "source_condition": source_condition,
        "seed": seed,
        "steps": stop_step,
        "total_steps": args.steps,
        "start_step": start_step,
        "parameter_count": count_parameters(model),
        "best_step": best_step,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir.resolve()),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sharded-evidence reuse controls.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=FORMAL_CONDITIONS,
        default=list(FORMAL_CONDITIONS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--evidence-count", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--stop-step", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--eval-seed", type=int, default=70_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/sharded_evidence_reuse_20260715"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = ShardedEvidenceConfig(
        evidence_count=args.evidence_count,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        loops=args.loops,
        dropout=args.dropout,
    )
    device = torch.device(args.device) if args.device != "auto" else pick_device()
    print(f"device={device} cfg={cfg}", flush=True)
    for condition in args.conditions:
        for seed in args.seeds:
            train_one(
                condition,
                cfg,
                args,
                seed=seed,
                device=device,
                out_dir=args.out_dir,
            )


if __name__ == "__main__":
    main()
