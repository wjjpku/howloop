from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.triadic_shortage import (
    VISIBILITY_CONDITIONS,
    TriadicShortageConfig,
    TriadicShortageModel,
    all_triples,
    make_visibility_mask,
    split_indices,
)


FIXED_CHECKPOINT_STEPS = {250, 500, 1000, 2000, 4000, 6000, 8000, 10000}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    distractor_logits = logits.clone()
    distractor_logits.scatter_(1, labels.unsqueeze(1), -torch.inf)
    return correct - distractor_logits.max(dim=1).values


@torch.no_grad()
def evaluate(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    condition: str,
    batch_size: int,
    seed: int,
    reset_before: torch.Tensor | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if indices.numel() < 1:
        raise ValueError("evaluation indices must be nonempty")
    model.eval()
    correct = torch.zeros(model.cfg.loops, device=operands.device)
    total = 0
    final_loss_sum = 0.0
    final_margin_sum = 0.0
    generator = torch.Generator(device=operands.device).manual_seed(seed)
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_operands = operands[batch_indices]
        batch_labels = labels[batch_indices]
        visibility = make_visibility_mask(
            condition,
            batch_operands.shape[0],
            model.cfg.loops,
            operands.device,
            generator=generator,
        )
        logits, _ = model(
            batch_operands,
            visibility,
            reset_before=reset_before,
        )
        correct += logits.argmax(dim=-1).eq(batch_labels.unsqueeze(1)).sum(dim=0)
        final_loss_sum += float(
            F.cross_entropy(logits[:, -1], batch_labels, reduction="sum").item()
        )
        final_margin_sum += float(_margin(logits[:, -1], batch_labels).sum().item())
        total += batch_operands.shape[0]
    per_loop = (correct.float() / total).detach().cpu().tolist()
    model.train()
    return {
        "per_loop_accuracy": [float(value) for value in per_loop],
        "final_accuracy": float(per_loop[-1]),
        "final_loss": final_loss_sum / total,
        "final_margin": final_margin_sum / total,
        "examples": total,
    }


def _checkpoint_payload(
    *,
    model: TriadicShortageModel,
    optimizer: torch.optim.Optimizer,
    cfg: TriadicShortageConfig,
    condition: str,
    seed: int,
    step: int,
    total_steps: int,
    history: list[dict[str, Any]],
    train_idx: torch.Tensor,
    heldout_idx: torch.Tensor,
    split_seed: int,
    train_fraction: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "condition": condition,
        "seed": seed,
        "step": step,
        "total_steps": total_steps,
        "history": history,
        "split": {
            "train_idx": train_idx.detach().cpu(),
            "heldout_idx": heldout_idx.detach().cpu(),
            "split_seed": split_seed,
            "train_fraction": train_fraction,
        },
        "rng_state": capture_rng_state(),
    }


def _config_from_args(args: argparse.Namespace) -> TriadicShortageConfig:
    d_mlp = 2 * args.d_model if args.d_mlp is None else args.d_mlp
    return TriadicShortageConfig(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=d_mlp,
        loops=args.loops,
        architecture=args.architecture,
        dropout=args.dropout,
    )


def train_one(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.condition not in VISIBILITY_CONDITIONS:
        raise ValueError(f"unknown condition: {args.condition}")
    stop_step = args.steps if args.stop_step is None else args.stop_step
    if not 0 < stop_step <= args.steps:
        raise ValueError("stop_step must be in (0, steps]")
    requested_cfg = _config_from_args(args)
    set_seed(args.seed)
    source_condition: str | None = None
    start_step = 0
    history: list[dict[str, Any]] = []
    if args.resume is None:
        cfg = requested_cfg
        split_seed = args.split_seed_offset + cfg.p * 1000 + args.seed
        train_idx_cpu, heldout_idx_cpu = split_indices(
            cfg.p**3,
            args.train_fraction,
            seed=split_seed,
        )
        resume_payload = None
    else:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = TriadicShortageConfig.from_dict(resume_payload["config"])
        if cfg != requested_cfg:
            raise ValueError(
                f"resume config mismatch: checkpoint={cfg.to_dict()} requested={requested_cfg.to_dict()}"
            )
        if int(resume_payload["seed"]) != args.seed:
            raise ValueError("resume seed does not match requested seed")
        start_step = int(resume_payload["step"])
        if not start_step < stop_step:
            raise ValueError("resume step must be earlier than stop_step")
        history = list(resume_payload.get("history", []))
        source_condition = str(resume_payload["condition"])
        split = resume_payload["split"]
        split_seed = int(split["split_seed"])
        train_idx_cpu = split["train_idx"].long()
        heldout_idx_cpu = split["heldout_idx"].long()
        args.train_fraction = float(split["train_fraction"])

    run_name = args.run_name or (
        f"{args.condition}_{cfg.architecture}_p{cfg.p}_d{cfg.d_model}_"
        f"L{cfg.loops}_seed{args.seed}"
    )
    run_dir = args.out_dir / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **cfg.to_dict(),
                "condition": args.condition,
                "seed": args.seed,
                "train_fraction": args.train_fraction,
                "steps": args.steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model = TriadicShortageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        restore_rng_state(resume_payload["rng_state"])

    operands, labels = all_triples(cfg.p, device=device)
    train_idx = train_idx_cpu.to(device)
    heldout_idx = heldout_idx_cpu.to(device)
    best_accuracy = max(
        (float(row["heldout_final_accuracy"]) for row in history),
        default=-1.0,
    )
    best_step = max(
        (
            int(row["step"])
            for row in history
            if float(row["heldout_final_accuracy"]) == best_accuracy
        ),
        default=0,
    )
    started = time.time()
    for step in range(start_step + 1, stop_step + 1):
        sample = torch.randint(0, train_idx.numel(), (args.batch_size,), device=device)
        batch_indices = train_idx[sample]
        batch_operands = operands[batch_indices]
        batch_labels = labels[batch_indices]
        visibility = make_visibility_mask(
            args.condition,
            args.batch_size,
            cfg.loops,
            device,
        )
        logits, _ = model(batch_operands, visibility)
        loss = F.cross_entropy(logits[:, -1], batch_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_eval = (
            step == start_step + 1
            or step % args.eval_every == 0
            or step == stop_step
        )
        if should_eval:
            train_metrics = evaluate(
                model,
                operands,
                labels,
                train_idx,
                condition=args.condition,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + step,
            )
            heldout_metrics = evaluate(
                model,
                operands,
                labels,
                heldout_idx,
                condition=args.condition,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed + 100_000 + step,
            )
            row = {
                "step": step,
                "batch_loss": float(loss.detach().cpu()),
                "train_final_accuracy": train_metrics["final_accuracy"],
                "heldout_final_accuracy": heldout_metrics["final_accuracy"],
                "heldout_final_loss": heldout_metrics["final_loss"],
                "heldout_final_margin": heldout_metrics["final_margin"],
                "heldout_per_loop_accuracy": json.dumps(
                    heldout_metrics["per_loop_accuracy"]
                ),
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            if heldout_metrics["final_accuracy"] > best_accuracy:
                best_accuracy = heldout_metrics["final_accuracy"]
                best_step = step
                best_payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    condition=args.condition,
                    seed=args.seed,
                    step=step,
                    total_steps=args.steps,
                    history=history,
                    train_idx=train_idx,
                    heldout_idx=heldout_idx,
                    split_seed=split_seed,
                    train_fraction=args.train_fraction,
                )
                _atomic_torch_save(best_payload, run_dir / "best.pt")
            if step % args.print_every == 0 or step == stop_step:
                loop_text = ",".join(
                    f"{value:.3f}" for value in heldout_metrics["per_loop_accuracy"]
                )
                print(
                    f"[{run_name} step={step:05d}] loss={float(loss.detach().cpu()):.4f} "
                    f"heldout={heldout_metrics['final_accuracy']:.3f} loops={loop_text}",
                    flush=True,
                )

        should_checkpoint = (
            step % args.checkpoint_every == 0
            or step in FIXED_CHECKPOINT_STEPS
            or step == stop_step
        )
        if should_checkpoint:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                condition=args.condition,
                seed=args.seed,
                step=step,
                total_steps=args.steps,
                history=history,
                train_idx=train_idx,
                heldout_idx=heldout_idx,
                split_seed=split_seed,
                train_fraction=args.train_fraction,
            )
            _atomic_torch_save(
                payload,
                run_dir / f"checkpoint_step_{step:07d}.pt",
            )

    final_metrics = {
        "train": evaluate(
            model,
            operands,
            labels,
            train_idx,
            condition=args.condition,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed + 200_000,
        ),
        "heldout": evaluate(
            model,
            operands,
            labels,
            heldout_idx,
            condition=args.condition,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed + 300_000,
        ),
        "all": evaluate(
            model,
            operands,
            labels,
            torch.arange(operands.shape[0], device=device),
            condition=args.condition,
            batch_size=args.eval_batch_size,
            seed=args.eval_seed + 400_000,
        ),
    }
    final_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        condition=args.condition,
        seed=args.seed,
        step=stop_step,
        total_steps=args.steps,
        history=history,
        train_idx=train_idx,
        heldout_idx=heldout_idx,
        split_seed=split_seed,
        train_fraction=args.train_fraction,
    )
    _atomic_torch_save(final_payload, run_dir / "final.pt")
    _write_history(run_dir / "history.csv", history)
    summary = {
        "run_name": run_name,
        "condition": args.condition,
        "source_condition": source_condition,
        "architecture": cfg.architecture,
        "p": cfg.p,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "loops": cfg.loops,
        "seed": args.seed,
        "start_step": start_step,
        "steps": stop_step,
        "total_steps": args.steps,
        "parameter_count": count_parameters(model),
        "best_step": best_step,
        "best_heldout_accuracy": best_accuracy,
        "final_metrics": final_metrics,
        "run_dir": str(run_dir.resolve()),
        "device": str(device),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train shortage-controlled p=17 triadic modular-addition models."
    )
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument("--condition", choices=VISIBILITY_CONDITIONS, default="full")
    parser.add_argument("--architecture", choices=["looped", "unshared"], default="looped")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=None)
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed-offset", type=int, default=9187)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--stop-step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-seed", type=int, default=51_001)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/triadic_shortage_reuse_20260716/runs"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_every < 1 or args.checkpoint_every < 1 or args.batch_size < 1:
        parser.error("evaluation, checkpoint, and batch intervals must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_one(args, pick_device(args.device))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
