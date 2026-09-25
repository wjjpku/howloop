from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.triadic_shortage import (
    TriadicShortageConfig,
    TriadicShortageModel,
    all_triples,
    make_visibility_mask,
    split_indices,
)
from reasoning_loop.triadic_shortage_train import (
    count_parameters,
    pick_device,
    set_seed,
)


MIXED_CONDITIONS = ("full", "full_once", "sequential_shuffled")
CONDITION_TO_MODE = {
    condition: mode_id
    for mode_id, condition in enumerate(MIXED_CONDITIONS)
}


def make_mixed_visibility(
    batch_size: int,
    loops: int,
    device: torch.device,
    *,
    mode_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 1 or loops < 3:
        raise ValueError("mixed visibility requires a positive batch and at least three loops")
    if mode_ids is None:
        mode_ids = torch.randint(
            0,
            len(MIXED_CONDITIONS),
            (batch_size,),
            device=device,
            generator=generator,
        )
    else:
        mode_ids = mode_ids.to(device=device, dtype=torch.long)
        if mode_ids.shape != (batch_size,):
            raise ValueError("mode_ids must have shape [batch]")
        if mode_ids.numel() and (
            mode_ids.min() < 0 or mode_ids.max() >= len(MIXED_CONDITIONS)
        ):
            raise ValueError("mode_ids must lie in the configured range")
    visibility = torch.zeros(
        (batch_size, loops, 3),
        dtype=torch.bool,
        device=device,
    )
    for mode_id, condition in enumerate(MIXED_CONDITIONS):
        selected = mode_ids.eq(mode_id)
        if selected.any():
            visibility[selected] = make_visibility_mask(
                condition,
                int(selected.sum().item()),
                loops,
                device,
                generator=generator,
            )
    return visibility, mode_ids


@torch.no_grad()
def _evaluate_condition(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    condition: str,
    batch_size: int,
    seed: int,
    mode_conditioning: bool,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    correct = torch.zeros(model.cfg.loops, device=device)
    total = 0
    loss_sum = 0.0
    model.eval()
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_operands = operands[batch_indices]
        batch_labels = labels[batch_indices]
        visibility = make_visibility_mask(
            condition,
            batch_indices.numel(),
            model.cfg.loops,
            device,
            generator=generator,
        )
        mode_ids = (
            torch.full(
                (batch_indices.numel(),),
                CONDITION_TO_MODE[condition],
                dtype=torch.long,
                device=device,
            )
            if mode_conditioning
            else None
        )
        logits, _ = model(
            batch_operands,
            visibility,
            mode_ids=mode_ids,
        )
        correct += logits.argmax(dim=-1).eq(batch_labels.unsqueeze(1)).sum(dim=0)
        loss_sum += float(
            F.cross_entropy(
                logits[:, -1],
                batch_labels,
                reduction="sum",
            ).item()
        )
        total += batch_indices.numel()
    per_loop = (correct / total).cpu().tolist()
    return {
        "per_loop_accuracy": [float(value) for value in per_loop],
        "final_accuracy": float(per_loop[-1]),
        "final_loss": loss_sum / total,
        "examples": total,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one triadic model on mixed input-access schedules."
    )
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=128)
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed-offset", type=int, default=9187)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mode-conditioning", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/resource_conditioned_reuse_20260716/triadic_mixed"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.loops < 3:
        parser.error("--loops must be at least 3")
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("steps and batch sizes must be positive")
    return args


def _checkpoint_payload(
    model: TriadicShortageModel,
    optimizer: torch.optim.Optimizer,
    *,
    cfg: TriadicShortageConfig,
    args: argparse.Namespace,
    step: int,
    train_idx: torch.Tensor,
    heldout_idx: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "condition": "mixed",
        "mode_conditioning": bool(args.mode_conditioning),
        "seed": int(args.seed),
        "step": int(step),
        "split": {
            "train_idx": train_idx.detach().cpu(),
            "heldout_idx": heldout_idx.detach().cpu(),
        },
    }


def train_one(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = TriadicShortageConfig(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        loops=args.loops,
        dropout=args.dropout,
        mode_count=len(MIXED_CONDITIONS) if args.mode_conditioning else 0,
    )
    split_seed = args.split_seed_offset + cfg.p * 1000 + args.seed
    train_idx_cpu, heldout_idx_cpu = split_indices(
        cfg.p**3,
        args.train_fraction,
        seed=split_seed,
    )
    run_name = args.run_name or (
        f"mixed_{'mode' if args.mode_conditioning else 'nomode'}_"
        f"d{cfg.d_model}_L{cfg.loops}_seed{args.seed}"
    )
    run_dir = args.out_dir / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    model = TriadicShortageModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    operands, labels = all_triples(cfg.p, device=device)
    train_idx = train_idx_cpu.to(device)
    heldout_idx = heldout_idx_cpu.to(device)
    history: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(
        70_000 + args.seed
    )
    for step in range(1, args.steps + 1):
        model.train()
        sample = torch.randint(
            0,
            train_idx.numel(),
            (args.batch_size,),
            device=device,
            generator=generator,
        )
        batch_indices = train_idx[sample]
        visibility, mode_ids = make_mixed_visibility(
            args.batch_size,
            cfg.loops,
            device,
            generator=generator,
        )
        logits, _ = model(
            operands[batch_indices],
            visibility,
            mode_ids=mode_ids if args.mode_conditioning else None,
        )
        loss = F.cross_entropy(logits[:, -1], labels[batch_indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = {
                condition: _evaluate_condition(
                    model,
                    operands,
                    labels,
                    heldout_idx,
                    condition=condition,
                    batch_size=args.eval_batch_size,
                    seed=90_000 + 100 * step + mode_id,
                    mode_conditioning=args.mode_conditioning,
                )
                for mode_id, condition in enumerate(MIXED_CONDITIONS)
            }
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "heldout_accuracy": {
                        condition: values["final_accuracy"]
                        for condition, values in metrics.items()
                    },
                }
            )
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    cfg=cfg,
                    args=args,
                    step=step,
                    train_idx=train_idx,
                    heldout_idx=heldout_idx,
                ),
                run_dir / f"checkpoint_step_{step:07d}.pt",
            )
    final_metrics = {
        condition: _evaluate_condition(
            model,
            operands,
            labels,
            heldout_idx,
            condition=condition,
            batch_size=args.eval_batch_size,
            seed=120_000 + mode_id,
            mode_conditioning=args.mode_conditioning,
        )
        for mode_id, condition in enumerate(MIXED_CONDITIONS)
    }
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            cfg=cfg,
            args=args,
            step=args.steps,
            train_idx=train_idx,
            heldout_idx=heldout_idx,
        ),
        run_dir / "final.pt",
    )
    summary = {
        "run_name": run_name,
        "config": cfg.to_dict(),
        "mode_conditioning": bool(args.mode_conditioning),
        "seed": int(args.seed),
        "steps": int(args.steps),
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
    summary = train_one(args, pick_device(args.device))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
