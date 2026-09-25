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
    TriadicShortageConfig,
    TriadicShortageModel,
    all_triples,
    make_visibility_mask,
    split_indices,
)


def stage_targets(operands: torch.Tensor, *, p: int) -> torch.Tensor:
    """Return the intended two-stage computation: a+b, then (a+b)*c."""
    if p < 2:
        raise ValueError("p must be at least 2")
    if operands.ndim != 2 or operands.shape[1] != 3:
        raise ValueError("operands must have shape [batch, 3]")
    if operands.numel() and (operands.min() < 0 or operands.max() >= p):
        raise ValueError("operand values must lie in [0, p)")
    partial_sum = (operands[:, 0] + operands[:, 1]).remainder(p)
    endpoint = (partial_sum * operands[:, 2]).remainder(p)
    return torch.stack((partial_sum, endpoint), dim=1)


def stage_accuracy_matrix(
    logits_by_loop: torch.Tensor,
    targets_by_stage: torch.Tensor,
) -> torch.Tensor:
    """Score every loop readout against the intermediate and endpoint targets."""
    if logits_by_loop.ndim != 3:
        raise ValueError("logits_by_loop must have shape [batch, loops, classes]")
    if targets_by_stage.ndim != 2 or targets_by_stage.shape[1] != 2:
        raise ValueError("targets_by_stage must have shape [batch, 2]")
    if logits_by_loop.shape[0] != targets_by_stage.shape[0]:
        raise ValueError("logits and targets must have the same batch size")
    prediction = logits_by_loop.argmax(dim=-1)
    return (
        prediction.unsqueeze(2)
        .eq(targets_by_stage.unsqueeze(1))
        .float()
        .mean(dim=0)
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _count_parameters(model: torch.nn.Module) -> int:
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


@torch.no_grad()
def evaluate_stage_composition(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    indices: torch.Tensor,
    *,
    train_loops: int,
    batch_size: int,
) -> dict[str, Any]:
    if not 1 <= train_loops <= model.cfg.loops:
        raise ValueError("train_loops must be within the configured loops")
    if indices.numel() < 1 or batch_size < 1:
        raise ValueError("indices and batch_size must be nonempty")
    model.eval()
    correct = torch.zeros(model.cfg.loops, 2, device=operands.device)
    total = 0
    endpoint_loss_sum = 0.0
    for start in range(0, indices.numel(), batch_size):
        batch_operands = operands[indices[start : start + batch_size]]
        targets = stage_targets(batch_operands, p=model.cfg.p)
        visibility = make_visibility_mask(
            "full",
            batch_operands.shape[0],
            model.cfg.loops,
            operands.device,
        )
        logits, _ = model(batch_operands, visibility)
        prediction = logits.argmax(dim=-1)
        correct += (
            prediction.unsqueeze(2)
            .eq(targets.unsqueeze(1))
            .sum(dim=0)
        )
        endpoint_loss_sum += float(
            F.cross_entropy(
                logits[:, train_loops - 1],
                targets[:, 1],
                reduction="sum",
            ).item()
        )
        total += batch_operands.shape[0]
    matrix = correct.float() / total
    result: dict[str, Any] = {
        "examples": total,
        "train_loops": train_loops,
        "accuracy_matrix": matrix.detach().cpu().tolist(),
        "loop1_sum_accuracy": float(matrix[0, 0].item()),
        "trained_endpoint_accuracy": float(matrix[train_loops - 1, 1].item()),
        "trained_endpoint_loss": endpoint_loss_sum / total,
    }
    if train_loops < model.cfg.loops:
        result["extra_loop_endpoint_accuracy"] = float(
            matrix[train_loops, 1].item()
        )
    return result


@torch.no_grad()
def stage_transplant_diagnostics(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    indices: torch.Tensor,
    *,
    train_loops: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if train_loops < 2:
        return {"status": "not_applicable", "reason": "requires at least two loops"}
    sample_count = min(batch_size, int(indices.numel()))
    if sample_count < 2:
        raise ValueError("transplant diagnostics require at least two examples")
    generator = torch.Generator(device=indices.device).manual_seed(seed)
    donor_choice = torch.randint(
        0,
        indices.numel(),
        (sample_count,),
        generator=generator,
        device=indices.device,
    )
    receiver_choice = torch.randint(
        0,
        indices.numel(),
        (sample_count,),
        generator=generator,
        device=indices.device,
    )
    donor = operands[indices[donor_choice]]
    receiver = operands[indices[receiver_choice]]
    donor_visibility = make_visibility_mask(
        "full", sample_count, model.cfg.loops, operands.device
    )
    receiver_visibility = make_visibility_mask(
        "full", sample_count, model.cfg.loops, operands.device
    )
    _, donor_states = model(
        donor,
        donor_visibility,
        active_loops=1,
    )
    donor_workspace = donor_states[:, 0]
    transplanted_logits, _ = model.continue_from_workspace(
        receiver,
        receiver_visibility,
        workspace=donor_workspace,
        start_loop=1,
        stop_loop=train_loops,
    )
    hybrid_target = (
        (donor[:, 0] + donor[:, 1]).remainder(model.cfg.p)
        * receiver[:, 2]
    ).remainder(model.cfg.p)
    receiver_target = stage_targets(receiver, p=model.cfg.p)[:, 1]
    permutation = torch.randperm(
        sample_count,
        generator=generator,
        device=operands.device,
    )
    shuffled_logits, _ = model.continue_from_workspace(
        receiver,
        receiver_visibility,
        workspace=donor_workspace[permutation],
        start_loop=1,
        stop_loop=train_loops,
    )
    c_only_visibility = torch.zeros_like(receiver_visibility)
    c_only_visibility[:, 1:train_loops, 2] = True
    c_only_logits, _ = model.continue_from_workspace(
        receiver,
        c_only_visibility,
        workspace=donor_workspace,
        start_loop=1,
        stop_loop=train_loops,
    )
    shuffled_c_only_logits, _ = model.continue_from_workspace(
        receiver,
        c_only_visibility,
        workspace=donor_workspace[permutation],
        start_loop=1,
        stop_loop=train_loops,
    )
    reset_c_only_logits, _ = model.continue_from_workspace(
        receiver,
        c_only_visibility,
        workspace=model._initial_workspace(sample_count),
        start_loop=1,
        stop_loop=train_loops,
    )
    donor_c_only_logits, _ = model.continue_from_workspace(
        donor,
        c_only_visibility,
        workspace=donor_workspace,
        start_loop=1,
        stop_loop=train_loops,
    )
    reset_before = torch.zeros(
        model.cfg.loops,
        dtype=torch.bool,
        device=operands.device,
    )
    reset_before[1] = True
    reset_logits, _ = model(
        receiver,
        receiver_visibility,
        reset_before=reset_before,
        active_loops=train_loops,
    )
    native_logits, _ = model(
        receiver,
        receiver_visibility,
        active_loops=train_loops,
    )

    def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
        return float(
            logits[:, -1].argmax(dim=-1).eq(target).float().mean().item()
        )

    return {
        "status": "ok",
        "examples": sample_count,
        "native_receiver_accuracy": accuracy(native_logits, receiver_target),
        "hybrid_transplant_accuracy": accuracy(
            transplanted_logits,
            hybrid_target,
        ),
        "hybrid_transplant_c_only_accuracy": accuracy(
            c_only_logits,
            hybrid_target,
        ),
        "donor_self_c_only_accuracy": accuracy(
            donor_c_only_logits,
            stage_targets(donor, p=model.cfg.p)[:, 1],
        ),
        "shuffled_workspace_hybrid_accuracy": accuracy(
            shuffled_logits,
            hybrid_target,
        ),
        "shuffled_workspace_c_only_accuracy": accuracy(
            shuffled_c_only_logits,
            hybrid_target,
        ),
        "reset_workspace_c_only_accuracy": accuracy(
            reset_c_only_logits,
            hybrid_target,
        ),
        "reset_before_second_loop_accuracy": accuracy(
            reset_logits,
            receiver_target,
        ),
    }


def _checkpoint_payload(
    *,
    model: TriadicShortageModel,
    cfg: TriadicShortageConfig,
    optimizer: torch.optim.Optimizer,
    train_loops: int,
    seed: int,
    step: int,
    history: list[dict[str, Any]],
    train_idx: torch.Tensor,
    heldout_idx: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "target": "sum_then_multiply",
        "train_loops": train_loops,
        "seed": seed,
        "step": step,
        "history": history,
        "split": {
            "train_idx": train_idx.detach().cpu(),
            "heldout_idx": heldout_idx.detach().cpu(),
        },
    }


def train_one(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if not 1 <= args.train_loops <= args.configured_loops:
        raise ValueError("train_loops must be within configured_loops")
    _set_seed(args.seed)
    d_mlp = 2 * args.d_model if args.d_mlp is None else args.d_mlp
    cfg = TriadicShortageConfig(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=d_mlp,
        loops=args.configured_loops,
        architecture=args.architecture,
        dropout=args.dropout,
    )
    split_seed = args.split_seed_offset + 1000 * cfg.p + args.seed
    train_idx_cpu, heldout_idx_cpu = split_indices(
        cfg.p**3,
        args.train_fraction,
        seed=split_seed,
    )
    run_name = args.run_name or (
        f"stage_{cfg.architecture}_p{cfg.p}_d{cfg.d_model}_"
        f"T{args.train_loops}_C{cfg.loops}_seed{args.seed}"
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
    operands, _ = all_triples(cfg.p, device=device)
    train_idx = train_idx_cpu.to(device)
    heldout_idx = heldout_idx_cpu.to(device)
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_step = 0
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        sample = torch.randint(
            0,
            train_idx.numel(),
            (args.batch_size,),
            device=device,
        )
        batch_operands = operands[train_idx[sample]]
        targets = stage_targets(batch_operands, p=cfg.p)
        visibility = make_visibility_mask(
            "full",
            args.batch_size,
            cfg.loops,
            device,
        )
        logits, _ = model(
            batch_operands,
            visibility,
            active_loops=args.train_loops,
        )
        loss = F.cross_entropy(logits[:, -1], targets[:, 1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            heldout = evaluate_stage_composition(
                model,
                operands,
                heldout_idx,
                train_loops=args.train_loops,
                batch_size=args.eval_batch_size,
            )
            row = {
                "step": step,
                "batch_loss": float(loss.detach().cpu()),
                "heldout_endpoint_accuracy": heldout[
                    "trained_endpoint_accuracy"
                ],
                "heldout_loop1_sum_accuracy": heldout["loop1_sum_accuracy"],
                "heldout_endpoint_loss": heldout["trained_endpoint_loss"],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            if row["heldout_endpoint_accuracy"] > best_accuracy:
                best_accuracy = float(row["heldout_endpoint_accuracy"])
                best_step = step
                _atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        cfg=cfg,
                        optimizer=optimizer,
                        train_loops=args.train_loops,
                        seed=args.seed,
                        step=step,
                        history=history,
                        train_idx=train_idx,
                        heldout_idx=heldout_idx,
                    ),
                    run_dir / "best.pt",
                )
            if step % args.print_every == 0 or step == args.steps:
                print(
                    f"[{run_name} step={step:05d}] "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"endpoint={heldout['trained_endpoint_accuracy']:.3f} "
                    f"loop1_sum={heldout['loop1_sum_accuracy']:.3f}",
                    flush=True,
                )

    final_metrics = {
        "train": evaluate_stage_composition(
            model,
            operands,
            train_idx,
            train_loops=args.train_loops,
            batch_size=args.eval_batch_size,
        ),
        "heldout": evaluate_stage_composition(
            model,
            operands,
            heldout_idx,
            train_loops=args.train_loops,
            batch_size=args.eval_batch_size,
        ),
        "all": evaluate_stage_composition(
            model,
            operands,
            torch.arange(operands.shape[0], device=device),
            train_loops=args.train_loops,
            batch_size=args.eval_batch_size,
        ),
    }
    diagnostics = stage_transplant_diagnostics(
        model,
        operands,
        heldout_idx,
        train_loops=args.train_loops,
        batch_size=args.diagnostic_batch_size,
        seed=args.eval_seed,
    )
    final_payload = _checkpoint_payload(
        model=model,
        cfg=cfg,
        optimizer=optimizer,
        train_loops=args.train_loops,
        seed=args.seed,
        step=args.steps,
        history=history,
        train_idx=train_idx,
        heldout_idx=heldout_idx,
    )
    _atomic_torch_save(final_payload, run_dir / "final.pt")
    summary = {
        "run_name": run_name,
        "target": "sum_then_multiply",
        "architecture": cfg.architecture,
        "p": cfg.p,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "configured_loops": cfg.loops,
        "train_loops": args.train_loops,
        "seed": args.seed,
        "steps": args.steps,
        "parameter_count": _count_parameters(model),
        "best_step": best_step,
        "best_heldout_accuracy": best_accuracy,
        "final_metrics": final_metrics,
        "diagnostics": diagnostics,
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
        description=(
            "Train a two-stage modular task for positive virtual-depth reuse: "
            "first compute a+b, then multiply the result by c."
        )
    )
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument(
        "--architecture",
        choices=["looped", "unshared"],
        default="looped",
    )
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=None)
    parser.add_argument("--configured-loops", type=int, default=3)
    parser.add_argument("--train-loops", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed-offset", type=int, default=24_611)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--diagnostic-batch-size", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-seed", type=int, default=78_103)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/resource_conditioned_reuse_20260717/"
            "triadic_stage_composition"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.p < 2
        or args.configured_loops < 1
        or args.train_loops < 1
        or args.steps < 1
        or args.batch_size < 1
        or args.eval_batch_size < 1
        or args.diagnostic_batch_size < 2
        or args.eval_every < 1
    ):
        parser.error("task, loop, training, and batch sizes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train_one(args, _pick_device(args.device))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
