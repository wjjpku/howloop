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

from reasoning_loop.typed_relation_composition import (
    RelationBatch,
    TypedRelationConfig,
    TypedRelationModel,
    make_relation_batch,
    relation_visibility,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
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


def _training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    objective: str,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    endpoint_loss = F.cross_entropy(logits[:, -1], targets[:, 1])
    stage_loss: torch.Tensor | None = None
    if logits.shape[1] >= 2:
        stage_loss = F.cross_entropy(logits[:, 0], targets[:, 0])
    if objective == "final":
        loss = endpoint_loss
    elif objective == "staged":
        if stage_loss is None:
            raise ValueError("staged objective requires at least two loops")
        loss = 0.5 * (endpoint_loss + stage_loss)
    elif objective == "anchor":
        if stage_loss is None:
            raise ValueError("anchor objective requires at least two loops")
        loss = endpoint_loss + anchor_weight * stage_loss
    else:
        raise ValueError(f"unknown objective: {objective}")
    return loss, {
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "stage_loss": (
            float(stage_loss.detach().cpu())
            if stage_loss is not None
            else float("nan")
        ),
    }


@torch.no_grad()
def evaluate(
    model: TypedRelationModel,
    *,
    train_loops: int,
    condition: str,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    composition_order: str = "f_then_g",
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    correct = torch.zeros(model.cfg.loops, 2, device=device)
    total = 0
    endpoint_loss_sum = 0.0
    for _ in range(batches):
        batch = make_relation_batch(
            batch_size=batch_size,
            node_count=model.cfg.node_count,
            device=device,
            generator=generator,
            composition_order=composition_order,
        )
        visibility = relation_visibility(
            condition,
            batch_size=batch_size,
            loops=model.cfg.loops,
            node_count=model.cfg.node_count,
            device=device,
            composition_order=composition_order,
        )
        logits, _ = model(batch, visibility)
        correct += (
            logits.argmax(dim=-1)
            .unsqueeze(2)
            .eq(batch.targets.unsqueeze(1))
            .sum(dim=0)
        )
        endpoint_loss_sum += float(
            F.cross_entropy(
                logits[:, train_loops - 1],
                batch.targets[:, 1],
                reduction="sum",
            ).item()
        )
        total += batch_size
    matrix = correct / total
    result: dict[str, Any] = {
        "condition": condition,
        "examples": total,
        "accuracy_matrix": matrix.detach().cpu().tolist(),
        "loop1_first_relation_accuracy": float(matrix[0, 0].item()),
        "trained_endpoint_accuracy": float(
            matrix[train_loops - 1, 1].item()
        ),
        "trained_endpoint_loss": endpoint_loss_sum / total,
    }
    if train_loops < model.cfg.loops:
        result["extra_loop_endpoint_accuracy"] = float(
            matrix[train_loops, 1].item()
        )
    return result


def _hybrid_target(
    donor: RelationBatch,
    receiver: RelationBatch,
) -> torch.Tensor:
    if donor.composition_order != receiver.composition_order:
        raise ValueError("donor and receiver must share composition_order")
    donor_first = donor.targets[:, 0]
    second_mapping = (
        receiver.g if donor.composition_order == "f_then_g" else receiver.f
    )
    return second_mapping.gather(1, donor_first[:, None]).squeeze(1)


@torch.no_grad()
def transplant_diagnostics(
    model: TypedRelationModel,
    *,
    examples: int,
    seed: int,
    device: torch.device,
    composition_order: str = "f_then_g",
) -> dict[str, Any]:
    if model.cfg.loops < 2:
        return {"status": "not_applicable", "reason": "requires two loops"}
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    donor = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    receiver = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    full = relation_visibility(
        "full",
        batch_size=examples,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=composition_order,
    )
    aligned = relation_visibility(
        "aligned",
        batch_size=examples,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=composition_order,
    )
    _, donor_full_states = model(donor, full, active_loops=1)
    _, donor_aligned_states = model(donor, aligned, active_loops=1)
    donor_full_workspace = donor_full_states[:, 0]
    donor_aligned_workspace = donor_aligned_states[:, 0]
    target = _hybrid_target(donor, receiver)
    full_logits, _ = model.continue_from_workspace(
        receiver,
        full,
        workspace=donor_full_workspace,
        start_loop=1,
        stop_loop=2,
    )
    aligned_logits, _ = model.continue_from_workspace(
        receiver,
        aligned,
        workspace=donor_aligned_workspace,
        start_loop=1,
        stop_loop=2,
    )
    permutation = torch.randperm(
        examples,
        device=device,
        generator=generator,
    )
    shuffled_logits, _ = model.continue_from_workspace(
        receiver,
        aligned,
        workspace=donor_aligned_workspace[permutation],
        start_loop=1,
        stop_loop=2,
    )
    reset_logits, _ = model.continue_from_workspace(
        receiver,
        aligned,
        workspace=model._initial_workspace(receiver.query),
        start_loop=1,
        stop_loop=2,
    )

    def accuracy(logits: torch.Tensor) -> float:
        return float(
            logits[:, -1]
            .argmax(dim=-1)
            .eq(target)
            .float()
            .mean()
            .item()
        )

    return {
        "status": "ok",
        "examples": examples,
        "hybrid_full_accuracy": accuracy(full_logits),
        "hybrid_aligned_accuracy": accuracy(aligned_logits),
        "shuffled_workspace_aligned_accuracy": accuracy(shuffled_logits),
        "reset_workspace_aligned_accuracy": accuracy(reset_logits),
    }


def _checkpoint_payload(
    *,
    model: TypedRelationModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.cfg.to_dict(),
        "train_loops": args.train_loops,
        "objective": args.objective,
        "anchor_weight": args.anchor_weight,
        "train_visibility": args.train_visibility,
        "composition_order": args.composition_order,
        "readout_mode": args.readout_mode,
        "seed": args.seed,
        "step": step,
        "history": history,
    }


def train_one(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = TypedRelationConfig(
        node_count=args.node_count,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        loops=args.configured_loops,
        architecture=args.architecture,
        dropout=args.dropout,
    )
    if not 1 <= args.train_loops <= cfg.loops:
        raise ValueError("train_loops must be within configured loops")
    run_name = args.run_name or (
        f"typed_{cfg.architecture}_{args.objective}_{args.train_visibility}_"
        f"N{cfg.node_count}_d{cfg.d_model}_T{args.train_loops}_seed{args.seed}"
    )
    run_dir = args.out_dir / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)

    model = TypedRelationModel(cfg).to(device)
    if args.readout_mode == "frozen_random":
        for parameter in model.readout_norm.parameters():
            parameter.requires_grad_(False)
        model.readout.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 97_003
    )
    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_step = 0
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        batch = make_relation_batch(
            batch_size=args.batch_size,
            node_count=cfg.node_count,
            device=device,
            generator=generator,
            composition_order=args.composition_order,
        )
        visibility = relation_visibility(
            args.train_visibility,
            batch_size=args.batch_size,
            loops=cfg.loops,
            node_count=cfg.node_count,
            device=device,
            composition_order=args.composition_order,
        )
        logits, _ = model(
            batch,
            visibility,
            active_loops=args.train_loops,
        )
        loss, loss_parts = _training_loss(
            logits,
            batch.targets,
            objective=args.objective,
            anchor_weight=args.anchor_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                train_loops=args.train_loops,
                condition="full",
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                seed=args.eval_seed,
                device=device,
                composition_order=args.composition_order,
            )
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                **loss_parts,
                "endpoint_accuracy": metrics["trained_endpoint_accuracy"],
                "loop1_first_relation_accuracy": metrics[
                    "loop1_first_relation_accuracy"
                ],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            (run_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            if row["endpoint_accuracy"] > best_accuracy:
                best_accuracy = float(row["endpoint_accuracy"])
                best_step = step
                _atomic_torch_save(
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        step=step,
                        history=history,
                    ),
                    run_dir / "best.pt",
                )
            if step % args.print_every == 0 or step == args.steps:
                print(
                    f"[{run_name} step={step:05d}] "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"endpoint={row['endpoint_accuracy']:.3f} "
                    f"stage1={row['loop1_first_relation_accuracy']:.3f}",
                    flush=True,
                )

    evaluation = {
        condition: evaluate(
            model,
            train_loops=args.train_loops,
            condition=condition,
            batch_size=args.eval_batch_size,
            batches=args.final_eval_batches,
            seed=args.eval_seed + 1000 * index,
            device=device,
            composition_order=args.composition_order,
        )
        for index, condition in enumerate(
            ("full", "aligned", "swapped", "f_only", "g_only")
        )
    }
    transplant = transplant_diagnostics(
        model,
        examples=args.transplant_examples,
        seed=args.eval_seed + 88_001,
        device=device,
        composition_order=args.composition_order,
    )
    _atomic_torch_save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            args=args,
            step=args.steps,
            history=history,
        ),
        run_dir / "final.pt",
    )
    summary = {
        "run_name": run_name,
        "architecture": cfg.architecture,
        "objective": args.objective,
        "anchor_weight": args.anchor_weight,
        "train_visibility": args.train_visibility,
        "composition_order": args.composition_order,
        "readout_mode": args.readout_mode,
        "node_count": cfg.node_count,
        "d_model": cfg.d_model,
        "d_mlp": cfg.d_mlp,
        "configured_loops": cfg.loops,
        "train_loops": args.train_loops,
        "seed": args.seed,
        "steps": args.steps,
        "parameter_count": _count_parameters(model),
        "best_step": best_step,
        "best_endpoint_accuracy": best_accuracy,
        "evaluation": evaluation,
        "transplant": transplant,
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
            "Train typed random-relation composition as a controlled "
            "serial-depth shortage."
        )
    )
    parser.add_argument("--node-count", type=int, default=16)
    parser.add_argument(
        "--architecture",
        choices=["looped", "unshared"],
        default="looped",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--configured-loops", type=int, default=3)
    parser.add_argument("--train-loops", type=int, default=2)
    parser.add_argument(
        "--objective",
        choices=["final", "anchor", "staged"],
        default="final",
    )
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument(
        "--train-visibility",
        choices=["full", "aligned"],
        default="full",
    )
    parser.add_argument(
        "--composition-order",
        choices=["f_then_g", "g_then_f"],
        default="f_then_g",
    )
    parser.add_argument(
        "--readout-mode",
        choices=["learned", "frozen_random"],
        default="learned",
        help=(
            "Whether the LayerNorm-plus-linear decoder is learned or held at "
            "its random initialization."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--final-eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--transplant-examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=47_021)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/typed_relation_composition_20260717"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.node_count < 2
        or args.d_model < 1
        or args.n_heads < 1
        or args.d_model % args.n_heads
        or args.d_mlp < 1
        or args.configured_loops < 1
        or args.train_loops < 1
        or args.steps < 1
        or args.batch_size < 1
        or args.eval_batch_size < 1
        or args.eval_batches < 1
        or args.final_eval_batches < 1
        or args.transplant_examples < 2
        or args.anchor_weight < 0
    ):
        parser.error("model, training, and evaluation sizes must be valid")
    if args.objective in {"anchor", "staged"} and args.train_loops < 2:
        parser.error("anchor and staged objectives require at least two loops")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = train_one(args, pick_device(args.device))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
