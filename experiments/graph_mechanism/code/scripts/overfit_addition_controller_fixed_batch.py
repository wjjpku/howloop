#!/usr/bin/env python3
"""Positive-control overfit test for an Addition recurrent interface controller."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.paper_length_telomere import (
    DenseAffineController,
    DiagonalLowRankController,
    _controlled_ce_batch,
    generate_paper_batch,
    load_backbone,
    pick_device,
    set_seed,
)
from scripts.compare_addition_readout_order_attention import (
    answer_positions_lsb_to_carry,
)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def metrics(model, controller, batch, spec, anchor_step: int) -> dict[str, float]:
    target_step = int(batch.target_steps[0])
    state = model.states(
        batch.inputs,
        steps=target_step,
        controller=controller,
        controller_start_step=anchor_step,
    )[-1]
    logits = model.decode(state).float()
    predictions = logits.argmax(dim=-1)
    answer_correct = predictions.eq(batch.targets) | ~batch.answer_mask
    positions = torch.tensor(
        answer_positions_lsb_to_carry(spec, int(batch.lengths[0])),
        device=batch.inputs.device,
    )
    arithmetic_correct = predictions.index_select(1, positions).eq(
        batch.targets.index_select(1, positions)
    )
    loss = torch.nn.functional.cross_entropy(
        logits[batch.answer_mask], batch.targets[batch.answer_mask]
    )
    return {
        "answer_ce": float(loss),
        "answer_region_em": float(answer_correct.all(dim=1).float().mean()),
        "full_arithmetic_em": float(arithmetic_correct.all(dim=1).float().mean()),
        "full_arithmetic_bit_accuracy": float(arithmetic_correct.float().mean()),
    }


def model_fingerprint(model) -> tuple[float, float]:
    total = 0.0
    square = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            values = parameter.detach().float()
            total += float(values.sum())
            square += float(values.square().sum())
    return total, square


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    set_seed(args.seed)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if args.parameterization == "diagonal_low_rank":
        controller = DiagonalLowRankController(model.config.d_model, args.rank).to(device)
        parameter_groups: Any = [
            {"params": [controller.A, controller.B, controller.bias]},
            {"params": [controller.diagonal], "lr": args.learning_rate * 0.1},
        ]
    else:
        controller = DenseAffineController(model.config.d_model).to(device)
        parameter_groups = controller.parameters()
    controller.train()
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.data_seed)
    batch = generate_paper_batch(
        spec,
        batch_size=args.batch_size,
        min_length=args.logical_length,
        max_length=args.logical_length,
        fixed_length=args.logical_length,
        generator=generator,
    ).to(device)
    target_step = int(batch.target_steps[0])
    controlled_steps = target_step - args.anchor_step
    if controlled_steps < 1:
        raise ValueError("the selected anchor never applies the controller")

    before_fingerprint = model_fingerprint(model)
    rows = [{"update": 0, **metrics(model, controller, batch, spec, args.anchor_step)}]
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _controlled_ce_batch(
            model=model,
            controller=controller,
            batch=batch,
            anchor_step=args.anchor_step,
            controlled_steps=controlled_steps,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(), args.grad_clip
        )
        optimizer.step()
        if update == 1 or update % args.eval_every == 0 or update == args.updates:
            row = {
                "update": update,
                "preclip_gradient_norm": float(gradient_norm),
                **metrics(model, controller, batch, spec, args.anchor_step),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row["answer_region_em"] == 1.0 and row["full_arithmetic_em"] == 1.0:
                break

    after_fingerprint = model_fingerprint(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "training.csv", rows)
    torch.save(
        {
            "controller_state": controller.state_dict(),
            "parameterization": args.parameterization,
            "rank": args.rank if args.parameterization == "diagonal_low_rank" else None,
            "logical_length": args.logical_length,
            "anchor_step": args.anchor_step,
        },
        args.out_dir / "controller.pt",
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "parameterization": args.parameterization,
        "rank": args.rank if args.parameterization == "diagonal_low_rank" else None,
        "logical_length": args.logical_length,
        "target_step": target_step,
        "anchor_step": args.anchor_step,
        "controlled_steps": controlled_steps,
        "batch_size": args.batch_size,
        "maximum_updates": args.updates,
        "actual_updates": int(rows[-1]["update"]),
        "learning_rate": args.learning_rate,
        "initial_metrics": rows[0],
        "final_metrics": rows[-1],
        "backbone_fingerprint_before": before_fingerprint,
        "backbone_fingerprint_after": after_fingerprint,
        "backbone_unchanged": before_fingerprint == after_fingerprint,
        "interpretation": (
            "A successful fixed-batch overfit establishes gradient, placement, "
            "serialization-independent capacity on the sampled states; it does not "
            "establish distribution-level length generalization."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--parameterization",
        choices=("diagonal_low_rank", "dense_affine"),
        default="diagonal_low_rank",
    )
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--logical-length", type=int, default=12)
    parser.add_argument("--anchor-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=522101)
    parser.add_argument("--data-seed", type=int, default=995101)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
