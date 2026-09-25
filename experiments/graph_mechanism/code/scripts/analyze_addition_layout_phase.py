#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    PAPER_TASKS,
    PaperBatch,
    atomic_json_dump,
    load_backbone,
    load_controller,
    pick_device,
    write_csv,
)


def binary_addition_batch(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    batch_size, width = first.shape
    result = torch.zeros((batch_size, width + 1), dtype=torch.long)
    carry = torch.zeros(batch_size, dtype=torch.long)
    for position in range(width - 1, -1, -1):
        total = first[:, position] + second[:, position] + carry
        result[:, position + 1] = total.remainder(2)
        carry = total.div(2, rounding_mode="floor")
    result[:, 0] = carry
    return result


def make_batch(first: torch.Tensor, second: torch.Tensor) -> PaperBatch:
    spec = PAPER_TASKS["addition"]
    batch_size, width = first.shape
    sequence_length = spec.sequence_length(width)
    token_ids = torch.full((batch_size, sequence_length), 3, dtype=torch.long)
    targets = torch.full((batch_size, sequence_length), 3, dtype=torch.long)
    answer_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
    token_ids[:, :width] = first
    token_ids[:, width] = 2
    token_ids[:, width + 1 : 2 * width + 1] = second
    answer_start = 2 * width + 1
    answer_end = answer_start + width + 1
    token_ids[:, answer_start] = 5
    targets[:, :answer_start] = 4
    targets[:, answer_start:answer_end] = binary_addition_batch(first, second)
    answer_mask[:, answer_start:] = True
    return PaperBatch(
        inputs=F.one_hot(token_ids, num_classes=spec.vocab_size).float(),
        targets=targets,
        answer_mask=answer_mask,
        lengths=torch.full((batch_size,), width, dtype=torch.long),
        target_steps=torch.full((batch_size,), width + 1, dtype=torch.long),
    )


def metrics(
    logits: torch.Tensor,
    batch: PaperBatch,
    *,
    layout_width: int,
    semantic_width: int,
) -> dict[str, float]:
    answer_start = 2 * layout_width + 1
    answer_end = answer_start + layout_width + 1
    predictions = logits.argmax(dim=-1)[:, answer_start:answer_end]
    targets = batch.targets[:, answer_start:answer_end]
    correct = predictions.eq(targets)
    semantic_correct = correct[:, -(semantic_width + 1) :]
    return {
        "full_layout_exact_match": float(correct.all(dim=1).float().mean()),
        "full_layout_token_accuracy": float(correct.float().mean()),
        "semantic_suffix_exact_match": float(
            semantic_correct.all(dim=1).float().mean()
        ),
        "semantic_suffix_token_accuracy": float(semantic_correct.float().mean()),
    }


@torch.inference_mode()
def evaluate_layout(
    *,
    model: torch.nn.Module,
    controller: torch.nn.Module,
    anchor_step: int,
    post_final_controller: bool,
    batch: PaperBatch,
    semantic_width: int,
    layout: str,
    selected_steps: tuple[int, ...],
    device: torch.device,
) -> list[dict[str, Any]]:
    live = batch.to(device)
    layout_width = int(batch.lengths[0])
    maximum_step = max(selected_steps)
    embedded = model.input_embeddings(live.inputs)
    raw_state = torch.zeros_like(embedded)
    controlled_state = torch.zeros_like(embedded)
    rows: list[dict[str, Any]] = []
    for step in range(1, maximum_step + 1):
        raw_state = model.recurrent_step(raw_state, embedded)
        if step > anchor_step:
            controlled_state = controller(controlled_state)
        controlled_state = model.recurrent_step(controlled_state, embedded)
        if step not in selected_steps:
            continue
        logits_by_variant = {
            "raw": model.decode(raw_state).float(),
            "J": model.decode(
                controller(controlled_state)
                if post_final_controller
                else controlled_state
            ).float(),
        }
        for variant, logits in logits_by_variant.items():
            rows.append(
                {
                    "semantic_length": semantic_width,
                    "layout": layout,
                    "layout_width": layout_width,
                    "step": step,
                    "semantic_target_step": semantic_width + 1,
                    "fixed_n10_step": 11,
                    "variant": variant,
                    **metrics(
                        logits,
                        live,
                        layout_width=layout_width,
                        semantic_width=semantic_width,
                    ),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("layout-phase analysis requires Addition")
    controller, controller_payload = load_controller(args.controller, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    for width in args.lengths:
        first = torch.randint(0, 2, (args.examples, width), generator=generator)
        second = torch.randint(0, 2, (args.examples, width), generator=generator)
        native = make_batch(first, second)
        padding = torch.zeros((args.examples, 10 - width), dtype=torch.long)
        padded = make_batch(
            torch.cat((padding, first), dim=1),
            torch.cat((padding, second), dim=1),
        )
        selected_steps = tuple(sorted({width + 1, 11}))
        rows.extend(
            evaluate_layout(
                model=model,
                controller=controller,
                anchor_step=int(controller_payload["anchor_step"]),
                post_final_controller=bool(
                    controller_payload.get("controller_post_final_j", False)
                ),
                batch=native,
                semantic_width=width,
                layout="native",
                selected_steps=selected_steps,
                device=device,
            )
        )
        rows.extend(
            evaluate_layout(
                model=model,
                controller=controller,
                anchor_step=int(controller_payload["anchor_step"]),
                post_final_controller=bool(
                    controller_payload.get("controller_post_final_j", False)
                ),
                batch=padded,
                semantic_width=width,
                layout="left_zero_padded_to_n10",
                selected_steps=selected_steps,
                device=device,
            )
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "layout_phase.csv", rows)
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller),
        "lengths": list(args.lengths),
        "examples_per_length": args.examples,
        "seed": args.seed,
        "comparison": (
            "same arithmetic values in native n layout versus left-zero-padded "
            "n=10 layout, each read at T(n)=n+1 and at the trained loop 11"
        ),
        "rows": rows,
        "claim_boundary": (
            "This separates behavioral layout/phase dependence; it does not localize "
            "the internal component implementing that dependence."
        ),
    }
    atomic_json_dump(payload, args.out_dir / "summary.json")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=tuple(range(1, 10)))
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=272001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
