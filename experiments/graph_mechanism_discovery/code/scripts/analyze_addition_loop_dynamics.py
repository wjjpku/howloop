#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.paper_length_telomere import (
    PaperBatch,
    atomic_json_dump,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
    write_csv,
)


def maximum_generated_carry_span(batch: PaperBatch, logical_length: int) -> torch.Tensor:
    token_ids = batch.inputs.argmax(dim=-1).detach().cpu()
    first = token_ids[:, :logical_length]
    second = token_ids[:, logical_length + 1 : 2 * logical_length + 1]
    generate = first & second
    propagate = first ^ second
    spans = torch.zeros(first.shape[0], dtype=torch.long, device=first.device)
    for row in range(first.shape[0]):
        maximum = 0
        for position in range(logical_length):
            if not bool(generate[row, position]):
                continue
            span = 1
            cursor = position - 1
            while cursor >= 0 and bool(propagate[row, cursor]):
                span += 1
                cursor -= 1
            maximum = max(maximum, span)
        spans[row] = maximum
    return spans


def carry_bucket(span: int) -> str:
    return str(span) if span <= 3 else "4+"


def empty_totals(position_count: int) -> dict[str, Any]:
    return {
        "examples": 0,
        "actual_exact": 0,
        "actual_correct_tokens": 0,
        "actual_tokens": 0,
        "carry_correct": 0,
        "padding_exact": 0,
        "low_order_frontier": 0,
        "position_correct_lsb_to_carry": [0] * position_count,
    }


def update_totals(
    totals: dict[str, Any],
    logits: torch.Tensor,
    batch: PaperBatch,
    *,
    logical_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    answer_start = 2 * logical_length + 1
    answer_end = answer_start + logical_length + 1
    predictions = logits.argmax(dim=-1)
    actual_correct = predictions[:, answer_start:answer_end].eq(
        batch.targets[:, answer_start:answer_end]
    )
    actual_exact = actual_correct.all(dim=1)
    padding_exact = predictions[:, answer_end:].eq(
        batch.targets[:, answer_end:]
    ).all(dim=1)
    reversed_correct = actual_correct.flip(dims=(1,))
    frontier = torch.cumprod(reversed_correct.to(torch.int64), dim=1).sum(dim=1)
    totals["examples"] += int(actual_correct.shape[0])
    totals["actual_exact"] += int(actual_exact.sum())
    totals["actual_correct_tokens"] += int(actual_correct.sum())
    totals["actual_tokens"] += int(actual_correct.numel())
    totals["carry_correct"] += int(actual_correct[:, 0].sum())
    totals["padding_exact"] += int(padding_exact.sum())
    totals["low_order_frontier"] += int(frontier.sum())
    position_correct = reversed_correct.sum(dim=0).tolist()
    totals["position_correct_lsb_to_carry"] = [
        left + int(right)
        for left, right in zip(
            totals["position_correct_lsb_to_carry"],
            position_correct,
            strict=True,
        )
    ]
    return actual_correct, actual_exact


def row_from_totals(
    *,
    logical_length: int,
    target_step: int,
    step: int,
    variant: str,
    totals: dict[str, Any],
) -> dict[str, Any]:
    examples = totals["examples"]
    return {
        "length": logical_length,
        "target_step": target_step,
        "step": step,
        "step_offset": step - target_step,
        "variant": variant,
        "examples": examples,
        "actual_answer_exact_match": totals["actual_exact"] / examples,
        "actual_answer_token_accuracy": (
            totals["actual_correct_tokens"] / totals["actual_tokens"]
        ),
        "final_carry_accuracy": totals["carry_correct"] / examples,
        "padding_exact_match": totals["padding_exact"] / examples,
        "mean_correct_low_order_digits": totals["low_order_frontier"] / examples,
        "mean_correct_low_order_fraction": (
            totals["low_order_frontier"] / (examples * (logical_length + 1))
        ),
        "position_accuracy_lsb_to_carry": json.dumps(
            [value / examples for value in totals["position_correct_lsb_to_carry"]]
        ),
    }


@torch.inference_mode()
def evaluate_length(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module,
    controller_anchor: int,
    post_final_controller: bool,
    logical_length: int,
    examples: int,
    batch_size: int,
    overloops: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_step = logical_length + int(spec.step_offset)
    maximum_step = target_step + overloops
    totals = {
        (variant, step): empty_totals(logical_length + 1)
        for variant in ("raw", "J")
        for step in range(1, maximum_step + 1)
    }
    chain_totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "examples": 0,
            "actual_exact": 0,
            "actual_correct_tokens": 0,
            "actual_tokens": 0,
        }
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009 * logical_length)
    remaining = examples
    while remaining:
        current_batch_size = min(batch_size, remaining)
        batch = generate_paper_batch(
            spec,
            batch_size=current_batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=generator,
        ).to(device)
        spans = maximum_generated_carry_span(batch, logical_length)
        embedded = model.input_embeddings(batch.inputs)
        raw_state = torch.zeros_like(embedded)
        controlled_state = torch.zeros_like(embedded)
        for step in range(1, maximum_step + 1):
            raw_state = model.recurrent_step(raw_state, embedded)
            if step > controller_anchor:
                controlled_state = controller(controlled_state)
            controlled_state = model.recurrent_step(controlled_state, embedded)
            logits_by_variant = {
                "raw": model.decode(raw_state).float(),
                "J": model.decode(
                    controller(controlled_state)
                    if post_final_controller
                    else controlled_state
                ).float(),
            }
            for variant, logits in logits_by_variant.items():
                actual_correct, actual_exact = update_totals(
                    totals[(variant, step)],
                    logits,
                    batch,
                    logical_length=logical_length,
                )
                if step != target_step:
                    continue
                for span in spans.unique().tolist():
                    selected_cpu = spans.eq(span)
                    selected = selected_cpu.to(actual_correct.device)
                    group = chain_totals[(variant, carry_bucket(int(span)))]
                    group["examples"] += int(selected_cpu.sum())
                    group["actual_exact"] += int(actual_exact[selected].sum())
                    group["actual_correct_tokens"] += int(
                        actual_correct[selected].sum()
                    )
                    group["actual_tokens"] += int(actual_correct[selected].numel())
        remaining -= current_batch_size
    rows = [
        row_from_totals(
            logical_length=logical_length,
            target_step=target_step,
            step=step,
            variant=variant,
            totals=totals[(variant, step)],
        )
        for variant in ("raw", "J")
        for step in range(1, maximum_step + 1)
    ]
    chain_rows = [
        {
            "length": logical_length,
            "target_step": target_step,
            "variant": variant,
            "maximum_generated_carry_span": bucket,
            "examples": group["examples"],
            "actual_answer_exact_match": group["actual_exact"] / group["examples"],
            "actual_answer_token_accuracy": (
                group["actual_correct_tokens"] / group["actual_tokens"]
            ),
        }
        for (variant, bucket), group in sorted(chain_totals.items())
        if group["examples"]
    ]
    return rows, chain_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["length"]), str(row["variant"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (length, variant), group in sorted(groups.items()):
        target = next(row for row in group if int(row["step_offset"]) == 0)
        best = max(
            group,
            key=lambda row: (
                float(row["actual_answer_exact_match"]),
                float(row["actual_answer_token_accuracy"]),
                -abs(int(row["step_offset"])),
            ),
        )
        summary.append(
            {
                "length": length,
                "variant": variant,
                "target_step": int(target["target_step"]),
                "target_actual_answer_exact_match": float(
                    target["actual_answer_exact_match"]
                ),
                "target_actual_answer_token_accuracy": float(
                    target["actual_answer_token_accuracy"]
                ),
                "target_final_carry_accuracy": float(target["final_carry_accuracy"]),
                "target_mean_correct_low_order_digits": float(
                    target["mean_correct_low_order_digits"]
                ),
                "best_step": int(best["step"]),
                "best_step_offset": int(best["step_offset"]),
                "best_actual_answer_exact_match": float(
                    best["actual_answer_exact_match"]
                ),
                "best_actual_answer_token_accuracy": float(
                    best["actual_answer_token_accuracy"]
                ),
            }
        )
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("loop dynamics analysis requires Addition")
    controller, controller_payload = load_controller(args.controller, device=device)
    all_rows: list[dict[str, Any]] = []
    all_chain_rows: list[dict[str, Any]] = []
    for logical_length in args.lengths:
        rows, chain_rows = evaluate_length(
            model=model,
            spec=spec,
            controller=controller,
            controller_anchor=int(controller_payload["anchor_step"]),
            post_final_controller=bool(
                controller_payload.get("controller_post_final_j", False)
            ),
            logical_length=logical_length,
            examples=args.examples,
            batch_size=args.batch_size,
            overloops=args.overloops,
            seed=args.seed,
            device=device,
        )
        all_rows.extend(rows)
        all_chain_rows.extend(chain_rows)
    summary_rows = summarize(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "trajectory.csv", all_rows)
    write_csv(args.out_dir / "carry_chain_groups.csv", all_chain_rows)
    write_csv(args.out_dir / "endpoint_summary.csv", summary_rows)
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller),
        "controller_anchor_step": int(controller_payload["anchor_step"]),
        "controller_post_final_j": bool(
            controller_payload.get("controller_post_final_j", False)
        ),
        "loss_placement": "backbone and controller both use final-only supervision",
        "target_loop_rule": "T(n)=n+1",
        "lengths": list(args.lengths),
        "examples_per_length": args.examples,
        "seed": args.seed,
        "endpoint_summary": summary_rows,
        "claim_boundary": (
            "Aggregate readout dynamics localize phase and carry behavior but are not "
            "a causal circuit without activation/component interventions."
        ),
    }
    atomic_json_dump(payload, args.out_dir / "summary.json")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=tuple(range(1, 13)))
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overloops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=271001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
