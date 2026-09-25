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


def make_carry_pairs(
    *,
    examples: int,
    width: int,
    target_operand_position: int,
    carry_source_position: int,
    generator: torch.Generator,
) -> tuple[PaperBatch, PaperBatch]:
    if not 0 <= target_operand_position < carry_source_position < width:
        raise ValueError("carry source must lie strictly right of target position")
    clean_first = torch.zeros((examples, width), dtype=torch.long)
    clean_second = torch.zeros((examples, width), dtype=torch.long)
    if target_operand_position:
        clean_first[:, :target_operand_position] = torch.randint(
            0, 2, (examples, target_operand_position), generator=generator
        )
        clean_second[:, :target_operand_position] = torch.randint(
            0, 2, (examples, target_operand_position), generator=generator
        )
    # The target column and every column through the source propagate carry.
    orientation = torch.randint(
        0,
        2,
        (examples, carry_source_position - target_operand_position),
        generator=generator,
    )
    clean_first[:, target_operand_position:carry_source_position] = orientation
    clean_second[:, target_operand_position:carry_source_position] = 1 - orientation
    # Clean generates a carry at the source. Corrupt kills it. Less-significant
    # columns are zero so no alternate carry can enter the chain.
    clean_first[:, carry_source_position] = 1
    clean_second[:, carry_source_position] = 1
    corrupt_first = clean_first.clone()
    corrupt_second = clean_second.clone()
    corrupt_first[:, carry_source_position] = 0
    corrupt_second[:, carry_source_position] = 0
    clean = make_batch(clean_first, clean_second)
    corrupt = make_batch(corrupt_first, corrupt_second)
    answer_start = 2 * width + 1
    target_answer_position = answer_start + target_operand_position + 1
    clean_target = clean.targets[:, target_answer_position]
    corrupt_target = corrupt.targets[:, target_answer_position]
    if not bool(clean_target.eq(1 - corrupt_target).all()):
        raise RuntimeError("constructed pairs do not flip the target sum digit")
    return clean, corrupt


@torch.inference_mode()
def trajectory(model: torch.nn.Module, batch: PaperBatch, steps: int) -> list[torch.Tensor]:
    embedded = model.input_embeddings(batch.inputs)
    state = torch.zeros_like(embedded)
    states: list[torch.Tensor] = []
    for _ in range(steps):
        state = model.recurrent_step(state, embedded)
        states.append(state.clone())
    return states


@torch.inference_mode()
def finish_from_state(
    model: torch.nn.Module,
    state: torch.Tensor,
    batch: PaperBatch,
    *,
    completed_steps: int,
    target_steps: int,
) -> torch.Tensor:
    embedded = model.input_embeddings(batch.inputs)
    live = state
    for _ in range(completed_steps, target_steps):
        live = model.recurrent_step(live, embedded)
    return model.decode(live).float()


def target_metrics(
    logits: torch.Tensor,
    *,
    target_position: int,
    clean_target: torch.Tensor,
    corrupt_target: torch.Tensor,
) -> dict[str, float]:
    target_logits = logits[:, target_position]
    clean_logits = target_logits.gather(-1, clean_target.unsqueeze(-1)).squeeze(-1)
    corrupt_logits = target_logits.gather(
        -1, corrupt_target.unsqueeze(-1)
    ).squeeze(-1)
    return {
        "mean_clean_minus_corrupt_logit": float((clean_logits - corrupt_logits).mean()),
        "clean_target_accuracy": float(
            target_logits.argmax(dim=-1).eq(clean_target).float().mean()
        ),
        "corrupt_target_accuracy": float(
            target_logits.argmax(dim=-1).eq(corrupt_target).float().mean()
        ),
    }


def position_groups(
    *,
    width: int,
    target_operand_position: int,
    carry_source_position: int,
) -> dict[str, list[int]]:
    answer_start = 2 * width + 1
    target_answer = answer_start + target_operand_position + 1
    chain = list(range(target_operand_position, carry_source_position + 1))
    operand_chain = chain + [width + 1 + position for position in chain]
    return {
        "target_answer_slot": [target_answer],
        "neighbor_answer_slot_control": [target_answer + 1],
        "all_actual_answer_slots": list(range(answer_start, answer_start + width + 1)),
        "operand_chain_slots": operand_chain,
        "full_sequence": list(range(PAPER_TASKS["addition"].sequence_length(width))),
    }


@torch.inference_mode()
def run_condition(
    *,
    model: torch.nn.Module,
    clean: PaperBatch,
    corrupt: PaperBatch,
    width: int,
    target_operand_position: int,
    carry_source_position: int,
    target_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_states = trajectory(model, clean, target_steps)
    corrupt_states = trajectory(model, corrupt, target_steps)
    answer_start = 2 * width + 1
    target_position = answer_start + target_operand_position + 1
    clean_target = clean.targets[:, target_position]
    corrupt_target = corrupt.targets[:, target_position]
    clean_logits = model.decode(clean_states[-1]).float()
    corrupt_logits = model.decode(corrupt_states[-1]).float()
    clean_metrics = target_metrics(
        clean_logits,
        target_position=target_position,
        clean_target=clean_target,
        corrupt_target=corrupt_target,
    )
    corrupt_metrics = target_metrics(
        corrupt_logits,
        target_position=target_position,
        clean_target=clean_target,
        corrupt_target=corrupt_target,
    )
    denominator = (
        clean_metrics["mean_clean_minus_corrupt_logit"]
        - corrupt_metrics["mean_clean_minus_corrupt_logit"]
    )
    groups = position_groups(
        width=width,
        target_operand_position=target_operand_position,
        carry_source_position=carry_source_position,
    )
    shuffled_indices = torch.roll(
        torch.arange(clean.inputs.shape[0], device=clean.inputs.device), shifts=1
    )
    rows: list[dict[str, Any]] = []
    for completed_steps in range(1, target_steps + 1):
        for group_name, positions in groups.items():
            for donor_mode in ("paired", "shuffled"):
                patched = corrupt_states[completed_steps - 1].clone()
                donor = clean_states[completed_steps - 1]
                if donor_mode == "shuffled":
                    donor = donor[shuffled_indices]
                patched[:, positions] = donor[:, positions]
                logits = finish_from_state(
                    model,
                    patched,
                    corrupt,
                    completed_steps=completed_steps,
                    target_steps=target_steps,
                )
                metrics = target_metrics(
                    logits,
                    target_position=target_position,
                    clean_target=clean_target,
                    corrupt_target=corrupt_target,
                )
                recovery = (
                    metrics["mean_clean_minus_corrupt_logit"]
                    - corrupt_metrics["mean_clean_minus_corrupt_logit"]
                ) / denominator
                rows.append(
                    {
                        "carry_span": carry_source_position - target_operand_position,
                        "target_operand_position": target_operand_position,
                        "carry_source_position": carry_source_position,
                        "completed_steps": completed_steps,
                        "remaining_steps": target_steps - completed_steps,
                        "position_group": group_name,
                        "position_count": len(positions),
                        "donor_mode": donor_mode,
                        **metrics,
                        "normalized_logit_recovery": float(recovery),
                    }
                )
    baselines = {
        "carry_span": carry_source_position - target_operand_position,
        "target_operand_position": target_operand_position,
        "carry_source_position": carry_source_position,
        "target_answer_position": target_position,
        "clean": clean_metrics,
        "corrupt": corrupt_metrics,
        "logit_recovery_denominator": denominator,
    }
    return rows, baselines


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("carry patching requires Addition")
    target_steps = args.width + int(spec.step_offset)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    all_rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for carry_span in args.carry_spans:
        source_position = args.target_operand_position + carry_span
        clean, corrupt = make_carry_pairs(
            examples=args.examples,
            width=args.width,
            target_operand_position=args.target_operand_position,
            carry_source_position=source_position,
            generator=generator,
        )
        clean = clean.to(device)
        corrupt = corrupt.to(device)
        rows, condition_baselines = run_condition(
            model=model,
            clean=clean,
            corrupt=corrupt,
            width=args.width,
            target_operand_position=args.target_operand_position,
            carry_source_position=source_position,
            target_steps=target_steps,
        )
        all_rows.extend(rows)
        baselines.append(condition_baselines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "patching.csv", all_rows)
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "loss_placement": "fixed n=10 final-only CE at loop 11",
        "examples_per_condition": args.examples,
        "width": args.width,
        "target_steps": target_steps,
        "carry_spans": list(args.carry_spans),
        "target_operand_position": args.target_operand_position,
        "patch_direction": "clean carry -> corrupt no-carry",
        "replacement": "paired clean activation or one-row-shuffled clean control",
        "hook_site": "residual state after a complete shared-block recurrence",
        "baselines": baselines,
        "claim_boundary": (
            "Position-level activation patching tests where carry information is "
            "causally available under this intervention. It does not identify a "
            "minimal head/MLP circuit."
        ),
    }
    atomic_json_dump(payload, args.out_dir / "summary.json")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--target-operand-position", type=int, default=4)
    parser.add_argument("--carry-spans", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=273001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
