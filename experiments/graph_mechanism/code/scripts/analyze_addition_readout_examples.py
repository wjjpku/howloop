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
)


EXAMPLES = (
    (1, "1", "1"),
    (2, "01", "01"),
    (3, "011", "001"),
    (4, "1111", "0001"),
    (5, "10101", "00111"),
    (6, "011111", "000001"),
    (7, "1010101", "0101011"),
    (8, "01111111", "00000001"),
    (9, "101101011", "010010101"),
)

TOKEN_SYMBOLS = {
    0: "0",
    1: "1",
    2: "|",  # operand separator
    3: "·",  # PAD/EOS
    4: "x",  # ignored target token
    5: "?",  # answer placeholder/query
}


def render_tokens(token_ids: torch.Tensor) -> str:
    return "".join(TOKEN_SYMBOLS[int(value)] for value in token_ids.flatten())


def addition_structure(first: str, second: str) -> dict[str, Any]:
    if len(first) != len(second) or set(first + second) - {"0", "1"}:
        raise ValueError("operands must be equal-width binary strings")
    width = len(first)
    target = f"{int(first, 2) + int(second, 2):0{width + 1}b}"
    generate = [0] * width
    propagate = [0] * width
    carry_in = [0] * width
    carry_out = [0] * width
    carry = 0
    for position in range(width - 1, -1, -1):
        left = int(first[position])
        right = int(second[position])
        generate[position] = left & right
        propagate[position] = left ^ right
        carry_in[position] = carry
        carry = generate[position] | (propagate[position] & carry)
        carry_out[position] = carry
    return {
        "first": first,
        "second": second,
        "target": target,
        "generate_msb_to_lsb": "".join(map(str, generate)),
        "propagate_msb_to_lsb": "".join(map(str, propagate)),
        "carry_in_msb_to_lsb": "".join(map(str, carry_in)),
        "carry_out_msb_to_lsb": "".join(map(str, carry_out)),
    }


def make_addition_batch(first: str, second: str) -> PaperBatch:
    spec = PAPER_TASKS["addition"]
    structure = addition_structure(first, second)
    width = len(first)
    sequence_length = spec.sequence_length(width)
    inputs = torch.full((1, sequence_length), 3, dtype=torch.long)
    targets = torch.full((1, sequence_length), 3, dtype=torch.long)
    answer_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
    first_tokens = torch.tensor([int(value) for value in first])
    second_tokens = torch.tensor([int(value) for value in second])
    answer_tokens = torch.tensor([int(value) for value in structure["target"]])
    inputs[0, :width] = first_tokens
    inputs[0, width] = 2
    inputs[0, width + 1 : 2 * width + 1] = second_tokens
    answer_start = 2 * width + 1
    answer_end = answer_start + width + 1
    inputs[0, answer_start] = 5
    targets[0, :answer_start] = 4
    targets[0, answer_start:answer_end] = answer_tokens
    answer_mask[0, answer_start:] = True
    return PaperBatch(
        inputs=F.one_hot(inputs, num_classes=spec.vocab_size).float(),
        targets=targets,
        answer_mask=answer_mask,
        lengths=torch.tensor([width]),
        target_steps=torch.tensor([width + 1]),
    )


@torch.inference_mode()
def readout_metrics(
    logits: torch.Tensor,
    batch: PaperBatch,
    *,
    logical_length: int,
) -> dict[str, Any]:
    answer_start = 2 * logical_length + 1
    answer_end = answer_start + logical_length + 1
    predictions = logits.argmax(dim=-1)
    actual_predictions = predictions[0, answer_start:answer_end]
    actual_targets = batch.targets[0, answer_start:answer_end]
    actual_correct = actual_predictions.eq(actual_targets)
    low_order_frontier = 0
    for correct in reversed(actual_correct.tolist()):
        if not correct:
            break
        low_order_frontier += 1
    correct_logits = logits[0, answer_start:answer_end].gather(
        -1, actual_targets.unsqueeze(-1)
    ).squeeze(-1)
    competing = logits[0, answer_start:answer_end].clone()
    competing.scatter_(-1, actual_targets.unsqueeze(-1), -torch.inf)
    margins = correct_logits - competing.max(dim=-1).values
    padding_predictions = predictions[0, answer_end:]
    padding_targets = batch.targets[0, answer_end:]
    return {
        "prediction": render_tokens(actual_predictions),
        "target": render_tokens(actual_targets),
        "actual_answer_exact_match": bool(actual_correct.all()),
        "actual_answer_token_accuracy": float(actual_correct.float().mean()),
        "final_carry_correct": bool(actual_correct[0]),
        "low_order_frontier": low_order_frontier,
        "minimum_actual_margin": float(margins.min()),
        "mean_actual_margin": float(margins.mean()),
        "padding_prediction": render_tokens(padding_predictions),
        "padding_exact_match": bool(padding_predictions.eq(padding_targets).all()),
    }


@torch.inference_mode()
def analyze_example(
    *,
    model: torch.nn.Module,
    controller: torch.nn.Module,
    first: str,
    second: str,
    device: torch.device,
    anchor_step: int,
    post_final_controller: bool,
    overloops: int,
) -> dict[str, Any]:
    structure = addition_structure(first, second)
    width = len(first)
    target_step = width + 1
    maximum_step = target_step + overloops
    batch = make_addition_batch(first, second).to(device)
    embedded = model.input_embeddings(batch.inputs)
    raw_state = torch.zeros_like(embedded)
    controlled_state = torch.zeros_like(embedded)
    trajectory: list[dict[str, Any]] = []
    for step in range(1, maximum_step + 1):
        raw_state = model.recurrent_step(raw_state, embedded)
        if step > anchor_step:
            controlled_state = controller(controlled_state)
        controlled_state = model.recurrent_step(controlled_state, embedded)
        raw_logits = model.decode(raw_state).float()
        controlled_readout_state = (
            controller(controlled_state)
            if post_final_controller
            else controlled_state
        )
        controlled_logits = model.decode(controlled_readout_state).float()
        trajectory.append(
            {
                "step": step,
                "step_offset": step - target_step,
                "raw": readout_metrics(
                    raw_logits, batch, logical_length=width
                ),
                "J": readout_metrics(
                    controlled_logits, batch, logical_length=width
                ),
            }
        )
    endpoint = trajectory[target_step - 1]
    return {
        "length": width,
        "target_step": target_step,
        **structure,
        "input_layout": f"{first}|{second}?",
        "endpoint_raw": endpoint["raw"],
        "endpoint_J": endpoint["J"],
        "trajectory": trajectory,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fixed-n10 Addition: readout examples for n=1..9",
        "",
        "Actual sum digits are shown separately from the four supervised PAD/EOS positions.",
        "The trained J path uses anchor=1 and the saved post-final-J setting.",
        "",
        "| n | input | target | raw @ T(n) | J @ T(n) | raw digit acc | J digit acc | raw/J low-order frontier |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for example in payload["examples"]:
        raw = example["endpoint_raw"]
        controlled = example["endpoint_J"]
        lines.append(
            f"| {example['length']} | `{example['input_layout']}` | "
            f"`{example['target']}` | `{raw['prediction']}` | "
            f"`{controlled['prediction']}` | "
            f"{raw['actual_answer_token_accuracy']:.3f} | "
            f"{controlled['actual_answer_token_accuracy']:.3f} | "
            f"{raw['low_order_frontier']}/{controlled['low_order_frontier']} |"
        )
    for example in payload["examples"]:
        lines.extend(
            [
                "",
                f"## n={example['length']}: {example['first']} + {example['second']} = {example['target']}",
                "",
                f"- generate: `{example['generate_msb_to_lsb']}`",
                f"- propagate: `{example['propagate_msb_to_lsb']}`",
                f"- carry-in: `{example['carry_in_msb_to_lsb']}`",
                f"- carry-out: `{example['carry_out_msb_to_lsb']}`",
                "",
                "| loop | offset from T(n) | raw | J | raw frontier | J frontier |",
                "|---:|---:|---|---|---:|---:|",
            ]
        )
        for row in example["trajectory"]:
            lines.append(
                f"| {row['step']} | {row['step_offset']:+d} | "
                f"`{row['raw']['prediction']}` | `{row['J']['prediction']}` | "
                f"{row['raw']['low_order_frontier']} | "
                f"{row['J']['low_order_frontier']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("readout examples require an Addition checkpoint")
    controller, controller_payload = load_controller(args.controller, device=device)
    anchor_step = int(controller_payload["anchor_step"])
    post_final_controller = bool(
        controller_payload.get("controller_post_final_j", False)
    )
    examples = [
        analyze_example(
            model=model,
            controller=controller,
            first=first,
            second=second,
            device=device,
            anchor_step=anchor_step,
            post_final_controller=post_final_controller,
            overloops=args.overloops,
        )
        for _, first, second in EXAMPLES
    ]
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller),
        "controller_parameterization": controller_payload.get(
            "controller_parameterization", "diagonal_low_rank"
        ),
        "anchor_step": anchor_step,
        "post_final_controller": post_final_controller,
        "loss_placement": "saved controller was trained with final full-answer-region CE only",
        "target_loop_rule": "T(n)=n+1",
        "examples": examples,
        "claim_boundary": (
            "Readout trajectories are behavioral/localization evidence. They do not "
            "by themselves establish a causal Addition circuit."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(payload, args.out_dir / "readout_examples.json")
    write_markdown(payload, args.out_dir / "READOUT_EXAMPLES.md")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overloops", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
