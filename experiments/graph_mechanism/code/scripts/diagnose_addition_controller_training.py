#!/usr/bin/env python3
"""Trace frozen-backbone Addition controller training and evaluation consistency."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.paper_length_telomere import (
    _controlled_ce_batch,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    masked_cross_entropy,
    pick_device,
)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixed_batch(spec: Any, length: int, examples: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + length)
    return generate_paper_batch(
        spec,
        batch_size=examples,
        min_length=length,
        max_length=length,
        fixed_length=length,
        generator=generator,
    ).to(device)


def forward_training_path(model, controller, batch, anchor_step: int):
    target_step = int(batch.target_steps[0])
    controlled_steps = target_step - anchor_step
    state = torch.zeros(
        batch.inputs.shape[0],
        batch.inputs.shape[1],
        model.config.d_model,
        device=batch.inputs.device,
    )
    for step_index in range(1, anchor_step + 1):
        embedded = model.input_embeddings(batch.inputs, step_index=step_index)
        state = model.recurrent_step(state, embedded)
    for step_index in range(anchor_step + 1, target_step + 1):
        embedded = model.input_embeddings(batch.inputs, step_index=step_index)
        state = controller(state)
        state = model.recurrent_step(state, embedded)
    return state, controlled_steps


@torch.inference_mode()
def snapshot_metrics(model, controller, batch, anchor_step: int) -> dict[str, float]:
    target_step = int(batch.target_steps[0])
    state = model.states(
        batch.inputs,
        steps=target_step,
        controller=controller,
        controller_start_step=anchor_step,
    )[-1]
    logits = model.decode(state).float()
    selected_predictions = logits.argmax(dim=-1)[batch.answer_mask]
    selected_targets = batch.targets[batch.answer_mask]
    return {
        "answer_ce": float(masked_cross_entropy(logits, batch)),
        "answer_region_em": exact_match(logits, batch),
        "answer_region_token_accuracy": float(
            selected_predictions.eq(selected_targets).float().mean()
        ),
    }


def gradient_metrics(model, controller, batch, anchor_step: int) -> dict[str, float]:
    controller.train()
    controller.zero_grad(set_to_none=True)
    loss, em = _controlled_ce_batch(
        model=model,
        controller=controller,
        batch=batch,
        anchor_step=anchor_step,
        controlled_steps=int(batch.target_steps[0]) - anchor_step,
    )
    loss.backward()
    result = {"loss": float(loss.detach()), "em": float(em)}
    for name, parameter in controller.named_parameters():
        gradient = parameter.grad
        result[f"grad_norm_{name}"] = (
            float(gradient.float().norm()) if gradient is not None else 0.0
        )
    controller.eval()
    return result


def snapshot_update(path: Path) -> int:
    match = re.search(r"controller_(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"cannot parse update from {path}")
    return int(match.group(1))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    snapshot_paths = sorted(args.snapshot_dir.glob("controller_*.pt"), key=snapshot_update)
    if not snapshot_paths:
        raise ValueError("no controller snapshots found")
    batches = {
        length: fixed_batch(spec, length, args.examples, args.seed, device)
        for length in args.lengths
    }

    rows: list[dict[str, Any]] = []
    for path in snapshot_paths:
        controller, payload = load_controller(path, device=device)
        anchor_step = int(payload["anchor_step"])
        controller.eval()
        for length, batch in batches.items():
            rows.append(
                {
                    "update": snapshot_update(path),
                    "length": length,
                    **snapshot_metrics(model, controller, batch, anchor_step),
                }
            )

    final_controller, final_payload = load_controller(
        snapshot_paths[-1], device=device
    )
    anchor_step = int(final_payload["anchor_step"])
    parity_batch = batches[args.parity_length]
    with torch.inference_mode():
        training_state, controlled_steps = forward_training_path(
            model, final_controller, parity_batch, anchor_step
        )
        evaluation_state = model.states(
            parity_batch.inputs,
            steps=int(parity_batch.target_steps[0]),
            controller=final_controller,
            controller_start_step=anchor_step,
        )[-1]
        training_logits = model.decode(training_state).float()
        evaluation_logits = model.decode(evaluation_state).float()
        path_consistency = {
            "length": args.parity_length,
            "target_step": int(parity_batch.target_steps[0]),
            "anchor_step": anchor_step,
            "controlled_steps": controlled_steps,
            "maximum_state_difference": float(
                (training_state - evaluation_state).abs().max()
            ),
            "maximum_logit_difference": float(
                (training_logits - evaluation_logits).abs().max()
            ),
            "training_path_ce": float(masked_cross_entropy(training_logits, parity_batch)),
            "evaluation_path_ce": float(
                masked_cross_entropy(evaluation_logits, parity_batch)
            ),
        }

    initial_controller, initial_payload = load_controller(
        snapshot_paths[0], device=device
    )
    initial_gradients = gradient_metrics(
        model,
        initial_controller,
        parity_batch,
        int(initial_payload["anchor_step"]),
    )
    final_gradients = gradient_metrics(
        model, final_controller, parity_batch, anchor_step
    )
    parameter_change: dict[str, float] = {}
    initial_parameters = dict(initial_controller.named_parameters())
    for name, parameter in final_controller.named_parameters():
        initial = initial_parameters[name]
        parameter_change[f"delta_norm_{name}"] = float(
            (parameter.detach().float() - initial.detach().float()).norm()
        )
        parameter_change[f"final_norm_{name}"] = float(
            parameter.detach().float().norm()
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "snapshot_metrics.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "snapshots": len(snapshot_paths),
        "evaluated_lengths": list(args.lengths),
        "examples_per_length": args.examples,
        "path_consistency": path_consistency,
        "initial_gradient_metrics": initial_gradients,
        "final_gradient_metrics": final_gradients,
        "parameter_change": parameter_change,
        "claim_boundary": (
            "This diagnoses code-path parity, gradient flow, and checkpoint dynamics; "
            "it does not by itself distinguish optimization failure from insufficient "
            "controller expressivity."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=(4, 8, 10, 11, 12, 14, 16, 18, 20))
    parser.add_argument("--parity-length", type=int, default=12)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=994101)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
