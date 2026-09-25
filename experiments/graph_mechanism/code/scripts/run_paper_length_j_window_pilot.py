#!/usr/bin/env python3
"""Causal application-window pilot for the inter-loop J controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.paper_length_telomere import (
    answer_cross_entropy,
    answer_predictive_entropy,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)


def application_windows(target_step: int) -> dict[str, tuple[int, int] | None]:
    if target_step < 100:
        raise ValueError("registered window pilot expects target step at least 100")
    return {
        "raw": None,
        "full_2_target": (2, target_step),
        "early_2_20": (2, 20),
        "middle_21_40": (21, 40),
        "late_41_target": (41, target_step),
        "first_2_40": (2, 40),
        "post_20": (21, target_step),
        "last_20": (target_step - 19, target_step),
        "single_last": (target_step, target_step),
    }


@torch.no_grad()
def evaluate_window(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module,
    logical_length: int,
    window: tuple[int, int] | None,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int | list[int] | None]:
    target_step = logical_length + spec.step_offset
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    correct = 0.0
    nll = 0.0
    entropy = 0.0
    for _ in range(batches):
        batch = generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=logical_length,
            max_length=logical_length,
            fixed_length=logical_length,
            generator=generator,
        ).to(device)
        embedded = model.input_embeddings(batch.inputs)
        state = torch.zeros_like(embedded)
        for step in range(1, target_step + 1):
            if window is not None and window[0] <= step <= window[1]:
                state = controller(state)
            state = model.recurrent_step(state, embedded)
        logits = model.decode(state).float()
        correct += exact_match(logits, batch) * batch_size
        nll += answer_cross_entropy(logits, batch)
        entropy += answer_predictive_entropy(logits, batch)
    return {
        "application_window": list(window) if window is not None else None,
        "controller_applications": (
            window[1] - window[0] + 1 if window is not None else 0
        ),
        "target_step_exact_match": correct / (batch_size * batches),
        "target_step_answer_nll": nll / batches,
        "target_step_predictive_entropy": entropy / batches,
        "examples": batch_size * batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, nargs="+", required=True)
    parser.add_argument("--controller-labels", nargs="+")
    parser.add_argument("--logical-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=261001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.controller_labels or [path.parent.name for path in args.controllers]
    if len(labels) != len(args.controllers):
        raise ValueError("controller labels must match controller paths")
    device = pick_device(args.device)
    model, spec, backbone = load_backbone(args.checkpoint, device=device)
    target_step = args.logical_length + spec.step_offset
    windows = application_windows(target_step)
    result: dict[str, Any] = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone["step"]),
        "backbone_seed": int(backbone["seed"]),
        "backbone_trained_logical_length_range": [1, spec.train_max_length],
        "logical_length": args.logical_length,
        "target_step": target_step,
        "evaluation_seed": args.seed,
        "examples_per_condition": args.batch_size * args.batches,
        "endpoint": "strict target-step task EM; no nearby-window oracle",
        "intervention": (
            "apply the trained J only inside the named loop window while "
            "retaining the frozen recurrent executor at every loop"
        ),
        "controllers": {},
    }
    for label, path in zip(labels, args.controllers, strict=True):
        controller, payload = load_controller(path, device=device)
        if Path(payload["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError(f"controller {label} belongs to another checkpoint")
        conditions: dict[str, Any] = {}
        for name, window in windows.items():
            conditions[name] = evaluate_window(
                model=model,
                spec=spec,
                controller=controller,
                logical_length=args.logical_length,
                window=window,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
                device=device,
            )
        result["controllers"][label] = {
            "controller": str(path),
            "trained_logical_length_range": [
                1,
                int(payload["controller_logical_max_length"]),
            ],
            "conditions": conditions,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
