#!/usr/bin/env python3
"""Localize candidate-head ablation and output rescue by recurrence window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)
from scripts.run_paper_length_head_output_patch import (
    add_metrics,
    capture_attended_heads,
    run_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--candidate-head", type=int, required=True)
    parser.add_argument("--control-head", type=int, required=True)
    parser.add_argument("--logical-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=274001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def window_name(start: int, end: int) -> str:
    return f"loops_{start}_{end}"


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.candidate_head == args.control_head:
        raise ValueError("candidate and control heads must differ")
    device = pick_device(args.device)
    model, spec, backbone = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if Path(controller_payload["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("controller belongs to another backbone checkpoint")
    anchor_step = int(controller_payload["anchor_step"])
    target_step = args.logical_length + spec.step_offset
    if anchor_step != 1 or target_step != 100:
        raise ValueError("registered pilot windows require anchor 1 and target 100")
    no_ab = ControllerView(controller, mode="no_AB").eval()
    windows = [(2, 20), (21, 40), (41, 60), (61, 80), (81, 100), (2, 100)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    totals: dict[str, dict[str, float]] = {}

    for _ in range(args.batches):
        batch = generate_paper_batch(
            spec,
            batch_size=args.batch_size,
            min_length=args.logical_length,
            max_length=args.logical_length,
            fixed_length=args.logical_length,
            generator=generator,
        ).to(device)
        heads = [args.candidate_head, args.control_head]
        with capture_attended_heads(model, heads) as full_heads:
            full_state = run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=controller,
                controller_start_step=anchor_step,
            )
        no_ab_state = run_trajectory(
            model=model,
            inputs=batch.inputs,
            steps=target_step,
            controller=no_ab,
            controller_start_step=anchor_step,
        )
        add_metrics(totals, "full_J", model.decode(full_state).float(), batch)
        add_metrics(totals, "no_AB", model.decode(no_ab_state).float(), batch)

        zero_heads = {
            head: [torch.zeros_like(value) for value in full_heads[head]]
            for head in heads
        }
        for start, end in windows:
            name = window_name(start, end)
            for label, head in (
                ("candidate", args.candidate_head),
                ("control", args.control_head),
            ):
                ablated = run_trajectory(
                    model=model,
                    inputs=batch.inputs,
                    steps=target_step,
                    controller=controller,
                    controller_start_step=anchor_step,
                    patch_head=head,
                    donor=zero_heads[head],
                    patch_start_step=start - 1,
                    patch_end_step=end,
                )
                add_metrics(
                    totals,
                    f"full_J_ablate_{label}_{name}",
                    model.decode(ablated).float(),
                    batch,
                )
                rescued = run_trajectory(
                    model=model,
                    inputs=batch.inputs,
                    steps=target_step,
                    controller=no_ab,
                    controller_start_step=anchor_step,
                    patch_head=head,
                    donor=full_heads[head],
                    patch_start_step=start - 1,
                    patch_end_step=end,
                )
                add_metrics(
                    totals,
                    f"no_AB_patch_{label}_{name}",
                    model.decode(rescued).float(),
                    batch,
                )

    examples = args.batch_size * args.batches
    metrics = {
        condition: {
            "target_step_exact_match": values["exact_match"] / examples,
            "target_step_answer_nll": values["answer_nll"] / args.batches,
            "target_step_predictive_entropy": values["entropy"] / args.batches,
            "target_step_answer_logit_margin": values["margin"] / args.batches,
        }
        for condition, values in totals.items()
    }
    full_em = metrics["full_J"]["target_step_exact_match"]
    no_ab_em = metrics["no_AB"]["target_step_exact_match"]
    rows: list[dict[str, Any]] = []
    for start, end in windows:
        name = window_name(start, end)
        row: dict[str, Any] = {
            "start_loop": start,
            "end_loop": end,
            "applications": end - start + 1,
        }
        for label in ("candidate", "control"):
            ablation = metrics[f"full_J_ablate_{label}_{name}"]
            rescue = metrics[f"no_AB_patch_{label}_{name}"]
            row[f"{label}_ablation_em"] = ablation["target_step_exact_match"]
            row[f"{label}_ablation_em_drop"] = (
                full_em - ablation["target_step_exact_match"]
            )
            row[f"{label}_ablation_nll"] = ablation["target_step_answer_nll"]
            row[f"{label}_patch_em"] = rescue["target_step_exact_match"]
            row[f"{label}_patch_em_gain"] = (
                rescue["target_step_exact_match"] - no_ab_em
            )
            row[f"{label}_patch_nll"] = rescue["target_step_answer_nll"]
        rows.append(row)

    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "controller": str(args.controller),
        "checkpoint_step": int(backbone["step"]),
        "backbone_seed": int(backbone["seed"]),
        "controller_seed": int(controller_payload["seed"]),
        "controller_anchor_step": anchor_step,
        "candidate_head": args.candidate_head,
        "control_head": args.control_head,
        "logical_length": args.logical_length,
        "target_step": target_step,
        "evaluation_seed": args.seed,
        "examples": examples,
        "same_example_same_loop": True,
        "windows_registered_before_run": [list(window) for window in windows],
        "metrics": metrics,
        "window_rows": rows,
        "claim_boundary": (
            "directional effects in several non-overlapping windows support "
            "repeated use on one backbone; general repeated reactivation "
            "requires independent backbone seeds"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
