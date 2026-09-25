#!/usr/bin/env python3
"""Causal head-output interchange between full-J and no-AB trajectories."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    answer_cross_entropy,
    answer_predictive_entropy,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)
from scripts.run_paper_length_head_reuse_screen import answer_logit_margin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--candidate-head", type=int, required=True)
    parser.add_argument("--control-head", type=int, required=True)
    parser.add_argument("--logical-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=273001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


@contextmanager
def capture_attended_heads(
    model: torch.nn.Module, heads: Sequence[int]
) -> Iterator[dict[int, list[torch.Tensor]]]:
    if len(model.layers) != 1:
        raise ValueError("this causal patch audit requires one physical layer")
    n_heads = int(model.config.n_heads)
    head_dim = int(model.config.d_model // n_heads)
    selected = tuple(sorted({int(head) for head in heads}))
    if any(head < 0 or head >= n_heads for head in selected):
        raise ValueError("attention head index is out of range")
    captured: dict[int, list[torch.Tensor]] = {head: [] for head in selected}

    def capture(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        attended = inputs[0].reshape(
            *inputs[0].shape[:-1], n_heads, head_dim
        )
        for head in selected:
            captured[head].append(attended[..., head, :].detach().clone())

    handle = model.layers[0].attention.output.register_forward_pre_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def patch_attended_head(
    model: torch.nn.Module,
    *,
    head: int,
    donor: Sequence[torch.Tensor],
    patch_start_step: int,
    patch_end_step: int,
    roll_examples: bool,
) -> Iterator[None]:
    if len(model.layers) != 1:
        raise ValueError("this causal patch audit requires one physical layer")
    n_heads = int(model.config.n_heads)
    head_dim = int(model.config.d_model // n_heads)
    if not 0 <= head < n_heads:
        raise ValueError("attention head index is out of range")
    calls = 0

    def patch(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        nonlocal calls
        step_index = calls + 1
        if calls >= len(donor):
            raise RuntimeError("recipient trajectory exceeds donor trajectory")
        attended = inputs[0]
        if patch_start_step < step_index <= patch_end_step:
            replacement = donor[calls]
            if roll_examples:
                replacement = replacement.roll(shifts=1, dims=0)
            shaped = attended.reshape(
                *attended.shape[:-1], n_heads, head_dim
            ).clone()
            shaped[..., head, :] = replacement
            attended = shaped.reshape_as(attended)
        calls += 1
        return (attended, *inputs[1:])

    handle = model.layers[0].attention.output.register_forward_pre_hook(patch)
    try:
        yield
        if calls != len(donor):
            raise RuntimeError(
                f"recipient used {calls} steps but donor has {len(donor)}"
            )
    finally:
        handle.remove()


def run_trajectory(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    steps: int,
    controller: torch.nn.Module,
    controller_start_step: int,
    patch_head: int | None = None,
    donor: Sequence[torch.Tensor] | None = None,
    patch_start_step: int | None = None,
    patch_end_step: int | None = None,
    roll_examples: bool = False,
) -> torch.Tensor:
    embedded = model.input_embeddings(inputs)
    state = torch.zeros_like(embedded)
    if (patch_head is None) != (donor is None):
        raise ValueError("patch_head and donor must be specified together")

    @contextmanager
    def maybe_patch() -> Iterator[None]:
        if patch_head is None or donor is None:
            yield
            return
        with patch_attended_head(
            model,
            head=patch_head,
            donor=donor,
            patch_start_step=(
                controller_start_step
                if patch_start_step is None
                else patch_start_step
            ),
            patch_end_step=steps if patch_end_step is None else patch_end_step,
            roll_examples=roll_examples,
        ):
            yield

    with maybe_patch():
        for step_index in range(1, steps + 1):
            if step_index > controller_start_step:
                state = controller(state)
            state = model.recurrent_step(state, embedded)
    return state


def add_metrics(
    totals: dict[str, dict[str, float]],
    condition: str,
    logits: torch.Tensor,
    batch: Any,
) -> None:
    values = totals.setdefault(
        condition,
        {"exact_match": 0.0, "answer_nll": 0.0, "entropy": 0.0, "margin": 0.0},
    )
    values["exact_match"] += exact_match(logits, batch) * batch.inputs.shape[0]
    values["answer_nll"] += answer_cross_entropy(logits, batch)
    values["entropy"] += answer_predictive_entropy(logits, batch)
    values["margin"] += answer_logit_margin(logits, batch)


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
    no_ab = ControllerView(controller, mode="no_AB").eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    totals: dict[str, dict[str, float]] = {}
    identity_max_abs = {"full_self_patch": 0.0, "no_AB_self_patch": 0.0}

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
        with capture_attended_heads(model, heads) as no_ab_heads:
            no_ab_state = run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=no_ab,
                controller_start_step=anchor_step,
            )

        states = {
            "full_J": full_state,
            "no_AB": no_ab_state,
            "no_AB_plus_full_candidate": run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=no_ab,
                controller_start_step=anchor_step,
                patch_head=args.candidate_head,
                donor=full_heads[args.candidate_head],
            ),
            "full_J_plus_no_AB_candidate": run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=controller,
                controller_start_step=anchor_step,
                patch_head=args.candidate_head,
                donor=no_ab_heads[args.candidate_head],
            ),
            "no_AB_plus_full_candidate_wrong_example": run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=no_ab,
                controller_start_step=anchor_step,
                patch_head=args.candidate_head,
                donor=full_heads[args.candidate_head],
                roll_examples=True,
            ),
            "no_AB_plus_full_control": run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=no_ab,
                controller_start_step=anchor_step,
                patch_head=args.control_head,
                donor=full_heads[args.control_head],
            ),
            "full_J_plus_no_AB_control": run_trajectory(
                model=model,
                inputs=batch.inputs,
                steps=target_step,
                controller=controller,
                controller_start_step=anchor_step,
                patch_head=args.control_head,
                donor=no_ab_heads[args.control_head],
            ),
        }
        full_self = run_trajectory(
            model=model,
            inputs=batch.inputs,
            steps=target_step,
            controller=controller,
            controller_start_step=anchor_step,
            patch_head=args.candidate_head,
            donor=full_heads[args.candidate_head],
        )
        no_ab_self = run_trajectory(
            model=model,
            inputs=batch.inputs,
            steps=target_step,
            controller=no_ab,
            controller_start_step=anchor_step,
            patch_head=args.candidate_head,
            donor=no_ab_heads[args.candidate_head],
        )
        identity_max_abs["full_self_patch"] = max(
            identity_max_abs["full_self_patch"],
            float((full_self - full_state).abs().max()),
        )
        identity_max_abs["no_AB_self_patch"] = max(
            identity_max_abs["no_AB_self_patch"],
            float((no_ab_self - no_ab_state).abs().max()),
        )
        for condition, state in states.items():
            add_metrics(totals, condition, model.decode(state).float(), batch)

    examples = args.batch_size * args.batches
    results = {
        condition: {
            "target_step_exact_match": values["exact_match"] / examples,
            "target_step_answer_nll": values["answer_nll"] / args.batches,
            "target_step_predictive_entropy": values["entropy"] / args.batches,
            "target_step_answer_logit_margin": values["margin"] / args.batches,
        }
        for condition, values in totals.items()
    }
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
        "patch_location": "selected attended-head slice before attention output projection",
        "patch_window": f"loops {anchor_step + 1}--{target_step}",
        "paired_same_example_same_loop": True,
        "identity_self_patch_max_abs": identity_max_abs,
        "results": results,
        "claim_boundary": (
            "same-example rescue plus reciprocal damage is component-output "
            "patch evidence on one frozen backbone; multi-seed confirmation "
            "is required for a general reuse claim"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
