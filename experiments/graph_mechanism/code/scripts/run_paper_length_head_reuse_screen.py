#!/usr/bin/env python3
"""Causal attention-head reuse screen for the paper-length telomere task."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

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


@contextmanager
def ablate_attention_heads(
    model: torch.nn.Module, heads: Sequence[int]
) -> Iterator[None]:
    selected = tuple(sorted({int(head) for head in heads}))
    n_heads = int(model.config.n_heads)
    head_dim = int(model.config.d_model // n_heads)
    if any(head < 0 or head >= n_heads for head in selected):
        raise ValueError("attention head index is out of range")
    handles: list[Any] = []

    def zero_selected_heads(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        attended = inputs[0]
        shaped = attended.reshape(*attended.shape[:-1], n_heads, head_dim).clone()
        if selected:
            shaped[..., list(selected), :] = 0
        return (shaped.reshape_as(attended), *inputs[1:])

    try:
        for layer in model.layers:
            handles.append(
                layer.attention.output.register_forward_pre_hook(
                    zero_selected_heads
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def answer_logit_margin(logits: torch.Tensor, batch: Any) -> float:
    selected_logits = logits[batch.answer_mask]
    selected_targets = batch.targets[batch.answer_mask]
    target_logits = selected_logits.gather(
        dim=-1, index=selected_targets.unsqueeze(-1)
    ).squeeze(-1)
    competitors = selected_logits.clone()
    competitors.scatter_(
        dim=-1,
        index=selected_targets.unsqueeze(-1),
        value=float("-inf"),
    )
    return float((target_logits - competitors.max(dim=-1).values).mean())


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module | None,
    controller_start_step: int | None,
    logical_length: int,
    ablated_heads: Sequence[int],
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    target_step = logical_length + spec.step_offset
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    correct = 0.0
    nll = 0.0
    entropy = 0.0
    margin = 0.0
    with ablate_attention_heads(model, ablated_heads):
        for _ in range(batches):
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=logical_length,
                max_length=logical_length,
                fixed_length=logical_length,
                generator=generator,
            ).to(device)
            final_state = None
            for final_state in model.iter_states(
                batch.inputs,
                steps=target_step,
                controller=controller,
                controller_start_step=controller_start_step,
            ):
                pass
            if final_state is None:
                raise RuntimeError("trajectory did not produce a state")
            logits = model.decode(final_state).float()
            correct += exact_match(logits, batch) * batch_size
            nll += answer_cross_entropy(logits, batch)
            entropy += answer_predictive_entropy(logits, batch)
            margin += answer_logit_margin(logits, batch)
    return {
        "ablated_heads": list(ablated_heads),
        "logical_length": logical_length,
        "target_step": target_step,
        "target_step_exact_match": correct / (batch_size * batches),
        "target_step_answer_nll": nll / batches,
        "target_step_predictive_entropy": entropy / batches,
        "target_step_answer_logit_margin": margin / batches,
        "examples": batch_size * batches,
    }


def with_deltas(value: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        **value,
        "delta_answer_nll": (
            value["target_step_answer_nll"]
            - baseline["target_step_answer_nll"]
        ),
        "exact_match_drop": (
            baseline["target_step_exact_match"]
            - value["target_step_exact_match"]
        ),
        "answer_logit_margin_drop": (
            baseline["target_step_answer_logit_margin"]
            - value["target_step_answer_logit_margin"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=271001)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, spec, backbone = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if Path(controller_payload["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("controller belongs to another backbone checkpoint")
    anchor_step = int(controller_payload["anchor_step"])
    n_heads = int(model.config.n_heads)
    if not 1 <= args.top_k <= n_heads:
        raise ValueError("top-k must lie within the model head count")

    raw20 = evaluate(
        model=model,
        spec=spec,
        controller=None,
        controller_start_step=None,
        logical_length=20,
        ablated_heads=(),
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device=device,
    )
    j100 = evaluate(
        model=model,
        spec=spec,
        controller=controller,
        controller_start_step=anchor_step,
        logical_length=100,
        ablated_heads=(),
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device=device,
    )
    individual: list[dict[str, Any]] = []
    for head in range(n_heads):
        in_domain = evaluate(
            model=model,
            spec=spec,
            controller=None,
            controller_start_step=None,
            logical_length=20,
            ablated_heads=(head,),
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            device=device,
        )
        long_j = evaluate(
            model=model,
            spec=spec,
            controller=controller,
            controller_start_step=anchor_step,
            logical_length=100,
            ablated_heads=(head,),
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            device=device,
        )
        individual.append(
            {
                "head": head,
                "raw_L20": with_deltas(in_domain, raw20),
                "J_L100": with_deltas(long_j, j100),
            }
        )

    ranked = sorted(
        individual,
        key=lambda row: (
            row["raw_L20"]["answer_logit_margin_drop"],
            row["raw_L20"]["delta_answer_nll"],
            row["raw_L20"]["exact_match_drop"],
        ),
        reverse=True,
    )
    top_heads = [int(row["head"]) for row in ranked[: args.top_k]]
    bottom_heads = [int(row["head"]) for row in ranked[-args.top_k :]]
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(args.seed + 999)
    random_heads = torch.randperm(n_heads, generator=random_generator)[
        : args.top_k
    ].tolist()

    groups: dict[str, list[int]] = {}
    for k in (1, 2, 4, args.top_k):
        if k <= args.top_k:
            groups[f"top_{k}"] = top_heads[:k]
    groups[f"bottom_{args.top_k}"] = bottom_heads
    groups[f"random_{args.top_k}"] = random_heads
    group_results: dict[str, Any] = {}
    for name, heads in groups.items():
        in_domain = evaluate(
            model=model,
            spec=spec,
            controller=None,
            controller_start_step=None,
            logical_length=20,
            ablated_heads=heads,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            device=device,
        )
        raw_long = evaluate(
            model=model,
            spec=spec,
            controller=None,
            controller_start_step=None,
            logical_length=100,
            ablated_heads=heads,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            device=device,
        )
        j_long = evaluate(
            model=model,
            spec=spec,
            controller=controller,
            controller_start_step=anchor_step,
            logical_length=100,
            ablated_heads=heads,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            device=device,
        )
        group_results[name] = {
            "heads": heads,
            "raw_L20": with_deltas(in_domain, raw20),
            "raw_L100": raw_long,
            "J_L100": with_deltas(j_long, j100),
        }

    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "controller": str(args.controller),
        "checkpoint_step": int(backbone["step"]),
        "backbone_seed": int(backbone["seed"]),
        "controller_seed": int(controller_payload["seed"]),
        "selection_rule": (
            "rank individual heads only by raw L20 correct-vs-strongest-"
            "incorrect answer-logit margin drop, using NLL and EM only as "
            "tie-breakers; freeze that ranking before reading J L100 effects"
        ),
        "intervention": (
            "zero the selected attended head slice before the frozen "
            "attention output projection on every recurrent use"
        ),
        "evaluation_seed": args.seed,
        "examples_per_condition": args.batch_size * args.batches,
        "baselines": {"raw_L20": raw20, "J_L100": j100},
        "individual_heads": individual,
        "raw_L20_ranking": [int(row["head"]) for row in ranked],
        "top_heads": top_heads,
        "group_results": group_results,
        "claim_boundary": (
            "a pilot overlap is reused-component evidence only after a "
            "larger paired audit and matched bottom/random controls"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
