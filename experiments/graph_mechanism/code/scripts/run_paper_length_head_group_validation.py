#!/usr/bin/env python3
"""Fresh-sample validation of a frozen in-domain-selected head group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.paper_length_telomere import (
    load_backbone,
    load_controller,
    pick_device,
)
from scripts.run_paper_length_head_reuse_screen import evaluate, with_deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--top-heads", type=int, nargs="+", required=True)
    parser.add_argument("--bottom-heads", type=int, nargs="+", required=True)
    parser.add_argument("--random-groups", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=271002)
    parser.add_argument("--random-seed", type=int, default=272001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    if list(args.top_heads) != list(discovery["top_heads"]):
        raise ValueError("fixed top heads do not match the discovery artifact")
    discovered_bottom = discovery["group_results"][
        f"bottom_{len(args.bottom_heads)}"
    ]["heads"]
    if list(args.bottom_heads) != list(discovered_bottom):
        raise ValueError("fixed bottom heads do not match discovery")

    device = pick_device(args.device)
    model, spec, backbone = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if Path(controller_payload["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("controller belongs to another backbone checkpoint")
    anchor_step = int(controller_payload["anchor_step"])
    n_heads = int(model.config.n_heads)

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
    raw100 = evaluate(
        model=model,
        spec=spec,
        controller=None,
        controller_start_step=None,
        logical_length=100,
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

    top_heads = list(args.top_heads)
    groups: dict[str, list[int]] = {
        "top_1": top_heads[:1],
        "top_2": top_heads[:2],
        "top_4": top_heads[:4],
        f"top_{len(top_heads)}": top_heads,
        f"bottom_{len(args.bottom_heads)}": list(args.bottom_heads),
    }
    remaining = [head for head in range(n_heads) if head not in top_heads]
    for group_index in range(args.random_groups):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.random_seed + group_index)
        permutation = torch.randperm(len(remaining), generator=generator)
        groups[f"random_{group_index + 1}"] = [
            remaining[index]
            for index in permutation[: len(top_heads)].tolist()
        ]

    results: dict[str, Any] = {}
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
        results[name] = {
            "heads": heads,
            "raw_L20": with_deltas(in_domain, raw20),
            "raw_L100": with_deltas(raw_long, raw100),
            "J_L100": with_deltas(j_long, j100),
        }

    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "controller": str(args.controller),
        "checkpoint_step": int(backbone["step"]),
        "backbone_seed": int(backbone["seed"]),
        "controller_seed": int(controller_payload["seed"]),
        "discovery": str(args.discovery),
        "selection_frozen_before_validation": True,
        "selection_rule": discovery["selection_rule"],
        "evaluation_seed": args.seed,
        "random_group_seed": args.random_seed,
        "examples_per_condition": args.batch_size * args.batches,
        "baselines": {
            "raw_L20": raw20,
            "raw_L100": raw100,
            "J_L100": j100,
        },
        "groups": results,
        "claim_boundary": (
            "single-backbone local validation can establish a candidate "
            "reused head group; a general mechanism requires independent "
            "backbone seeds and component-output patching"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
