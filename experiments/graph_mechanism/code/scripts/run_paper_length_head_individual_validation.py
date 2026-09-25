#!/usr/bin/env python3
"""Matched single-head validation of a frozen in-domain head ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
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
    parser.add_argument("--random-heads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=271003)
    parser.add_argument("--random-seed", type=int, default=272002)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def summarize_cohort(
    names: list[str], results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    metrics = {
        "raw_L20_margin_drop": [
            results[name]["raw_L20"]["answer_logit_margin_drop"] for name in names
        ],
        "raw_L20_em_drop": [
            results[name]["raw_L20"]["exact_match_drop"] for name in names
        ],
        "J_L100_margin_drop": [
            results[name]["J_L100"]["answer_logit_margin_drop"] for name in names
        ],
        "J_L100_em_drop": [
            results[name]["J_L100"]["exact_match_drop"] for name in names
        ],
        "J_L100_nll_increase": [
            results[name]["J_L100"]["delta_answer_nll"] for name in names
        ],
    }
    return {
        "members": names,
        "count": len(names),
        "metrics": {
            metric: {
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
            }
            for metric, values in metrics.items()
        },
    }


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

    baselines = {
        "raw_L20": evaluate(
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
        ),
        "raw_L100": evaluate(
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
        ),
        "J_L100": evaluate(
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
        ),
    }

    excluded = set(args.top_heads) | set(args.bottom_heads)
    remaining = [head for head in range(n_heads) if head not in excluded]
    if args.random_heads > len(remaining):
        raise ValueError("requested more random controls than available heads")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.random_seed)
    permutation = torch.randperm(len(remaining), generator=generator)
    random_heads = [
        remaining[index] for index in permutation[: args.random_heads].tolist()
    ]

    cohorts = {
        "top": list(args.top_heads),
        "bottom": list(args.bottom_heads),
        "random": random_heads,
    }
    results: dict[str, dict[str, Any]] = {}
    cohort_names: dict[str, list[str]] = {}
    for cohort, heads in cohorts.items():
        cohort_names[cohort] = []
        for head in heads:
            name = f"{cohort}_head_{head}"
            cohort_names[cohort].append(name)
            condition_results: dict[str, Any] = {"head": head, "cohort": cohort}
            for condition, baseline in baselines.items():
                logical_length = 20 if condition == "raw_L20" else 100
                uses_controller = condition == "J_L100"
                measured = evaluate(
                    model=model,
                    spec=spec,
                    controller=controller if uses_controller else None,
                    controller_start_step=anchor_step if uses_controller else None,
                    logical_length=logical_length,
                    ablated_heads=[head],
                    batch_size=args.batch_size,
                    batches=args.batches,
                    seed=args.seed,
                    device=device,
                )
                condition_results[condition] = with_deltas(measured, baseline)
            results[name] = condition_results

    cohort_summaries = {
        cohort: summarize_cohort(names, results)
        for cohort, names in cohort_names.items()
    }
    candidate_name = f"top_head_{args.top_heads[0]}"
    candidate = results[candidate_name]
    matched_controls = cohort_names["bottom"] + cohort_names["random"]
    comparisons = {}
    for metric in (
        "answer_logit_margin_drop",
        "exact_match_drop",
        "delta_answer_nll",
    ):
        candidate_value = candidate["J_L100"][metric]
        control_values = [results[name]["J_L100"][metric] for name in matched_controls]
        comparisons[metric] = {
            "candidate": candidate_value,
            "controls_exceeded": sum(candidate_value > value for value in control_values),
            "controls_total": len(control_values),
            "control_max": max(control_values),
            "control_mean": mean(control_values),
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
        "random_control_seed": args.random_seed,
        "examples_per_condition": args.batch_size * args.batches,
        "baselines": baselines,
        "cohorts": cohorts,
        "individual_results": results,
        "cohort_summaries": cohort_summaries,
        "candidate_head": args.top_heads[0],
        "candidate_vs_matched_controls": comparisons,
        "claim_boundary": (
            "single-head ablation can identify a candidate component that is "
            "causally required for the controller-corrected trajectory; reuse "
            "still requires component-output patching, and generality requires "
            "independent backbone seeds"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
