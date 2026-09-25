from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
)
from reasoning_loop.graph_path_telomere_explicit_diagonal_rrr import (
    last_active_age,
)


CONDITIONS = (
    "learned_J",
    "young_norm_before_J",
    "young_norm_after_J",
    "young_norm_only",
    "exact_young_boundary",
    "wrong_current_young_boundary",
    "cross_graph_young_boundary",
    "zero_boundary",
    "no_J",
)


def _token_normmatched(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.float().norm(dim=-1, keepdim=True) / value.float().norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return value * scale.to(dtype=value.dtype)


def _segment(values: list[float], start: int, stop: int) -> float | None:
    selected = values[start:stop]
    return sum(selected) / len(selected) if selected else None


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    young_age = last_active_age(phase_positions)
    positions = intervention_groups(cfg.node_count)["all"]
    checkpoint, artifact_positions, modules, payload = load_task_lora_modules(
        args.artifact, device=device
    )
    if checkpoint != str(args.checkpoint):
        raise ValueError("controller and backbone checkpoints differ")
    if artifact_positions != positions:
        raise ValueError("controller positions differ from the full-token protocol")
    if payload.get("placement") != "loop_boundary":
        raise ValueError("boundary norm control requires a loop-boundary controller")
    operator = modules[args.label]

    streams = _canonical_training_streams(payload)
    seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
        device=device, node_count=cfg.node_count, streams=streams
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    _, sampled_unseen, distribution = stratified_samples(
        seen, unseen, count=args.sample_per_partition, seed=args.sample_seed
    )
    successors_all, starts_all = _expand_all_starts(sampled_unseen, device=device)

    correct = {condition: [0] * args.continuation_loops for condition in CONDITIONS}
    norm_ratio_sum: dict[tuple[str, int], float] = defaultdict(float)
    total = 0
    indices = list(positions)
    for offset in range(0, successors_all.shape[0], args.batch_size):
        successors = successors_all[offset : offset + args.batch_size]
        starts = starts_all[offset : offset + args.batch_size]
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        states = {condition: initial.clone() for condition in CONDITIONS}

        for cycle in range(1, args.continuation_loops + 1):
            current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
            young = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=young_age,
                phase_position=phase_positions[young_age],
            )
            wrong_current = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=advance_nodes(successors, current, steps=jump),
                age=young_age,
                phase_position=phase_positions[young_age],
            )
            cross_graph = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                # All-start expansion stores eight consecutive starts per graph;
                # roll by node_count to change graph while retaining start order.
                successors=successors.roll(cfg.node_count, dims=0),
                current=current,
                age=young_age,
                phase_position=phase_positions[young_age],
            )
            controlled: list[torch.Tensor] = []
            for condition in CONDITIONS:
                source = states[condition]
                live = source[:, indices]
                target = young[:, indices]
                if condition == "learned_J":
                    mapped = operator(live)
                elif condition == "young_norm_before_J":
                    mapped = operator(_token_normmatched(live, target))
                elif condition == "young_norm_after_J":
                    mapped = _token_normmatched(operator(live), target)
                elif condition == "young_norm_only":
                    mapped = _token_normmatched(live, target)
                elif condition == "exact_young_boundary":
                    mapped = target
                elif condition == "wrong_current_young_boundary":
                    mapped = wrong_current[:, indices]
                elif condition == "cross_graph_young_boundary":
                    mapped = cross_graph[:, indices]
                elif condition == "zero_boundary":
                    mapped = torch.zeros_like(live)
                elif condition == "no_J":
                    mapped = live
                else:
                    raise RuntimeError(condition)
                result = source.clone()
                result[:, indices] = mapped.to(dtype=result.dtype)
                controlled.append(result)
                norm_ratio_sum[(condition, cycle)] += float(
                    (
                        mapped.float().norm(dim=-1)
                        / target.float().norm(dim=-1).clamp_min(1e-8)
                    ).mean().item()
                    * successors.shape[0]
                )
            step = run_one_loop(
                model,
                torch.cat(controlled, dim=0),
                loop_index=cfg.max_loops + cycle - 1,
            )
            target_node = advance_nodes(successors, endpoint, steps=jump * cycle)
            for condition, state_chunk, logits_chunk in zip(
                CONDITIONS,
                step.state.chunk(len(CONDITIONS), dim=0),
                step.logits.chunk(len(CONDITIONS), dim=0),
                strict=True,
            ):
                correct[condition][cycle - 1] += int(
                    logits_chunk.argmax(dim=-1).eq(target_node).sum().item()
                )
                states[condition] = state_chunk
        total += successors.shape[0]

    curves: dict[str, Any] = {}
    for condition in CONDITIONS:
        accuracy = [value / total for value in correct[condition]]
        curves[condition] = {
            "accuracy_by_cycle": accuracy,
            "auc_1_24": _segment(accuracy, 0, 24),
            "auc_25_48": _segment(accuracy, 24, 48),
            "auc_49_64": _segment(accuracy, 48, 64),
            "auc_65_96": _segment(accuracy, 64, 96),
            "auc_97_128": _segment(accuracy, 96, 128),
            "final_accuracy": accuracy[-1],
            "mean_norm_ratio_at_cycles": {
                str(cycle): norm_ratio_sum[(condition, cycle)] / total
                for cycle in (1, 32, 64, 96, 128)
                if cycle <= args.continuation_loops
            },
        }

    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_description": payload.get("backbone_loss_description"),
        "controller_placement": "loop_boundary",
        "controller_label": args.label,
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "task_step_size": jump,
        "young_interface_age": young_age,
        "training_graph_draws": total_draws,
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs": len(unseen),
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "permutations": len(sampled_unseen),
            "all_start_examples": total,
            "cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "intervention": (
            "per-token residual norm matched to exact same-graph, same-current "
            f"H{young_age} boundary; direction and all frozen weights otherwise retained"
        ),
        "curves": curves,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary-native matched-norm control.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))
