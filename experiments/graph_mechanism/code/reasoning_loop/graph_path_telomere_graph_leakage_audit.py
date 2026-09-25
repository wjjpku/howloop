from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import (
    exact_interfaces,
    load_unit_j_map,
)


Permutation = tuple[int, ...]


PRIMARY_J_TRAINING_STREAMS = (
    # Initial natural adjacent-state regression.
    ("calibration", 175001, 0, 1, 256, 4),
    # DAgger regression rounds.
    ("dagger", 175002, 1000, 8, 128, 8),
    # H24 task fine-tuning, followed by the H32/H48/H64 curriculum.
    ("task_h24", 175003, 1000, 32, 64, 8),
    ("task_h32", 177003, 1000, 16, 32, 9),
    ("task_h48", 178003, 1000, 16, 32, 9),
    ("task_h64", 179003, 1000, 16, 32, 9),
)


def cycle_type(permutation: Permutation) -> tuple[int, ...]:
    seen: set[int] = set()
    lengths: list[int] = []
    for start in range(len(permutation)):
        if start in seen:
            continue
        current = start
        length = 0
        while current not in seen:
            seen.add(current)
            current = permutation[current]
            length += 1
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _draw_successors(
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    batches: int,
    node_count: int,
) -> list[Permutation]:
    """Replay fixed_depth_batch's two RNG calls without running the model."""

    set_seed(seed)
    result: list[Permutation] = []
    for _ in range(batches):
        successors = torch.rand(
            batch_size,
            node_count,
            device=device,
        ).argsort(dim=-1)
        # fixed_depth_batch samples start immediately after successors.
        torch.randint(
            0,
            node_count,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )
        result.extend(tuple(row) for row in successors.cpu().tolist())
    return result


def reconstruct_primary_training_graphs(
    *,
    device: torch.device,
    node_count: int,
    streams: Iterable[tuple[str, int, int, int, int, int]] = (
        PRIMARY_J_TRAINING_STREAMS
    ),
) -> tuple[set[Permutation], dict[str, int], int]:
    all_seen: set[Permutation] = set()
    unique_after_stage: dict[str, int] = {}
    total_draws = 0
    for label, base_seed, stride, rounds, batch_size, batches in streams:
        for round_index in range(1, rounds + 1):
            seed = base_seed if stride == 0 else base_seed + stride * round_index
            draws = _draw_successors(
                device=device,
                seed=seed,
                batch_size=batch_size,
                batches=batches,
                node_count=node_count,
            )
            total_draws += len(draws)
            all_seen.update(draws)
        unique_after_stage[label] = len(all_seen)
    return all_seen, unique_after_stage, total_draws


def stratified_samples(
    left: Iterable[Permutation],
    right: Iterable[Permutation],
    *,
    count: int,
    seed: int,
) -> tuple[list[Permutation], list[Permutation], dict[str, int]]:
    """Sample equal cycle-type counts from seen and unseen graph pools."""

    left_groups: dict[tuple[int, ...], list[Permutation]] = defaultdict(list)
    right_groups: dict[tuple[int, ...], list[Permutation]] = defaultdict(list)
    for item in left:
        left_groups[cycle_type(item)].append(item)
    for item in right:
        right_groups[cycle_type(item)].append(item)
    common = sorted(set(left_groups) & set(right_groups))
    rng = random.Random(seed)
    for group in common:
        rng.shuffle(left_groups[group])
        rng.shuffle(right_groups[group])

    capacities = {
        group: min(len(left_groups[group]), len(right_groups[group]))
        for group in common
    }
    total_capacity = sum(capacities.values())
    if count > total_capacity:
        raise ValueError(f"requested {count} graphs but capacity is {total_capacity}")

    allocation = {
        group: min(
            capacities[group],
            int(count * capacities[group] / total_capacity),
        )
        for group in common
    }
    assigned = sum(allocation.values())
    for group in sorted(common, key=lambda value: capacities[value], reverse=True):
        while assigned < count and allocation[group] < capacities[group]:
            allocation[group] += 1
            assigned += 1

    sampled_left: list[Permutation] = []
    sampled_right: list[Permutation] = []
    distribution: dict[str, int] = {}
    for group in common:
        take = allocation[group]
        sampled_left.extend(left_groups[group][:take])
        sampled_right.extend(right_groups[group][:take])
        if take:
            distribution["+".join(map(str, group))] = take
    rng.shuffle(sampled_left)
    rng.shuffle(sampled_right)
    return sampled_left, sampled_right, distribution


def _expand_all_starts(
    permutations: list[Permutation],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = len(permutations[0])
    successors = torch.tensor(permutations, dtype=torch.long, device=device)
    successors = successors.repeat_interleave(node_count, dim=0)
    starts = torch.arange(node_count, device=device).repeat(len(permutations))
    return successors, starts


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().flatten(1),
            right.float().flatten(1),
            dim=1,
        )
        .mean()
        .item()
    )


def _relative_mse(source: torch.Tensor, target: torch.Tensor) -> float:
    denominator = (
        target.float()
        - target.float().mean(dim=(0, 1), keepdim=True)
    ).square().mean().clamp_min(1e-8)
    return float(((source.float() - target.float()).square().mean() / denominator).item())


@torch.no_grad()
def audit_partition(
    *,
    label: str,
    permutations: list[Permutation],
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    age_map,
    device: torch.device,
    batch_size: int,
    continuation_loops: int,
    operating_age: int = 7,
) -> dict:
    jump = phase_positions[3] - phase_positions[2]
    successors_all, starts_all = _expand_all_starts(
        permutations,
        device=device,
    )
    exact_condition = f"exact_H{operating_age}_plus_full_Block2"
    exact_blocked_condition = (
        f"exact_H{operating_age}_Block2_answer_updates_zero"
    )
    correct: dict[str, list[int]] = {
        "learned_J_plus_full_Block2": [0] * continuation_loops,
        exact_condition: [0] * continuation_loops,
        "no_J_plus_full_Block2": [0] * continuation_loops,
    }
    nonendpoint_correct: dict[str, list[int]] = {
        condition: [0] * continuation_loops for condition in correct
    }
    nonendpoint_counts = [0] * continuation_loops
    direct_correct = {
        "learned_J_Block2_answer_updates_zero": 0,
        exact_blocked_condition: 0,
        "no_J_Block2_answer_updates_zero": 0,
    }
    direct_nonendpoint_correct = {
        condition: 0 for condition in direct_correct
    }
    direct_nonendpoint_count = 0
    alignment_sums: dict[str, float] = defaultdict(float)
    count = 0
    answer_index = positions.index(cfg.seq_len - 1)

    for offset in range(0, successors_all.shape[0], batch_size):
        successors = successors_all[offset : offset + batch_size]
        starts = starts_all[offset : offset + batch_size]
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )

        current_oracles = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=endpoint,
            ages=(operating_age,),
            loop_index=cfg.max_loops,
        )[operating_age]
        next_node = advance_nodes(successors, endpoint, steps=jump)
        next_oracles = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=next_node,
            ages=(operating_age,),
            loop_index=cfg.max_loops,
        )[operating_age]

        captured: dict[str, torch.Tensor] = {}

        def capture_map(value: torch.Tensor) -> torch.Tensor:
            captured["source"] = value.detach()
            transformed = age_map(value)
            captured["transformed"] = transformed.detach()
            return transformed

        learned_first = run_one_loop(
            model,
            initial,
            loop_index=cfg.max_loops,
            block2_position_transform=(positions, capture_map),
        )
        transformed = captured["transformed"]
        source = captured["source"]
        for suffix, index in (
            ("all_positions", slice(None)),
            ("answer_position", answer_index),
        ):
            transformed_part = transformed[:, index]
            source_part = source[:, index]
            current_part = current_oracles[:, index]
            next_part = next_oracles[:, index]
            alignment_sums[
                f"mse_to_current_H{operating_age}_{suffix}"
            ] += (
                _relative_mse(transformed_part.unsqueeze(1) if transformed_part.ndim == 2 else transformed_part,
                              current_part.unsqueeze(1) if current_part.ndim == 2 else current_part)
                * successors.shape[0]
            )
            alignment_sums[f"mse_to_next_H{operating_age}_{suffix}"] += (
                _relative_mse(transformed_part.unsqueeze(1) if transformed_part.ndim == 2 else transformed_part,
                              next_part.unsqueeze(1) if next_part.ndim == 2 else next_part)
                * successors.shape[0]
            )
            alignment_sums[f"cos_delta_with_age_reset_{suffix}"] += (
                _cosine(
                    transformed_part - source_part,
                    current_part - source_part,
                )
                * successors.shape[0]
            )
            alignment_sums[f"cos_delta_with_graph_step_{suffix}"] += (
                _cosine(
                    transformed_part - source_part,
                    next_part - source_part,
                )
                * successors.shape[0]
            )
            current_error = (
                transformed_part.float() - current_part.float()
            ).flatten(1).square().mean(dim=1)
            next_error = (
                transformed_part.float() - next_part.float()
            ).flatten(1).square().mean(dim=1)
            alignment_sums[
                f"fraction_closer_to_current_H{operating_age}_{suffix}"
            ] += float(current_error.lt(next_error).sum().item())

        zeros = torch.zeros(
            successors.shape[0],
            cfg.d_model,
            device=device,
            dtype=initial.dtype,
        )
        direct_conditions = {
            "learned_J_Block2_answer_updates_zero": run_one_loop(
                model,
                initial,
                loop_index=cfg.max_loops,
                block2_position_transform=(positions, age_map),
                attention_answer_override=zeros,
                mlp_answer_override=zeros,
            ),
            exact_blocked_condition: run_one_loop(
                model,
                initial,
                loop_index=cfg.max_loops,
                block2_position_override=(positions, current_oracles),
                attention_answer_override=zeros,
                mlp_answer_override=zeros,
            ),
            "no_J_Block2_answer_updates_zero": run_one_loop(
                model,
                initial,
                loop_index=cfg.max_loops,
                attention_answer_override=zeros,
                mlp_answer_override=zeros,
            ),
        }
        for condition, step in direct_conditions.items():
            prediction = step.logits.argmax(dim=-1)
            is_correct = prediction.eq(next_node)
            direct_correct[condition] += int(is_correct.sum().item())
            direct_nonendpoint = next_node.ne(endpoint)
            direct_nonendpoint_correct[condition] += int(
                (is_correct & direct_nonendpoint).sum().item()
            )
        direct_nonendpoint_count += int(next_node.ne(endpoint).sum().item())

        states = {
            "learned_J_plus_full_Block2": initial.clone(),
            exact_condition: initial.clone(),
            "no_J_plus_full_Block2": initial.clone(),
        }
        for cycle in range(1, continuation_loops + 1):
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            target = advance_nodes(
                successors,
                endpoint,
                steps=jump * cycle,
            )
            nonendpoint = target.ne(endpoint)
            nonendpoint_counts[cycle - 1] += int(nonendpoint.sum().item())
            oracle = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=successors,
                current=current,
                ages=(operating_age,),
                loop_index=cfg.max_loops + cycle - 1,
            )[operating_age]
            next_states = {}
            for condition, state in states.items():
                kwargs = {}
                if condition == "learned_J_plus_full_Block2":
                    kwargs["block2_position_transform"] = (positions, age_map)
                elif condition == exact_condition:
                    kwargs["block2_position_override"] = (positions, oracle)
                step = run_one_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops + cycle - 1,
                    **kwargs,
                )
                prediction = step.logits.argmax(dim=-1)
                is_correct = prediction.eq(target)
                correct[condition][cycle - 1] += int(is_correct.sum().item())
                nonendpoint_correct[condition][cycle - 1] += int(
                    (is_correct & nonendpoint).sum().item()
                )
                next_states[condition] = step.state
            states = next_states
        count += successors.shape[0]

    curves = {}
    for condition, values in correct.items():
        nonendpoint_accuracy = [
            numerator / denominator
            for numerator, denominator in zip(
                nonendpoint_correct[condition],
                nonendpoint_counts,
                strict=True,
            )
        ]
        curves[condition] = {
            "accuracy_by_cycle": [value / count for value in values],
            "auc_1_24": sum(values[:24]) / (count * min(24, len(values))),
            "auc_25_48": (
                sum(values[24:48])
                / (count * min(24, max(0, len(values) - 24)))
                if len(values) > 24
                else None
            ),
            "auc_49_64": (
                sum(values[48:64])
                / (count * min(16, max(0, len(values) - 48)))
                if len(values) > 48
                else None
            ),
            "nonendpoint_accuracy_by_cycle": nonendpoint_accuracy,
            "nonendpoint_auc_1_24": sum(nonendpoint_accuracy[:24])
            / min(24, len(nonendpoint_accuracy)),
            "nonendpoint_auc_25_48": (
                sum(nonendpoint_accuracy[24:48])
                / min(24, max(0, len(nonendpoint_accuracy) - 24))
                if len(nonendpoint_accuracy) > 24
                else None
            ),
            "nonendpoint_auc_49_64": (
                sum(nonendpoint_accuracy[48:64])
                / min(16, max(0, len(nonendpoint_accuracy) - 48))
                if len(nonendpoint_accuracy) > 48
                else None
            ),
        }
    alignment = {}
    for key, value in alignment_sums.items():
        alignment[key] = value / count
    return {
        "partition": label,
        "permutations": len(permutations),
        "examples_all_starts": count,
        "curves": curves,
        "one_step_executor_blocked_accuracy": {
            key: value / count for key, value in direct_correct.items()
        },
        "one_step_executor_blocked_nonendpoint_accuracy": {
            key: value / direct_nonendpoint_count
            for key, value in direct_nonendpoint_correct.items()
        },
        "interface_alignment": alignment,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--j-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--operating-age", type=int, default=7)
    parser.add_argument(
        "--training-streams-json",
        type=Path,
        help=(
            "Optional JSON list of {label, base_seed, stride, rounds, "
            "batch_size, batches} records used to reconstruct J-training "
            "graphs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    if cfg.node_count != 8 or cfg.max_depth != 8 or cfg.max_loops != 8:
        raise ValueError("audit is fixed to the D8L8 N8 experiment")
    phase_summary = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_summary["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    age_map, map_checkpoint = load_unit_j_map(
        args.j_artifact,
        label=args.j_label,
        device=device,
    )
    if map_checkpoint != str(args.checkpoint):
        raise ValueError("J and frozen model checkpoints differ")

    if args.training_streams_json is None:
        training_streams = PRIMARY_J_TRAINING_STREAMS
    else:
        stream_payload = json.loads(
            args.training_streams_json.read_text(encoding="utf-8")
        )
        training_streams = tuple(
            (
                str(item["label"]),
                int(item["base_seed"]),
                int(item["stride"]),
                int(item["rounds"]),
                int(item["batch_size"]),
                int(item["batches"]),
            )
            for item in stream_payload
        )
    seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
        device=device,
        node_count=cfg.node_count,
        streams=training_streams,
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    sampled_seen, sampled_unseen, distribution = stratified_samples(
        seen,
        unseen,
        count=args.sample_per_partition,
        seed=args.sample_seed,
    )
    results = [
        audit_partition(
            label=label,
            permutations=sample,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            age_map=age_map,
            device=device,
            batch_size=args.batch_size,
            continuation_loops=args.continuation_loops,
            operating_age=args.operating_age,
        )
        for label, sample in (
            ("seen_during_J_training", sampled_seen),
            ("strictly_unseen_by_J_training", sampled_unseen),
        )
    ]
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "J_artifact": str(args.j_artifact),
        "J_label": args.j_label,
        "graph_universe": len(universe),
        "training_graph_draws": total_draws,
        "training_streams": [
            {
                "label": label,
                "base_seed": base_seed,
                "stride": stride,
                "rounds": rounds,
                "batch_size": batch_size,
                "batches": batches,
            }
            for label, base_seed, stride, rounds, batch_size, batches in (
                training_streams
            )
        ],
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs": len(unseen),
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "per_partition": args.sample_per_partition,
            "all_starts_per_graph": cfg.node_count,
            "matched_cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "continuation_loops": args.continuation_loops,
        "operating_age": args.operating_age,
        "graph_steps_per_continuation_loop": (
            phase_positions[3] - phase_positions[2]
        ),
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
