from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_observability_audit import (
    wilson_interval,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import _all_targets
from reasoning_loop.graph_path_telomere_per_node_component import (
    _component_values,
    sample_cosine,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def permutation_orbit_lengths(
    successors: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    if successors.ndim != 2:
        raise ValueError("successors must have shape [batch, node]")
    if start.shape != (successors.shape[0],):
        raise ValueError("start must have shape [batch]")
    current = start
    lengths = torch.zeros_like(start)
    for step in range(1, successors.shape[1] + 1):
        current = successors.gather(1, current[:, None]).squeeze(1)
        first_return = current.eq(start) & lengths.eq(0)
        lengths[first_return] = step
    if bool(lengths.eq(0).any()):
        raise ValueError("successors are not valid permutations")
    return lengths


def cohort_cycles(
    orbit_length: int,
    relative_phase: int,
    extra_loops: int,
    jump: int = 2,
) -> list[int]:
    return [
        cycle
        for cycle in range(1, extra_loops + 1)
        if (jump * cycle) % orbit_length == relative_phase
    ]


def exact_mcnemar_p(n10: int, n01: int) -> float:
    discordant = n10 + n01
    if discordant == 0:
        return 1.0
    tail = min(n10, n01)
    probability = sum(
        math.comb(discordant, value)
        for value in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_orbit_control(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    map_artifact: Path,
    map_label: str,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("orbit-control audit is fixed to D8L8")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    jump = phase_positions[3] - phase_positions[2]
    age_map, positions = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    expected_positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if positions != expected_positions:
        raise ValueError("map positions do not match the Block2 interface")

    conditions = ("no_control", "feedback_map", "exact_interface")
    component_fields = (
        "answer_input_cosine",
        "graph_input_cosine",
        "metadata_input_cosine",
        "interface_input_cosine",
        "q_cosine",
        "k_cosine",
        "v_cosine",
        "context_cosine",
        "attention_out_cosine",
        "residual_mid_cosine",
        "mlp_cosine",
    )
    orbit8_patch_conditions = (
        "feedback_baseline",
        "q_head2",
        "qkv_head2",
        "context_head2",
        "mlp_out_answer",
        "exact_interface",
    )
    count_buckets: dict[tuple[str, int, int, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    component_buckets: dict[
        tuple[int, int, int, str], list[float | int]
    ] = defaultdict(lambda: [0.0, 0])
    orbit8_patch_buckets: dict[
        tuple[str, int, int], list[int]
    ] = defaultdict(lambda: [0, 0])
    pair_buckets: dict[tuple[str, int, int], list[int]] = defaultdict(
        lambda: [0, 0, 0, 0]
    )
    cohort_schedule: dict[tuple[int, int], list[int]] = {}
    for orbit_length in range(1, cfg.node_count + 1):
        for relative_phase in range(orbit_length):
            cycles = cohort_cycles(
                orbit_length,
                relative_phase,
                extra_loops,
                jump=jump,
            )
            if relative_phase != 0 and len(cycles) >= 2:
                cohort_schedule[(orbit_length, relative_phase)] = cycles
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    executor_head = 2

    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * extra_loops,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        orbit_lengths = permutation_orbit_lengths(successors, endpoint)
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {condition: initial.clone() for condition in conditions}
        first_correct: dict[
            tuple[str, int, int], torch.Tensor
        ] = {}
        for cycle in range(1, extra_loops + 1):
            current = all_targets[
                :, cfg.max_depth + jump * (cycle - 1)
            ]
            target = all_targets[:, cfg.max_depth + jump * cycle]
            oracle_input = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=2,
                phase_position=phase_positions[2],
            )
            oracle = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            steps = {
                "no_control": run_one_loop(
                    model,
                    states["no_control"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
                "feedback_map": run_one_loop(
                    model,
                    states["feedback_map"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, age_map) if cycle > 1 else None
                    ),
                ),
                "exact_interface": run_one_loop(
                    model,
                    states["exact_interface"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=(
                        None
                        if cycle == 1
                        else (
                            positions,
                            oracle.block2_hidden_in[:, list(positions)],
                        )
                    ),
                ),
            }
            predictions = {
                condition: step.logits.argmax(dim=-1)
                for condition, step in steps.items()
            }
            component_values = _component_values(
                steps["feedback_map"],
                oracle,
                answer_position=answer_position,
                destination_positions=destination_positions,
            )
            feedback_step = steps["feedback_map"]
            role_positions = {
                "answer_input_cosine": groups["answer"],
                "graph_input_cosine": groups["graph"],
                "metadata_input_cosine": groups["query_metadata"],
                "interface_input_cosine": positions,
            }
            for field, role in role_positions.items():
                component_values[field] = sample_cosine(
                    feedback_step.block2_hidden_in[:, list(role)],
                    oracle.block2_hidden_in[:, list(role)],
                )
            component_values["attention_out_cosine"] = sample_cosine(
                feedback_step.block2_attention_out[:, answer_position],
                oracle.block2_attention_out[:, answer_position],
            )
            component_values["residual_mid_cosine"] = sample_cosine(
                feedback_step.block2_residual_mid[:, answer_position],
                oracle.block2_residual_mid[:, answer_position],
            )
            for (orbit_length, relative_phase), cycles in (
                cohort_schedule.items()
            ):
                if (jump * cycle) % orbit_length != relative_phase:
                    continue
                selected = orbit_lengths.eq(orbit_length)
                count = int(selected.sum())
                for condition, prediction in predictions.items():
                    correct_values = prediction[selected].eq(target[selected])
                    bucket = count_buckets[
                        (
                            condition,
                            cycle,
                            orbit_length,
                            relative_phase,
                        )
                    ]
                    bucket[0] += int(correct_values.sum())
                    bucket[1] += count
                    pair_key = (condition, orbit_length, relative_phase)
                    if cycle == cycles[0]:
                        first_correct[pair_key] = correct_values
                    elif cycle == cycles[-1]:
                        initial_correct = first_correct[pair_key]
                        pair = pair_buckets[pair_key]
                        pair[0] += int(
                            (initial_correct & correct_values).sum()
                        )
                        pair[1] += int(
                            (initial_correct & ~correct_values).sum()
                        )
                        pair[2] += int(
                            (~initial_correct & correct_values).sum()
                        )
                        pair[3] += int(
                            (~initial_correct & ~correct_values).sum()
                        )
                for field, values in component_values.items():
                    component_bucket = component_buckets[
                        (
                            cycle,
                            orbit_length,
                            relative_phase,
                            field,
                        )
                    ]
                    component_bucket[0] += float(values[selected].sum())
                    component_bucket[1] += count

            orbit8_phase = (jump * cycle) % cfg.node_count
            orbit8_selected = orbit_lengths.eq(cfg.node_count)
            orbit8_count = int(orbit8_selected.sum())
            if orbit8_phase != 0 and orbit8_count:
                feedback_state = states["feedback_map"][orbit8_selected]
                oracle_q = oracle.block2_q[
                    orbit8_selected,
                    executor_head,
                    answer_position,
                ]
                oracle_k = oracle.block2_k[
                    orbit8_selected,
                    executor_head,
                ]
                oracle_v = oracle.block2_v[
                    orbit8_selected,
                    executor_head,
                ]
                orbit8_steps = {
                    "feedback_baseline": steps["feedback_map"],
                    "q_head2": run_one_loop(
                        model,
                        feedback_state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            (positions, age_map) if cycle > 1 else None
                        ),
                        query_override={executor_head: oracle_q},
                    ),
                    "qkv_head2": run_one_loop(
                        model,
                        feedback_state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            (positions, age_map) if cycle > 1 else None
                        ),
                        query_override={executor_head: oracle_q},
                        key_override={executor_head: oracle_k},
                        value_override={executor_head: oracle_v},
                    ),
                    "context_head2": run_one_loop(
                        model,
                        feedback_state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            (positions, age_map) if cycle > 1 else None
                        ),
                        context_answer_override={
                            executor_head: oracle.block2_head_context[
                                orbit8_selected,
                                executor_head,
                                answer_position,
                            ]
                        },
                    ),
                    "mlp_out_answer": run_one_loop(
                        model,
                        feedback_state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            (positions, age_map) if cycle > 1 else None
                        ),
                        mlp_answer_override=oracle.block2_mlp_out[
                            orbit8_selected,
                            answer_position,
                        ],
                    ),
                    "exact_interface": run_one_loop(
                        model,
                        feedback_state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            (positions, age_map) if cycle > 1 else None
                        ),
                        block2_position_override=(
                            positions,
                            oracle.block2_hidden_in[
                                orbit8_selected
                            ][:, list(positions)],
                        ),
                    ),
                }
                orbit8_target = target[orbit8_selected]
                for condition in orbit8_patch_conditions:
                    step = orbit8_steps[condition]
                    if condition == "feedback_baseline":
                        prediction = step.logits[
                            orbit8_selected
                        ].argmax(dim=-1)
                    else:
                        prediction = step.logits.argmax(dim=-1)
                    correct = int(prediction.eq(orbit8_target).sum())
                    patch_bucket = orbit8_patch_buckets[
                        condition,
                        cycle,
                        orbit8_phase,
                    ]
                    patch_bucket[0] += correct
                    patch_bucket[1] += orbit8_count
            for condition in conditions:
                states[condition] = steps[condition].state

    curve_rows: list[dict[str, Any]] = []
    for (orbit_length, relative_phase), cycles in cohort_schedule.items():
        for visit_index, cycle in enumerate(cycles, start=1):
            for condition in conditions:
                correct, count = count_buckets[
                    condition,
                    cycle,
                    orbit_length,
                    relative_phase,
                ]
                accuracy = correct / count if count else float("nan")
                low, high = wilson_interval(correct, count)
                curve_rows.append(
                    {
                        "condition": condition,
                        "orbit_length": orbit_length,
                        "relative_phase": relative_phase,
                        "cycle": cycle,
                        "visit_index": visit_index,
                        "accuracy": accuracy,
                        "ci95_low": low,
                        "ci95_high": high,
                        "correct": correct,
                        "count": count,
                    }
                )

    component_rows: list[dict[str, Any]] = []
    for (orbit_length, relative_phase), cycles in cohort_schedule.items():
        for visit_index, cycle in enumerate(cycles, start=1):
            row: dict[str, Any] = {
                "orbit_length": orbit_length,
                "relative_phase": relative_phase,
                "cycle": cycle,
                "visit_index": visit_index,
            }
            for field in component_fields:
                total, count = component_buckets[
                    cycle,
                    orbit_length,
                    relative_phase,
                    field,
                ]
                row[field] = total / count if count else float("nan")
            component_rows.append(row)

    orbit8_patch_rows: list[dict[str, Any]] = []
    for cycle in range(1, extra_loops + 1):
        relative_phase = (jump * cycle) % cfg.node_count
        if relative_phase == 0:
            continue
        visits = cohort_cycles(
            cfg.node_count,
            relative_phase,
            extra_loops,
            jump=jump,
        )
        visit_index = visits.index(cycle) + 1
        for condition in orbit8_patch_conditions:
            correct, count = orbit8_patch_buckets[
                condition,
                cycle,
                relative_phase,
            ]
            accuracy = correct / count if count else float("nan")
            low, high = wilson_interval(correct, count)
            orbit8_patch_rows.append(
                {
                    "condition": condition,
                    "orbit_length": cfg.node_count,
                    "relative_phase": relative_phase,
                    "cycle": cycle,
                    "visit_index": visit_index,
                    "accuracy": accuracy,
                    "ci95_low": low,
                    "ci95_high": high,
                    "correct": correct,
                    "count": count,
                }
            )

    pair_rows: list[dict[str, Any]] = []
    for (orbit_length, relative_phase), cycles in cohort_schedule.items():
        for condition in conditions:
            n11, n10, n01, n00 = pair_buckets[
                condition,
                orbit_length,
                relative_phase,
            ]
            count = n11 + n10 + n01 + n00
            first_accuracy = (n11 + n10) / count
            last_accuracy = (n11 + n01) / count
            pair_rows.append(
                {
                    "condition": condition,
                    "orbit_length": orbit_length,
                    "relative_phase": relative_phase,
                    "first_cycle": cycles[0],
                    "last_cycle": cycles[-1],
                    "visits": len(cycles),
                    "count": count,
                    "first_accuracy": first_accuracy,
                    "last_accuracy": last_accuracy,
                    "last_minus_first": last_accuracy - first_accuracy,
                    "n11": n11,
                    "n10_first_only": n10,
                    "n01_last_only": n01,
                    "n00": n00,
                    "mcnemar_exact_p": exact_mcnemar_p(n10, n01),
                }
            )

    condition_summaries: list[dict[str, Any]] = []
    for condition in conditions:
        selected = [
            row for row in pair_rows if row["condition"] == condition
        ]
        weights = np.asarray([row["count"] for row in selected], dtype=float)
        first = np.asarray(
            [row["first_accuracy"] for row in selected], dtype=float
        )
        last = np.asarray(
            [row["last_accuracy"] for row in selected], dtype=float
        )
        condition_summaries.append(
            {
                "condition": condition,
                "cohorts": len(selected),
                "weighted_first_accuracy": float(
                    np.average(first, weights=weights)
                ),
                "weighted_last_accuracy": float(
                    np.average(last, weights=weights)
                ),
                "weighted_last_minus_first": float(
                    np.average(last - first, weights=weights)
                ),
                "declining_cohorts": sum(
                    row["last_minus_first"] < 0 for row in selected
                ),
                "significant_declining_cohorts_p_lt_0.05": sum(
                    row["last_minus_first"] < 0
                    and row["mcnemar_exact_p"] < 0.05
                    for row in selected
                ),
            }
        )

    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "seed": seed,
        "jump": jump,
        "condition_summaries": condition_summaries,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "orbit_control_curves.csv", curve_rows)
    _write_csv(out_dir / "orbit_control_paired.csv", pair_rows)
    _write_csv(
        out_dir / "orbit_control_component_curves.csv",
        component_rows,
    )
    _write_csv(
        out_dir / "orbit8_component_patch_curves.csv",
        orbit8_patch_rows,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Separate D8L8 hidden-state aging from permutation-orbit phase "
            "by comparing repeated visits to the same semantic target."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--map-label", default="w28_r256_round4")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument("--seed", type=int, default=120101)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = evaluate_orbit_control(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        map_artifact=args.map_artifact,
        map_label=args.map_label,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
