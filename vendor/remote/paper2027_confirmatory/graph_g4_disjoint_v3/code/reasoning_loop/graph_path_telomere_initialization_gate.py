from __future__ import annotations

import argparse
import csv
import json
import os
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
from reasoning_loop.graph_path_telomere_localized_query import (
    _routing_metrics,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    _component_similarity,
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_initialization_gate(
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
    executor_head: int,
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
        raise ValueError("experiment is fixed to the D8L8 two-block model")
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
    age_map, interface = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    expected_interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if interface != expected_interface:
        raise ValueError("map positions do not match the Block2 interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    answer_graph = tuple(sorted(set(answer) | set(groups["graph"])))
    answer_position = answer[0]
    destination_positions = groups["destination"]
    conditions = (
        "raw_h8_no_control",
        "raw_h8_R_each_cycle",
        "raw_h8_R6_first_then_R",
        "raw_h8_exact_answer_first_then_R",
        "raw_h8_exact_answer_graph_first_then_R",
        "raw_h8_exact_interface_first_then_R",
        "raw_h8_shuffled_interface_first_then_R",
        "raw_h8_exact_interface_every_cycle",
        "exact_h2_R",
        "exact_h2_no_control",
    )
    rows: list[dict[str, Any]] = []
    set_seed(seed)
    for batch_index in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * extra_loops,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        raw_h8 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        exact_h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {
            condition: (
                exact_h2.clone()
                if condition.startswith("exact_h2")
                else raw_h8.clone()
            )
            for condition in conditions
        }
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
            oracle_values = oracle.block2_hidden_in[:, list(interface)]
            steps = {
                "raw_h8_no_control": run_one_loop(
                    model,
                    states["raw_h8_no_control"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
                "raw_h8_R_each_cycle": run_one_loop(
                    model,
                    states["raw_h8_R_each_cycle"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(interface, age_map),
                ),
                "raw_h8_R6_first_then_R": run_one_loop(
                    model,
                    states["raw_h8_R6_first_then_R"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (
                            interface,
                            lambda value: age_map.repeated(value, 6),
                        )
                        if cycle == 1
                        else (interface, age_map)
                    ),
                ),
                "raw_h8_exact_answer_first_then_R": run_one_loop(
                    model,
                    states["raw_h8_exact_answer_first_then_R"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (interface, age_map)
                    ),
                    block2_position_override=(
                        (
                            answer,
                            oracle.block2_hidden_in[:, list(answer)],
                        )
                        if cycle == 1
                        else None
                    ),
                ),
                "raw_h8_exact_answer_graph_first_then_R": run_one_loop(
                    model,
                    states["raw_h8_exact_answer_graph_first_then_R"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (interface, age_map)
                    ),
                    block2_position_override=(
                        (
                            answer_graph,
                            oracle.block2_hidden_in[
                                :, list(answer_graph)
                            ],
                        )
                        if cycle == 1
                        else None
                    ),
                ),
                "raw_h8_exact_interface_first_then_R": run_one_loop(
                    model,
                    states["raw_h8_exact_interface_first_then_R"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (interface, age_map)
                    ),
                    block2_position_override=(
                        (interface, oracle_values)
                        if cycle == 1
                        else None
                    ),
                ),
                "raw_h8_shuffled_interface_first_then_R": run_one_loop(
                    model,
                    states["raw_h8_shuffled_interface_first_then_R"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (interface, age_map)
                    ),
                    block2_position_override=(
                        (interface, oracle_values.roll(1, dims=0))
                        if cycle == 1
                        else None
                    ),
                ),
                "raw_h8_exact_interface_every_cycle": run_one_loop(
                    model,
                    states["raw_h8_exact_interface_every_cycle"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=(interface, oracle_values),
                ),
                "exact_h2_R": run_one_loop(
                    model,
                    states["exact_h2_R"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (interface, age_map)
                    ),
                ),
                "exact_h2_no_control": run_one_loop(
                    model,
                    states["exact_h2_no_control"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
            }
            for condition, step in steps.items():
                metrics = _masked_metrics(
                    step.logits,
                    target,
                    endpoint=endpoint,
                )
                mass, hit = _routing_metrics(
                    step.block2_pattern,
                    current=current,
                    destination_positions=destination_positions,
                    executor_head=executor_head,
                    answer_position=answer_position,
                )
                rows.append(
                    {
                        "batch": batch_index,
                        "cycle": cycle,
                        "condition": condition,
                        "accuracy": metrics["accuracy"],
                        "margin": metrics["margin"],
                        "valid_count": metrics["valid_count"],
                        "head2_correct_destination_mass": mass,
                        "head2_correct_destination_argmax": hit,
                        **_component_similarity(
                            step,
                            oracle,
                            answer_position=answer_position,
                            destination_positions=destination_positions,
                            executor_head=executor_head,
                        ),
                    }
                )
            states = {
                condition: steps[condition].state
                for condition in conditions
            }

    aggregate: dict[str, dict[str, Any]] = {}
    component_fields = (
        "head2_correct_destination_mass",
        "head2_correct_destination_argmax",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    for condition in conditions:
        accuracy: list[float] = []
        components = {field: [] for field in component_fields}
        for cycle in range(1, extra_loops + 1):
            parts = [
                row
                for row in rows
                if row["condition"] == condition and row["cycle"] == cycle
            ]
            count = sum(float(part["valid_count"]) for part in parts)
            accuracy.append(
                sum(
                    float(part["accuracy"]) * float(part["valid_count"])
                    for part in parts
                )
                / count
            )
            for field in component_fields:
                components[field].append(
                    float(np.mean([float(part[field]) for part in parts]))
                )
        aggregate[condition] = {
            "accuracy": accuracy,
            "auc": float(np.mean(accuracy)),
            "auc_8": float(np.mean(accuracy[:8])),
            "auc_16": float(np.mean(accuracy[:16])),
            "auc_32": float(np.mean(accuracy[:32])),
            **components,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "initialization_gate_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "map_rank": age_map.update_rank,
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "seed": seed,
        "closed_loop": aggregate,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "initialization_gate_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate natural H8-to-H2 initialization from subsequent "
            "R-composed feedback maintenance."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--map-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument("--seed", type=int, default=102101)
    parser.add_argument("--executor-head", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_initialization_gate(
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
        executor_head=args.executor_head,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
