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
from reasoning_loop.graph_path_telomere_shared_power_eval import (
    _load_map,
)
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
def evaluate_long_component_patches(
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
    probe_cycles: Sequence[int],
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
    probe_cycles = tuple(sorted({int(value) for value in probe_cycles}))
    if not probe_cycles or probe_cycles[0] < 1:
        raise ValueError("probe cycles must be positive")
    if probe_cycles[-1] > extra_loops:
        raise ValueError("probe cycle exceeds rollout horizon")
    age_map, positions = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if positions != interface:
        raise ValueError("map positions do not match the Block2 interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    jump = phase_positions[3] - phase_positions[2]
    conditions = (
        "baseline_map",
        "exact_answer_input",
        "exact_interface",
        "q_head2",
        "k_head2",
        "v_head2",
        "qkv_head2",
        "pattern_head2",
        "context_head2",
        "attention_out_answer",
        "mlp_out_answer",
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
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
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
            transform = (positions, age_map) if cycle > 1 else None
            baseline = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=transform,
            )
            if cycle in probe_cycles:
                answer_values = oracle.block2_hidden_in[
                    :, list(groups["answer"])
                ]
                interface_values = oracle.block2_hidden_in[
                    :, list(interface)
                ]
                patched = {
                    "baseline_map": baseline,
                    "exact_answer_input": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        block2_position_override=(
                            groups["answer"],
                            answer_values,
                        ),
                    ),
                    "exact_interface": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        block2_position_override=(
                            interface,
                            interface_values,
                        ),
                    ),
                    "q_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        query_override={
                            executor_head: oracle.block2_q[
                                :, executor_head, answer_position
                            ]
                        },
                    ),
                    "k_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        key_override={
                            executor_head: oracle.block2_k[
                                :, executor_head
                            ]
                        },
                    ),
                    "v_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        value_override={
                            executor_head: oracle.block2_v[
                                :, executor_head
                            ]
                        },
                    ),
                    "qkv_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        query_override={
                            executor_head: oracle.block2_q[
                                :, executor_head, answer_position
                            ]
                        },
                        key_override={
                            executor_head: oracle.block2_k[
                                :, executor_head
                            ]
                        },
                        value_override={
                            executor_head: oracle.block2_v[
                                :, executor_head
                            ]
                        },
                    ),
                    "pattern_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        pattern_answer_override={
                            executor_head: oracle.block2_pattern[
                                :, executor_head, answer_position
                            ]
                        },
                    ),
                    "context_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        context_answer_override={
                            executor_head: oracle.block2_head_context[
                                :, executor_head, answer_position
                            ]
                        },
                    ),
                    "attention_out_answer": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        attention_answer_override=(
                            oracle.block2_attention_out[:, answer_position]
                        ),
                    ),
                    "mlp_out_answer": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        mlp_answer_override=(
                            oracle.block2_mlp_out[:, answer_position]
                        ),
                    ),
                }
                for condition in conditions:
                    step = patched[condition]
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
            state = baseline.state

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    fields = (
        "accuracy",
        "margin",
        "head2_correct_destination_mass",
        "head2_correct_destination_argmax",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    for condition in conditions:
        aggregate[condition] = {}
        for cycle in probe_cycles:
            parts = [
                row
                for row in rows
                if row["condition"] == condition and row["cycle"] == cycle
            ]
            aggregate[condition][str(cycle)] = {
                field: float(
                    np.average(
                        [float(part[field]) for part in parts],
                        weights=(
                            [float(part["valid_count"]) for part in parts]
                            if field in {"accuracy", "margin"}
                            else None
                        ),
                    )
                )
                for field in fields
            }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "component_patch_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "map_rank": age_map.update_rank,
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "probe_cycles": list(probe_cycles),
        "seed": seed,
        "aggregate": aggregate,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "component_patch_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "At selected long-horizon failures of the learned shared R, patch "
            "individual Block2-head2 and MLP components from exact-young donors."
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
    parser.add_argument(
        "--probe-cycles",
        type=int,
        nargs="+",
        default=(16, 24, 32, 40, 48, 64),
    )
    parser.add_argument("--seed", type=int, default=96101)
    parser.add_argument("--executor-head", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_long_component_patches(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        map_artifact=args.map_artifact,
        map_label=args.map_label,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        probe_cycles=args.probe_cycles,
        seed=args.seed,
        executor_head=args.executor_head,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
