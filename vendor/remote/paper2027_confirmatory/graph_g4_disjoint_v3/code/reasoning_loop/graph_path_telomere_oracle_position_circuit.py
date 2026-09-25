from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    LoopStep,
    _routing_metrics,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
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


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().flatten(1),
            right.float().flatten(1),
            dim=-1,
        ).mean()
    )


def _position_union(*parts: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({position for part in parts for position in part}))


def intervention_groups(node_count: int) -> dict[str, tuple[int, ...]]:
    groups = explicit_depth_position_groups(node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    metadata = groups["query_metadata"]
    all_positions = tuple(range(1 + 3 * node_count + 4))
    def complement(part: tuple[int, ...]) -> tuple[int, ...]:
        excluded = set(part)
        return tuple(
            position
            for position in all_positions
            if position not in excluded
        )

    return {
        "answer": answer,
        "destination_only": groups["destination"],
        "answer_edge": _position_union(answer, groups["edge_marker"]),
        "answer_source": _position_union(answer, groups["source"]),
        "answer_destination": _position_union(
            answer,
            groups["destination"],
        ),
        "answer_metadata": _position_union(answer, metadata),
        "answer_graph": _position_union(answer, graph),
        "answer_graph_metadata": _position_union(answer, graph, metadata),
        "all_except_answer": complement(answer),
        "all_except_graph": complement(graph),
        "all_except_metadata": complement(metadata),
        "all_except_edge": complement(groups["edge_marker"]),
        "all_except_source": complement(groups["source"]),
        "all_except_destination": complement(groups["destination"]),
        "all": all_positions,
    }


def _component_similarity(
    step: LoopStep,
    oracle: LoopStep,
    *,
    answer_position: int,
    destination_positions: tuple[int, ...],
    executor_head: int,
) -> dict[str, float]:
    destinations = list(destination_positions)
    return {
        "head2_q_answer_cosine": _cosine(
            step.block2_q[:, executor_head, answer_position],
            oracle.block2_q[:, executor_head, answer_position],
        ),
        "head2_k_destination_cosine": _cosine(
            step.block2_k[:, executor_head, destinations],
            oracle.block2_k[:, executor_head, destinations],
        ),
        "head2_v_destination_cosine": _cosine(
            step.block2_v[:, executor_head, destinations],
            oracle.block2_v[:, executor_head, destinations],
        ),
        "head2_context_answer_cosine": _cosine(
            step.block2_head_context[
                :, executor_head, answer_position
            ],
            oracle.block2_head_context[
                :, executor_head, answer_position
            ],
        ),
        "block2_mlp_answer_cosine": _cosine(
            step.block2_mlp_out[:, answer_position],
            oracle.block2_mlp_out[:, answer_position],
        ),
    }


@torch.no_grad()
def evaluate_oracle_position_circuit(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
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
    groups = explicit_depth_position_groups(cfg.node_count)
    patch_groups = intervention_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    condition_names = (
        "no_intervention",
        "oracle_full_state",
        *[f"b2_{name}" for name in patch_groups],
        "b2_answer_destination_shuffled",
        "b2_answer_graph_shuffled",
        "loop_answer",
        "loop_answer_graph",
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
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {
            condition: initial.clone()
            for condition in condition_names
            if condition != "oracle_full_state"
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
            oracle_step = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            steps: dict[str, LoopStep] = {
                "oracle_full_state": oracle_step,
                "no_intervention": run_one_loop(
                    model,
                    states["no_intervention"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
            }
            for name, positions in patch_groups.items():
                condition = f"b2_{name}"
                override = (
                    None
                    if cycle == 1
                    else (
                        positions,
                        oracle_step.block2_hidden_in[:, list(positions)],
                    )
                )
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=override,
                )
            for base_name in ("answer_destination", "answer_graph"):
                condition = f"b2_{base_name}_shuffled"
                positions = patch_groups[base_name]
                override = (
                    None
                    if cycle == 1
                    else (
                        positions,
                        oracle_step.block2_hidden_in[
                            :, list(positions)
                        ].roll(1, dims=0),
                    )
                )
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=override,
                )
            for condition, positions in (
                ("loop_answer", patch_groups["answer"]),
                ("loop_answer_graph", patch_groups["answer_graph"]),
            ):
                override = (
                    None
                    if cycle == 1
                    else (
                        positions,
                        oracle_input[:, list(positions)],
                    )
                )
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    loop_input_position_override=override,
                )
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
                            oracle_step,
                            answer_position=answer_position,
                            destination_positions=destination_positions,
                            executor_head=executor_head,
                        ),
                    }
                )
            for condition in states:
                states[condition] = steps[condition].state

    curves: dict[str, dict[str, Any]] = {}
    for condition in condition_names:
        accuracy: list[float] = []
        component_curves: dict[str, list[float]] = {
            key: []
            for key in (
                "head2_correct_destination_mass",
                "head2_correct_destination_argmax",
                "head2_q_answer_cosine",
                "head2_k_destination_cosine",
                "head2_v_destination_cosine",
                "head2_context_answer_cosine",
                "block2_mlp_answer_cosine",
            )
        }
        for cycle in range(1, extra_loops + 1):
            parts = [
                row
                for row in rows
                if row["condition"] == condition
                and int(row["cycle"]) == cycle
            ]
            count = sum(float(part["valid_count"]) for part in parts)
            accuracy.append(
                sum(
                    float(part["accuracy"]) * float(part["valid_count"])
                    for part in parts
                )
                / count
            )
            for field, values in component_curves.items():
                values.append(
                    float(np.mean([float(part[field]) for part in parts]))
                )
        curves[condition] = {
            "accuracy": accuracy,
            "auc": float(np.mean(accuracy)),
            "auc_4": float(np.mean(accuracy[:4])),
            "auc_8": float(np.mean(accuracy[:8])),
            "auc_16": float(np.mean(accuracy[:16])),
            **component_curves,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "position_circuit_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": {
            "node_count": cfg.node_count,
            "max_depth": cfg.max_depth,
            "max_loops": cfg.max_loops,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
        },
        "device": str(device),
        "phase_positions": phase_positions,
        "evaluation_graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "intervention_site": (
            "exact same-graph/current H2 donor at Block2 input unless "
            "condition starts with loop_"
        ),
        "position_groups": {
            name: list(positions)
            for name, positions in patch_groups.items()
        },
        "curves": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "position_circuit_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Localize long-horizon D8L8 telomere state by exact-young "
            "position-group interventions."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=32)
    parser.add_argument("--seed", type=int, default=78101)
    parser.add_argument("--executor-head", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_oracle_position_circuit(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
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
