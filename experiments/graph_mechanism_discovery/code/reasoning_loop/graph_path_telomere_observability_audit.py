from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


def wilson_interval(
    correct: int,
    count: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    proportion = correct / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    low = 0.0 if correct == 0 else max(0.0, center - radius)
    high = 1.0 if correct == count else min(1.0, center + radius)
    return low, high


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _empty_bucket() -> dict[str, float]:
    return {
        "correct": 0.0,
        "count": 0.0,
        "probability_sum": 0.0,
        "margin_sum": 0.0,
    }


def _update_bucket(
    bucket: dict[str, float],
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    if not bool(valid.any()):
        return
    selected_logits = logits[valid].float()
    selected_target = target[valid]
    prediction = selected_logits.argmax(dim=-1)
    bucket["correct"] += float(prediction.eq(selected_target).sum())
    bucket["count"] += float(selected_target.numel())
    bucket["probability_sum"] += float(
        selected_logits.softmax(dim=-1)
        .gather(1, selected_target[:, None])
        .sum()
    )
    bucket["margin_sum"] += float(
        target_margin(selected_logits, selected_target).sum()
    )


def _finish_bucket(
    bucket: dict[str, float],
) -> dict[str, float | int]:
    count = int(bucket["count"])
    correct = int(bucket["correct"])
    low, high = wilson_interval(correct, count)
    return {
        "accuracy": correct / count if count else float("nan"),
        "ci95_low": low,
        "ci95_high": high,
        "mean_probability": (
            bucket["probability_sum"] / count
            if count
            else float("nan")
        ),
        "mean_margin": (
            bucket["margin_sum"] / count
            if count
            else float("nan")
        ),
        "correct": correct,
        "valid_count": count,
    }


@torch.no_grad()
def run_observability_audit(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    one_step_operator_path: Path,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
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
        raise ValueError("observability audit is fixed to D8L8")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    if len(phase_positions) != cfg.max_loops + 1:
        raise ValueError("phase trajectory length does not match model")
    jump = phase_positions[3] - phase_positions[2]
    if jump <= 0:
        raise ValueError("selected executor phase must make progress")

    natural = {
        age: _empty_bucket()
        for age in range(1, cfg.max_loops + 1)
    }
    natural_per_node = {
        (age, node): _empty_bucket()
        for age in range(1, cfg.max_loops + 1)
        for node in range(cfg.node_count)
    }
    aligned_readout = _empty_bucket()
    executor = _empty_bucket()
    executor_per_node = {
        node: _empty_bucket() for node in range(cfg.node_count)
    }
    circuit_conditions = (
        "baseline",
        "oracle_full_state",
        "q_head2",
        "q_head2_shuffled",
        "k_head2",
        "v_head2",
        "qkv_head2",
        "pattern_head2",
        "context_head2",
        "attention_out_answer",
        "mlp_out_answer",
        "exact_interface",
    )
    circuit_trajectories = (
        "raw_no_control",
        "one_step_answer_map",
    )
    circuit = {
        (trajectory, condition): _empty_bucket()
        for trajectory in circuit_trajectories
        for condition in circuit_conditions
    }
    circuit_per_node = {
        (trajectory, condition, node): _empty_bucket()
        for trajectory in circuit_trajectories
        for condition in circuit_conditions
        for node in range(cfg.node_count)
    }
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    operator_payload = torch.load(
        one_step_operator_path,
        map_location=device,
        weights_only=False,
    )
    if operator_payload.get("kind") != "graph_path_simple_one_step_telomere":
        raise ValueError("one-step operator artifact has the wrong kind")
    one_step_weight = operator_payload["weight"].to(device).float()
    one_step_bias = operator_payload["bias"].to(device).float()
    if one_step_weight.shape != (cfg.d_model, cfg.d_model):
        raise ValueError("one-step operator has the wrong weight shape")
    if one_step_bias.shape != (cfg.d_model,):
        raise ValueError("one-step operator has the wrong bias shape")

    set_seed(seed)
    for _ in range(batches):
        tokens, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 3 * jump,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        states = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )
        for age in range(1, cfg.max_loops + 1):
            position = phase_positions[age]
            target = all_targets[:, position]
            valid = (
                torch.ones_like(target, dtype=torch.bool)
                if position == cfg.max_depth
                else target.ne(endpoint)
            )
            logits = logits_from_raw_state(model, states[age])
            _update_bucket(natural[age], logits, target, valid)
            for node in range(cfg.node_count):
                _update_bucket(
                    natural_per_node[(age, node)],
                    logits,
                    target,
                    valid & target.eq(node),
                )

        current = endpoint
        aligned = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        current_logits = logits_from_raw_state(model, aligned)
        _update_bucket(
            aligned_readout,
            current_logits,
            current,
            torch.ones_like(current, dtype=torch.bool),
        )
        step = run_one_loop(
            model,
            aligned,
            loop_index=cfg.max_loops,
        )
        next_target = advance_nodes(successors, current, steps=jump)
        valid = next_target.ne(current)
        _update_bucket(executor, step.logits, next_target, valid)
        for node in range(cfg.node_count):
            _update_bucket(
                executor_per_node[node],
                step.logits,
                next_target,
                valid & next_target.eq(node),
            )

        raw_second_step = run_one_loop(
            model,
            step.state,
            loop_index=cfg.max_loops + 1,
        )
        raw_aged_state = raw_second_step.state
        controlled_after_first = step.state.clone()
        controlled_after_first[:, answer_position] = (
            controlled_after_first[:, answer_position].float()
            @ one_step_weight
            + one_step_bias
        ).to(dtype=controlled_after_first.dtype)
        controlled_second_step = run_one_loop(
            model,
            controlled_after_first,
            loop_index=cfg.max_loops + 1,
        )
        controlled_aged_state = controlled_second_step.state.clone()
        controlled_aged_state[:, answer_position] = (
            controlled_aged_state[:, answer_position].float()
            @ one_step_weight
            + one_step_bias
        ).to(dtype=controlled_aged_state.dtype)
        cycle3_current = advance_nodes(
            successors,
            current,
            steps=2 * jump,
        )
        cycle3_target = advance_nodes(
            successors,
            current,
            steps=3 * jump,
        )
        cycle3_oracle_input = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=cycle3_current,
            age=2,
            phase_position=phase_positions[2],
        )
        oracle = run_one_loop(
            model,
            cycle3_oracle_input,
            loop_index=cfg.max_loops + 2,
        )
        oracle_q = oracle.block2_q[:, 2, answer_position]
        def component_steps(
            aged_state: torch.Tensor,
        ) -> dict[str, Any]:
            return {
                "baseline": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                ),
                "oracle_full_state": oracle,
                "q_head2": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    query_override={2: oracle_q},
                ),
                "q_head2_shuffled": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    query_override={2: oracle_q.roll(1, dims=0)},
                ),
                "k_head2": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    key_override={2: oracle.block2_k[:, 2]},
                ),
                "v_head2": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    value_override={2: oracle.block2_v[:, 2]},
                ),
                "qkv_head2": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    query_override={2: oracle_q},
                    key_override={2: oracle.block2_k[:, 2]},
                    value_override={2: oracle.block2_v[:, 2]},
                ),
                "pattern_head2": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    pattern_answer_override={
                        2: oracle.block2_pattern[:, 2, answer_position]
                    },
                ),
                "context_head2": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    context_answer_override={
                        2: oracle.block2_head_context[
                            :, 2, answer_position
                        ]
                    },
                ),
                "attention_out_answer": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    attention_answer_override=(
                        oracle.block2_attention_out[:, answer_position]
                    ),
                ),
                "mlp_out_answer": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    mlp_answer_override=(
                        oracle.block2_mlp_out[:, answer_position]
                    ),
                ),
                "exact_interface": run_one_loop(
                    model,
                    aged_state,
                    loop_index=cfg.max_loops + 2,
                    block2_position_override=(
                        interface,
                        oracle.block2_hidden_in[:, list(interface)],
                    ),
                ),
            }

        cycle3_valid = cycle3_target.ne(endpoint)
        trajectory_states = {
            "raw_no_control": raw_aged_state,
            "one_step_answer_map": controlled_aged_state,
        }
        for trajectory, aged_state in trajectory_states.items():
            for condition, condition_step in component_steps(
                aged_state
            ).items():
                _update_bucket(
                    circuit[(trajectory, condition)],
                    condition_step.logits,
                    cycle3_target,
                    cycle3_valid,
                )
                for node in range(cfg.node_count):
                    _update_bucket(
                        circuit_per_node[
                            (trajectory, condition, node)
                        ],
                        condition_step.logits,
                        cycle3_target,
                        cycle3_valid & cycle3_target.eq(node),
                    )

    natural_rows = []
    for age in range(1, cfg.max_loops + 1):
        natural_rows.append(
            {
                "age": age,
                "expected_path_position": phase_positions[age],
                "collision_controlled": (
                    phase_positions[age] != cfg.max_depth
                ),
                **_finish_bucket(natural[age]),
            }
        )
    per_node_rows = []
    for age in range(1, cfg.max_loops + 1):
        for node in range(cfg.node_count):
            per_node_rows.append(
                {
                    "metric": "natural_readout",
                    "age": age,
                    "expected_path_position": phase_positions[age],
                    "target_node": node,
                    **_finish_bucket(natural_per_node[(age, node)]),
                }
            )
    for node in range(cfg.node_count):
        per_node_rows.append(
            {
                "metric": "age2_executor",
                "age": 2,
                "expected_path_position": (
                    phase_positions[2] + jump
                ),
                "target_node": node,
                **_finish_bucket(executor_per_node[node]),
            }
        )
    circuit_rows = [
        {
            "trajectory": trajectory,
            "condition": condition,
            **_finish_bucket(circuit[(trajectory, condition)]),
        }
        for trajectory in circuit_trajectories
        for condition in circuit_conditions
    ]
    circuit_per_node_rows = [
        {
            "trajectory": trajectory,
            "condition": condition,
            "target_node": node,
            **_finish_bucket(
                circuit_per_node[(trajectory, condition, node)]
            ),
        }
        for trajectory in circuit_trajectories
        for condition in circuit_conditions
        for node in range(cfg.node_count)
    ]

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "one_step_operator": str(one_step_operator_path),
        "model": {
            "max_depth": cfg.max_depth,
            "max_loops": cfg.max_loops,
            "physical_blocks": cfg.n_layers,
            "d_model": cfg.d_model,
            "node_count": cfg.node_count,
        },
        "seed": seed,
        "sample_size": batch_size * batches,
        "phase_positions": phase_positions,
        "natural_readout": natural_rows,
        "matched_age2_current_readout": _finish_bucket(
            aligned_readout
        ),
        "matched_age2_next_executor": {
            "jump": jump,
            "collision_controlled": True,
            **_finish_bucket(executor),
        },
        "executor_per_node_min_accuracy": min(
            float(_finish_bucket(bucket)["accuracy"])
            for bucket in executor_per_node.values()
        ),
        "executor_per_node_max_accuracy": max(
            float(_finish_bucket(bucket)["accuracy"])
            for bucket in executor_per_node.values()
        ),
        "cycle3_component_circuit": circuit_rows,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "natural_readout_rows.csv", natural_rows)
    _write_csv(out_dir / "per_node_rows.csv", per_node_rows)
    _write_csv(out_dir / "cycle3_component_rows.csv", circuit_rows)
    _write_csv(
        out_dir / "cycle3_component_per_node_rows.csv",
        circuit_per_node_rows,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether D8L8 intermediate states and the selected "
            "two-hop executor are observable above the measurement floor."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--one-step-operator", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=114101)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_observability_audit(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        one_step_operator_path=args.one_step_operator,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
