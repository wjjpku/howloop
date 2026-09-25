from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

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
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_observability_audit import (
    wilson_interval,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import _all_targets
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


def sample_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.shape[0] == 0:
        raise ValueError("cosine inputs must be nonempty and shape matched")
    return F.cosine_similarity(
        left.float().flatten(1),
        right.float().flatten(1),
        dim=-1,
    )


def _component_values(
    step: LoopStep,
    oracle: LoopStep,
    *,
    answer_position: int,
    destination_positions: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    destinations = list(destination_positions)
    return {
        "q_cosine": sample_cosine(
            step.block2_q[:, 2, answer_position],
            oracle.block2_q[:, 2, answer_position],
        ),
        "k_cosine": sample_cosine(
            step.block2_k[:, 2, destinations],
            oracle.block2_k[:, 2, destinations],
        ),
        "v_cosine": sample_cosine(
            step.block2_v[:, 2, destinations],
            oracle.block2_v[:, 2, destinations],
        ),
        "context_cosine": sample_cosine(
            step.block2_head_context[:, 2, answer_position],
            oracle.block2_head_context[:, 2, answer_position],
        ),
        "mlp_cosine": sample_cosine(
            step.block2_mlp_out[:, answer_position],
            oracle.block2_mlp_out[:, answer_position],
        ),
    }


@torch.no_grad()
def evaluate_per_node_components(
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
        raise ValueError("per-node component audit is fixed to D8L8")
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
    expected_positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if positions != expected_positions:
        raise ValueError("map positions do not match the Block2 interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    conditions = (
        "feedback_baseline",
        "q_head2",
        "context_head2",
        "attention_out_answer",
        "mlp_out_answer",
        "exact_interface",
        "oracle_full_state",
    )
    count_buckets = {
        (condition, cycle, node): [0, 0]
        for condition in conditions
        for cycle in probe_cycles
        for node in range(cfg.node_count)
    }
    component_fields = (
        "q_cosine",
        "k_cosine",
        "v_cosine",
        "context_cosine",
        "mlp_cosine",
    )
    component_buckets = {
        (cycle, node, field): [0.0, 0]
        for cycle in probe_cycles
        for node in range(cfg.node_count)
        for field in component_fields
    }

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
            valid = target.ne(endpoint)
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
                patched = {
                    "feedback_baseline": baseline,
                    "q_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        query_override={
                            2: oracle.block2_q[:, 2, answer_position]
                        },
                    ),
                    "context_head2": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        context_answer_override={
                            2: oracle.block2_head_context[
                                :, 2, answer_position
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
                    "exact_interface": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=transform,
                        block2_position_override=(
                            positions,
                            oracle.block2_hidden_in[:, list(positions)],
                        ),
                    ),
                    "oracle_full_state": oracle,
                }
                components = _component_values(
                    baseline,
                    oracle,
                    answer_position=answer_position,
                    destination_positions=destination_positions,
                )
                for node in range(cfg.node_count):
                    selected = valid & target.eq(node)
                    count = int(selected.sum())
                    for condition, step in patched.items():
                        prediction = step.logits.argmax(dim=-1)
                        correct = int(
                            prediction[selected]
                            .eq(target[selected])
                            .sum()
                        )
                        bucket = count_buckets[(condition, cycle, node)]
                        bucket[0] += correct
                        bucket[1] += count
                    for field, values in components.items():
                        bucket = component_buckets[(cycle, node, field)]
                        bucket[0] += float(values[selected].sum())
                        bucket[1] += count
            state = baseline.state

    patch_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for cycle in probe_cycles:
            for node in range(cfg.node_count):
                correct, count = count_buckets[(condition, cycle, node)]
                accuracy = correct / count if count else float("nan")
                low, high = wilson_interval(correct, count)
                patch_rows.append(
                    {
                        "condition": condition,
                        "cycle": cycle,
                        "target_node": node,
                        "accuracy": accuracy,
                        "ci95_low": low,
                        "ci95_high": high,
                        "correct": correct,
                        "valid_count": count,
                    }
                )
    component_rows: list[dict[str, Any]] = []
    for cycle in probe_cycles:
        for node in range(cfg.node_count):
            row: dict[str, Any] = {
                "cycle": cycle,
                "target_node": node,
            }
            for field in component_fields:
                total, count = component_buckets[(cycle, node, field)]
                row[field] = total / count if count else float("nan")
            component_rows.append(row)

    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "probe_cycles": list(probe_cycles),
        "seed": seed,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_node_component_patch_rows.csv", patch_rows)
    _write_csv(out_dir / "per_node_component_similarity_rows.csv", component_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the component failure of difficult target nodes under "
            "the D8L8 shared rejuvenation feedback map."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--map-label", default="w28_r256_round4")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--extra-loops", type=int, default=32)
    parser.add_argument(
        "--probe-cycles",
        type=int,
        nargs="+",
        default=(2, 4, 8, 16, 32),
    )
    parser.add_argument("--seed", type=int, default=118101)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = evaluate_per_node_components(
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
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
