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
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
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


def prefix_lifespan(curve: Sequence[float], threshold: float) -> int:
    for index, value in enumerate(curve):
        if value < threshold:
            return index
    return len(curve)


def relative_prefix_lifespan(
    curve: Sequence[float],
    oracle: Sequence[float],
    fraction: float,
) -> int:
    if len(curve) != len(oracle):
        raise ValueError("curve and oracle must have equal length")
    return prefix_lifespan(
        [
            value / max(reference, 1e-12)
            for value, reference in zip(curve, oracle, strict=True)
        ],
        fraction,
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
def evaluate_per_node_lifespan(
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
        raise ValueError("per-node lifespan audit is fixed to D8L8")
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
    counts = {
        (condition, cycle, node): [0, 0]
        for condition in conditions
        for cycle in range(1, extra_loops + 1)
        for node in range(cfg.node_count)
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
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {
            condition: initial.clone() for condition in conditions
        }
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
            for condition, step in steps.items():
                prediction = step.logits.argmax(dim=-1)
                for node in range(cfg.node_count):
                    selected = valid & target.eq(node)
                    count = int(selected.sum())
                    correct = int(
                        prediction[selected].eq(target[selected]).sum()
                    )
                    bucket = counts[(condition, cycle, node)]
                    bucket[0] += correct
                    bucket[1] += count
            for condition in conditions:
                states[condition] = steps[condition].state

    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[int, list[float]]] = {
        condition: {node: [] for node in range(cfg.node_count)}
        for condition in conditions
    }
    for condition in conditions:
        for cycle in range(1, extra_loops + 1):
            for node in range(cfg.node_count):
                correct, count = counts[(condition, cycle, node)]
                accuracy = correct / count if count else float("nan")
                low, high = wilson_interval(correct, count)
                curves[condition][node].append(accuracy)
                rows.append(
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

    node_summaries: list[dict[str, Any]] = []
    for node in range(cfg.node_count):
        oracle_curve = curves["exact_interface"][node]
        for condition in conditions:
            curve = curves[condition][node]
            node_summaries.append(
                {
                    "condition": condition,
                    "target_node": node,
                    "auc64": float(np.mean(curve)),
                    "oracle_auc64": float(np.mean(oracle_curve)),
                    "auc_fraction_of_oracle": (
                        float(np.mean(curve))
                        / max(float(np.mean(oracle_curve)), 1e-12)
                    ),
                    "lifespan_at_absolute_0.8": prefix_lifespan(
                        curve,
                        0.8,
                    ),
                    "lifespan_at_0.8_of_oracle": (
                        relative_prefix_lifespan(
                            curve,
                            oracle_curve,
                            0.8,
                        )
                    ),
                    "lifespan_within_0.1_of_oracle": prefix_lifespan(
                        [
                            1.0
                            if value >= reference - 0.1
                            else 0.0
                            for value, reference in zip(
                                curve,
                                oracle_curve,
                                strict=True,
                            )
                        ],
                        0.5,
                    ),
                    "accuracy_cycle32": curve[31],
                    "accuracy_cycle64": curve[63],
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
        "node_summaries": node_summaries,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_node_cycle_rows.csv", rows)
    _write_csv(out_dir / "per_node_summary.csv", node_summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit 64-cycle D8L8 rejuvenation separately for each target node."
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
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument("--seed", type=int, default=116101)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = evaluate_per_node_lifespan(
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
