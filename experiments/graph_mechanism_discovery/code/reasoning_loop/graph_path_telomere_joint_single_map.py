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
from reasoning_loop.graph_path_telomere_learned_initializer import (
    collect_initializer_pairs,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    _relative_mse,
    _routing_metrics,
    collect_aligned_pairs,
    fit_reduced_rank_update_family,
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
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _weighted_position_samples,
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


def _load_initializer(
    artifact: Path,
    *,
    device: torch.device,
) -> tuple[VectorAffine, tuple[int, ...]]:
    payload = torch.load(artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_learned_initializer":
        raise ValueError("unexpected initializer artifact kind")
    item = payload["map"]
    weight = item["weight"].to(device=device, dtype=torch.float32)
    return (
        VectorAffine(
            weight=weight,
            bias=item["bias"].to(device=device, dtype=torch.float32),
            update_rank=int(item["rank"]),
            fit_dimension=int(weight.shape[0]),
            retained_fit_energy=float(item["retained_fit_energy"]),
        ),
        tuple(int(value) for value in payload["positions"]),
    )


@torch.no_grad()
def collect_raw_h8_rollout_pairs(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_repeat: int,
    age_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    rollout_cycles: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    jump = phase_positions[3] - phase_positions[2]
    sources: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * rollout_cycles,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        for cycle in range(1, rollout_cycles + 1):
            current = all_targets[
                :, cfg.max_depth + jump * (cycle - 1)
            ]
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
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(positions, age_map),
            )
            sources.append(
                _weighted_position_samples(
                    step.block2_hidden_pre_intervention,
                    positions,
                    answer_position=answer_position,
                    answer_repeat=answer_repeat,
                ).float()
            )
            targets_out.append(
                _weighted_position_samples(
                    oracle.block2_hidden_in,
                    positions,
                    answer_position=answer_position,
                    answer_repeat=answer_repeat,
                ).float()
            )
            state = step.state
    return torch.cat(sources), torch.cat(targets_out)


@torch.no_grad()
def evaluate_joint_map(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    joint_map: VectorAffine,
    initializer: VectorAffine,
    feedback: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
    executor_head: int,
) -> list[dict[str, Any]]:
    destination_positions = explicit_depth_position_groups(
        cfg.node_count
    )["destination"]
    jump = phase_positions[3] - phase_positions[2]
    conditions = (
        "raw_h8_no_control",
        "joint_single_map",
        "shuffled_joint_single_map",
        "learned_two_stage",
        "exact_init_then_feedback",
        "exact_interface_every_cycle",
        "exact_h2_joint_local",
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
                if condition == "exact_h2_joint_local"
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
            oracle_values = oracle.block2_hidden_in[:, list(positions)]
            steps = {
                "raw_h8_no_control": run_one_loop(
                    model,
                    states["raw_h8_no_control"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
                "joint_single_map": run_one_loop(
                    model,
                    states["joint_single_map"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(positions, joint_map),
                ),
                "shuffled_joint_single_map": run_one_loop(
                    model,
                    states["shuffled_joint_single_map"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        positions,
                        lambda value: joint_map(value).roll(1, dims=0),
                    ),
                ),
                "learned_two_stage": run_one_loop(
                    model,
                    states["learned_two_stage"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, initializer)
                        if cycle == 1
                        else (positions, feedback)
                    ),
                ),
                "exact_init_then_feedback": run_one_loop(
                    model,
                    states["exact_init_then_feedback"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (positions, feedback)
                    ),
                    block2_position_override=(
                        (positions, oracle_values)
                        if cycle == 1
                        else None
                    ),
                ),
                "exact_interface_every_cycle": run_one_loop(
                    model,
                    states["exact_interface_every_cycle"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=(positions, oracle_values),
                ),
                "exact_h2_joint_local": run_one_loop(
                    model,
                    states["exact_h2_joint_local"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (positions, joint_map)
                    ),
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
    return rows


def _curve_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        accuracy: list[float] = []
        cycles = sorted(
            {
                int(row["cycle"])
                for row in rows
                if row["condition"] == condition
            }
        )
        for cycle in cycles:
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
        result[condition] = {
            "accuracy": accuracy,
            "auc": float(np.mean(accuracy)),
            "auc_8": float(np.mean(accuracy[:8])),
            "auc_16": float(np.mean(accuracy[:16])),
            "auc_32": float(np.mean(accuracy[:32])),
        }
    return result


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    feedback_artifact: Path,
    feedback_label: str,
    initializer_artifact: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    dagger_batch_size: int,
    dagger_batches: int,
    dagger_rounds: int,
    dagger_rollout_cycles: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    answer_repeat: int,
    initializer_repeat: int,
    ridge: float,
    calibration_seed: int,
    dagger_seed: int,
    evaluation_seed: int,
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
    positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    answer_position = explicit_depth_position_groups(
        cfg.node_count
    )["answer"][0]
    feedback, feedback_positions = _load_map(
        feedback_artifact,
        label=feedback_label,
        device=device,
    )
    initializer, initializer_positions = _load_initializer(
        initializer_artifact,
        device=device,
    )
    if feedback_positions != positions or initializer_positions != positions:
        raise ValueError("reference maps do not match the interface")
    if initializer_repeat < 1:
        raise ValueError("initializer_repeat must be positive")
    init_source, init_target = collect_initializer_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
    )
    aligned = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed + 1,
        executor_head=2,
    )
    local_source = _weighted_position_samples(
        aligned.z3_block2_full,
        positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
    )
    local_target = _weighted_position_samples(
        aligned.z2_block2_full,
        positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
    )
    train_source = torch.cat(
        (*([init_source] * initializer_repeat), local_source)
    )
    train_target = torch.cat(
        (*([init_target] * initializer_repeat), local_target)
    )
    family = fit_reduced_rank_update_family(
        train_source,
        train_target,
        ridge=ridge,
    )
    joint_map = family.map_for_rank(cfg.d_model)
    training_rows: list[dict[str, Any]] = [
        {
            "round": 0,
            "training_position_pairs": int(train_source.shape[0]),
            "new_rollout_position_pairs": 0,
            "rollout_relative_mse_before_refit": float("nan"),
            "initializer_relative_mse": _relative_mse(
                joint_map(init_source),
                init_target,
            ),
            "local_relative_mse": _relative_mse(
                joint_map(local_source),
                local_target,
            ),
        }
    ]
    for round_index in range(1, dagger_rounds + 1):
        rollout_source, rollout_target = collect_raw_h8_rollout_pairs(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            answer_position=answer_position,
            answer_repeat=answer_repeat,
            age_map=joint_map,
            device=device,
            batch_size=dagger_batch_size,
            batches=dagger_batches,
            rollout_cycles=dagger_rollout_cycles,
            seed=dagger_seed + round_index,
        )
        rollout_error = _relative_mse(
            joint_map(rollout_source),
            rollout_target,
        )
        train_source = torch.cat((train_source, rollout_source))
        train_target = torch.cat((train_target, rollout_target))
        joint_map = fit_reduced_rank_update_family(
            train_source,
            train_target,
            ridge=ridge,
        ).map_for_rank(cfg.d_model)
        training_rows.append(
            {
                "round": round_index,
                "training_position_pairs": int(train_source.shape[0]),
                "new_rollout_position_pairs": int(
                    rollout_source.shape[0]
                ),
                "rollout_relative_mse_before_refit": rollout_error,
                "initializer_relative_mse": _relative_mse(
                    joint_map(init_source),
                    init_target,
                ),
                "local_relative_mse": _relative_mse(
                    joint_map(local_source),
                    local_target,
                ),
            }
        )
    closed_rows = evaluate_joint_map(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        joint_map=joint_map,
        initializer=initializer,
        feedback=feedback,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=2,
    )
    curves = _curve_summary(closed_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "graph_path_telomere_joint_single_map",
            "positions": positions,
            "map": {
                "weight": joint_map.weight.cpu(),
                "bias": joint_map.bias.cpu(),
                "rank": joint_map.update_rank,
                "retained_fit_energy": joint_map.retained_fit_energy,
            },
        },
        out_dir / "joint_single_map.pt",
    )
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map": {
            "form": "one shared 256x256+b affine map over 28 positions",
            "rank": joint_map.update_rank,
            "answer_repeat": answer_repeat,
            "initializer_repeat": initializer_repeat,
            "ridge": ridge,
        },
        "training": {
            "natural_initializer_graphs": (
                calibration_batch_size * calibration_batches
            ),
            "natural_local_graphs": (
                calibration_batch_size * calibration_batches
            ),
            "dagger_rounds": dagger_rounds,
            "dagger_rollout_cycles": dagger_rollout_cycles,
            "dagger_graphs_per_round": dagger_batch_size * dagger_batches,
            "loss": "joint weighted interface state MSE only",
            "excluded": [
                "task CE",
                "power loss",
                "64-cycle loss",
            ],
        },
        "training_rows": training_rows,
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "extra_loops": extra_loops,
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "map": "joint_single_map.pt",
            "training": "training_rows.csv",
            "closed_loop": "closed_loop_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one shared affine map jointly on raw H8 initialization, "
            "local H3-to-H2, and its own raw-H8 rollout states."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--initializer-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--dagger-batch-size", type=int, default=64)
    parser.add_argument("--dagger-batches", type=int, default=1)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--dagger-rollout-cycles", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument("--answer-repeat", type=int, default=28)
    parser.add_argument("--initializer-repeat", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=106101)
    parser.add_argument("--dagger-seed", type=int, default=106102)
    parser.add_argument("--evaluation-seed", type=int, default=106103)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        feedback_artifact=args.feedback_artifact,
        feedback_label=args.feedback_label,
        initializer_artifact=args.initializer_artifact,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        dagger_batch_size=args.dagger_batch_size,
        dagger_batches=args.dagger_batches,
        dagger_rounds=args.dagger_rounds,
        dagger_rollout_cycles=args.dagger_rollout_cycles,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        answer_repeat=args.answer_repeat,
        initializer_repeat=args.initializer_repeat,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        dagger_seed=args.dagger_seed,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
