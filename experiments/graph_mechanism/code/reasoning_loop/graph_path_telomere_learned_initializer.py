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
    VectorAffine,
    _relative_mse,
    _routing_metrics,
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


@torch.no_grad()
def collect_initializer_pairs(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_repeat: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []
    set_seed(seed)
    for _ in range(batches):
        _, targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = targets[:, cfg.max_depth - 1]
        h8 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=8,
            phase_position=phase_positions[8],
        )
        h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        z8 = run_one_loop(
            model,
            h8,
            loop_index=cfg.max_loops,
        ).block2_hidden_pre_intervention
        z2 = run_one_loop(
            model,
            h2,
            loop_index=cfg.max_loops,
        ).block2_hidden_in
        sources.append(
            _weighted_position_samples(
                z8,
                positions,
                answer_position=answer_position,
                answer_repeat=answer_repeat,
            ).float()
        )
        targets_out.append(
            _weighted_position_samples(
                z2,
                positions,
                answer_position=answer_position,
                answer_repeat=answer_repeat,
            ).float()
        )
    return torch.cat(sources), torch.cat(targets_out)


@torch.no_grad()
def evaluate_learned_initializer(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
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
        "raw_h8_feedback_each_cycle",
        "learned_init_only",
        "learned_init_then_feedback",
        "shuffled_learned_init_then_feedback",
        "exact_interface_then_feedback",
        "exact_h2_feedback",
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
        h8 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {
            condition: (
                h2.clone()
                if condition == "exact_h2_feedback"
                else h8.clone()
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
                "raw_h8_feedback_each_cycle": run_one_loop(
                    model,
                    states["raw_h8_feedback_each_cycle"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(positions, feedback),
                ),
                "learned_init_only": run_one_loop(
                    model,
                    states["learned_init_only"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, initializer)
                        if cycle == 1
                        else None
                    ),
                ),
                "learned_init_then_feedback": run_one_loop(
                    model,
                    states["learned_init_then_feedback"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, initializer)
                        if cycle == 1
                        else (positions, feedback)
                    ),
                ),
                "shuffled_learned_init_then_feedback": run_one_loop(
                    model,
                    states["shuffled_learned_init_then_feedback"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (
                            positions,
                            lambda value: initializer(value).roll(1, dims=0),
                        )
                        if cycle == 1
                        else (positions, feedback)
                    ),
                ),
                "exact_interface_then_feedback": run_one_loop(
                    model,
                    states["exact_interface_then_feedback"],
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
                "exact_h2_feedback": run_one_loop(
                    model,
                    states["exact_h2_feedback"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        None if cycle == 1 else (positions, feedback)
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
        cycles = sorted(
            {
                int(row["cycle"])
                for row in rows
                if row["condition"] == condition
            }
        )
        accuracy: list[float] = []
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
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    heldout_batches: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    answer_repeat: int,
    ridge: float,
    calibration_seed: int,
    heldout_seed: int,
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
    if feedback_positions != positions:
        raise ValueError("feedback positions do not match the interface")
    train_source, train_target = collect_initializer_pairs(
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
    initializer = fit_reduced_rank_update_family(
        train_source,
        train_target,
        ridge=ridge,
    ).map_for_rank(cfg.d_model)
    heldout_source, heldout_target = collect_initializer_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=heldout_seed,
    )
    heldout_prediction = initializer(heldout_source)
    heldout = {
        "relative_mse": _relative_mse(
            heldout_prediction,
            heldout_target,
        ),
        "sample_pairs": int(heldout_source.shape[0]),
    }
    closed_rows = evaluate_learned_initializer(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
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
            "kind": "graph_path_telomere_learned_initializer",
            "positions": positions,
            "map": {
                "weight": initializer.weight.cpu(),
                "bias": initializer.bias.cpu(),
                "rank": initializer.update_rank,
                "retained_fit_energy": initializer.retained_fit_energy,
            },
        },
        out_dir / "learned_initializer.pt",
    )
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "feedback_artifact": str(feedback_artifact),
        "feedback_label": feedback_label,
        "initializer": {
            "form": "one shared 256x256+b affine map over 28 positions",
            "training_relation": "natural H8 Block2 pre-input -> exact H2 Block2 input",
            "training_loss": "weighted positionwise state MSE only",
            "answer_repeat": answer_repeat,
            "ridge": ridge,
            "rank": initializer.update_rank,
            "graphs": calibration_batch_size * calibration_batches,
            "sample_pairs": int(train_source.shape[0]),
            "excluded": [
                "task CE",
                "rollout loss",
                "closed-loop loss",
                "power loss",
            ],
        },
        "heldout": heldout,
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "extra_loops": extra_loops,
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "initializer": "learned_initializer.pt",
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
            "Learn a one-time H8-to-H2 affine initializer, then hand off to "
            "the separately learned local feedback map."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--heldout-batch-size", type=int, default=256)
    parser.add_argument("--heldout-batches", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument("--answer-repeat", type=int, default=28)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=104101)
    parser.add_argument("--heldout-seed", type=int, default=104102)
    parser.add_argument("--evaluation-seed", type=int, default=104103)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        feedback_artifact=args.feedback_artifact,
        feedback_label=args.feedback_label,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        heldout_batch_size=args.heldout_batch_size,
        heldout_batches=args.heldout_batches,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        answer_repeat=args.answer_repeat,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        heldout_seed=args.heldout_seed,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
