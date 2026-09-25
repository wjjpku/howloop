from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

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
    _curve_summary,
    _write_csv,
)
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
    _position_samples_with_weights,
    _weighted_position_samples,
    fit_weighted_full_affine,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _booster_cycles(
    *,
    first_cycle: int,
    period: int,
    extra_loops: int,
) -> tuple[int, ...]:
    if first_cycle < 2:
        raise ValueError("first_cycle must be at least 2")
    if period < 1:
        raise ValueError("period must be positive")
    if first_cycle > extra_loops:
        raise ValueError("first_cycle exceeds extra_loops")
    return tuple(range(first_cycle, extra_loops + 1, period))


@torch.no_grad()
def collect_booster_pairs(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_repeat: int,
    feedback: VectorAffine,
    switch_cycle: int,
    device: torch.device,
    batch_size: int,
    batches: int,
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
            path_positions=cfg.max_depth + jump * switch_cycle,
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
        for cycle in range(1, switch_cycle):
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    None
                    if cycle == 1
                    else (positions, feedback)
                ),
            )
            state = step.state
        current = all_targets[
            :, cfg.max_depth + jump * (switch_cycle - 1)
        ]
        source = run_one_loop(
            model,
            state,
            loop_index=cfg.max_loops + switch_cycle - 1,
        ).block2_hidden_pre_intervention
        oracle_input = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        target = run_one_loop(
            model,
            oracle_input,
            loop_index=cfg.max_loops + switch_cycle - 1,
        ).block2_hidden_in
        sources.append(
            _weighted_position_samples(
                source,
                positions,
                answer_position=answer_position,
                answer_repeat=answer_repeat,
            ).float()
        )
        targets_out.append(
            _weighted_position_samples(
                target,
                positions,
                answer_position=answer_position,
                answer_repeat=answer_repeat,
            ).float()
        )
    return torch.cat(sources), torch.cat(targets_out)


@torch.no_grad()
def collect_periodic_dagger_pairs(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_weight: int,
    feedback: VectorAffine,
    booster: VectorAffine,
    first_cycle: int,
    period: int,
    rollout_cycles: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scheduled = set(
        _booster_cycles(
            first_cycle=first_cycle,
            period=period,
            extra_loops=rollout_cycles,
        )
    )
    jump = phase_positions[3] - phase_positions[2]
    sources: list[torch.Tensor] = []
    targets_out: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
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
            age=2,
            phase_position=phase_positions[2],
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
                block2_position_transform=(
                    (positions, booster)
                    if cycle in scheduled
                    else (
                        None
                        if cycle == 1
                        else (positions, feedback)
                    )
                ),
            )
            if cycle in scheduled:
                source, source_weights = _position_samples_with_weights(
                    step.block2_hidden_pre_intervention,
                    positions,
                    answer_position=answer_position,
                    answer_weight=answer_weight,
                )
                target, target_weights = _position_samples_with_weights(
                    oracle.block2_hidden_in,
                    positions,
                    answer_position=answer_position,
                    answer_weight=answer_weight,
                )
                if not torch.equal(source_weights, target_weights):
                    raise RuntimeError("source and target weights differ")
                sources.append(source.float())
                targets_out.append(target.float())
                weights.append(source_weights)
            state = step.state
    return torch.cat(sources), torch.cat(targets_out), torch.cat(weights)


@torch.no_grad()
def evaluate_periodic_booster(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    feedback: VectorAffine,
    booster: VectorAffine,
    switch_cycle: int,
    period: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
    executor_head: int,
) -> list[dict[str, Any]]:
    periodic_cycles = set(
        _booster_cycles(
            first_cycle=switch_cycle,
            period=period,
            extra_loops=extra_loops,
        )
    )
    destination_positions = explicit_depth_position_groups(
        cfg.node_count
    )["destination"]
    jump = phase_positions[3] - phase_positions[2]
    conditions = (
        "feedback_only",
        "learned_booster_once",
        "learned_booster_periodic",
        "shuffled_booster_periodic",
        "exact_interface_once",
        "exact_interface_periodic",
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
        states = {condition: initial.clone() for condition in conditions}
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
            steps = {}
            for condition in conditions:
                transform = None
                override = None
                if cycle > 1:
                    transform = (positions, feedback)
                if (
                    condition == "learned_booster_once"
                    and cycle == switch_cycle
                ):
                    transform = (positions, booster)
                elif (
                    condition == "learned_booster_periodic"
                    and cycle in periodic_cycles
                ):
                    transform = (positions, booster)
                elif (
                    condition == "shuffled_booster_periodic"
                    and cycle in periodic_cycles
                ):
                    transform = (
                        positions,
                        lambda value: booster(value).roll(1, dims=0),
                    )
                elif (
                    condition == "exact_interface_once"
                    and cycle == switch_cycle
                ):
                    transform = None
                    override = (positions, oracle_values)
                elif (
                    condition == "exact_interface_periodic"
                    and cycle in periodic_cycles
                ):
                    transform = None
                    override = (positions, oracle_values)
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=transform,
                    block2_position_override=override,
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
    switch_cycle: int,
    period: int,
    answer_repeat: int,
    ridge: float,
    calibration_seed: int,
    heldout_seed: int,
    evaluation_seed: int,
    dagger_rounds: int,
    dagger_batch_size: int,
    dagger_batches: int,
    dagger_rollout_cycles: int,
    dagger_seed: int,
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
    train_source, train_target = collect_booster_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
        feedback=feedback,
        switch_cycle=switch_cycle,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
    )
    booster = fit_reduced_rank_update_family(
        train_source,
        train_target,
        ridge=ridge,
    ).map_for_rank(cfg.d_model)
    aggregate_sources = [train_source]
    aggregate_targets = [train_target]
    aggregate_weights = [
        torch.ones(
            train_source.shape[0],
            device=train_source.device,
            dtype=train_source.dtype,
        )
    ]
    dagger_history: list[dict[str, Any]] = []
    for round_index in range(1, dagger_rounds + 1):
        source, target, weights = collect_periodic_dagger_pairs(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            answer_position=answer_position,
            answer_weight=answer_repeat,
            feedback=feedback,
            booster=booster,
            first_cycle=switch_cycle,
            period=period,
            rollout_cycles=dagger_rollout_cycles,
            device=device,
            batch_size=dagger_batch_size,
            batches=dagger_batches,
            seed=dagger_seed + round_index - 1,
        )
        aggregate_sources.append(source)
        aggregate_targets.append(target)
        aggregate_weights.append(weights)
        booster = fit_weighted_full_affine(
            torch.cat(aggregate_sources),
            torch.cat(aggregate_targets),
            torch.cat(aggregate_weights),
            ridge=ridge,
        )
        dagger_history.append(
            {
                "round": round_index,
                "new_unique_pairs": int(source.shape[0]),
                "aggregate_unique_pairs": int(
                    sum(value.shape[0] for value in aggregate_sources)
                ),
                "rollout_cycles": dagger_rollout_cycles,
            }
        )
    heldout_source, heldout_target = collect_booster_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        answer_repeat=answer_repeat,
        feedback=feedback,
        switch_cycle=switch_cycle,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=heldout_seed,
    )
    heldout_prediction = booster(heldout_source)
    closed_rows = evaluate_periodic_booster(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        feedback=feedback,
        booster=booster,
        switch_cycle=switch_cycle,
        period=period,
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
            "kind": "graph_path_telomere_periodic_booster",
            "positions": positions,
            "map": {
                "weight": booster.weight.cpu(),
                "bias": booster.bias.cpu(),
                "rank": booster.update_rank,
                "retained_fit_energy": booster.retained_fit_energy,
            },
        },
        out_dir / "periodic_booster.pt",
    )
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "feedback_artifact": str(feedback_artifact),
        "feedback_label": feedback_label,
        "booster": {
            "form": "one shared 256x256+b affine map over 28 positions",
            "training_relation": (
                f"feedback pre-map state at cycle {switch_cycle} "
                "-> exact H2 Block2 input"
            ),
            "training_loss": "weighted positionwise state MSE only",
            "answer_repeat": answer_repeat,
            "ridge": ridge,
            "rank": booster.update_rank,
            "graphs": calibration_batch_size * calibration_batches,
            "sample_pairs": int(train_source.shape[0]),
            "switch_cycle": switch_cycle,
            "period": period,
            "dagger_rounds": dagger_rounds,
            "dagger_rollout_cycles": dagger_rollout_cycles,
            "dagger_history": dagger_history,
            "excluded": [
                "task CE",
                "closed-loop task loss",
                "evaluation accuracy loss",
            ],
        },
        "heldout": {
            "relative_mse": _relative_mse(
                heldout_prediction,
                heldout_target,
            ),
            "sample_pairs": int(heldout_source.shape[0]),
        },
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "extra_loops": extra_loops,
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "booster": "periodic_booster.pt",
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
            "Fit one late affine reset on the expiring D8L8 feedback "
            "trajectory and test one-time and periodic reuse."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--heldout-batch-size", type=int, default=128)
    parser.add_argument("--heldout-batches", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=128)
    parser.add_argument("--switch-cycle", type=int, default=32)
    parser.add_argument("--period", type=int, default=32)
    parser.add_argument("--answer-repeat", type=int, default=28)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=132101)
    parser.add_argument("--heldout-seed", type=int, default=132102)
    parser.add_argument("--evaluation-seed", type=int, default=132103)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--dagger-batches", type=int, default=4)
    parser.add_argument("--dagger-rollout-cycles", type=int, default=96)
    parser.add_argument("--dagger-seed", type=int, default=136101)
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
        switch_cycle=args.switch_cycle,
        period=args.period,
        answer_repeat=args.answer_repeat,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        heldout_seed=args.heldout_seed,
        evaluation_seed=args.evaluation_seed,
        dagger_rounds=args.dagger_rounds,
        dagger_batch_size=args.dagger_batch_size,
        dagger_batches=args.dagger_batches,
        dagger_rollout_cycles=args.dagger_rollout_cycles,
        dagger_seed=args.dagger_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
