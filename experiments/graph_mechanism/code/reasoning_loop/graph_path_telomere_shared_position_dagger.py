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


def _flatten_positions(
    state: torch.Tensor,
    positions: tuple[int, ...],
) -> torch.Tensor:
    return state[:, list(positions)].reshape(-1, state.shape[-1])


def _weighted_position_samples(
    state: torch.Tensor,
    positions: tuple[int, ...],
    *,
    answer_position: int,
    answer_repeat: int,
) -> torch.Tensor:
    if answer_repeat < 1:
        raise ValueError("answer_repeat must be positive")
    parts = [_flatten_positions(state, positions)]
    if answer_repeat > 1:
        answer = state[:, answer_position].repeat_interleave(
            answer_repeat - 1,
            dim=0,
        )
        parts.append(answer)
    return torch.cat(parts)


def _position_samples_with_weights(
    state: torch.Tensor,
    positions: tuple[int, ...],
    *,
    answer_position: int,
    answer_weight: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if answer_weight < 1:
        raise ValueError("answer_weight must be positive")
    if answer_position not in positions:
        raise ValueError("answer_position must be one of positions")
    samples = _flatten_positions(state, positions)
    weights = torch.ones(
        state.shape[0],
        len(positions),
        device=state.device,
        dtype=torch.float32,
    )
    answer_index = positions.index(answer_position)
    weights[:, answer_index] = float(answer_weight)
    return samples, weights.reshape(-1)


def _weighted_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have equal shape")
    if weights.shape != (prediction.shape[0],):
        raise ValueError("weights must have one entry per sample")
    weights = weights.float()
    normalized = weights / weights.sum().clamp_min(1e-12)
    error = (prediction.float() - target.float()).square().mean(dim=1)
    target_mean = (normalized[:, None] * target.float()).sum(dim=0)
    centered_energy = (
        (target.float() - target_mean).square().mean(dim=1)
    )
    return float(
        (normalized * error).sum()
        / (normalized * centered_energy).sum().clamp_min(1e-12)
    )


def fit_weighted_full_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge: float,
) -> VectorAffine:
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must share [sample, feature] shape")
    if weights.shape != (source.shape[0],):
        raise ValueError("weights must have one entry per sample")
    if bool(weights.le(0).any()):
        raise ValueError("weights must be positive")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    source = source.float()
    target = target.float()
    weights = weights.float()
    total_weight = weights.sum()
    normalized = weights / total_weight
    source_mean = (normalized[:, None] * source).sum(dim=0)
    target_mean = (normalized[:, None] * target).sum(dim=0)
    centered_source = source - source_mean
    centered_update = (
        (target - source) - (target_mean - source_mean)
    )
    weighted_source = centered_source * weights.sqrt()[:, None]
    weighted_update = centered_update * weights.sqrt()[:, None]
    gram = weighted_source.transpose(0, 1) @ weighted_source
    cross = weighted_source.transpose(0, 1) @ weighted_update
    scale = gram.diagonal().mean().clamp_min(1e-6)
    update = torch.linalg.solve(
        gram
        + ridge
        * scale
        * torch.eye(
            source.shape[1],
            device=source.device,
            dtype=source.dtype,
        ),
        cross,
    )
    weight = torch.eye(
        source.shape[1],
        device=source.device,
        dtype=source.dtype,
    ) + update
    bias = target_mean - source_mean @ weight
    return VectorAffine(
        weight=weight,
        bias=bias,
        update_rank=source.shape[1],
        fit_dimension=source.shape[1],
        retained_fit_energy=1.0,
    )


@torch.no_grad()
def collect_shared_position_rollout_pairs(
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
    if rollout_cycles < 2:
        raise ValueError("rollout_cycles must be at least two")
    jump = phase_positions[3] - phase_positions[2]
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
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
            oracle_step = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    (positions, age_map) if cycle > 1 else None
                ),
            )
            if cycle > 1:
                sources.append(
                    _weighted_position_samples(
                        step.block2_hidden_pre_intervention,
                        positions,
                        answer_position=answer_position,
                        answer_repeat=answer_repeat,
                    ).float()
                )
                targets.append(
                    _weighted_position_samples(
                        oracle_step.block2_hidden_in,
                        positions,
                        answer_position=answer_position,
                        answer_repeat=answer_repeat,
                    ).float()
                )
            state = step.state
    return torch.cat(sources), torch.cat(targets)


@torch.no_grad()
def collect_shared_position_rollout_pairs_weighted(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_weight: int,
    age_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    rollout_cycles: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rollout_cycles < 2:
        raise ValueError("rollout_cycles must be at least two")
    jump = phase_positions[3] - phase_positions[2]
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
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
            oracle_step = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    (positions, age_map) if cycle > 1 else None
                ),
            )
            if cycle > 1:
                source, sample_weights = _position_samples_with_weights(
                    step.block2_hidden_pre_intervention,
                    positions,
                    answer_position=answer_position,
                    answer_weight=answer_weight,
                )
                target, target_weights = _position_samples_with_weights(
                    oracle_step.block2_hidden_in,
                    positions,
                    answer_position=answer_position,
                    answer_weight=answer_weight,
                )
                if not torch.equal(sample_weights, target_weights):
                    raise RuntimeError("source and target weights differ")
                sources.append(source.float())
                targets.append(target.float())
                weights.append(sample_weights)
            state = step.state
    return torch.cat(sources), torch.cat(targets), torch.cat(weights)


@torch.no_grad()
def heldout_shared_map_metrics(
    *,
    model,
    cfg,
    positions: tuple[int, ...],
    answer_position: int,
    heldout,
    age_map: VectorAffine,
) -> dict[str, float]:
    source = _flatten_positions(heldout.z3_block2_full, positions)
    target = _flatten_positions(heldout.z2_block2_full, positions)
    prediction = age_map(source)
    prediction_shaped = prediction.reshape(
        heldout.h3_full.shape[0],
        len(positions),
        cfg.d_model,
    )
    step = run_one_loop(
        model,
        heldout.h3_full,
        loop_index=cfg.max_loops,
        block2_position_override=(positions, prediction_shaped),
    )
    metrics = _masked_metrics(
        step.logits,
        heldout.next_two_hop,
        endpoint=heldout.current,
    )
    return {
        "heldout_natural_relative_mse": _relative_mse(
            prediction,
            target,
        ),
        "heldout_natural_next_accuracy": float(metrics["accuracy"]),
        "heldout_natural_next_margin": float(metrics["margin"]),
        "heldout_answer_relative_mse": _relative_mse(
            age_map(heldout.z3_block2_full[:, answer_position]),
            heldout.z2_block2_full[:, answer_position],
        ),
    }


@torch.no_grad()
def evaluate_shared_maps(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    maps: dict[str, VectorAffine],
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
    executor_head: int,
) -> list[dict[str, Any]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    jump = phase_positions[3] - phase_positions[2]
    conditions = (
        "no_intervention",
        "oracle_full_state",
        "oracle_interface",
        "shuffled_oracle_interface",
        *[f"shared_r{label}" for label in maps],
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
            for condition in conditions
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
            steps = {
                "oracle_full_state": oracle_step,
                "no_intervention": run_one_loop(
                    model,
                    states["no_intervention"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
            }
            oracle_values = oracle_step.block2_hidden_in[:, list(positions)]
            steps["oracle_interface"] = run_one_loop(
                model,
                states["oracle_interface"],
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_override=(
                    None
                    if cycle == 1
                    else (positions, oracle_values)
                ),
            )
            steps["shuffled_oracle_interface"] = run_one_loop(
                model,
                states["shuffled_oracle_interface"],
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_override=(
                    None
                    if cycle == 1
                    else (positions, oracle_values.roll(1, dims=0))
                ),
            )
            for label, age_map in maps.items():
                condition = f"shared_r{label}"
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, age_map) if cycle > 1 else None
                    ),
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
    return rows


def _curve_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    conditions = sorted({str(row["condition"]) for row in rows})
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
            for field in component_fields:
                components[field].append(
                    float(np.mean([float(part[field]) for part in parts]))
                )
        result[condition] = {
            "accuracy": accuracy,
            "auc": float(np.mean(accuracy)),
            "auc_4": float(np.mean(accuracy[:4])),
            "auc_8": float(np.mean(accuracy[:8])),
            "auc_16": float(np.mean(accuracy[:16])),
            **components,
        }
    return result


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    heldout_batches: int,
    dagger_batch_size: int,
    dagger_batches: int,
    dagger_rounds: int,
    dagger_rollout_cycles: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    ranks: Sequence[int],
    answer_repeats: Sequence[int],
    ridge: float,
    calibration_seed: int,
    heldout_seed: int,
    dagger_seed: int,
    evaluation_seed: int,
    memory_efficient_full_rank: bool = False,
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
    if memory_efficient_full_rank and any(
        int(rank) != cfg.d_model for rank in ranks
    ):
        raise ValueError(
            "memory-efficient weighted fit supports full rank only"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
        executor_head=2,
    )
    heldout = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=heldout_seed,
        executor_head=2,
    )
    snapshots: dict[str, VectorAffine] = {}
    training_rows: list[dict[str, Any]] = []
    natural_pair_counts: dict[int, int] = {}
    for answer_repeat in answer_repeats:
        natural_weights: torch.Tensor | None = None
        if memory_efficient_full_rank:
            natural_source, natural_weights = (
                _position_samples_with_weights(
                    calibration.z3_block2_full,
                    positions,
                    answer_position=answer_position,
                    answer_weight=int(answer_repeat),
                )
            )
            natural_target, target_weights = (
                _position_samples_with_weights(
                    calibration.z2_block2_full,
                    positions,
                    answer_position=answer_position,
                    answer_weight=int(answer_repeat),
                )
            )
            if not torch.equal(natural_weights, target_weights):
                raise RuntimeError("natural source and target weights differ")
            natural_pair_counts[int(answer_repeat)] = int(
                natural_weights.sum()
            )
        else:
            natural_source = _weighted_position_samples(
                calibration.z3_block2_full,
                positions,
                answer_position=answer_position,
                answer_repeat=int(answer_repeat),
            )
            natural_target = _weighted_position_samples(
                calibration.z2_block2_full,
                positions,
                answer_position=answer_position,
                answer_repeat=int(answer_repeat),
            )
            natural_pair_counts[int(answer_repeat)] = int(
                natural_source.shape[0]
            )
        for rank in ranks:
            train_source = natural_source
            train_target = natural_target
            train_weights = natural_weights
            if memory_efficient_full_rank:
                if train_weights is None:
                    raise RuntimeError("missing weighted-fit weights")
                age_map = fit_weighted_full_affine(
                    train_source,
                    train_target,
                    train_weights,
                    ridge=ridge,
                )
            else:
                family = fit_reduced_rank_update_family(
                    train_source,
                    train_target,
                    ridge=ridge,
                )
                age_map = family.map_for_rank(int(rank))
            label = f"w{answer_repeat}_r{rank}_round0"
            snapshots[label] = age_map
            metrics = heldout_shared_map_metrics(
                model=model,
                cfg=cfg,
                positions=positions,
                answer_position=answer_position,
                heldout=heldout,
                age_map=age_map,
            )
            training_rows.append(
                {
                    "answer_repeat": answer_repeat,
                    "rank": rank,
                    "round": 0,
                    "training_position_pairs": int(
                        (
                            train_weights.sum()
                            if train_weights is not None
                            else train_source.shape[0]
                        )
                    ),
                    "new_rollout_position_pairs": 0,
                    "rollout_relative_mse_before_refit": float("nan"),
                    **metrics,
                }
            )
            for round_index in range(1, dagger_rounds + 1):
                rollout_weights: torch.Tensor | None = None
                if memory_efficient_full_rank:
                    (
                        rollout_source,
                        rollout_target,
                        rollout_weights,
                    ) = collect_shared_position_rollout_pairs_weighted(
                        model=model,
                        cfg=cfg,
                        phase_positions=phase_positions,
                        positions=positions,
                        answer_position=answer_position,
                        answer_weight=int(answer_repeat),
                        age_map=age_map,
                        device=device,
                        batch_size=dagger_batch_size,
                        batches=dagger_batches,
                        rollout_cycles=dagger_rollout_cycles,
                        seed=(
                            dagger_seed
                            + 100000 * int(answer_repeat)
                            + 1000 * int(rank)
                            + round_index
                        ),
                    )
                    rollout_error = _weighted_relative_mse(
                        age_map(rollout_source),
                        rollout_target,
                        rollout_weights,
                    )
                else:
                    rollout_source, rollout_target = (
                        collect_shared_position_rollout_pairs(
                            model=model,
                            cfg=cfg,
                            phase_positions=phase_positions,
                            positions=positions,
                            answer_position=answer_position,
                            answer_repeat=int(answer_repeat),
                            age_map=age_map,
                            device=device,
                            batch_size=dagger_batch_size,
                            batches=dagger_batches,
                            rollout_cycles=dagger_rollout_cycles,
                            seed=(
                                dagger_seed
                                + 100000 * int(answer_repeat)
                                + 1000 * int(rank)
                                + round_index
                            ),
                        )
                    )
                    rollout_error = _relative_mse(
                        age_map(rollout_source),
                        rollout_target,
                    )
                train_source = torch.cat((train_source, rollout_source))
                train_target = torch.cat((train_target, rollout_target))
                if memory_efficient_full_rank:
                    if train_weights is None or rollout_weights is None:
                        raise RuntimeError("missing rollout weights")
                    train_weights = torch.cat(
                        (train_weights, rollout_weights)
                    )
                    age_map = fit_weighted_full_affine(
                        train_source,
                        train_target,
                        train_weights,
                        ridge=ridge,
                    )
                else:
                    family = fit_reduced_rank_update_family(
                        train_source,
                        train_target,
                        ridge=ridge,
                    )
                    age_map = family.map_for_rank(int(rank))
                label = (
                    f"w{answer_repeat}_r{rank}_round{round_index}"
                )
                snapshots[label] = age_map
                metrics = heldout_shared_map_metrics(
                    model=model,
                    cfg=cfg,
                    positions=positions,
                    answer_position=answer_position,
                    heldout=heldout,
                    age_map=age_map,
                )
                training_rows.append(
                    {
                        "answer_repeat": answer_repeat,
                        "rank": rank,
                        "round": round_index,
                        "training_position_pairs": int(
                            (
                                train_weights.sum()
                                if train_weights is not None
                                else train_source.shape[0]
                            )
                        ),
                        "new_rollout_position_pairs": int(
                            (
                                rollout_weights.sum()
                                if rollout_weights is not None
                                else rollout_source.shape[0]
                            )
                        ),
                        "rollout_relative_mse_before_refit": (
                            rollout_error
                        ),
                        **metrics,
                    }
                )
    closed_rows = evaluate_shared_maps(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        maps=snapshots,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=2,
    )
    curves = _curve_summary(closed_rows)
    map_curves = {
        condition: data
        for condition, data in curves.items()
        if condition.startswith("shared_r")
    }
    best = max(map_curves.items(), key=lambda item: item[1]["auc"])
    torch.save(
        {
            "kind": "graph_path_telomere_shared_position_dagger",
            "positions": positions,
            "maps": {
                label: {
                    "weight": age_map.weight.cpu(),
                    "bias": age_map.bias.cpu(),
                    "rank": age_map.update_rank,
                    "retained_fit_energy": age_map.retained_fit_energy,
                }
                for label, age_map in snapshots.items()
            },
        },
        out_dir / "shared_position_maps.pt",
    )
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
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
        "shared_map_positions": list(positions),
        "shared_map_parameter_count": {
            label: age_map.factor_parameter_count
            for label, age_map in snapshots.items()
        },
        "training": {
            "natural_graphs": calibration_batch_size
            * calibration_batches,
            "natural_position_pairs_by_answer_repeat": natural_pair_counts,
            "dagger_rounds": dagger_rounds,
            "rollout_cycles_per_round": dagger_rollout_cycles,
            "rollout_graphs_per_round": dagger_batch_size
            * dagger_batches,
            "ranks": [int(rank) for rank in ranks],
            "answer_repeats": [
                int(answer_repeat)
                for answer_repeat in answer_repeats
            ],
            "ridge": ridge,
            "loss": "shared positionwise state MSE only",
            "memory_efficient_full_rank_weighting": (
                memory_efficient_full_rank
            ),
            "excluded": [
                "task CE",
                "32-loop loss",
                "component-specific attention loss",
            ],
        },
        "heldout_graphs": heldout_batch_size * heldout_batches,
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "extra_loops": extra_loops,
        "training_rows": training_rows,
        "closed_loop": curves,
        "best_map_by_eval_auc_descriptive_only": {
            "condition": best[0],
            "auc": best[1]["auc"],
        },
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "maps": "shared_position_maps.pt",
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
            "Train one shared affine rejuvenation matrix over all non-BOS "
            "Block2-interface positions with short-horizon DAgger."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--heldout-batch-size", type=int, default=256)
    parser.add_argument("--heldout-batches", type=int, default=4)
    parser.add_argument("--dagger-batch-size", type=int, default=256)
    parser.add_argument("--dagger-batches", type=int, default=2)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--dagger-rollout-cycles", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=32)
    parser.add_argument("--ranks", type=int, nargs="+", default=(32, 256))
    parser.add_argument(
        "--answer-repeats",
        type=int,
        nargs="+",
        default=(1,),
    )
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=80101)
    parser.add_argument("--heldout-seed", type=int, default=80102)
    parser.add_argument("--dagger-seed", type=int, default=80103)
    parser.add_argument("--evaluation-seed", type=int, default=80104)
    parser.add_argument(
        "--memory-efficient-full-rank",
        action="store_true",
        help=(
            "Use exact sample weights and sufficient-statistic full-rank "
            "regression instead of materializing repeated answer rows."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        heldout_batch_size=args.heldout_batch_size,
        heldout_batches=args.heldout_batches,
        dagger_batch_size=args.dagger_batch_size,
        dagger_batches=args.dagger_batches,
        dagger_rounds=args.dagger_rounds,
        dagger_rollout_cycles=args.dagger_rollout_cycles,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        ranks=args.ranks,
        answer_repeats=args.answer_repeats,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        heldout_seed=args.heldout_seed,
        dagger_seed=args.dagger_seed,
        evaluation_seed=args.evaluation_seed,
        memory_efficient_full_rank=args.memory_efficient_full_rank,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
