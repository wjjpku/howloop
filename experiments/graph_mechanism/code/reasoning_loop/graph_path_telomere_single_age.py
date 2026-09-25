from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_lifespan_extension import (
    _accumulate,
    _blank_bucket,
    _write_csv,
    extension_position_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_telomere_rejuvenator import (
    FlattenedAffine,
    PositionwiseAffine,
    fit_rejuvenator,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


@dataclass(frozen=True)
class RectangularAffine:
    weight: torch.Tensor
    bias: torch.Tensor
    input_position_count: int
    output_position_count: int
    feature_count: int

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        expected = (self.input_position_count, self.feature_count)
        if state.shape[1:] != expected:
            raise ValueError("state shape does not match contextual map")
        flat = state.flatten(1)
        output = flat @ self.weight + self.bias
        return output.reshape(
            state.shape[0],
            self.output_position_count,
            self.feature_count,
        )


AgeMap = PositionwiseAffine | FlattenedAffine | RectangularAffine


def homogeneous_age_matrix(age_map: AgeMap) -> torch.Tensor:
    """Return one row-vector homogeneous matrix for an affine age map."""
    if not isinstance(age_map, FlattenedAffine):
        raise ValueError(
            "a homogeneous age matrix requires a flattened affine map"
        )
    weight = age_map.affine.weight[0]
    bias = age_map.affine.bias[0]
    feature_count = weight.shape[0]
    if weight.shape != (feature_count, feature_count):
        raise ValueError("the flattened affine map must be square")
    matrix = torch.eye(
        feature_count + 1,
        device=weight.device,
        dtype=weight.dtype,
    )
    matrix[:-1, :-1] = weight
    matrix[-1, :-1] = bias
    return matrix


def predecessor_for_phase(
    successors: torch.Tensor,
    current: torch.Tensor,
    *,
    phase_position: int,
) -> torch.Tensor:
    if phase_position < 0:
        raise ValueError("phase position must be nonnegative")
    # Every graph is a permutation, so f^{-p} = f^{(-p mod N)}.
    inverse_steps = (-phase_position) % successors.shape[1]
    return advance_nodes(
        successors,
        current,
        steps=inverse_steps,
    )


def _state_at_age(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    successors: torch.Tensor,
    current: torch.Tensor,
    age: int,
    phase_position: int,
) -> torch.Tensor:
    start = predecessor_for_phase(
        successors,
        current,
        phase_position=phase_position,
    )
    tokens, _, _, _ = fixed_depth_batch(
        cfg,
        current.shape[0],
        current.device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=start,
    )
    return cache_states_with_initial(
        model,
        tokens,
        loops=max(1, age),
    )[age]


def _validate_phase_positions(
    cfg: GraphPathConfig,
    phase_positions: list[int],
) -> None:
    if len(phase_positions) != cfg.max_loops + 1:
        raise ValueError(
            "phase positions must include initial plus every trained loop"
        )
    if any(position < 0 for position in phase_positions):
        raise ValueError("phase positions must be nonnegative")


@torch.no_grad()
def collect_clean_adjacent_age_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    input_positions: tuple[int, ...],
    positions: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    minimum_target_age: int = 0,
) -> dict[str, torch.Tensor]:
    _validate_phase_positions(cfg, phase_positions)
    if not 0 <= minimum_target_age < cfg.max_loops:
        raise ValueError("minimum target age is outside the trained range")
    sources = []
    targets = []
    ages = []
    index = list(positions)
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        desired_current = path_targets[:, cfg.max_depth - 1]
        for source_age in range(
            minimum_target_age + 1,
            cfg.max_loops + 1,
        ):
            source = _state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=desired_current,
                age=source_age,
                phase_position=phase_positions[source_age],
            )
            target = _state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=desired_current,
                age=source_age - 1,
                phase_position=phase_positions[source_age - 1],
            )
            sources.append(source[:, list(input_positions)])
            targets.append(target[:, index])
            ages.append(
                torch.full(
                    (batch_size,),
                    source_age,
                    device=device,
                    dtype=torch.long,
                )
            )
    return {
        "source": torch.cat(sources, dim=0),
        "target": torch.cat(targets, dim=0),
        "source_age": torch.cat(ages, dim=0),
    }


def fit_single_age_map(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> AgeMap:
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target sample counts must match")
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("source and target must be rank-three tensors")
    if source.shape[2] != target.shape[2]:
        raise ValueError("source and target feature counts must match")
    if source.shape[1] != target.shape[1]:
        source_flat = source.float().flatten(1)
        target_flat = target.float().flatten(1)
        source_mean = source_flat.mean(dim=0)
        target_mean = target_flat.mean(dim=0)
        centered_source = source_flat - source_mean
        centered_target = target_flat - target_mean
        gram = centered_source.T @ centered_source
        cross = centered_source.T @ centered_target
        scale = gram.diagonal().mean().clamp_min(1e-6)
        identity = torch.eye(
            gram.shape[0],
            device=gram.device,
            dtype=gram.dtype,
        )
        weight = torch.linalg.solve(
            gram + ridge * scale * identity,
            cross,
        )
        bias = target_mean - source_mean @ weight
        return RectangularAffine(
            weight=weight,
            bias=bias,
            input_position_count=source.shape[1],
            output_position_count=target.shape[1],
            feature_count=source.shape[2],
        )
    pairs = {
        "initial_source": source,
        "initial_target": target,
        "cycle_source": source,
        "cycle_target": target,
    }
    age_map, _ = fit_rejuvenator(pairs, ridge=ridge)
    return age_map


def relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    numerator = (prediction - target).square().mean()
    denominator = (
        target - target.mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


def adjacent_age_fit_rows(
    *,
    age_map: AgeMap,
    pairs: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    prediction = age_map(pairs["source"])
    rows = []
    ages = pairs["source_age"]
    for source_age in sorted(int(value) for value in ages.unique()):
        selected = ages.eq(source_age)
        rows.append(
            {
                "source_age": source_age,
                "target_age": source_age - 1,
                "relative_mse": relative_mse(
                    prediction[selected],
                    pairs["target"][selected],
                ),
                "sample_count": int(selected.sum()),
            }
        )
    return rows


@torch.no_grad()
def collect_aligned_age_chains(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    positions: tuple[int, ...],
    target_age: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> torch.Tensor:
    _validate_phase_positions(cfg, phase_positions)
    if not 0 <= target_age < cfg.max_loops:
        raise ValueError("target age is outside the trained range")
    chains = []
    index = list(positions)
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        age_states = []
        for age in range(target_age, cfg.max_loops + 1):
            state = _state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=age,
                phase_position=phase_positions[age],
            )
            age_states.append(state[:, index])
        chains.append(torch.stack(age_states, dim=1))
    return torch.cat(chains, dim=0)


def optimize_power_consistency(
    *,
    age_map: AgeMap,
    chains: torch.Tensor,
    rollout_source: torch.Tensor,
    rollout_target: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[AgeMap, list[dict[str, float]]]:
    if steps < 0 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("invalid power-optimization hyperparameters")
    if steps == 0:
        return age_map, []
    if not isinstance(age_map, FlattenedAffine):
        raise ValueError(
            "power consistency currently requires a flattened affine map"
        )
    flat_chains = chains.flatten(2).float()
    flat_rollout_source = rollout_source.flatten(1).float()
    flat_rollout_target = rollout_target.flatten(1).float()
    weight = torch.nn.Parameter(
        age_map.affine.weight[0].detach().clone()
    )
    bias = torch.nn.Parameter(
        age_map.affine.bias[0].detach().clone()
    )
    optimizer = torch.optim.AdamW(
        (weight, bias),
        lr=learning_rate,
        weight_decay=0.0,
    )
    feature_scale = (
        flat_chains.flatten(0, 1)
        .std(dim=0)
        .clamp_min(0.05)
    )
    identity = torch.eye(
        weight.shape[0],
        device=weight.device,
        dtype=weight.dtype,
    )
    history = []
    torch.manual_seed(seed)
    with torch.enable_grad():
        for step in range(1, steps + 1):
            chain_index = torch.randint(
                flat_chains.shape[0],
                (min(batch_size, flat_chains.shape[0]),),
                device=flat_chains.device,
            )
            batch = flat_chains[chain_index]
            predicted = batch[:, -1]
            power_loss = torch.zeros(
                (),
                device=weight.device,
                dtype=weight.dtype,
            )
            power_count = batch.shape[1] - 1
            for rewind in range(1, batch.shape[1]):
                predicted = predicted @ weight + bias
                target = batch[:, -1 - rewind]
                power_loss = power_loss + (
                    (predicted - target) / feature_scale
                ).square().mean()
            power_loss = power_loss / power_count

            adjacent_source = batch[:, 1:].reshape(
                -1, batch.shape[-1]
            )
            adjacent_target = batch[:, :-1].reshape(
                -1, batch.shape[-1]
            )
            adjacent_prediction = adjacent_source @ weight + bias
            adjacent_loss = (
                (adjacent_prediction - adjacent_target) / feature_scale
            ).square().mean()

            rollout_index = torch.randint(
                flat_rollout_source.shape[0],
                (
                    min(
                        batch_size,
                        flat_rollout_source.shape[0],
                    ),
                ),
                device=flat_rollout_source.device,
            )
            rollout_prediction = (
                flat_rollout_source[rollout_index] @ weight + bias
            )
            rollout_loss = (
                (
                    rollout_prediction
                    - flat_rollout_target[rollout_index]
                )
                / feature_scale
            ).square().mean()
            stability_loss = (weight - identity).square().mean()
            loss = (
                power_loss
                + 0.5 * adjacent_loss
                + 0.5 * rollout_loss
                + 1e-4 * stability_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((weight, bias), 5.0)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == steps:
                history.append(
                    {
                        "step": float(step),
                        "loss": float(loss.detach()),
                        "power_loss": float(power_loss.detach()),
                        "adjacent_loss": float(adjacent_loss.detach()),
                        "rollout_loss": float(rollout_loss.detach()),
                    }
                )
    affine = PositionwiseAffine(
        weight=weight.detach().unsqueeze(0),
        bias=bias.detach().unsqueeze(0),
    )
    return (
        FlattenedAffine(
            affine=affine,
            position_count=age_map.position_count,
            feature_count=age_map.feature_count,
        ),
        history,
    )


def apply_age_map(
    state: torch.Tensor,
    *,
    positions: tuple[int, ...],
    input_positions: tuple[int, ...],
    age_map: AgeMap,
    applications: int,
    mode: str,
) -> torch.Tensor:
    if applications < 0:
        raise ValueError("applications must be nonnegative")
    if mode not in {"matched", "shuffled", "reverse"}:
        raise ValueError(f"unknown age-map mode: {mode}")
    index = list(positions)
    result = state.clone()
    for _ in range(applications):
        source = result[:, list(input_positions)]
        rejuvenated = age_map(source)
        if mode == "shuffled":
            rejuvenated = rejuvenated.roll(1, dims=0)
        elif mode == "reverse":
            rejuvenated = 2 * result[:, index] - rejuvenated
        result[:, index] = rejuvenated
    return result


@torch.no_grad()
def collect_closed_loop_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    phase_positions: list[int],
    input_positions: tuple[int, ...],
    positions: tuple[int, ...],
    age_map: AgeMap,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    rewind_steps = cfg.max_loops - target_age
    index = list(positions)
    sources = []
    targets = []
    set_seed(seed)
    for _ in range(batches):
        tokens, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * extra_loops,
        )
        state = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )[-1]
        current = path_targets[:, cfg.max_depth - 1]
        for application in range(rewind_steps):
            intended_age = cfg.max_loops - application - 1
            target = _state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=intended_age,
                phase_position=phase_positions[intended_age],
            )
            sources.append(state[:, list(input_positions)])
            targets.append(target[:, index])
            state = apply_age_map(
                state,
                positions=positions,
                input_positions=input_positions,
                age_map=age_map,
                applications=1,
                mode="matched",
            )
        for extra_loop in range(1, extra_loops + 1):
            state = apply_shared_stack(
                model,
                state,
                loop_index=cfg.max_loops + extra_loop - 1,
            )
            if extra_loop == extra_loops:
                continue
            current = path_targets[
                :, cfg.max_depth + jump * extra_loop - 1
            ]
            target = _state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=target_age,
                phase_position=reference_position,
            )
            sources.append(state[:, list(input_positions)])
            targets.append(target[:, index])
            state = apply_age_map(
                state,
                positions=positions,
                input_positions=input_positions,
                age_map=age_map,
                applications=1,
                mode="matched",
            )
    return torch.cat(sources, dim=0), torch.cat(targets, dim=0)


@torch.no_grad()
def refine_single_age_map(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    phase_positions: list[int],
    input_positions: tuple[int, ...],
    positions: tuple[int, ...],
    clean_pairs: dict[str, torch.Tensor],
    age_map: AgeMap,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    iterations: int,
    ridge: float,
    seed: int,
) -> tuple[AgeMap, list[dict[str, float]]]:
    refined = age_map
    history = []
    for iteration in range(iterations):
        rollout_source, rollout_target = collect_closed_loop_pairs(
            model=model,
            cfg=cfg,
            candidate=candidate,
            phase_positions=phase_positions,
            input_positions=input_positions,
            positions=positions,
            age_map=refined,
            device=device,
            batch_size=batch_size,
            batches=batches,
            extra_loops=extra_loops,
            seed=seed + iteration,
        )
        before = relative_mse(
            refined(rollout_source),
            rollout_target,
        )
        training_source = torch.cat(
            (clean_pairs["source"], rollout_source),
            dim=0,
        )
        training_target = torch.cat(
            (clean_pairs["target"], rollout_target),
            dim=0,
        )
        refined = fit_single_age_map(
            training_source,
            training_target,
            ridge=ridge,
        )
        after = relative_mse(
            refined(rollout_source),
            rollout_target,
        )
        history.append(
            {
                "iteration": float(iteration + 1),
                "rollout_relative_mse_before": before,
                "rollout_relative_mse_after": after,
                "clean_pairs": float(clean_pairs["source"].shape[0]),
                "rollout_pairs": float(rollout_source.shape[0]),
            }
        )
    return refined, history


@torch.no_grad()
def evaluate_single_age_control(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    positions: tuple[int, ...],
    input_positions: tuple[int, ...],
    age_map: AgeMap,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    target_age = int(candidate["reference_age"])
    jump = int(candidate["programmed_jump"])
    rewind_steps = cfg.max_loops - target_age
    conditions = {
        "baseline": (0, "matched"),
        "single_age": (rewind_steps, "matched"),
        "under_rewind": (max(0, rewind_steps - 1), "matched"),
        "over_rewind": (rewind_steps + 1, "matched"),
        "batch_shuffled": (rewind_steps, "shuffled"),
        "reverse": (rewind_steps, "reverse"),
    }
    accumulators = {
        (condition, extra_loop): _blank_bucket()
        for condition in conditions
        for extra_loop in range(1, extra_loops + 1)
    }
    set_seed(seed)
    for _ in range(batches):
        tokens, path_targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * extra_loops,
        )
        terminal = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )[-1]
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        states = {
            condition: apply_age_map(
                terminal,
                positions=positions,
                input_positions=input_positions,
                age_map=age_map,
                applications=initial_applications,
                mode=mode,
            )
            for condition, (
                initial_applications,
                mode,
            ) in conditions.items()
        }
        for extra_loop in range(1, extra_loops + 1):
            target = all_targets[
                :, cfg.max_depth + jump * extra_loop
            ]
            for condition, (_, mode) in conditions.items():
                state = states[condition]
                if extra_loop > 1 and condition != "baseline":
                    state = apply_age_map(
                        state,
                        positions=positions,
                        input_positions=input_positions,
                        age_map=age_map,
                        applications=1,
                        mode=mode,
                    )
                state = apply_shared_stack(
                    model,
                    state,
                    loop_index=cfg.max_loops + extra_loop - 1,
                )
                states[condition] = state
                metrics = _masked_metrics(
                    logits_from_raw_state(model, state),
                    target,
                    endpoint=endpoint,
                )
                _accumulate(
                    accumulators[(condition, extra_loop)],
                    metrics,
                )
    rows = []
    for condition in conditions:
        for extra_loop in range(1, extra_loops + 1):
            bucket = accumulators[(condition, extra_loop)]
            count = int(bucket["count"])
            rows.append(
                {
                    "condition": condition,
                    "initial_age_map_applications": conditions[
                        condition
                    ][0],
                    "cycle_age_map_applications": (
                        0 if condition == "baseline" else 1
                    ),
                    "extra_loop": extra_loop,
                    "target_offset": jump * extra_loop,
                    "accuracy": (
                        bucket["correct"] / count
                        if count
                        else float("nan")
                    ),
                    "probability": (
                        bucket["probability"] / count
                        if count
                        else float("nan")
                    ),
                    "margin": (
                        bucket["margin"] / count
                        if count
                        else float("nan")
                    ),
                    "valid_count": count,
                }
            )
    return rows


def _curve(
    rows: list[dict[str, Any]],
    condition: str,
) -> list[float]:
    return [
        float(row["accuracy"])
        for row in sorted(
            (
                row
                for row in rows
                if row["condition"] == condition
            ),
            key=lambda row: int(row["extra_loop"]),
        )
    ]


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    lifespan_summary_path: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device: torch.device,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    dagger_batch_size: int,
    dagger_batches: int,
    dagger_iterations: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    ridge: float,
    input_group_mode: str,
    train_all_ages: bool,
    power_training_steps: int,
    power_training_batch_size: int,
    power_training_learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    lifespan = json.loads(
        lifespan_summary_path.read_text(encoding="utf-8")
    )
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    candidate = lifespan["candidate"]
    group_summary = lifespan["minimal_four_loop_replacement"]
    group = (
        str(group_summary["group"])
        if group_summary is not None
        else str(lifespan["best_exact_replacement"]["group"])
    )
    positions = extension_position_groups(cfg)[group]
    if input_group_mode == "selected":
        input_group = group
    elif input_group_mode == "query_work":
        input_group = "query_work"
    else:
        raise ValueError("input group mode must be selected or query_work")
    input_positions = extension_position_groups(cfg)[input_group]
    minimum_target_age = (
        0 if train_all_ages else int(candidate["reference_age"])
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    clean_pairs = collect_clean_adjacent_age_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        input_positions=input_positions,
        positions=positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=seed,
        minimum_target_age=minimum_target_age,
    )
    age_map = fit_single_age_map(
        clean_pairs["source"],
        clean_pairs["target"],
        ridge=ridge,
    )
    age_map, dagger_history = refine_single_age_map(
        model=model,
        cfg=cfg,
        candidate=candidate,
        phase_positions=phase_positions,
        input_positions=input_positions,
        positions=positions,
        clean_pairs=clean_pairs,
        age_map=age_map,
        device=device,
        batch_size=dagger_batch_size,
        batches=dagger_batches,
        extra_loops=extra_loops,
        iterations=dagger_iterations,
        ridge=ridge,
        seed=seed + 100,
    )
    power_chains = collect_aligned_age_chains(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        target_age=int(candidate["reference_age"]),
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=seed + 200,
    )
    power_rollout_source, power_rollout_target = (
        collect_closed_loop_pairs(
            model=model,
            cfg=cfg,
            candidate=candidate,
            phase_positions=phase_positions,
            input_positions=input_positions,
            positions=positions,
            age_map=age_map,
            device=device,
            batch_size=dagger_batch_size,
            batches=dagger_batches,
            extra_loops=extra_loops,
            seed=seed + 201,
        )
    )
    age_map, power_history = optimize_power_consistency(
        age_map=age_map,
        chains=power_chains,
        rollout_source=power_rollout_source,
        rollout_target=power_rollout_target,
        steps=power_training_steps,
        batch_size=power_training_batch_size,
        learning_rate=power_training_learning_rate,
        seed=seed + 202,
    )
    heldout_pairs = collect_clean_adjacent_age_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        input_positions=input_positions,
        positions=positions,
        device=device,
        batch_size=heldout_batch_size,
        batches=1,
        seed=seed + 1,
        minimum_target_age=minimum_target_age,
    )
    fit_rows = adjacent_age_fit_rows(
        age_map=age_map,
        pairs=heldout_pairs,
    )
    evaluation_rows = evaluate_single_age_control(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        input_positions=input_positions,
        age_map=age_map,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=seed + 2,
    )
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    homogeneous_matrix = homogeneous_age_matrix(age_map)
    matrix_artifact = "single_age_J.pt"
    torch.save(
        {
            "format_version": 1,
            "convention": (
                "append a constant one to each row-vector state; "
                "[h, 1] @ homogeneous_matrix = [J(h), 1]"
            ),
            "model_name": name,
            "checkpoint": str(checkpoint),
            "group": group,
            "positions": list(positions),
            "target_age": int(candidate["reference_age"]),
            "initial_applications": (
                cfg.max_loops - int(candidate["reference_age"])
            ),
            "cycle_applications": 1,
            "homogeneous_matrix": homogeneous_matrix.cpu(),
            "weight": age_map.affine.weight[0].cpu(),
            "bias": age_map.affine.bias[0].cpu(),
        },
        run_dir / matrix_artifact,
    )
    _write_csv(run_dir / "adjacent_age_fit_rows.csv", fit_rows)
    _write_csv(run_dir / "single_age_control_rows.csv", evaluation_rows)
    curves = {
        condition: _curve(evaluation_rows, condition)
        for condition in {
            str(row["condition"]) for row in evaluation_rows
        }
    }
    single_curve = curves["single_age"]
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "candidate": candidate,
        "group": group,
        "input_group": input_group,
        "input_position_count": len(input_positions),
        "position_count": len(positions),
        "map_structure": (
            "contextual_rectangular_affine"
            if len(input_positions) != len(positions)
            else "flattened_affine"
        ),
        "single_operator": True,
        "matrix_artifact": matrix_artifact,
        "homogeneous_matrix_shape": list(homogeneous_matrix.shape),
        "train_all_ages": train_all_ages,
        "minimum_training_target_age": minimum_target_age,
        "target_age": int(candidate["reference_age"]),
        "initial_rewind_applications": (
            cfg.max_loops - int(candidate["reference_age"])
        ),
        "cycle_rewind_applications": 1,
        "phase_positions": phase_positions,
        "ridge": ridge,
        "calibration_pairs": int(clean_pairs["source"].shape[0]),
        "heldout_pairs": int(heldout_pairs["source"].shape[0]),
        "evaluation_samples": (
            evaluation_batch_size * evaluation_batches
        ),
        "heldout_relative_mse_by_transition": fit_rows,
        "dagger_history": dagger_history,
        "power_training_steps": power_training_steps,
        "power_training_history": power_history,
        "accuracy_by_extra_loop": single_curve,
        "mean_accuracy": float(np.mean(single_curve)),
        "control_curves": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path, Path, Path]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,LIFESPAN_SUMMARY,PHASE_SUMMARY"
        )
    paths = parts[1].split(",", 2)
    if len(paths) != 3:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,LIFESPAN_SUMMARY,PHASE_SUMMARY"
        )
    return parts[0], Path(paths[0]), Path(paths[1]), Path(paths[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one shared affine operator J that rewinds a recurrent "
            "state by exactly one effective age, then reuse J for both "
            "terminal rewind and continued overloop control."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--heldout-batch-size", type=int, default=512)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--dagger-batches", type=int, default=4)
    parser.add_argument("--dagger-iterations", type=int, default=5)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=128,
    )
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument(
        "--input-group-mode",
        choices=("selected", "query_work"),
        default="selected",
    )
    parser.add_argument("--train-all-ages", action="store_true")
    parser.add_argument("--power-training-steps", type=int, default=0)
    parser.add_argument(
        "--power-training-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--power-training-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = {}
    for run_index, (
        name,
        checkpoint,
        lifespan_summary,
        phase_summary,
    ) in enumerate(args.run):
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            lifespan_summary_path=lifespan_summary,
            phase_summary_path=phase_summary,
            out_dir=args.out_dir,
            device=device,
            calibration_batch_size=args.calibration_batch_size,
            calibration_batches=args.calibration_batches,
            heldout_batch_size=args.heldout_batch_size,
            dagger_batch_size=args.dagger_batch_size,
            dagger_batches=args.dagger_batches,
            dagger_iterations=args.dagger_iterations,
            evaluation_batch_size=args.evaluation_batch_size,
            evaluation_batches=args.evaluation_batches,
            extra_loops=args.extra_loops,
            ridge=args.ridge,
            input_group_mode=args.input_group_mode,
            train_all_ages=args.train_all_ages,
            power_training_steps=args.power_training_steps,
            power_training_batch_size=args.power_training_batch_size,
            power_training_learning_rate=(
                args.power_training_learning_rate
            ),
            seed=args.seed + 1000 * run_index,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
