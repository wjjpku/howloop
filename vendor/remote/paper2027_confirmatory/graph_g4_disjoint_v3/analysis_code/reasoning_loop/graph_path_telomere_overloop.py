from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    _project_context,
    explicit_depth_position_groups,
    run_instrumented,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


ResetMode = Literal["centroid", "rank1", "reverse", "orthogonal"]


@dataclass(frozen=True)
class PhaseProfile:
    centroids: torch.Tensor
    positions: tuple[int, ...]
    slope: torch.Tensor
    unit: torch.Tensor
    step_norm: float


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def telomere_position_groups(cfg: GraphPathConfig) -> dict[str, tuple[int, ...]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    return {
        "start": groups["start"],
        "depth": groups["depth"],
        "answer": groups["answer"],
        "registers": groups["start"] + groups["depth"] + groups["answer"],
        "all": tuple(range(cfg.seq_len)),
    }


def cache_states_with_initial(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    loops: int,
) -> list[torch.Tensor]:
    if loops < 1:
        raise ValueError("loops must be positive")
    state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    states = [state]
    for loop_index in range(loops):
        state = apply_shared_stack(model, state, loop_index=loop_index)
        states.append(state)
    return states


def fit_phase_profile(
    centroids: torch.Tensor,
    positions: tuple[int, ...],
) -> PhaseProfile:
    if centroids.ndim != 3:
        raise ValueError("centroids must have shape [age, position, d_model]")
    if centroids.shape[0] < 2:
        raise ValueError("at least two ages are required")
    if not positions:
        raise ValueError("positions must not be empty")
    index = torch.as_tensor(positions, device=centroids.device)
    selected = centroids[:, index]
    ages = torch.arange(
        selected.shape[0],
        device=selected.device,
        dtype=selected.dtype,
    )
    centered_age = ages - ages.mean()
    centered_state = selected - selected.mean(dim=0, keepdim=True)
    slope = (
        centered_state * centered_age[:, None, None]
    ).sum(dim=0) / centered_age.square().sum()
    step_norm_tensor = slope.flatten().norm()
    if float(step_norm_tensor) <= 1e-12:
        raise ValueError("phase slope is degenerate")
    unit = slope / step_norm_tensor
    return PhaseProfile(
        centroids=centroids,
        positions=positions,
        slope=slope,
        unit=unit,
        step_norm=float(step_norm_tensor),
    )


def phase_coordinates(
    states_or_centroids: torch.Tensor,
    profile: PhaseProfile,
) -> torch.Tensor:
    if states_or_centroids.ndim not in {3, 4}:
        raise ValueError(
            "states must have [age, position, d_model] or "
            "[age, batch, position, d_model] shape"
        )
    index = torch.as_tensor(
        profile.positions, device=states_or_centroids.device
    )
    if states_or_centroids.ndim == 3:
        selected = states_or_centroids[:, index]
        return (selected * profile.unit).sum(dim=(-2, -1))
    selected = states_or_centroids[:, :, index]
    return (selected * profile.unit).sum(dim=(-2, -1))


def phase_shift(
    profile: PhaseProfile,
    *,
    reference_age: int,
    receiver_age: int,
    mode: ResetMode,
    random_seed: int,
) -> torch.Tensor:
    if not 0 <= reference_age < profile.centroids.shape[0]:
        raise ValueError("reference_age is outside the profile")
    if not 0 <= receiver_age < profile.centroids.shape[0]:
        raise ValueError("receiver_age is outside the profile")
    direct = (
        profile.centroids[reference_age, list(profile.positions)]
        - profile.centroids[receiver_age, list(profile.positions)]
    )
    if mode == "centroid":
        return direct
    scalar = (direct * profile.unit).sum()
    rank1 = scalar * profile.unit
    if mode == "rank1":
        return rank1
    if mode == "reverse":
        return -direct
    generator = torch.Generator(device=direct.device)
    generator.manual_seed(random_seed)
    control = torch.randn(
        direct.shape,
        device=direct.device,
        dtype=direct.dtype,
        generator=generator,
    )
    flat_control = control.flatten()
    flat_direct = direct.flatten()
    flat_phase = profile.unit.flatten()
    denominator = flat_phase.square().sum().clamp_min(1e-12)
    flat_control = flat_control - (
        flat_control.dot(flat_phase) / denominator
    ) * flat_phase
    control_norm = flat_control.norm().clamp_min(1e-12)
    return (flat_control * (flat_direct.norm() / control_norm)).reshape_as(
        direct
    )


def add_group_shift(
    state: torch.Tensor,
    *,
    positions: tuple[int, ...],
    shift: torch.Tensor,
) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError("state must have [batch, position, d_model] shape")
    if shift.shape != (len(positions), state.shape[-1]):
        raise ValueError("shift has the wrong shape")
    result = state.clone()
    result[:, list(positions)] += shift.unsqueeze(0)
    return result


def advance_nodes(
    successors: torch.Tensor,
    start: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    current = start
    for _ in range(steps):
        current = successors.gather(1, current[:, None]).squeeze(1)
    return current


def clamp_phase_coordinate(
    state: torch.Tensor,
    *,
    profile: PhaseProfile,
    target_coordinate: float,
) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError("state must have [batch, position, d_model] shape")
    positions = list(profile.positions)
    selected = state[:, positions]
    coordinate = (selected * profile.unit).sum(dim=(-2, -1))
    result = state.clone()
    result[:, positions] += (
        torch.as_tensor(
            target_coordinate, device=state.device, dtype=state.dtype
        )
        - coordinate
    )[:, None, None] * profile.unit.unsqueeze(0)
    return result


def orthogonal_phase_control(
    state: torch.Tensor,
    *,
    profile: PhaseProfile,
    target_coordinate: float,
    orthogonal_unit: torch.Tensor,
) -> torch.Tensor:
    positions = list(profile.positions)
    selected = state[:, positions]
    coordinate = (selected * profile.unit).sum(dim=(-2, -1))
    magnitude = (
        torch.as_tensor(
            target_coordinate, device=state.device, dtype=state.dtype
        )
        - coordinate
    ).abs()
    result = state.clone()
    result[:, positions] += magnitude[:, None, None] * orthogonal_unit.unsqueeze(
        0
    )
    return result


def _orthogonal_unit(
    profile: PhaseProfile,
    *,
    random_seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=profile.unit.device)
    generator.manual_seed(random_seed)
    control = torch.randn(
        profile.unit.shape,
        device=profile.unit.device,
        dtype=profile.unit.dtype,
        generator=generator,
    )
    control = control - (control * profile.unit).sum() * profile.unit
    return control / control.flatten().norm().clamp_min(1e-12)


def _all_targets(
    start: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((start[:, None], targets), dim=1)


def _masked_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    endpoint: torch.Tensor | None = None,
) -> dict[str, float | int]:
    valid = (
        torch.ones_like(target, dtype=torch.bool)
        if endpoint is None
        else target.ne(endpoint)
    )
    count = int(valid.sum())
    if count == 0:
        return {
            "accuracy": float("nan"),
            "probability": float("nan"),
            "margin": float("nan"),
            "valid_count": 0,
        }
    selected_logits = logits[valid]
    selected_target = target[valid]
    return {
        "accuracy": float(
            selected_logits.argmax(-1).eq(selected_target).float().mean()
        ),
        "probability": float(
            selected_logits.softmax(-1)
            .gather(1, selected_target[:, None])
            .mean()
        ),
        "margin": float(target_margin(selected_logits, selected_target).mean()),
        "valid_count": count,
    }


def _best_path_position(
    logits: torch.Tensor,
    all_targets: torch.Tensor,
    *,
    endpoint_position: int,
) -> tuple[int, float]:
    endpoint = all_targets[:, endpoint_position]
    candidates: list[tuple[int, float]] = []
    prediction = logits.argmax(-1)
    for position in range(all_targets.shape[1]):
        target = all_targets[:, position]
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if position == endpoint_position
            else target.ne(endpoint)
        )
        accuracy = (
            float(prediction[valid].eq(target[valid]).float().mean())
            if bool(valid.any())
            else float("nan")
        )
        if math.isfinite(accuracy):
            candidates.append((position, accuracy))
    return max(candidates, key=lambda item: item[1])


def _collect_centroids(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> torch.Tensor:
    set_seed(seed)
    accumulator = torch.zeros(
        cfg.max_loops + 1,
        cfg.seq_len,
        cfg.d_model,
        device=device,
    )
    count = 0
    for _ in range(batches):
        tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        states = cache_states_with_initial(
            model, tokens, loops=cfg.max_loops
        )
        for age, state in enumerate(states):
            accumulator[age] += state.sum(dim=0)
        count += batch_size
    return accumulator / count


def _trajectory(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    batch_size: int,
    path_positions: int,
    seed: int,
) -> tuple[list[int], list[float]]:
    set_seed(seed)
    tokens, targets, _, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
    )
    states = cache_states_with_initial(model, tokens, loops=cfg.max_loops)
    all_targets = _all_targets(start, targets)
    positions = [0]
    accuracies = [1.0]
    for state in states[1:]:
        position, accuracy = _best_path_position(
            logits_from_raw_state(model, state),
            all_targets,
            endpoint_position=cfg.max_depth,
        )
        positions.append(position)
        accuracies.append(accuracy)
    return positions, accuracies


def _phase_decode_rows(
    *,
    centroids: torch.Tensor,
    heldout_centroids: torch.Tensor,
    profiles: dict[str, PhaseProfile],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ages = np.arange(centroids.shape[0], dtype=float)
    for group, profile in profiles.items():
        train_coordinate = (
            phase_coordinates(centroids, profile).detach().cpu().numpy()
        )
        test_coordinate = (
            phase_coordinates(heldout_centroids, profile)
            .detach()
            .cpu()
            .numpy()
        )
        correlation = float(np.corrcoef(ages, test_coordinate)[0, 1])
        adjacent = np.diff(test_coordinate)
        monotone_fraction = float((adjacent > 0).mean())
        for age in range(len(ages)):
            rows.append(
                {
                    "group": group,
                    "age": age,
                    "remaining_trained_loops": len(ages) - 1 - age,
                    "train_coordinate": float(train_coordinate[age]),
                    "heldout_coordinate": float(test_coordinate[age]),
                    "heldout_age_correlation": correlation,
                    "heldout_monotone_fraction": monotone_fraction,
                    "phase_step_norm": profile.step_norm,
                }
            )
    return rows


def _reset_selection_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    profiles: dict[str, PhaseProfile],
    trajectory_positions: list[int],
    trajectory_accuracies: list[float],
    device: torch.device,
    batch_size: int,
    path_positions: int,
    seed: int,
) -> list[dict[str, Any]]:
    set_seed(seed)
    tokens, targets, _, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
    )
    terminal = cache_states_with_initial(
        model, tokens, loops=cfg.max_loops
    )[-1]
    all_targets = _all_targets(start, targets)
    endpoint = all_targets[:, cfg.max_depth]
    rows: list[dict[str, Any]] = []
    for group, profile in profiles.items():
        for reference_age in range(cfg.max_loops):
            clean_jump = (
                trajectory_positions[reference_age + 1]
                - trajectory_positions[reference_age]
            )
            resolved = (
                trajectory_accuracies[reference_age] >= 0.50
                and trajectory_accuracies[reference_age + 1] >= 0.50
                and clean_jump in {0, 1, 2}
            )
            for mode in ("centroid", "rank1", "reverse", "orthogonal"):
                shift = phase_shift(
                    profile,
                    reference_age=reference_age,
                    receiver_age=cfg.max_loops,
                    mode=mode,
                    random_seed=seed + 1000 * reference_age,
                )
                patched = add_group_shift(
                    terminal, positions=profile.positions, shift=shift
                )
                updated = apply_shared_stack(
                    model, patched, loop_index=cfg.max_loops
                )
                logits = logits_from_raw_state(model, updated)
                for target_offset in range(3):
                    metrics = _masked_metrics(
                        logits,
                        all_targets[:, cfg.max_depth + target_offset],
                        endpoint=(
                            None if target_offset == 0 else endpoint
                        ),
                    )
                    rows.append(
                        {
                            "group": group,
                            "reference_age": reference_age,
                            "reference_next_loop": reference_age + 1,
                            "reference_path_before": trajectory_positions[
                                reference_age
                            ],
                            "reference_path_after": trajectory_positions[
                                reference_age + 1
                            ],
                            "reference_jump": clean_jump,
                            "reference_resolved": resolved,
                            "mode": mode,
                            "target_offset": target_offset,
                            **metrics,
                        }
                    )
    return rows


def _matched_phase_selection_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    profiles: dict[str, PhaseProfile],
    trajectory_positions: list[int],
    trajectory_accuracies: list[float],
    device: torch.device,
    batch_size: int,
    path_positions: int,
    seed: int,
) -> list[dict[str, Any]]:
    set_seed(seed)
    receiver_tokens, targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
    )
    receiver_terminal = cache_states_with_initial(
        model, receiver_tokens, loops=cfg.max_loops
    )[-1]
    all_targets = _all_targets(start, targets)
    endpoint = all_targets[:, cfg.max_depth]
    rows: list[dict[str, Any]] = []
    for reference_age in range(cfg.max_loops):
        reference_position = trajectory_positions[reference_age]
        clean_jump = (
            trajectory_positions[reference_age + 1]
            - reference_position
        )
        resolved = (
            trajectory_accuracies[reference_age] >= 0.50
            and trajectory_accuracies[reference_age + 1] >= 0.50
            and 0 <= reference_position <= cfg.max_depth
            and clean_jump in {0, 1, 2}
        )
        if not 0 <= reference_position <= cfg.max_depth:
            continue
        reference_start = advance_nodes(
            successors,
            start,
            steps=cfg.max_depth - reference_position,
        )
        reference_tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
            successors=successors,
            start=reference_start,
        )
        reference_state = cache_states_with_initial(
            model, reference_tokens, loops=max(1, reference_age)
        )[reference_age]
        for group, profile in profiles.items():
            positions = list(profile.positions)
            matched_delta = (
                reference_state[:, positions]
                - receiver_terminal[:, positions]
            )
            for mode in ("matched", "batch_shuffled", "reverse"):
                delta = (
                    matched_delta
                    if mode == "matched"
                    else matched_delta.roll(1, dims=0)
                    if mode == "batch_shuffled"
                    else -matched_delta
                )
                patched = receiver_terminal.clone()
                patched[:, positions] += delta
                updated = apply_shared_stack(
                    model, patched, loop_index=cfg.max_loops
                )
                logits = logits_from_raw_state(model, updated)
                for target_offset in range(3):
                    metrics = _masked_metrics(
                        logits,
                        all_targets[:, cfg.max_depth + target_offset],
                        endpoint=(
                            None if target_offset == 0 else endpoint
                        ),
                    )
                    rows.append(
                        {
                            "group": group,
                            "reference_age": reference_age,
                            "reference_next_loop": reference_age + 1,
                            "reference_path_before": reference_position,
                            "reference_path_after": trajectory_positions[
                                reference_age + 1
                            ],
                            "reference_jump": clean_jump,
                            "reference_resolved": resolved,
                            "mode": mode,
                            "target_offset": target_offset,
                            **metrics,
                        }
                    )
    return rows


def choose_reset(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["mode"] == "centroid"
        and row["reference_resolved"]
        and row["reference_jump"] in {1, 2}
        and row["target_offset"] == row["reference_jump"]
        and math.isfinite(float(row["accuracy"]))
    ]
    if not candidates:
        raise ValueError("no resolved active phase is available")
    one_hop = [row for row in candidates if row["reference_jump"] == 1]
    pool = one_hop or candidates
    return max(
        pool,
        key=lambda row: (
            float(row["accuracy"]),
            -telomere_group_order(str(row["group"])),
            int(row["reference_age"]),
        ),
    )


def choose_matched_reset(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["mode"] == "matched"
        and row["reference_resolved"]
        and row["reference_jump"] in {1, 2}
        and row["target_offset"] == row["reference_jump"]
        and math.isfinite(float(row["accuracy"]))
    ]
    if not candidates:
        raise ValueError("no resolved matched phase reset is available")
    one_hop = [row for row in candidates if row["reference_jump"] == 1]
    pool = one_hop or candidates
    return max(
        pool,
        key=lambda row: (
            float(row["accuracy"]),
            -telomere_group_order(str(row["group"])),
            int(row["reference_age"]),
        ),
    )


def telomere_group_order(group: str) -> int:
    order = {
        "answer": 0,
        "start": 1,
        "depth": 2,
        "registers": 3,
        "all": 4,
    }
    if group not in order:
        raise ValueError(f"unknown group: {group}")
    return order[group]


def _evaluation_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    profile: PhaseProfile,
    selected: dict[str, Any],
    matched_selected: dict[str, Any],
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    reference_age = int(selected["reference_age"])
    programmed_jump = int(selected["reference_jump"])
    matched_reference_age = int(matched_selected["reference_age"])
    matched_reference_position = int(
        matched_selected["reference_path_before"]
    )
    matched_programmed_jump = int(matched_selected["reference_jump"])
    matched_group = str(matched_selected["group"])
    matched_positions = list(
        telomere_position_groups(cfg)[matched_group]
    )
    path_positions = cfg.max_depth + 2 * extra_loops
    target_coordinate = float(
        phase_coordinates(profile.centroids, profile)[reference_age]
    )
    terminal_coordinate = float(
        phase_coordinates(profile.centroids, profile)[cfg.max_loops]
    )
    older_coordinate = 2.0 * terminal_coordinate - target_coordinate
    orthogonal_unit = _orthogonal_unit(
        profile, random_seed=seed + 987654
    )
    conditions = (
        "baseline",
        "centroid_once",
        "rank1_once",
        "rank1_clamp",
        "centroid_each",
        "reverse_rank1_clamp",
        "orthogonal_each",
        "matched_delta_once",
        "matched_delta_each",
        "matched_shuffled_each",
        "matched_reverse_each",
    )
    accumulators: dict[
        tuple[str, int, str], dict[str, float]
    ] = {}
    for condition in conditions:
        for extra_loop in range(1, extra_loops + 1):
            for target_kind in ("strict_one_hop", "phase_programmed", "endpoint"):
                accumulators[(condition, extra_loop, target_kind)] = {
                    "correct": 0.0,
                    "probability": 0.0,
                    "margin": 0.0,
                    "count": 0.0,
                    "best_position_sum": 0.0,
                    "best_position_accuracy_sum": 0.0,
                    "batch_count": 0.0,
                }

    set_seed(seed)
    centroid_delta = phase_shift(
        profile,
        reference_age=reference_age,
        receiver_age=cfg.max_loops,
        mode="centroid",
        random_seed=seed,
    )
    rank1_delta = phase_shift(
        profile,
        reference_age=reference_age,
        receiver_age=cfg.max_loops,
        mode="rank1",
        random_seed=seed,
    )
    for _ in range(batches):
        tokens, targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        terminal = cache_states_with_initial(
            model, tokens, loops=cfg.max_loops
        )[-1]
        all_targets = _all_targets(start, targets)
        endpoint = all_targets[:, cfg.max_depth]
        for condition in conditions:
            state = terminal.clone()
            if condition == "centroid_once":
                state = add_group_shift(
                    state,
                    positions=profile.positions,
                    shift=centroid_delta,
                )
            elif condition == "rank1_once":
                state = add_group_shift(
                    state,
                    positions=profile.positions,
                    shift=rank1_delta,
                )
            if condition == "matched_delta_once":
                reference_start = advance_nodes(
                    successors,
                    start,
                    steps=cfg.max_depth - matched_reference_position,
                )
                reference_tokens, _, _, _ = fixed_depth_batch(
                    cfg,
                    batch_size,
                    device,
                    path_positions=cfg.max_depth,
                    successors=successors,
                    start=reference_start,
                )
                reference_state = cache_states_with_initial(
                    model,
                    reference_tokens,
                    loops=max(1, matched_reference_age),
                )[matched_reference_age]
                state[:, matched_positions] += (
                    reference_state[:, matched_positions]
                    - terminal[:, matched_positions]
                )
            for extra_loop in range(1, extra_loops + 1):
                if condition == "rank1_clamp":
                    state = clamp_phase_coordinate(
                        state,
                        profile=profile,
                        target_coordinate=target_coordinate,
                    )
                elif condition == "centroid_each":
                    state = add_group_shift(
                        state,
                        positions=profile.positions,
                        shift=centroid_delta,
                    )
                elif condition == "reverse_rank1_clamp":
                    state = clamp_phase_coordinate(
                        state,
                        profile=profile,
                        target_coordinate=older_coordinate,
                    )
                elif condition == "orthogonal_each":
                    state = orthogonal_phase_control(
                        state,
                        profile=profile,
                        target_coordinate=target_coordinate,
                        orthogonal_unit=orthogonal_unit,
                    )
                elif condition in {
                    "matched_delta_each",
                    "matched_shuffled_each",
                    "matched_reverse_each",
                }:
                    current_position = (
                        cfg.max_depth
                        + matched_programmed_jump * (extra_loop - 1)
                    )
                    reference_start = advance_nodes(
                        successors,
                        start,
                        steps=current_position - matched_reference_position,
                    )
                    anchor_start = advance_nodes(
                        successors,
                        start,
                        steps=current_position - cfg.max_depth,
                    )
                    reference_tokens, _, _, _ = fixed_depth_batch(
                        cfg,
                        batch_size,
                        device,
                        path_positions=cfg.max_depth,
                        successors=successors,
                        start=reference_start,
                    )
                    anchor_tokens, _, _, _ = fixed_depth_batch(
                        cfg,
                        batch_size,
                        device,
                        path_positions=cfg.max_depth,
                        successors=successors,
                        start=anchor_start,
                    )
                    reference_state = cache_states_with_initial(
                        model,
                        reference_tokens,
                        loops=max(1, matched_reference_age),
                    )[matched_reference_age]
                    anchor_state = cache_states_with_initial(
                        model,
                        anchor_tokens,
                        loops=cfg.max_loops,
                    )[-1]
                    matched_delta = (
                        reference_state[:, matched_positions]
                        - anchor_state[:, matched_positions]
                    )
                    delta = (
                        matched_delta.roll(1, dims=0)
                        if condition == "matched_shuffled_each"
                        else -matched_delta
                        if condition == "matched_reverse_each"
                        else matched_delta
                    )
                    state[:, matched_positions] += delta
                state = apply_shared_stack(
                    model,
                    state,
                    loop_index=cfg.max_loops + extra_loop - 1,
                )
                logits = logits_from_raw_state(model, state)
                target_positions = {
                    "strict_one_hop": cfg.max_depth + extra_loop,
                    "phase_programmed": (
                        cfg.max_depth
                        + (
                            matched_programmed_jump
                            if condition.startswith("matched_")
                            else programmed_jump
                        )
                        * extra_loop
                    ),
                    "endpoint": cfg.max_depth,
                }
                best_position, best_accuracy = _best_path_position(
                    logits,
                    all_targets,
                    endpoint_position=cfg.max_depth,
                )
                for target_kind, target_position in target_positions.items():
                    metrics = _masked_metrics(
                        logits,
                        all_targets[:, target_position],
                        endpoint=(
                            None if target_kind == "endpoint" else endpoint
                        ),
                    )
                    bucket = accumulators[
                        (condition, extra_loop, target_kind)
                    ]
                    valid_count = int(metrics["valid_count"])
                    if valid_count:
                        bucket["correct"] += (
                            float(metrics["accuracy"]) * valid_count
                        )
                        bucket["probability"] += (
                            float(metrics["probability"]) * valid_count
                        )
                        bucket["margin"] += (
                            float(metrics["margin"]) * valid_count
                        )
                        bucket["count"] += valid_count
                    bucket["best_position_sum"] += best_position
                    bucket["best_position_accuracy_sum"] += best_accuracy
                    bucket["batch_count"] += 1
    rows: list[dict[str, Any]] = []
    for (condition, extra_loop, target_kind), bucket in accumulators.items():
        count = bucket["count"]
        batch_count = bucket["batch_count"]
        condition_jump = (
            matched_programmed_jump
            if condition.startswith("matched_")
            else programmed_jump
        )
        rows.append(
            {
                "condition": condition,
                "extra_loop": extra_loop,
                "target_kind": target_kind,
                "target_offset": (
                    extra_loop
                    if target_kind == "strict_one_hop"
                    else condition_jump * extra_loop
                    if target_kind == "phase_programmed"
                    else 0
                ),
                "programmed_jump": condition_jump,
                "accuracy": (
                    bucket["correct"] / count if count else float("nan")
                ),
                "probability": (
                    bucket["probability"] / count if count else float("nan")
                ),
                "margin": (
                    bucket["margin"] / count if count else float("nan")
                ),
                "valid_count": int(count),
                "mean_best_path_position": (
                    bucket["best_position_sum"] / batch_count
                ),
                "mean_best_path_accuracy": (
                    bucket["best_position_accuracy_sum"] / batch_count
                ),
            }
        )
    return rows


def _component_phase_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    profile: PhaseProfile,
    group: str,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    set_seed(seed)
    tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
    )
    _, trace = run_instrumented(model, tokens, max_loops=cfg.max_loops)
    positions = list(profile.positions)

    def age_units(update: torch.Tensor) -> float:
        mean_update = update[:, positions].mean(dim=0)
        return float((mean_update * profile.unit).sum() / profile.step_norm)

    rows: list[dict[str, Any]] = []
    for site in trace.sites:
        block = model.blocks[site.block_index]
        if not isinstance(block, TransformerBlock):
            raise TypeError("legacy TransformerBlock required")
        updates = {
            "attention_total": site.attention_out,
            "mlp": site.mlp_out,
        }
        for component, update in updates.items():
            rows.append(
                {
                    "group": group,
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "component": component,
                    "phase_age_units_written": age_units(update),
                }
            )
        for head in range(cfg.n_heads):
            isolated = torch.zeros_like(site.head_context)
            isolated[:, head] = site.head_context[:, head]
            head_update = _project_context(block.attn, isolated)
            rows.append(
                {
                    "group": group,
                    "loop": site.loop_index + 1,
                    "block": site.block_index + 1,
                    "component": f"H{head}",
                    "phase_age_units_written": age_units(head_update),
                }
            )
    return rows


def _lookup(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    extra_loop: int,
    target_kind: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["condition"] == condition
        and row["extra_loop"] == extra_loop
        and row["target_kind"] == target_kind
    ]
    if len(matches) != 1:
        raise ValueError("evaluation row lookup is not unique")
    return matches[0]


def _plot_phase(
    rows: list[dict[str, Any]],
    *,
    name: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for group in sorted({str(row["group"]) for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        ax.plot(
            [row["age"] for row in selected],
            [row["heldout_coordinate"] for row in selected],
            marker="o",
            label=group,
        )
    ax.set_xlabel("completed loops (age)")
    ax.set_ylabel("held-out phase coordinate")
    ax.set_title(f"{name}: candidate telomere coordinates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_closed_loop(
    rows: list[dict[str, Any]],
    *,
    name: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    selected = [
        row for row in rows if row["target_kind"] == "phase_programmed"
    ]
    for condition in (
        "baseline",
        "centroid_once",
        "rank1_once",
        "rank1_clamp",
        "centroid_each",
        "reverse_rank1_clamp",
        "orthogonal_each",
        "matched_delta_each",
        "matched_shuffled_each",
    ):
        curve = [row for row in selected if row["condition"] == condition]
        ax.plot(
            [row["extra_loop"] for row in curve],
            [row["accuracy"] for row in curve],
            marker="o",
            label=condition,
        )
    ax.set_xlabel("extra loop")
    ax.set_ylabel("collision-controlled continuation accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"{name}: causal overloop control")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    calibration_batch_size: int,
    calibration_batches: int,
    selection_batch_size: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    component_batch_size: int,
    extra_loops: int,
    seed: int,
    objective_label: str = "not_recorded_in_checkpoint",
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.block_schedule != "all_blocks" or cfg.n_layers != 2:
        raise ValueError("telomere study expects two shared all-block layers")
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    path_positions = cfg.max_depth + 2 * extra_loops

    centroids = _collect_centroids(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=seed,
    )
    heldout_centroids = _collect_centroids(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=calibration_batch_size,
        batches=max(1, calibration_batches // 2),
        seed=seed + 1,
    )
    groups = telomere_position_groups(cfg)
    profiles = {
        group: fit_phase_profile(centroids, positions)
        for group, positions in groups.items()
    }
    decode_rows = _phase_decode_rows(
        centroids=centroids,
        heldout_centroids=heldout_centroids,
        profiles=profiles,
    )
    _write_csv(run_dir / "phase_decode_rows.csv", decode_rows)

    trajectory_positions, trajectory_accuracies = _trajectory(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=selection_batch_size,
        path_positions=path_positions,
        seed=seed + 2,
    )
    selection_rows = _reset_selection_rows(
        model=model,
        cfg=cfg,
        profiles=profiles,
        trajectory_positions=trajectory_positions,
        trajectory_accuracies=trajectory_accuracies,
        device=device,
        batch_size=selection_batch_size,
        path_positions=path_positions,
        seed=seed + 3,
    )
    _write_csv(run_dir / "terminal_reset_selection_rows.csv", selection_rows)
    selected = choose_reset(selection_rows)
    matched_selection_rows = _matched_phase_selection_rows(
        model=model,
        cfg=cfg,
        profiles=profiles,
        trajectory_positions=trajectory_positions,
        trajectory_accuracies=trajectory_accuracies,
        device=device,
        batch_size=selection_batch_size,
        path_positions=path_positions,
        seed=seed + 30,
    )
    _write_csv(
        run_dir / "matched_phase_selection_rows.csv",
        matched_selection_rows,
    )
    matched_selected = choose_matched_reset(matched_selection_rows)
    selected_group = str(selected["group"])
    selected_profile = profiles[selected_group]

    evaluation_rows = _evaluation_rows(
        model=model,
        cfg=cfg,
        profile=selected_profile,
        selected=selected,
        matched_selected=matched_selected,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=seed + 4,
    )
    _write_csv(run_dir / "closed_loop_rows.csv", evaluation_rows)
    component_groups = {
        selected_group,
        str(matched_selected["group"]),
    }
    component_rows: list[dict[str, Any]] = []
    for group in sorted(component_groups):
        component_rows.extend(
            _component_phase_rows(
                model=model,
                cfg=cfg,
                profile=profiles[group],
                group=group,
                device=device,
                batch_size=component_batch_size,
                seed=seed + 5,
            )
        )
    _write_csv(run_dir / "component_phase_write_rows.csv", component_rows)

    target_kind = "phase_programmed"
    baseline_first = _lookup(
        evaluation_rows,
        condition="baseline",
        extra_loop=1,
        target_kind=target_kind,
    )
    centroid_first = _lookup(
        evaluation_rows,
        condition="centroid_once",
        extra_loop=1,
        target_kind=target_kind,
    )
    rank1_first = _lookup(
        evaluation_rows,
        condition="rank1_once",
        extra_loop=1,
        target_kind=target_kind,
    )
    orthogonal_first = _lookup(
        evaluation_rows,
        condition="orthogonal_each",
        extra_loop=1,
        target_kind=target_kind,
    )
    reverse_first = _lookup(
        evaluation_rows,
        condition="reverse_rank1_clamp",
        extra_loop=1,
        target_kind=target_kind,
    )
    rank1_curve = [
        _lookup(
            evaluation_rows,
            condition="rank1_clamp",
            extra_loop=extra_loop,
            target_kind=target_kind,
        )
        for extra_loop in range(1, extra_loops + 1)
    ]
    baseline_curve = [
        _lookup(
            evaluation_rows,
            condition="baseline",
            extra_loop=extra_loop,
            target_kind=target_kind,
        )
        for extra_loop in range(1, extra_loops + 1)
    ]
    matched_curve = [
        _lookup(
            evaluation_rows,
            condition="matched_delta_each",
            extra_loop=extra_loop,
            target_kind=target_kind,
        )
        for extra_loop in range(1, extra_loops + 1)
    ]
    matched_control_curve = [
        _lookup(
            evaluation_rows,
            condition="matched_shuffled_each",
            extra_loop=extra_loop,
            target_kind=target_kind,
        )
        for extra_loop in range(1, extra_loops + 1)
    ]
    centroid_gain = float(centroid_first["accuracy"]) - float(
        baseline_first["accuracy"]
    )
    rank1_gain = float(rank1_first["accuracy"]) - float(
        baseline_first["accuracy"]
    )
    causal_control_pass = (
        centroid_gain >= 0.15
        and float(centroid_first["accuracy"])
        >= float(orthogonal_first["accuracy"]) + 0.10
        and float(centroid_first["accuracy"])
        >= float(reverse_first["accuracy"]) + 0.10
    )
    scalar_control_pass = (
        rank1_gain >= 0.15
        and float(rank1_first["accuracy"])
        >= float(orthogonal_first["accuracy"]) + 0.10
    )
    clamp_mean_gain = float(
        np.mean(
            [
                float(active["accuracy"]) - float(base["accuracy"])
                for active, base in zip(
                    rank1_curve, baseline_curve, strict=True
                )
            ]
        )
    )
    matched_mean_control_gain = float(
        np.mean(
            [
                float(active["accuracy"]) - float(control["accuracy"])
                for active, control in zip(
                    matched_curve, matched_control_curve, strict=True
                )
            ]
        )
    )
    heldout_correlation = next(
        float(row["heldout_age_correlation"])
        for row in decode_rows
        if row["group"] == selected_group
    )
    heldout_monotone = next(
        float(row["heldout_monotone_fraction"])
        for row in decode_rows
        if row["group"] == selected_group
    )
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "config": asdict(cfg),
        "loss_mode": objective_label,
        "query_depth": cfg.max_depth,
        "trained_loops": cfg.max_loops,
        "trajectory_positions_including_initial": trajectory_positions,
        "trajectory_accuracies_including_initial": trajectory_accuracies,
        "selected_phase_reset": {
            "group": selected_group,
            "reference_age": int(selected["reference_age"]),
            "reference_next_loop": int(selected["reference_next_loop"]),
            "reference_path_before": int(selected["reference_path_before"]),
            "reference_path_after": int(selected["reference_path_after"]),
            "programmed_jump": int(selected["reference_jump"]),
            "selection_accuracy": float(selected["accuracy"]),
        },
        "selected_matched_phase_reset": {
            "group": str(matched_selected["group"]),
            "reference_age": int(matched_selected["reference_age"]),
            "reference_next_loop": int(
                matched_selected["reference_next_loop"]
            ),
            "reference_path_before": int(
                matched_selected["reference_path_before"]
            ),
            "reference_path_after": int(
                matched_selected["reference_path_after"]
            ),
            "programmed_jump": int(matched_selected["reference_jump"]),
            "selection_accuracy": float(matched_selected["accuracy"]),
        },
        "heldout_phase_decode": {
            "age_correlation": heldout_correlation,
            "monotone_fraction": heldout_monotone,
        },
        "causal_tests": {
            "baseline_first_programmed_accuracy": float(
                baseline_first["accuracy"]
            ),
            "centroid_once_first_programmed_accuracy": float(
                centroid_first["accuracy"]
            ),
            "rank1_once_first_programmed_accuracy": float(
                rank1_first["accuracy"]
            ),
            "orthogonal_first_programmed_accuracy": float(
                orthogonal_first["accuracy"]
            ),
            "reverse_first_programmed_accuracy": float(
                reverse_first["accuracy"]
            ),
            "centroid_gain_over_baseline": centroid_gain,
            "rank1_gain_over_baseline": rank1_gain,
            "rank1_clamp_mean_gain_over_baseline": clamp_mean_gain,
            "matched_delta_mean_gain_over_shuffled": (
                matched_mean_control_gain
            ),
        },
        "claim_ledger": {
            "monotone_candidate_supported": (
                heldout_correlation >= 0.90
                and heldout_monotone >= 0.75
            ),
            "content_preserving_phase_control_supported": causal_control_pass,
            "scalar_telomere_control_supported": scalar_control_pass,
            "closed_loop_overloop_supported": clamp_mean_gain >= 0.15,
            "content_matched_closed_loop_supported": (
                matched_mean_control_gain >= 0.15
            ),
            "interpretation": (
                "scalar_telomere"
                if scalar_control_pass and clamp_mean_gain >= 0.15
                else "content_dependent_telomere"
                if matched_mean_control_gain >= 0.15
                else "distributed_or_curved_phase_state"
                if causal_control_pass
                else "phase_is_decodable_but_not_independently_controllable"
            ),
        },
        "controls": {
            "calibration_evaluation_split": True,
            "content_preservation": (
                "phase reset adds calibration mean differences while "
                "preserving each evaluation example's residual deviation"
            ),
            "direction_controls": [
                "reverse/older phase",
                "norm-matched orthogonal direction",
            ],
            "continuation_metric": (
                "future target accuracy excludes examples whose future "
                "node collides with the trained f^8 endpoint"
            ),
        },
        "sample_sizes": {
            "calibration": calibration_batch_size * calibration_batches,
            "heldout_decode": calibration_batch_size
            * max(1, calibration_batches // 2),
            "selection": selection_batch_size,
            "evaluation": evaluation_batch_size * evaluation_batches,
            "component": component_batch_size,
        },
        "extra_loops": extra_loops,
        "cuda_peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_phase(
        decode_rows,
        name=name,
        path=run_dir / "phase_coordinate_by_loop.png",
    )
    _plot_closed_loop(
        evaluation_rows,
        name=name,
        path=run_dir / "closed_loop_continuation.png",
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    name, checkpoint = text.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def parse_label_spec(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("objective label must be NAME=LABEL")
    name, label = text.split("=", 1)
    if not name or not label:
        raise argparse.ArgumentTypeError("objective label must be NAME=LABEL")
    return name, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a monotone phase/telomere state controls graph-path "
            "jump size and overloop shutdown."
        )
    )
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument(
        "--objective-label",
        action="append",
        type=parse_label_spec,
        default=[],
        help=(
            "record training objective provenance as NAME=LABEL; checkpoints "
            "do not otherwise retain this information"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--selection-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--component-batch-size", type=int, default=128)
    parser.add_argument("--extra-loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries: dict[str, Any] = {}
    objective_labels = dict(args.objective_label)
    for name, checkpoint in args.run:
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            calibration_batch_size=args.calibration_batch_size,
            calibration_batches=args.calibration_batches,
            selection_batch_size=args.selection_batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            evaluation_batches=args.evaluation_batches,
            component_batch_size=args.component_batch_size,
            extra_loops=args.extra_loops,
            seed=args.seed,
            objective_label=objective_labels.get(
                name, "not_recorded_in_checkpoint"
            ),
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
