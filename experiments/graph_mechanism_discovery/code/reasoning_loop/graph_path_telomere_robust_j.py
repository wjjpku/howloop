from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
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
from reasoning_loop.graph_path_telomere_initializer_semantic_readout import (
    permutation_orbit,
    semantic_offsets,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    _relative_mse,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _position_samples_with_weights,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


@dataclass
class WeightedAffineStats:
    """Streaming sufficient statistics for weighted affine update regression."""

    dimension: int
    sum_weight: torch.Tensor = field(init=False)
    sum_source: torch.Tensor = field(init=False)
    sum_target: torch.Tensor = field(init=False)
    sum_source_source: torch.Tensor = field(init=False)
    sum_source_update: torch.Tensor = field(init=False)
    sum_target_target: torch.Tensor = field(init=False)
    raw_rows: int = 0
    effective_rows: float = 0.0
    group_effective_rows: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def __post_init__(self) -> None:
        self.sum_weight = torch.zeros((), dtype=torch.float64)
        self.sum_source = torch.zeros(self.dimension, dtype=torch.float64)
        self.sum_target = torch.zeros(self.dimension, dtype=torch.float64)
        self.sum_source_source = torch.zeros(
            self.dimension,
            self.dimension,
            dtype=torch.float64,
        )
        self.sum_source_update = torch.zeros(
            self.dimension,
            self.dimension,
            dtype=torch.float64,
        )
        self.sum_target_target = torch.zeros(
            self.dimension,
            self.dimension,
            dtype=torch.float64,
        )

    def add(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        *,
        group: str,
    ) -> None:
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError("source and target must share [sample, feature]")
        if source.shape[1] != self.dimension:
            raise ValueError("feature dimension does not match accumulator")
        if weights.shape != (source.shape[0],):
            raise ValueError("weights must have one entry per sample")
        if bool(weights.le(0).any()):
            raise ValueError("weights must be positive")
        # Form the expensive 256x256 products on the source device (normally
        # the A100), then keep only the small sufficient statistics on CPU.
        source_work = source.detach().float()
        target_work = target.detach().float()
        weights_work = weights.detach().to(
            device=source.device,
            dtype=torch.float32,
        )
        update_work = target_work - source_work
        weighted_source = source_work * weights_work[:, None]
        self.sum_weight += weights_work.sum().double().cpu()
        self.sum_source += weighted_source.sum(dim=0).double().cpu()
        self.sum_target += (
            target_work * weights_work[:, None]
        ).sum(dim=0).double().cpu()
        self.sum_source_source += (
            source_work.T @ weighted_source
        ).double().cpu()
        self.sum_source_update += (
            source_work.T @ (update_work * weights_work[:, None])
        ).double().cpu()
        self.sum_target_target += (
            target_work.T
            @ (target_work * weights_work[:, None])
        ).double().cpu()
        self.raw_rows += int(source.shape[0])
        effective = float(weights_work.sum())
        self.effective_rows += effective
        self.group_effective_rows[group] += effective

    def fit(self, *, ridge: float, device: torch.device) -> VectorAffine:
        if ridge < 0:
            raise ValueError("ridge must be nonnegative")
        if float(self.sum_weight) <= 0:
            raise ValueError("cannot fit empty statistics")
        mean_source = self.sum_source / self.sum_weight
        mean_target = self.sum_target / self.sum_weight
        sum_update = self.sum_target - self.sum_source
        gram = self.sum_source_source - torch.outer(
            self.sum_source,
            self.sum_source,
        ) / self.sum_weight
        cross = self.sum_source_update - torch.outer(
            self.sum_source,
            sum_update,
        ) / self.sum_weight
        scale = gram.diagonal().mean().clamp_min(1e-12)
        update = torch.linalg.solve(
            gram
            + ridge
            * scale
            * torch.eye(self.dimension, dtype=torch.float64),
            cross,
        )
        weight = torch.eye(self.dimension, dtype=torch.float64) + update
        bias = mean_target - mean_source @ weight
        return VectorAffine(
            weight=weight.to(device=device, dtype=torch.float32),
            bias=bias.to(device=device, dtype=torch.float32),
            update_rank=self.dimension,
            fit_dimension=self.dimension,
            retained_fit_energy=1.0,
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legal_schedule(
    *,
    name: str,
    horizon: int,
    start_age: int,
    seed: int,
) -> tuple[bool, ...]:
    """Return an ID schedule, forcing a reset before the represented age > 8."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 2 <= start_age <= 8:
        raise ValueError("start_age must be in [2, 8]")
    rng = np.random.default_rng(seed)
    age = start_age
    decisions: list[bool] = []
    for cycle in range(1, horizon + 1):
        if name == "every1":
            proposed = True
        elif name.startswith("period"):
            period = int(name.removeprefix("period"))
            proposed = (cycle - 1) % period == 0
        elif name == "late":
            proposed = age == 8
        elif name == "burst":
            proposed = (cycle - 1) % 6 in (0, 1)
        elif name.startswith("random"):
            probability = int(name.removeprefix("random")) / 100.0
            proposed = bool(rng.random() < probability)
        else:
            raise ValueError(f"unknown schedule: {name}")
        apply = proposed or age == 8
        decisions.append(apply)
        age = 3 if apply else age + 1
        if age > 8:
            raise RuntimeError("schedule left the registered ID age range")
    return tuple(decisions)


def schedule_age_trace(
    *,
    schedule: Sequence[bool],
    start_age: int,
) -> tuple[int, ...]:
    age = start_age
    trace = []
    for apply in schedule:
        trace.append(age)
        age = 3 if apply else age + 1
        if age > 8:
            raise ValueError("schedule leaves the registered ID age range")
    return tuple(trace)


def _training_case(
    *,
    batch_index: int,
    rollout_horizons: Sequence[int],
    schedule_names: Sequence[str],
    seed: int,
) -> tuple[str, int, int]:
    schedule_index = batch_index % len(schedule_names)
    age_index = (batch_index // len(schedule_names)) % 7
    schedule_name = str(schedule_names[schedule_index])
    start_age = 2 + age_index
    horizon_index = (
        schedule_index + age_index + seed // 1000
    ) % len(rollout_horizons)
    horizon = int(rollout_horizons[horizon_index])
    return schedule_name, start_age, horizon


def _add_position_pair(
    stats: WeightedAffineStats,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    positions: tuple[int, ...],
    answer_position: int,
    answer_weight: int,
    group: str,
) -> None:
    source_samples, weights = _position_samples_with_weights(
        source,
        positions,
        answer_position=answer_position,
        answer_weight=answer_weight,
    )
    target_samples, target_weights = _position_samples_with_weights(
        target,
        positions,
        answer_position=answer_position,
        answer_weight=answer_weight,
    )
    if not torch.equal(weights, target_weights):
        raise RuntimeError("source and target weights differ")
    stats.add(
        source_samples.float(),
        target_samples.float(),
        weights,
        group=group,
    )


@torch.no_grad()
def collect_natural_age_anchors(
    *,
    stats: WeightedAffineStats,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_position: int,
    answer_weight: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, float]:
    set_seed(seed)
    before = dict(stats.group_effective_rows)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        exact_h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        oracle = run_one_loop(model, exact_h2, loop_index=cfg.max_loops)
        for age in range(2, 9):
            state = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=age,
                phase_position=phase_positions[age],
            )
            step = run_one_loop(model, state, loop_index=cfg.max_loops)
            _add_position_pair(
                stats,
                source=step.block2_hidden_pre_intervention,
                target=oracle.block2_hidden_in,
                positions=positions,
                answer_position=answer_position,
                answer_weight=answer_weight,
                group=f"natural_H{age}",
            )
    return {
        key: value - before.get(key, 0.0)
        for key, value in stats.group_effective_rows.items()
    }


@torch.no_grad()
def collect_on_policy_schedule(
    *,
    stats: WeightedAffineStats,
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
    rollout_horizons: Sequence[int],
    schedule_names: Sequence[str],
    seed: int,
) -> list[dict[str, Any]]:
    jump = phase_positions[3] - phase_positions[2]
    rows: list[dict[str, Any]] = []
    set_seed(seed)
    for batch_index in range(batches):
        # One 9x7 round covers every schedule/age pair. Consecutive DAgger
        # rounds rotate those pairs through all registered horizons.
        schedule_name, start_age, horizon = _training_case(
            batch_index=batch_index,
            rollout_horizons=rollout_horizons,
            schedule_names=schedule_names,
            seed=seed,
        )
        schedule = legal_schedule(
            name=schedule_name,
            horizon=horizon,
            start_age=start_age,
            seed=seed * 1009 + batch_index,
        )
        age_trace = schedule_age_trace(
            schedule=schedule,
            start_age=start_age,
        )
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=start_age,
            phase_position=phase_positions[start_age],
        )
        new_effective = 0.0
        application_count = 0
        for cycle, apply in enumerate(schedule, start=1):
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            exact_h2 = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=2,
                phase_position=phase_positions[2],
            )
            oracle = run_one_loop(
                model,
                exact_h2,
                loop_index=cfg.max_loops + cycle - 1,
            )
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    (positions, age_map) if apply else None
                ),
            )
            if apply:
                before = stats.effective_rows
                _add_position_pair(
                    stats,
                    source=step.block2_hidden_pre_intervention,
                    target=oracle.block2_hidden_in,
                    positions=positions,
                    answer_position=answer_position,
                    answer_weight=answer_weight,
                    group=f"on_policy_{schedule_name}_H{age_trace[cycle - 1]}",
                )
                new_effective += stats.effective_rows - before
                application_count += 1
            state = step.state
        rows.append(
            {
                "batch": batch_index,
                "schedule": schedule_name,
                "start_age": start_age,
                "horizon": horizon,
                "applications": application_count,
                "new_effective_position_pairs": new_effective,
            }
        )
    return rows


class TrainableVectorAffine(torch.nn.Module):
    def __init__(self, initial: VectorAffine) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(initial.weight.detach().clone())
        self.bias = torch.nn.Parameter(initial.bias.detach().clone())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.weight + self.bias

    def frozen(self) -> VectorAffine:
        return VectorAffine(
            weight=self.weight.detach().clone(),
            bias=self.bias.detach().clone(),
            update_rank=self.weight.shape[0],
            fit_dimension=self.weight.shape[0],
            retained_fit_energy=1.0,
        )


def fine_tune_task_j(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    initial_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches_per_round: int,
    rounds: int,
    rollout_horizons: Sequence[int],
    schedule_names: Sequence[str],
    learning_rate: float,
    state_loss_weight: float,
    seed: int,
) -> tuple[VectorAffine, list[dict[str, Any]]]:
    """Directly optimize current-step CE on ID on-policy compositions."""

    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if state_loss_weight < 0:
        raise ValueError("state_loss_weight must be nonnegative")
    if rounds == 0:
        return initial_map, []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    module = TrainableVectorAffine(initial_map).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    jump = phase_positions[3] - phase_positions[2]
    rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        round_seed = seed + 1000 * round_index
        set_seed(round_seed)
        totals: dict[str, float] = defaultdict(float)
        for batch_index in range(batches_per_round):
            schedule_name, start_age, horizon = _training_case(
                batch_index=batch_index,
                rollout_horizons=rollout_horizons,
                schedule_names=schedule_names,
                seed=round_seed,
            )
            schedule = legal_schedule(
                name=schedule_name,
                horizon=horizon,
                start_age=start_age,
                seed=round_seed * 1009 + batch_index,
            )
            application_count = sum(schedule)
            if application_count < 1:
                totals["skipped_no_action_episodes"] += 1
                continue
            with torch.no_grad():
                _, path_targets, successors, _ = fixed_depth_batch(
                    cfg,
                    batch_size,
                    device,
                    path_positions=cfg.max_depth,
                )
                endpoint = path_targets[:, cfg.max_depth - 1]
                state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=endpoint,
                    age=start_age,
                    phase_position=phase_positions[start_age],
                )
            optimizer.zero_grad(set_to_none=True)
            task_losses: list[torch.Tensor] = []
            state_losses: list[torch.Tensor] = []
            graph_started = False
            for cycle, apply in enumerate(schedule, start=1):
                current = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * (cycle - 1),
                )
                target = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * cycle,
                )
                if not apply and not graph_started:
                    with torch.no_grad():
                        state = run_one_loop(
                            model,
                            state,
                            loop_index=cfg.max_loops + cycle - 1,
                        ).state
                    continue
                oracle_interface = None
                if apply:
                    graph_started = True
                if apply:
                    with torch.no_grad():
                        exact_h2 = _aligned_state_at_age(
                            model=model,
                            cfg=cfg,
                            successors=successors,
                            current=current,
                            age=2,
                            phase_position=phase_positions[2],
                        )
                        oracle = run_one_loop(
                            model,
                            exact_h2,
                            loop_index=cfg.max_loops + cycle - 1,
                        )
                        oracle_interface = oracle.block2_hidden_in[
                            :, list(positions)
                        ]
                # The public circuit-analysis helper is decorated with
                # torch.no_grad(). Its wrapped implementation is otherwise
                # identical and is needed here so gradients reach J through
                # the entire registered F/J composition.
                differentiable_run_one_loop = run_one_loop.__wrapped__
                step = differentiable_run_one_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, module) if apply else None
                    ),
                )
                task_loss = F.cross_entropy(step.logits.float(), target)
                task_losses.append(task_loss)
                prediction = step.logits.argmax(dim=-1)
                totals["supervised_steps"] += 1
                totals["examples"] += batch_size
                totals["correct"] += int(prediction.eq(target).sum())
                totals["task_loss"] += float(task_loss.detach())
                if apply:
                    if oracle_interface is None:
                        raise RuntimeError("missing oracle at J application")
                    live_interface = step.block2_hidden_in[
                        :, list(positions)
                    ]
                    target_scale = (
                        oracle_interface.float()
                        - oracle_interface.float().mean(
                            dim=(0, 1),
                            keepdim=True,
                        )
                    ).square().mean().clamp_min(1e-6)
                    state_loss = (
                        live_interface.float() - oracle_interface.float()
                    ).square().mean() / target_scale
                    state_losses.append(state_loss)
                    totals["applications"] += 1
                    totals["state_loss"] += float(state_loss.detach())
                state = step.state
            if not task_losses or not state_losses:
                raise RuntimeError("action episode produced no trainable loss")
            loss = torch.stack(task_losses).mean() + state_loss_weight * (
                torch.stack(state_losses).mean()
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            totals["optimizer_steps"] += 1
            totals["grad_norm_sum"] += float(grad_norm)
        rows.append(
            {
                "round": round_index,
                "optimizer_steps": int(totals["optimizer_steps"]),
                "skipped_no_action_episodes": int(
                    totals["skipped_no_action_episodes"]
                ),
                "J_applications": int(totals["applications"]),
                "supervised_composition_steps": int(
                    totals["supervised_steps"]
                ),
                "examples": int(totals["examples"]),
                "composition_accuracy": (
                    totals["correct"] / totals["examples"]
                ),
                "task_ce": (
                    totals["task_loss"] / totals["supervised_steps"]
                ),
                "interface_relative_mse": (
                    totals["state_loss"] / totals["applications"]
                ),
                "grad_norm_mean": (
                    totals["grad_norm_sum"] / totals["optimizer_steps"]
                ),
                "learning_rate": learning_rate,
                "state_loss_weight": state_loss_weight,
            }
        )
    return module.frozen(), rows


def _condition_schedule(
    condition: str,
    *,
    horizon: int,
    seed: int,
) -> tuple[bool, ...]:
    if condition in {"no_control", "exact_oracle_every1"}:
        return tuple(condition == "exact_oracle_every1" for _ in range(horizon))
    base = condition
    if condition in {"J_every1", "shuffled_J_every1"}:
        base = "every1"
    elif condition.startswith("J_"):
        base = condition.removeprefix("J_")
    return legal_schedule(name=base, horizon=horizon, start_age=8, seed=seed)


def _aggregate_behavior(
    sums: dict[tuple[str, int], dict[str, float]],
    mode_counts: dict[tuple[str, int, int], int],
    *,
    node_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (condition, cycle), values in sorted(sums.items()):
        orbit_count = values["orbit8_count"]
        mode_offset, mode_count = max(
            (
                (offset, mode_counts[(condition, cycle, offset)])
                for offset in range(node_count)
            ),
            key=lambda item: item[1],
        )
        rows.append(
            {
                "condition": condition,
                "cycle": cycle,
                "J_applied": bool(values["J_applied"]),
                "accuracy_all": values["correct_all"] / values["count"],
                "accuracy_orbit8": (
                    values["correct_orbit8"] / orbit_count
                    if orbit_count
                    else float("nan")
                ),
                "orbit8_count": int(orbit_count),
                "accuracy_nonendpoint": (
                    values["correct_nonendpoint"]
                    / values["nonendpoint_count"]
                    if values["nonendpoint_count"]
                    else float("nan")
                ),
                "nonendpoint_count": int(values["nonendpoint_count"]),
                "orbit8_top1_mode_offset": mode_offset,
                "orbit8_top1_mode_fraction": (
                    mode_count / orbit_count if orbit_count else float("nan")
                ),
                "interface_relative_mse": (
                    values["interface_error_sum"] / values["batches"]
                ),
                "output_norm_mean": values["output_norm_sum"] / values["count"],
            }
        )
    return rows


@torch.no_grad()
def evaluate_robust_j(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    age_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    jump = phase_positions[3] - phase_positions[2]
    conditions = (
        "no_control",
        "J_every1",
        "J_period2",
        "J_period3",
        "J_period6",
        "J_late",
        "J_burst",
        "J_random25",
        "J_random50",
        "J_random75",
        "shuffled_J_every1",
        "exact_oracle_every1",
    )
    schedules = {
        condition: _condition_schedule(
            condition,
            horizon=continuation_loops,
            seed=seed + 10007 * index,
        )
        for index, condition in enumerate(conditions)
    }
    sums: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    mode_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        orbit, orbit_lengths = permutation_orbit(successors, endpoint)
        orbit8 = orbit_lengths.eq(cfg.node_count)
        states = {condition: initial.clone() for condition in conditions}
        for cycle in range(1, continuation_loops + 1):
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            target = advance_nodes(
                successors,
                endpoint,
                steps=jump * cycle,
            )
            nonendpoint = target.ne(endpoint)
            exact_h2 = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=2,
                phase_position=phase_positions[2],
            )
            oracle = run_one_loop(
                model,
                exact_h2,
                loop_index=cfg.max_loops + cycle - 1,
            )
            oracle_interface = oracle.block2_hidden_in[:, list(positions)]
            next_states = {}
            for condition in conditions:
                apply = schedules[condition][cycle - 1]
                transform = None
                override = None
                if condition == "exact_oracle_every1":
                    override = (positions, oracle_interface)
                elif apply:
                    if condition == "shuffled_J_every1":
                        transform = (
                            positions,
                            lambda value: age_map(value).roll(1, dims=0),
                        )
                    else:
                        transform = (positions, age_map)
                step = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=transform,
                    block2_position_override=override,
                )
                next_states[condition] = step.state
                prediction = step.logits.argmax(dim=-1)
                offsets = semantic_offsets(
                    prediction,
                    orbit,
                    orbit_lengths,
                )
                values = sums[(condition, cycle)]
                values["J_applied"] = float(apply)
                values["count"] += batch_size
                values["correct_all"] += int(prediction.eq(target).sum())
                values["orbit8_count"] += int(orbit8.sum())
                values["correct_orbit8"] += int(
                    (prediction.eq(target) & orbit8).sum()
                )
                values["nonendpoint_count"] += int(nonendpoint.sum())
                values["correct_nonendpoint"] += int(
                    (prediction.eq(target) & nonendpoint).sum()
                )
                values["interface_error_sum"] += _relative_mse(
                    step.block2_hidden_in[:, list(positions)],
                    oracle_interface,
                )
                values["output_norm_sum"] += float(
                    step.state.float().norm(dim=-1).mean(dim=1).sum()
                )
                values["batches"] += 1
                orbit_offsets = offsets[orbit8]
                for offset in range(cfg.node_count):
                    mode_counts[(condition, cycle, offset)] += int(
                        orbit_offsets.eq(offset).sum()
                    )
            states = next_states
    return _aggregate_behavior(sums, mode_counts, node_count=cfg.node_count)


@torch.no_grad()
def evaluate_one_shot_grid(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    age_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply J once after every legal start-age/delay combination."""

    jump = phase_positions[3] - phase_positions[2]
    storage: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        for start_age in range(2, 9):
            # Delay d means d natural loops, followed by one J-controlled loop.
            for delay in range(0, 9 - start_age):
                state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=endpoint,
                    age=start_age,
                    phase_position=phase_positions[start_age],
                )
                for skipped in range(delay):
                    state = run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + skipped,
                    ).state
                current = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * delay,
                )
                target = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * (delay + 1),
                )
                exact_h2 = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=2,
                    phase_position=phase_positions[2],
                )
                oracle = run_one_loop(
                    model,
                    exact_h2,
                    loop_index=cfg.max_loops + delay,
                )
                step = run_one_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops + delay,
                    block2_position_transform=(positions, age_map),
                )
                values = storage[(start_age, delay)]
                values["count"] += batch_size
                values["correct"] += int(
                    step.logits.argmax(dim=-1).eq(target).sum()
                )
                values["interface_error"] += _relative_mse(
                    step.block2_hidden_in[:, list(positions)],
                    oracle.block2_hidden_in[:, list(positions)],
                )
                values["batches"] += 1
    return [
        {
            "start_age": start_age,
            "delay_before_J": delay,
            "source_age_at_J": start_age + delay,
            "accuracy_at_J_step": values["correct"] / values["count"],
            "interface_relative_mse": (
                values["interface_error"] / values["batches"]
            ),
            "graphs": int(values["count"]),
        }
        for (start_age, delay), values in sorted(storage.items())
    ]


def summarize_curves(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        parts = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: int(row["cycle"]),
        )
        accuracy = [float(row["accuracy_orbit8"]) for row in parts]
        summary[condition] = {
            "accuracy_orbit8": accuracy,
            "auc_8": float(np.mean(accuracy[:8])),
            "auc_24": float(np.mean(accuracy[:24])),
            "auc_all": float(np.mean(accuracy)),
            "minimum": float(np.min(accuracy)),
            "final": accuracy[-1],
            "interface_relative_mse_mean": float(
                np.mean([float(row["interface_relative_mse"]) for row in parts])
            ),
        }
    return summary


def _plot_curves(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = (
        "no_control",
        "J_every1",
        "J_period2",
        "J_period3",
        "J_period6",
        "J_random50",
        "shuffled_J_every1",
        "exact_oracle_every1",
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for condition in selected:
        parts = [row for row in rows if row["condition"] == condition]
        cycles = [int(row["cycle"]) for row in parts]
        axes[0].plot(
            cycles,
            [float(row["accuracy_orbit8"]) for row in parts],
            label=condition,
            linewidth=1.7,
        )
        axes[1].plot(
            cycles,
            [float(row["interface_relative_mse"]) for row in parts],
            label=condition,
            linewidth=1.7,
        )
    axes[0].set_ylabel("Current-step accuracy (orbit length 8)")
    axes[1].set_ylabel("Relative MSE to exact H2 Block-2 interface")
    for axis in axes:
        axis.set_xlabel("Continuation loop")
        axis.grid(alpha=0.2)
    axes[0].set_ylim(-0.04, 1.04)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, ncol=2)
    figure.savefig(out_dir / "robust_j_accuracy_and_interface.png", dpi=180)
    plt.close(figure)


def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    dagger_batch_size: int,
    dagger_batches_per_round: int,
    dagger_rounds: int,
    rollout_horizons: Sequence[int],
    schedule_names: Sequence[str],
    evaluation_batch_size: int,
    evaluation_batches: int,
    continuation_loops: int,
    answer_weight: int,
    ridge: float,
    task_finetune_batch_size: int,
    task_finetune_batches_per_round: int,
    task_finetune_rounds: int,
    task_learning_rate: float,
    task_state_loss_weight: float,
    calibration_seed: int,
    dagger_seed: int,
    task_finetune_seed: int,
    evaluation_seed: int,
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
    shared_gpu: bool,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = WeightedAffineStats(cfg.d_model)
    collect_natural_age_anchors(
        stats=stats,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_position=answer_position,
        answer_weight=answer_weight,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
    )
    age_map = stats.fit(ridge=ridge, device=device)
    training_rows: list[dict[str, Any]] = [
        {
            "round": 0,
            "raw_position_rows": stats.raw_rows,
            "effective_position_rows": stats.effective_rows,
            "new_effective_position_rows": stats.effective_rows,
            "dagger_batches": 0,
        }
    ]
    schedule_rows: list[dict[str, Any]] = []
    for round_index in range(1, dagger_rounds + 1):
        before = stats.effective_rows
        new_rows = collect_on_policy_schedule(
            stats=stats,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            answer_position=answer_position,
            answer_weight=answer_weight,
            age_map=age_map,
            device=device,
            batch_size=dagger_batch_size,
            batches=dagger_batches_per_round,
            rollout_horizons=rollout_horizons,
            schedule_names=schedule_names,
            seed=dagger_seed + 1000 * round_index,
        )
        for row in new_rows:
            row["round"] = round_index
        schedule_rows.extend(new_rows)
        age_map = stats.fit(ridge=ridge, device=device)
        training_rows.append(
            {
                "round": round_index,
                "raw_position_rows": stats.raw_rows,
                "effective_position_rows": stats.effective_rows,
                "new_effective_position_rows": stats.effective_rows - before,
                "dagger_batches": dagger_batches_per_round,
            }
        )
    age_map, task_training_rows = fine_tune_task_j(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        initial_map=age_map,
        device=device,
        batch_size=task_finetune_batch_size,
        batches_per_round=task_finetune_batches_per_round,
        rounds=task_finetune_rounds,
        rollout_horizons=rollout_horizons,
        schedule_names=schedule_names,
        learning_rate=task_learning_rate,
        state_loss_weight=task_state_loss_weight,
        seed=task_finetune_seed,
    )
    evaluation_rows = evaluate_robust_j(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        age_map=age_map,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        continuation_loops=continuation_loops,
        seed=evaluation_seed,
    )
    one_shot_rows = evaluate_one_shot_grid(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        age_map=age_map,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        seed=evaluation_seed + 1,
    )
    curves = summarize_curves(evaluation_rows)
    torch.save(
        {
            "kind": "graph_path_telomere_robust_j",
            "checkpoint": str(checkpoint),
            "positions": positions,
            "answer_position": answer_position,
            "map": {
                "weight": age_map.weight.detach().cpu(),
                "bias": age_map.bias.detach().cpu(),
                "rank": age_map.update_rank,
            },
            "training_seeds": {
                "calibration": calibration_seed,
                "dagger": dagger_seed,
                "task_finetune": task_finetune_seed,
            },
        },
        out_dir / "robust_j.pt",
    )
    _write_csv(out_dir / "training_rounds.csv", training_rows)
    _write_csv(out_dir / "training_schedules.csv", schedule_rows)
    _write_csv(out_dir / "task_finetune_rounds.csv", task_training_rows)
    _write_csv(out_dir / "evaluation_rows.csv", evaluation_rows)
    _write_csv(out_dir / "one_shot_age_delay_grid.csv", one_shot_rows)
    _plot_curves(evaluation_rows, out_dir=out_dir)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": "frozen model final-only; J weighted interface MSE",
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "J": {
            "kind": "single token-wise full affine with bias",
            "interface": "residual stream immediately before physical Block 2",
            "positions": positions,
            "parameter_count": cfg.d_model * cfg.d_model + cfg.d_model,
            "ridge": ridge,
            "answer_weight": answer_weight,
        },
        "training": {
            "natural_ages": list(range(2, 9)),
            "natural_graphs": calibration_batch_size * calibration_batches,
            "dagger_rounds": dagger_rounds,
            "dagger_graphs_per_round": (
                dagger_batch_size * dagger_batches_per_round
            ),
            "task_finetune_rounds": task_finetune_rounds,
            "task_finetune_graphs_per_round": (
                task_finetune_batch_size
                * task_finetune_batches_per_round
            ),
            "task_learning_rate": task_learning_rate,
            "task_state_loss_weight": task_state_loss_weight,
            "task_finetune_rows": task_training_rows,
            "rollout_horizons": list(rollout_horizons),
            "schedule_names": list(schedule_names),
            "raw_position_rows": stats.raw_rows,
            "effective_position_rows": stats.effective_rows,
            "group_effective_rows": dict(stats.group_effective_rows),
            "excluded_as_ood": [
                "Gaussian hidden-state noise",
                "arbitrary layer/interface insertion",
                "other checkpoints",
            ],
        },
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "evaluation_seed": evaluation_seed,
        "curves": curves,
        "one_shot_age_delay_grid": one_shot_rows,
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "shared_gpu": shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "artifact": "robust_j.pt",
            "training_rounds": "training_rounds.csv",
            "training_schedules": "training_schedules.csv",
            "task_finetune_rounds": "task_finetune_rounds.csv",
            "evaluation": "evaluation_rows.csv",
            "one_shot_age_delay_grid": "one_shot_age_delay_grid.csv",
            "figure": "robust_j_accuracy_and_interface.png",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one robust schedule-diverse affine J on D8L8."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--dagger-batches-per-round", type=int, default=63)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=(4, 8, 16, 24),
    )
    parser.add_argument(
        "--schedule-names",
        nargs="+",
        default=(
            "every1",
            "period2",
            "period3",
            "period6",
            "late",
            "burst",
            "random25",
            "random50",
            "random75",
        ),
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--answer-weight", type=int, default=28)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--task-finetune-batch-size", type=int, default=64)
    parser.add_argument(
        "--task-finetune-batches-per-round",
        type=int,
        default=63,
    )
    parser.add_argument("--task-finetune-rounds", type=int, default=4)
    parser.add_argument("--task-learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--task-state-loss-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument("--calibration-seed", type=int, default=151001)
    parser.add_argument("--dagger-seed", type=int, default=151002)
    parser.add_argument("--task-finetune-seed", type=int, default=151004)
    parser.add_argument("--evaluation-seed", type=int, default=151003)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=3.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
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
        dagger_batch_size=args.dagger_batch_size,
        dagger_batches_per_round=args.dagger_batches_per_round,
        dagger_rounds=args.dagger_rounds,
        rollout_horizons=args.rollout_horizons,
        schedule_names=args.schedule_names,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        continuation_loops=args.continuation_loops,
        answer_weight=args.answer_weight,
        ridge=args.ridge,
        task_finetune_batch_size=args.task_finetune_batch_size,
        task_finetune_batches_per_round=(
            args.task_finetune_batches_per_round
        ),
        task_finetune_rounds=args.task_finetune_rounds,
        task_learning_rate=args.task_learning_rate,
        task_state_loss_weight=args.task_state_loss_weight,
        calibration_seed=args.calibration_seed,
        dagger_seed=args.dagger_seed,
        task_finetune_seed=args.task_finetune_seed,
        evaluation_seed=args.evaluation_seed,
        physical_gpu=args.physical_gpu,
        prelaunch_used_mib=args.prelaunch_used_mib,
        prelaunch_free_mib=args.prelaunch_free_mib,
        declared_peak_gib=args.declared_peak_gib,
        reserve_gib=args.reserve_gib,
        shared_gpu=args.shared_gpu,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
