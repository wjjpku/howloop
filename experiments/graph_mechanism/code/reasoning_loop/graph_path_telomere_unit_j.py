from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

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
from reasoning_loop.graph_path_telomere_robust_j import (
    TrainableVectorAffine,
    WeightedAffineStats,
)
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _position_samples_with_weights,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


POLICIES = (
    "reset3",
    "maintain4",
    "unit_every",
    "periodic_reset",
    "alternating",
    "random_sparse",
    "random_power",
)

TASK_PARAMETERIZATIONS = (
    "full",
    "lora",
    "identity_low_rank",
    "scalar_low_rank",
    "diagonal_low_rank",
    "weight_low_rank",
)

TASK_TIME_WEIGHTINGS = (
    "uniform",
    "linear",
    "quadratic",
    "late_half",
    "terminal",
)


def _truncated_factors(
    matrix: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("low-rank initialization requires a square matrix")
    if not 1 <= rank <= matrix.shape[0]:
        raise ValueError("rank must be in [1, d_model]")
    left, singular, right = torch.linalg.svd(
        matrix.detach().float(),
        full_matrices=False,
    )
    root = singular[:rank].clamp_min(0).sqrt()
    return (
        left[:, :rank] * root.unsqueeze(0),
        root.unsqueeze(1) * right[:rank],
    )


class TrainableStructuredAffine(torch.nn.Module):
    """Train J with a full, low-rank, diagonal+low-rank, or LoRA form."""

    def __init__(
        self,
        initial: VectorAffine,
        *,
        parameterization: str,
        rank: int,
        train_bias: bool = True,
    ) -> None:
        super().__init__()
        if parameterization not in TASK_PARAMETERIZATIONS:
            raise ValueError(f"unknown task parameterization: {parameterization}")
        dimension = int(initial.weight.shape[0])
        if tuple(initial.weight.shape) != (dimension, dimension):
            raise ValueError("initial affine weight must be square")
        if not 1 <= rank <= dimension:
            raise ValueError("task rank must be in [1, d_model]")
        self.parameterization = parameterization
        self.rank = rank
        self.dimension = dimension
        self.bias = torch.nn.Parameter(initial.bias.detach().float().clone())
        self.bias.requires_grad_(train_bias)

        weight = initial.weight.detach().float()
        if parameterization == "full":
            self.weight = torch.nn.Parameter(weight.clone())
            return

        if parameterization == "lora":
            self.register_buffer("base_weight", weight.clone())
            self.left = torch.nn.Parameter(
                torch.zeros(
                    dimension,
                    rank,
                    device=weight.device,
                    dtype=weight.dtype,
                )
            )
            self.right = torch.nn.Parameter(
                torch.randn(
                    rank,
                    dimension,
                    device=weight.device,
                    dtype=weight.dtype,
                )
                / dimension**0.5
            )
            return

        if parameterization == "identity_low_rank":
            self.register_buffer(
                "base_weight",
                torch.eye(
                    dimension,
                    device=weight.device,
                    dtype=weight.dtype,
                ),
            )
            residual = weight - self.base_weight
        elif parameterization == "scalar_low_rank":
            scalar = torch.trace(weight) / dimension
            self.scalar = torch.nn.Parameter(scalar.clone())
            residual = weight - scalar * torch.eye(
                dimension,
                device=weight.device,
                dtype=weight.dtype,
            )
        elif parameterization == "diagonal_low_rank":
            diagonal = torch.diagonal(weight).clone()
            self.diagonal = torch.nn.Parameter(diagonal)
            residual = weight - torch.diag(diagonal)
        elif parameterization == "weight_low_rank":
            self.register_buffer("base_weight", torch.zeros_like(weight))
            residual = weight
        else:
            raise AssertionError("unreachable parameterization")
        left, right = _truncated_factors(residual, rank)
        self.left = torch.nn.Parameter(left)
        self.right = torch.nn.Parameter(right)

    def dense_weight(self) -> torch.Tensor:
        if self.parameterization == "full":
            return self.weight
        if self.parameterization in (
            "lora",
            "identity_low_rank",
            "weight_low_rank",
        ):
            base = self.base_weight
        elif self.parameterization == "scalar_low_rank":
            base = self.scalar * torch.eye(
                self.dimension,
                device=self.scalar.device,
                dtype=self.scalar.dtype,
            )
        elif self.parameterization == "diagonal_low_rank":
            base = torch.diag(self.diagonal)
        else:
            raise AssertionError("unreachable parameterization")
        return base + self.left @ self.right

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.dense_weight() + self.bias

    def frozen(self) -> VectorAffine:
        represented_rank = (
            self.rank
            if self.parameterization in (
                "identity_low_rank",
                "weight_low_rank",
            )
            else self.dimension
        )
        return VectorAffine(
            weight=self.dense_weight().detach().clone(),
            bias=self.bias.detach().clone(),
            update_rank=represented_rank,
            fit_dimension=self.dimension,
            retained_fit_energy=1.0,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def _time_weight(cycle: int, horizon: int, mode: str) -> float:
    if not 1 <= cycle <= horizon:
        raise ValueError("cycle must be in [1, horizon]")
    progress = cycle / horizon
    if mode == "uniform":
        return 1.0
    if mode == "linear":
        return progress
    if mode == "quadratic":
        return progress * progress
    if mode == "late_half":
        return 1.0 if cycle > horizon / 2 else 0.25
    if mode == "terminal":
        return 1.0 if cycle == horizon else 0.0
    raise ValueError(f"unknown task time weighting: {mode}")


def _weighted_losses(
    losses: Sequence[tuple[torch.Tensor, int, int]],
    *,
    mode: str,
) -> torch.Tensor:
    if not losses:
        raise ValueError("cannot aggregate an empty loss sequence")
    weights = torch.tensor(
        [_time_weight(cycle, horizon, mode) for _, cycle, horizon in losses],
        device=losses[0][0].device,
        dtype=losses[0][0].dtype,
    )
    if not bool(weights.gt(0).any()):
        raise ValueError("time weighting produced no positive loss")
    values = torch.stack([loss for loss, _, _ in losses])
    return (values * weights).sum() / weights.sum()


def _weighted_interface_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    answer_relative_position: int,
    answer_weight: float,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("interface tensors must share [batch, position, feature]")
    if answer_weight <= 0:
        raise ValueError("task state answer weight must be positive")
    weights = torch.ones(
        prediction.shape[1],
        device=prediction.device,
        dtype=torch.float32,
    )
    weights[answer_relative_position] = answer_weight
    weights = weights.view(1, -1, 1)
    denominator = prediction.shape[0] * weights.sum() * prediction.shape[2]
    target_float = target.float()
    target_mean = (target_float * weights).sum(dim=(0, 1), keepdim=True) / (
        prediction.shape[0] * weights.sum()
    )
    target_scale = (
        (target_float - target_mean).square() * weights
    ).sum() / denominator
    mse = ((prediction.float() - target_float).square() * weights).sum() / (
        denominator
    )
    return mse / target_scale.clamp_min(1e-6)


@torch.no_grad()
def _diagnostic_interface_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    answer_relative_position: int,
) -> torch.Tensor:
    return _weighted_interface_loss(
        prediction.detach(),
        target.detach(),
        answer_relative_position=answer_relative_position,
        answer_weight=1.0,
    )


def structured_parameter_count(
    dimension: int,
    *,
    parameterization: str,
    rank: int,
) -> int:
    if parameterization == "full":
        return dimension * dimension + dimension
    count = 2 * dimension * rank + dimension
    if parameterization == "scalar_low_rank":
        count += 1
    elif parameterization == "diagonal_low_rank":
        count += dimension
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_unit_j_map(
    artifact: Path,
    *,
    label: str,
    device: torch.device,
) -> tuple[VectorAffine, str]:
    payload = torch.load(artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_unit_j":
        raise ValueError("unexpected unit-J artifact kind")
    if label not in payload["maps"]:
        raise ValueError(f"unit-J artifact has no map label {label!r}")
    item = payload["maps"][label]
    weight = item["weight"].to(device=device, dtype=torch.float32)
    return (
        VectorAffine(
            weight=weight,
            bias=item["bias"].to(device=device, dtype=torch.float32),
            update_rank=int(item["rank"]),
            fit_dimension=int(weight.shape[0]),
            retained_fit_energy=1.0,
        ),
        str(payload["checkpoint"]),
    )


def apply_power(
    value: torch.Tensor,
    operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    dose: int,
) -> torch.Tensor:
    if dose < 0:
        raise ValueError("dose must be nonnegative")
    result = value
    for _ in range(dose):
        result = operator(result)
    return result


def unit_j_dose_schedule(
    *,
    name: str,
    horizon: int,
    start_age: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return per-loop J doses and source ages, always staying in H2..H8."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 2 <= start_age <= 8:
        raise ValueError("start_age must be in [2, 8]")
    rng = np.random.default_rng(seed)
    age = start_age
    doses: list[int] = []
    ages: list[int] = []
    for cycle in range(1, horizon + 1):
        ages.append(age)
        if name == "reset3":
            dose = max(0, age - 2)
        elif name == "maintain4":
            dose = max(0, age - 3)
        elif name == "unit_every":
            dose = 0 if age == 2 else 1
        elif name == "periodic_reset":
            dose = age - 2 if age == 8 else 0
        elif name == "alternating":
            dose = min(age - 2, 1 + (cycle - 1) % 3)
        elif name == "random_sparse":
            dose = (
                int(rng.integers(1, age - 1))
                if age == 8 or (age > 2 and rng.random() < 0.35)
                else 0
            )
        elif name == "random_power":
            dose = (
                0
                if age == 2
                else int(rng.integers(0 if age < 8 else 1, age - 1))
            )
        else:
            raise ValueError(f"unknown unit-J policy: {name}")
        if not 0 <= dose <= age - 2:
            raise RuntimeError("policy proposed an invalid rejuvenation dose")
        next_age = age - dose + 1
        if not 3 <= next_age <= 8:
            raise RuntimeError("policy leaves the registered age range")
        doses.append(dose)
        age = next_age
    return tuple(doses), tuple(ages)


def _training_case(
    *,
    batch_index: int,
    rollout_horizons: Sequence[int],
    policies: Sequence[str],
    start_ages: Sequence[int],
    seed: int,
) -> tuple[str, int, int]:
    if not start_ages or any(not 2 <= int(age) <= 8 for age in start_ages):
        raise ValueError("start_ages must be a nonempty subset of H2..H8")
    policy_index = batch_index % len(policies)
    age_index = (batch_index // len(policies)) % len(start_ages)
    policy = str(policies[policy_index])
    start_age = int(start_ages[age_index])
    repeat_index = batch_index // (len(policies) * len(start_ages))
    horizon = int(
        rollout_horizons[
            (
                policy_index
                + age_index
                + repeat_index
                + seed // 1000
            )
            % len(rollout_horizons)
        ]
    )
    return policy, start_age, horizon


def _selected_samples(
    value: torch.Tensor,
    *,
    answer_relative_position: int,
    answer_weight: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative_positions = tuple(range(value.shape[1]))
    return _position_samples_with_weights(
        value,
        relative_positions,
        answer_position=answer_relative_position,
        answer_weight=answer_weight,
    )


def _add_selected_pair(
    stats: WeightedAffineStats,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    answer_relative_position: int,
    answer_weight: int,
    group: str,
) -> None:
    source_samples, weights = _selected_samples(
        source,
        answer_relative_position=answer_relative_position,
        answer_weight=answer_weight,
    )
    target_samples, target_weights = _selected_samples(
        target,
        answer_relative_position=answer_relative_position,
        answer_weight=answer_weight,
    )
    if not torch.equal(weights, target_weights):
        raise RuntimeError("source and target weights differ")
    stats.add(
        source_samples,
        target_samples,
        weights,
        group=group,
    )


@torch.no_grad()
def exact_interfaces(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    successors: torch.Tensor,
    current: torch.Tensor,
    ages: Sequence[int],
    loop_index: int,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for age in sorted(set(int(value) for value in ages)):
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=age,
            phase_position=phase_positions[age],
        )
        result[age] = run_one_loop(
            model,
            state,
            loop_index=loop_index,
        ).block2_hidden_in[:, list(positions)]
    return result


@torch.no_grad()
def collect_adjacent_natural_pairs(
    *,
    stats: WeightedAffineStats,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_relative_position: int,
    answer_weight: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    identity_weight: int,
) -> None:
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        interfaces = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=current,
            ages=range(2, 9),
            loop_index=cfg.max_loops,
        )
        if identity_weight > 0:
            identity_weights = answer_weight * identity_weight
            _add_selected_pair(
                stats,
                source=interfaces[2],
                target=interfaces[2],
                answer_relative_position=answer_relative_position,
                answer_weight=identity_weights,
                group="natural_H2_identity",
            )
        for age in range(3, 9):
            _add_selected_pair(
                stats,
                source=interfaces[age],
                target=interfaces[age - 1],
                answer_relative_position=answer_relative_position,
                answer_weight=answer_weight,
                group=f"natural_H{age}_to_H{age - 1}",
            )


@torch.no_grad()
def collect_direct_operating_natural_pairs(
    *,
    stats: WeightedAffineStats,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_relative_position: int,
    answer_weight: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    identity_weight: int,
    operating_age: int,
) -> None:
    if not 2 <= operating_age <= 7:
        raise ValueError("direct operating age must be in [2, 7]")
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        interfaces = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=current,
            ages=range(operating_age, 9),
            loop_index=cfg.max_loops,
        )
        if identity_weight > 0:
            _add_selected_pair(
                stats,
                source=interfaces[operating_age],
                target=interfaces[operating_age],
                answer_relative_position=answer_relative_position,
                answer_weight=answer_weight * identity_weight,
                group=f"natural_H{operating_age}_identity",
            )
        for source_age in range(operating_age + 1, 9):
            _add_selected_pair(
                stats,
                source=interfaces[source_age],
                target=interfaces[operating_age],
                answer_relative_position=answer_relative_position,
                answer_weight=answer_weight,
                group=(
                    f"natural_H{source_age}_to_operating_H{operating_age}"
                ),
            )


@torch.no_grad()
def collect_power_dagger_pairs(
    *,
    stats: WeightedAffineStats,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_relative_position: int,
    answer_weight: int,
    age_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    rollout_horizons: Sequence[int],
    policies: Sequence[str],
    start_ages: Sequence[int],
    seed: int,
    direct_operating_target_age: int | None = None,
) -> list[dict[str, Any]]:
    jump = phase_positions[3] - phase_positions[2]
    rows: list[dict[str, Any]] = []
    set_seed(seed)
    for batch_index in range(batches):
        policy, start_age, horizon = _training_case(
            batch_index=batch_index,
            rollout_horizons=rollout_horizons,
            policies=policies,
            start_ages=start_ages,
            seed=seed,
        )
        doses, ages = unit_j_dose_schedule(
            name=policy,
            horizon=horizon,
            start_age=start_age,
            seed=seed * 1009 + batch_index,
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
        new_pairs = 0.0
        for cycle, (dose, age) in enumerate(
            zip(doses, ages, strict=True),
            start=1,
        ):
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            captures: list[torch.Tensor] = []

            def transform(value: torch.Tensor) -> torch.Tensor:
                live = value
                for _ in range(dose):
                    captures.append(live)
                    live = age_map(live)
                return live

            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    (positions, transform) if dose else None
                ),
            )
            if dose:
                if direct_operating_target_age is None:
                    target_ages = tuple(range(age - dose, age))
                else:
                    if dose != 1:
                        raise ValueError(
                            "direct operating DAgger requires one J per loop"
                        )
                    target_ages = (direct_operating_target_age,)
                targets = exact_interfaces(
                    model=model,
                    cfg=cfg,
                    phase_positions=phase_positions,
                    positions=positions,
                    successors=successors,
                    current=current,
                    ages=target_ages,
                    loop_index=cfg.max_loops + cycle - 1,
                )
                for power_index, source in enumerate(captures, start=1):
                    before = stats.effective_rows
                    target_age = (
                        age - power_index
                        if direct_operating_target_age is None
                        else direct_operating_target_age
                    )
                    _add_selected_pair(
                        stats,
                        source=source,
                        target=targets[target_age],
                        answer_relative_position=answer_relative_position,
                        answer_weight=answer_weight,
                        group=(
                            (
                                f"dagger_{policy}_H{target_age + 1}"
                                f"_to_H{target_age}"
                            )
                            if direct_operating_target_age is None
                            else (
                                f"dagger_{policy}_direct_to_operating_"
                                f"H{target_age}"
                            )
                        ),
                    )
                    new_pairs += stats.effective_rows - before
            state = step.state
        rows.append(
            {
                "batch": batch_index,
                "policy": policy,
                "start_age": start_age,
                "horizon": horizon,
                "total_J_applications": sum(doses),
                "loops_with_J": sum(int(dose > 0) for dose in doses),
                "new_effective_position_pairs": new_pairs,
            }
        )
    return rows


def fine_tune_power_compositions(
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
    policies: Sequence[str],
    start_ages: Sequence[int],
    learning_rate: float,
    seed: int,
    direct_operating_target_age: int | None = None,
    parameterization: str = "full",
    rank: int = 256,
    time_weighting: str = "uniform",
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    detach_interval: int = 0,
    train_bias: bool = True,
) -> tuple[VectorAffine, list[dict[str, Any]]]:
    if rounds == 0:
        return initial_map, []
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if time_weighting not in TASK_TIME_WEIGHTINGS:
        raise ValueError(f"unknown task time weighting: {time_weighting}")
    if weight_decay < 0:
        raise ValueError("task weight decay must be nonnegative")
    if grad_clip <= 0:
        raise ValueError("task grad clip must be positive")
    if detach_interval < 0:
        raise ValueError("task detach interval must be nonnegative")
    set_seed(seed)
    module = TrainableStructuredAffine(
        initial_map,
        parameterization=parameterization,
        rank=rank,
        train_bias=train_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    differentiable_loop = run_one_loop.__wrapped__
    jump = phase_positions[3] - phase_positions[2]
    rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        round_seed = seed + 1000 * round_index
        set_seed(round_seed)
        totals: dict[str, float] = defaultdict(float)
        for batch_index in range(batches_per_round):
            policy, start_age, horizon = _training_case(
                batch_index=batch_index,
                rollout_horizons=rollout_horizons,
                policies=policies,
                start_ages=start_ages,
                seed=round_seed,
            )
            doses, ages = unit_j_dose_schedule(
                name=policy,
                horizon=horizon,
                start_age=start_age,
                seed=round_seed * 1009 + batch_index,
            )
            if sum(doses) == 0:
                totals["skipped_no_action"] += 1
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
            task_losses: list[tuple[torch.Tensor, int, int]] = []
            graph_started = False
            for cycle, (dose, age) in enumerate(
                zip(doses, ages, strict=True),
                start=1,
            ):
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
                if dose == 0 and not graph_started:
                    with torch.no_grad():
                        state = run_one_loop(
                            model,
                            state,
                            loop_index=cfg.max_loops + cycle - 1,
                        ).state
                    continue
                if dose:
                    graph_started = True
                captures: list[torch.Tensor] = []

                def transform(value: torch.Tensor) -> torch.Tensor:
                    live = value
                    for _ in range(dose):
                        live = module(live)
                        captures.append(live)
                    return live

                step = differentiable_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        (positions, transform) if dose else None
                    ),
                )
                task_loss = F.cross_entropy(step.logits.float(), target)
                task_losses.append((task_loss, cycle, horizon))
                totals["supervised_steps"] += 1
                totals["examples"] += batch_size
                totals["correct"] += int(
                    step.logits.argmax(dim=-1).eq(target).sum()
                )
                totals["task_loss"] += float(task_loss.detach())
                if dose:
                    with torch.no_grad():
                        if direct_operating_target_age is None:
                            target_ages = tuple(range(age - dose, age))
                        else:
                            if dose != 1:
                                raise ValueError(
                                    "direct operating training requires "
                                    "one J per loop"
                                )
                            target_ages = (direct_operating_target_age,)
                        targets = exact_interfaces(
                            model=model,
                            cfg=cfg,
                            phase_positions=phase_positions,
                            positions=positions,
                            successors=successors,
                            current=current,
                            ages=target_ages,
                            loop_index=cfg.max_loops + cycle - 1,
                        )
                    for power_index, live in enumerate(captures, start=1):
                        target_age = (
                            age - power_index
                            if direct_operating_target_age is None
                            else direct_operating_target_age
                        )
                        oracle = targets[target_age]
                        diagnostic_state_mse = _diagnostic_interface_mse(
                            live,
                            oracle,
                            answer_relative_position=positions.index(
                                cfg.seq_len - 1
                            ),
                        )
                        totals["diagnostic_state_mse"] += float(
                            diagnostic_state_mse.detach()
                        )
                        totals["J_applications"] += 1
                state = step.state
                if (
                    detach_interval > 0
                    and cycle % detach_interval == 0
                    and cycle < horizon
                ):
                    state = state.detach()
            if not task_losses:
                raise RuntimeError("composition episode has no trainable loss")
            task_ce_objective = _weighted_losses(
                task_losses,
                mode=time_weighting,
            )
            task_ce_objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(),
                max_norm=grad_clip,
            )
            optimizer.step()
            totals["optimizer_steps"] += 1
            totals["grad_norm"] += float(grad_norm)
        rows.append(
            {
                "round": round_index,
                "optimizer_steps": int(totals["optimizer_steps"]),
                "skipped_no_action_episodes": int(
                    totals["skipped_no_action"]
                ),
                "J_applications": int(totals["J_applications"]),
                "supervised_composition_steps": int(
                    totals["supervised_steps"]
                ),
                "composition_accuracy": (
                    totals["correct"] / totals["examples"]
                ),
                "task_ce": (
                    totals["task_loss"] / totals["supervised_steps"]
                ),
                "diagnostic_H3_relative_mse": (
                    totals["diagnostic_state_mse"]
                    / totals["J_applications"]
                ),
                "grad_norm_mean": (
                    totals["grad_norm"] / totals["optimizer_steps"]
                ),
                "learning_rate": learning_rate,
                "objective": "trajectory_ce_only",
                "time_weighting": time_weighting,
                "parameterization": parameterization,
                "rank": rank,
                "trainable_J_parameters": module.trainable_parameter_count,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "detach_interval": detach_interval,
                "train_bias": train_bias,
            }
        )
    return module.frozen(), rows


def fine_tune_supervised_powers(
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
    learning_rate: float,
    identity_loss_weight: float,
    seed: int,
) -> tuple[VectorAffine, list[dict[str, Any]]]:
    """Fit all 21 exact natural-age powers on fresh in-distribution graphs."""

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
    rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        set_seed(seed + 1000 * round_index)
        totals: dict[str, float] = defaultdict(float)
        for _ in range(batches_per_round):
            with torch.no_grad():
                _, path_targets, successors, _ = fixed_depth_batch(
                    cfg,
                    batch_size,
                    device,
                    path_positions=cfg.max_depth,
                )
                current = path_targets[:, cfg.max_depth - 1]
                interfaces = exact_interfaces(
                    model=model,
                    cfg=cfg,
                    phase_positions=phase_positions,
                    positions=positions,
                    successors=successors,
                    current=current,
                    ages=range(2, 9),
                    loop_index=cfg.max_loops,
                )
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            identity_prediction = module(interfaces[2])
            identity_scale = (
                interfaces[2].float()
                - interfaces[2].float().mean(
                    dim=(0, 1),
                    keepdim=True,
                )
            ).square().mean().clamp_min(1e-6)
            identity_loss = (
                identity_prediction - interfaces[2].float()
            ).square().mean() / identity_scale
            if identity_loss_weight > 0:
                losses.append(identity_loss_weight * identity_loss)
            totals["identity_loss"] += float(identity_loss.detach())
            for source_age in range(3, 9):
                live = interfaces[source_age]
                for target_age in range(source_age - 1, 1, -1):
                    live = module(live)
                    target = interfaces[target_age]
                    scale = (
                        target.float()
                        - target.float().mean(
                            dim=(0, 1),
                            keepdim=True,
                        )
                    ).square().mean().clamp_min(1e-6)
                    power_loss = (
                        live - target.float()
                    ).square().mean() / scale
                    losses.append(power_loss)
                    totals["power_loss"] += float(power_loss.detach())
                    totals["power_pairs"] += 1
                    if target_age == 2:
                        totals["to_H2_loss"] += float(power_loss.detach())
                        totals["to_H2_pairs"] += 1
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                module.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            totals["optimizer_steps"] += 1
            totals["grad_norm"] += float(grad_norm)
        rows.append(
            {
                "round": round_index,
                "optimizer_steps": int(totals["optimizer_steps"]),
                "all_power_relative_mse": (
                    totals["power_loss"] / totals["power_pairs"]
                ),
                "to_H2_relative_mse": (
                    totals["to_H2_loss"] / totals["to_H2_pairs"]
                ),
                "H2_identity_relative_mse": (
                    totals["identity_loss"] / totals["optimizer_steps"]
                ),
                "grad_norm_mean": (
                    totals["grad_norm"] / totals["optimizer_steps"]
                ),
                "learning_rate": learning_rate,
                "identity_loss_weight": identity_loss_weight,
            }
        )
    return module.frozen(), rows


@torch.no_grad()
def evaluate_power_grid(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    maps: dict[str, VectorAffine],
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> list[dict[str, Any]]:
    jump = phase_positions[3] - phase_positions[2]
    storage: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
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
        current = path_targets[:, cfg.max_depth - 1]
        target = advance_nodes(successors, current, steps=jump)
        interfaces = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=current,
            ages=range(2, 9),
            loop_index=cfg.max_loops,
        )
        for source_age in range(3, 9):
            state = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=source_age,
                phase_position=phase_positions[source_age],
            )
            for dose in range(1, source_age - 1):
                target_age = source_age - dose
                oracle_step = run_one_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops,
                    block2_position_override=(
                        positions,
                        interfaces[target_age],
                    ),
                )
                oracle_values = storage[("oracle", source_age, dose)]
                oracle_values["count"] += batch_size
                oracle_values["correct"] += int(
                    oracle_step.logits.argmax(dim=-1).eq(target).sum()
                )
                oracle_values["interface_error"] += 0.0
                oracle_values["batches"] += 1
                for label, age_map in maps.items():
                    step = run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops,
                        block2_position_transform=(
                            positions,
                            lambda value, m=age_map, d=dose: apply_power(
                                value,
                                m,
                                dose=d,
                            ),
                        ),
                    )
                    values = storage[(label, source_age, dose)]
                    values["count"] += batch_size
                    values["correct"] += int(
                        step.logits.argmax(dim=-1).eq(target).sum()
                    )
                    values["interface_error"] += _relative_mse(
                        step.block2_hidden_in[:, list(positions)],
                        interfaces[target_age],
                    )
                    values["batches"] += 1
    return [
        {
            "map": label,
            "source_age": source_age,
            "dose": dose,
            "target_age": source_age - dose,
            "accuracy": values["correct"] / values["count"],
            "interface_relative_mse": (
                values["interface_error"] / values["batches"]
            ),
            "graphs": int(values["count"]),
        }
        for (label, source_age, dose), values in sorted(storage.items())
    ]


def _policy_for_condition(condition: str) -> str:
    for policy in POLICIES:
        if condition.endswith(f"_{policy}"):
            return policy
    raise ValueError(f"condition has no policy: {condition}")


@torch.no_grad()
def evaluate_closed_loop(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    maps: dict[str, VectorAffine],
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
    operating_age: int,
    direct_operating_target_age: int | None = None,
) -> list[dict[str, Any]]:
    if not 2 <= operating_age <= 7:
        raise ValueError("operating_age must be in [2, 7]")
    jump = phase_positions[3] - phase_positions[2]
    conditions = ["no_control", "exact_H2_every", "exact_operating_every"]
    for label in maps:
        conditions.extend(
            (
                f"{label}_reset3",
                f"{label}_unit_every",
                f"{label}_periodic_reset",
                f"{label}_random_power",
            )
        )
    conditions.extend(
        ("shuffled_final_reset3", "shuffled_final_unit_every")
    )
    shuffled_map_label = tuple(maps)[-1]
    schedules: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for index, condition in enumerate(conditions):
        if condition in {
            "no_control",
            "exact_H2_every",
            "exact_operating_every",
        }:
            continue
        policy = _policy_for_condition(condition)
        schedules[condition] = unit_j_dose_schedule(
            name=policy,
            horizon=continuation_loops,
            start_age=8,
            seed=seed + 10007 * index,
        )
    sums: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    modes: dict[tuple[str, int, int], int] = defaultdict(int)
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
            oracle_interfaces = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=successors,
                current=current,
                ages=(2, operating_age),
                loop_index=cfg.max_loops + cycle - 1,
            )
            next_states = {}
            for condition in conditions:
                dose = 0
                target_age = 8
                transform = None
                override = None
                if condition == "exact_H2_every":
                    override = (positions, oracle_interfaces[2])
                    target_age = 2
                elif condition == "exact_operating_every":
                    override = (
                        positions,
                        oracle_interfaces[operating_age],
                    )
                    target_age = operating_age
                elif condition != "no_control":
                    doses, ages = schedules[condition]
                    dose = doses[cycle - 1]
                    source_age = ages[cycle - 1]
                    target_age = source_age - dose
                    if (
                        direct_operating_target_age is not None
                        and condition.endswith("_unit_every")
                    ):
                        target_age = direct_operating_target_age
                    label = (
                        shuffled_map_label
                        if condition.startswith("shuffled_final_")
                        else condition.split("_", 1)[0]
                    )
                    age_map = maps[label]
                    if dose:
                        if condition.startswith("shuffled_final_"):
                            transform = (
                                positions,
                                lambda value, m=age_map, d=dose: (
                                    apply_power(value, m, dose=d).roll(
                                        1,
                                        dims=0,
                                    )
                                ),
                            )
                        else:
                            transform = (
                                positions,
                                lambda value, m=age_map, d=dose: apply_power(
                                    value,
                                    m,
                                    dose=d,
                                ),
                            )
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
                values["dose"] = dose
                values["target_interface_age"] = target_age
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
                values["output_norm"] += float(
                    step.state.float().norm(dim=-1).mean(dim=1).sum()
                )
                orbit_offsets = offsets[orbit8]
                for offset in range(cfg.node_count):
                    modes[(condition, cycle, offset)] += int(
                        orbit_offsets.eq(offset).sum()
                    )
            states = next_states
    rows: list[dict[str, Any]] = []
    for (condition, cycle), values in sorted(sums.items()):
        orbit_count = values["orbit8_count"]
        mode, mode_count = max(
            (
                (offset, modes[(condition, cycle, offset)])
                for offset in range(cfg.node_count)
            ),
            key=lambda item: item[1],
        )
        rows.append(
            {
                "condition": condition,
                "cycle": cycle,
                "dose": int(values["dose"]),
                "target_interface_age": int(values["target_interface_age"]),
                "accuracy_all": values["correct_all"] / values["count"],
                "accuracy_orbit8": (
                    values["correct_orbit8"] / orbit_count
                    if orbit_count
                    else float("nan")
                ),
                "accuracy_nonendpoint": (
                    values["correct_nonendpoint"]
                    / values["nonendpoint_count"]
                    if values["nonendpoint_count"]
                    else float("nan")
                ),
                "orbit8_top1_mode_offset": mode,
                "orbit8_top1_mode_fraction": (
                    mode_count / orbit_count if orbit_count else float("nan")
                ),
                "output_norm_mean": values["output_norm"] / values["count"],
            }
        )
    return rows


def summarize_curves(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        parts = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: int(row["cycle"]),
        )
        accuracy = [float(row["accuracy_orbit8"]) for row in parts]
        condition_summary: dict[str, Any] = {
            "accuracy_orbit8": accuracy,
            "auc_8": float(np.mean(accuracy[:8])),
            "auc_24": float(np.mean(accuracy[:24])),
            "auc_all": float(np.mean(accuracy)),
            "minimum": float(np.min(accuracy)),
            "final": accuracy[-1],
        }
        for metric in ("accuracy_all", "accuracy_nonendpoint"):
            values = [float(row[metric]) for row in parts]
            condition_summary[metric] = {
                "values": values,
                "auc_1_24": float(np.mean(values[:24])),
                "auc_25_48": (
                    float(np.mean(values[24:48]))
                    if len(values) > 24
                    else None
                ),
                "auc_49_64": (
                    float(np.mean(values[48:64]))
                    if len(values) > 48
                    else None
                ),
                "auc_all": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "final": values[-1],
            }
        result[condition] = condition_summary
    return result


def _plot_curves(rows: list[dict[str, Any]], *, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        constrained_layout=True,
    )
    selected_conditions = {
        "no_control",
        "exact_operating_every",
        "reg_unit_every",
        "power_unit_every",
        "task_unit_every",
        "shuffled_final_unit_every",
    }
    for axis, metric, title in (
        (axes[0], "accuracy_orbit8", "orbit length 8"),
        (axes[1], "accuracy_nonendpoint", "endpoint-return cases excluded"),
    ):
        for condition in sorted({str(row["condition"]) for row in rows}):
            if condition not in selected_conditions:
                continue
            parts = [row for row in rows if row["condition"] == condition]
            axis.plot(
                [int(row["cycle"]) for row in parts],
                [float(row[metric]) for row in parts],
                label=condition,
                linewidth=1.9,
            )
        axis.axhline(0.9, color="black", linestyle=":", linewidth=1.1)
        axis.set_xlabel("Continuation loop")
        axis.set_ylabel("Current-step accuracy")
        axis.set_title(title)
        axis.set_ylim(-0.04, 1.04)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(out_dir / "unit_j_closed_loop_accuracy.png", dpi=180)
    plt.close(figure)


def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device_name: str,
    frozen_model_loss_placement: str,
    calibration_batch_size: int,
    calibration_batches: int,
    dagger_batch_size: int,
    dagger_batches_per_round: int,
    dagger_rounds: int,
    rollout_horizons: Sequence[int],
    policies: Sequence[str],
    train_start_ages: Sequence[int],
    initial_map_artifact: Path | None,
    initial_map_label: str,
    power_batch_size: int,
    power_batches_per_round: int,
    power_rounds: int,
    power_learning_rate: float,
    power_identity_loss_weight: float,
    task_batch_size: int,
    task_batches_per_round: int,
    task_rounds: int,
    task_learning_rate: float,
    evaluation_batch_size: int,
    evaluation_batches: int,
    continuation_loops: int,
    position_group: str,
    operating_age: int,
    answer_weight: int,
    identity_weight: int,
    ridge: float,
    calibration_seed: int,
    dagger_seed: int,
    task_seed: int,
    power_seed: int,
    evaluation_seed: int,
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
    shared_gpu: bool,
    direct_operating_target_age: int | None,
    task_parameterization: str = "full",
    task_rank: int = 256,
    task_time_weighting: str = "uniform",
    task_weight_decay: float = 0.0,
    task_grad_clip: float = 1.0,
    task_detach_interval: int = 0,
    task_train_bias: bool = True,
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
        raise ValueError("experiment is fixed to D8L8 with two blocks")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    available_position_groups = intervention_groups(cfg.node_count)
    if position_group not in available_position_groups:
        raise ValueError(f"unknown position group: {position_group}")
    positions = available_position_groups[position_group]
    answer_position = explicit_depth_position_groups(
        cfg.node_count
    )["answer"][0]
    answer_relative_position = positions.index(answer_position)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = WeightedAffineStats(cfg.d_model)
    if direct_operating_target_age is None:
        collect_adjacent_natural_pairs(
            stats=stats,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            answer_relative_position=answer_relative_position,
            answer_weight=answer_weight,
            device=device,
            batch_size=calibration_batch_size,
            batches=calibration_batches,
            seed=calibration_seed,
            identity_weight=identity_weight,
        )
    else:
        collect_direct_operating_natural_pairs(
            stats=stats,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            answer_relative_position=answer_relative_position,
            answer_weight=answer_weight,
            device=device,
            batch_size=calibration_batch_size,
            batches=calibration_batches,
            seed=calibration_seed,
            identity_weight=identity_weight,
            operating_age=direct_operating_target_age,
        )
    regression_map = stats.fit(ridge=ridge, device=device)
    dagger_rows: list[dict[str, Any]] = []
    training_rows = [
        {
            "round": 0,
            "raw_position_rows": stats.raw_rows,
            "effective_position_rows": stats.effective_rows,
            "new_effective_position_rows": stats.effective_rows,
        }
    ]
    for round_index in range(1, dagger_rounds + 1):
        before = stats.effective_rows
        new_rows = collect_power_dagger_pairs(
            stats=stats,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            answer_relative_position=answer_relative_position,
            answer_weight=answer_weight,
            age_map=regression_map,
            device=device,
            batch_size=dagger_batch_size,
            batches=dagger_batches_per_round,
            rollout_horizons=rollout_horizons,
            policies=policies,
            start_ages=train_start_ages,
            seed=dagger_seed + 1000 * round_index,
            direct_operating_target_age=direct_operating_target_age,
        )
        for row in new_rows:
            row["round"] = round_index
        dagger_rows.extend(new_rows)
        regression_map = stats.fit(ridge=ridge, device=device)
        training_rows.append(
            {
                "round": round_index,
                "raw_position_rows": stats.raw_rows,
                "effective_position_rows": stats.effective_rows,
                "new_effective_position_rows": (
                    stats.effective_rows - before
                ),
            }
        )
    power_map, power_rows_training = fine_tune_supervised_powers(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        initial_map=regression_map,
        device=device,
        batch_size=power_batch_size,
        batches_per_round=power_batches_per_round,
        rounds=power_rounds,
        learning_rate=power_learning_rate,
        identity_loss_weight=power_identity_loss_weight,
        seed=power_seed,
    )
    loaded_initial_checkpoint = None
    if initial_map_artifact is not None:
        power_map, loaded_initial_checkpoint = load_unit_j_map(
            initial_map_artifact,
            label=initial_map_label,
            device=device,
        )
        if loaded_initial_checkpoint != str(checkpoint):
            raise ValueError(
                "initial J was trained for a different frozen checkpoint"
            )
    task_map, task_rows = fine_tune_power_compositions(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        initial_map=power_map,
        device=device,
        batch_size=task_batch_size,
        batches_per_round=task_batches_per_round,
        rounds=task_rounds,
        rollout_horizons=rollout_horizons,
        policies=policies,
        start_ages=train_start_ages,
        learning_rate=task_learning_rate,
        seed=task_seed,
        direct_operating_target_age=direct_operating_target_age,
        parameterization=task_parameterization,
        rank=task_rank,
        time_weighting=task_time_weighting,
        weight_decay=task_weight_decay,
        grad_clip=task_grad_clip,
        detach_interval=task_detach_interval,
        train_bias=task_train_bias,
    )
    maps = {"reg": regression_map, "power": power_map}
    if task_rounds > 0 or initial_map_artifact is not None:
        maps["task"] = task_map
    power_grid_rows = evaluate_power_grid(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        maps=maps,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        seed=evaluation_seed,
    )
    closed_rows = evaluate_closed_loop(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        maps=maps,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        continuation_loops=continuation_loops,
        seed=evaluation_seed + 1,
        operating_age=operating_age,
        direct_operating_target_age=direct_operating_target_age,
    )
    curves = summarize_curves(closed_rows)
    artifact_path = out_dir / "unit_j_maps.pt"
    artifact_tmp_path = out_dir / "unit_j_maps.pt.tmp"
    torch.save(
        {
            "kind": "graph_path_telomere_unit_j",
            "checkpoint": str(checkpoint),
            "positions": positions,
            "maps": {
                label: {
                    "weight": age_map.weight.detach().cpu(),
                    "bias": age_map.bias.detach().cpu(),
                    "rank": age_map.update_rank,
                }
                for label, age_map in maps.items()
            },
        },
        artifact_tmp_path,
    )
    os.replace(artifact_tmp_path, artifact_path)
    _write_csv(out_dir / "regression_rounds.csv", training_rows)
    _write_csv(out_dir / "dagger_schedules.csv", dagger_rows)
    _write_csv(
        out_dir / "supervised_power_rounds.csv",
        power_rows_training,
    )
    _write_csv(out_dir / "task_rounds.csv", task_rows)
    _write_csv(out_dir / "power_grid.csv", power_grid_rows)
    _write_csv(out_dir / "closed_loop.csv", closed_rows)
    _plot_curves(closed_rows, out_dir=out_dir)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            f"frozen model: {frozen_model_loss_placement}; "
            "unit J trajectory current-step CE only; exact-H3 relative "
            "MSE is diagnostic only"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "J_definition": (
            (
                "one model-specific affine age step at pre-Block2:"
                " J(z_Hk)=z_H(k-1)"
            )
            if direct_operating_target_age is None
            else (
                "one model-specific affine operating reset at pre-Block2: "
                f"J(z_live)=z_H{direct_operating_target_age}"
            )
        ),
        "position_group": position_group,
        "positions": positions,
        "operating_age": operating_age,
        "J_parameter_count": cfg.d_model * cfg.d_model + cfg.d_model,
        "J_dense_parameter_count": cfg.d_model * cfg.d_model + cfg.d_model,
        "J_trainable_parameter_count": structured_parameter_count(
            cfg.d_model,
            parameterization=task_parameterization,
            rank=task_rank,
        ) - (0 if task_train_bias else cfg.d_model),
        "training": {
            "natural_adjacent_ages": (
                list(range(3, 9))
                if direct_operating_target_age is None
                else None
            ),
            "natural_direct_source_ages": (
                list(range(direct_operating_target_age + 1, 9))
                if direct_operating_target_age is not None
                else None
            ),
            "direct_operating_target_age": direct_operating_target_age,
            "natural_graphs": calibration_batch_size * calibration_batches,
            "H2_identity_weight": identity_weight,
            "answer_weight": answer_weight,
            "ridge": ridge,
            "dagger_rounds": dagger_rounds,
            "dagger_graphs_per_round": (
                dagger_batch_size * dagger_batches_per_round
            ),
            "task_rounds": task_rounds,
            "supervised_power_rounds": power_rounds,
            "supervised_power_graphs_per_round": (
                power_batch_size * power_batches_per_round
            ),
            "supervised_power_learning_rate": power_learning_rate,
            "supervised_power_identity_loss_weight": (
                power_identity_loss_weight
            ),
            "supervised_power_rows": power_rows_training,
            "task_graphs_per_round": (
                task_batch_size * task_batches_per_round
            ),
            "task_learning_rate": task_learning_rate,
            "task_objective": "trajectory_ce_only",
            "task_hidden_state_metric": (
                "diagnostic unweighted relative MSE to exact H3; "
                "excluded from the optimization objective"
            ),
            "task_parameterization": task_parameterization,
            "task_rank": task_rank,
            "task_time_weighting": task_time_weighting,
            "task_weight_decay": task_weight_decay,
            "task_grad_clip": task_grad_clip,
            "task_detach_interval": task_detach_interval,
            "task_train_bias": task_train_bias,
            "rollout_horizons": list(rollout_horizons),
            "policies": list(policies),
            "train_start_ages": list(train_start_ages),
            "task_rows": task_rows,
            "initial_map_artifact": (
                str(initial_map_artifact)
                if initial_map_artifact is not None
                else None
            ),
            "initial_map_label": (
                initial_map_label if initial_map_artifact is not None else None
            ),
            "group_effective_rows": dict(stats.group_effective_rows),
            "seeds": {
                "calibration": calibration_seed,
                "dagger": dagger_seed,
                "task": task_seed,
                "power": power_seed,
            },
        },
        "evaluation_graphs": evaluation_batch_size * evaluation_batches,
        "evaluation_seed": evaluation_seed,
        "power_grid": power_grid_rows,
        "curves": curves,
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
            "artifact": "unit_j_maps.pt",
            "power_grid": "power_grid.csv",
            "closed_loop": "closed_loop.csv",
            "figure": "unit_j_closed_loop_accuracy.png",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one reusable one-age rejuvenation affine J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--frozen-model-loss-placement",
        type=str,
        default="final CE at recurrent loop 8 only",
    )
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--dagger-batches-per-round", type=int, default=49)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument(
        "--rollout-horizons",
        type=int,
        nargs="+",
        default=(4, 8, 16, 24),
    )
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument(
        "--train-start-ages",
        type=int,
        nargs="+",
        default=tuple(range(2, 9)),
    )
    parser.add_argument("--initial-map-artifact", type=Path)
    parser.add_argument("--initial-map-label", type=str, default="task")
    parser.add_argument("--power-batch-size", type=int, default=128)
    parser.add_argument("--power-batches-per-round", type=int, default=64)
    parser.add_argument("--power-rounds", type=int, default=4)
    parser.add_argument("--power-learning-rate", type=float, default=0.00001)
    parser.add_argument(
        "--power-identity-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--task-batch-size", type=int, default=32)
    parser.add_argument("--task-batches-per-round", type=int, default=49)
    parser.add_argument("--task-rounds", type=int, default=2)
    parser.add_argument("--task-learning-rate", type=float, default=0.000003)
    parser.add_argument(
        "--task-parameterization",
        choices=TASK_PARAMETERIZATIONS,
        default="full",
    )
    parser.add_argument("--task-rank", type=int, default=256)
    parser.add_argument(
        "--task-time-weighting",
        choices=TASK_TIME_WEIGHTINGS,
        default="uniform",
    )
    parser.add_argument("--task-weight-decay", type=float, default=0.0)
    parser.add_argument("--task-grad-clip", type=float, default=1.0)
    parser.add_argument("--task-detach-interval", type=int, default=0)
    parser.add_argument("--task-freeze-bias", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--position-group", type=str, default="all")
    parser.add_argument("--operating-age", type=int, default=2)
    parser.add_argument(
        "--direct-operating-target-age",
        type=int,
        help=(
            "Fit and train one J application directly to this operating-age "
            "pre-Block2 interface instead of an adjacent one-age rollback."
        ),
    )
    parser.add_argument("--answer-weight", type=int, default=28)
    parser.add_argument("--identity-weight", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=161001)
    parser.add_argument("--dagger-seed", type=int, default=161002)
    parser.add_argument("--task-seed", type=int, default=161003)
    parser.add_argument("--power-seed", type=int, default=161005)
    parser.add_argument("--evaluation-seed", type=int, default=161004)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=1.0)
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
        frozen_model_loss_placement=args.frozen_model_loss_placement,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        dagger_batch_size=args.dagger_batch_size,
        dagger_batches_per_round=args.dagger_batches_per_round,
        dagger_rounds=args.dagger_rounds,
        rollout_horizons=args.rollout_horizons,
        policies=args.policies,
        train_start_ages=args.train_start_ages,
        initial_map_artifact=args.initial_map_artifact,
        initial_map_label=args.initial_map_label,
        power_batch_size=args.power_batch_size,
        power_batches_per_round=args.power_batches_per_round,
        power_rounds=args.power_rounds,
        power_learning_rate=args.power_learning_rate,
        power_identity_loss_weight=args.power_identity_loss_weight,
        task_batch_size=args.task_batch_size,
        task_batches_per_round=args.task_batches_per_round,
        task_rounds=args.task_rounds,
        task_learning_rate=args.task_learning_rate,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        continuation_loops=args.continuation_loops,
        position_group=args.position_group,
        operating_age=args.operating_age,
        answer_weight=args.answer_weight,
        identity_weight=args.identity_weight,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        dagger_seed=args.dagger_seed,
        task_seed=args.task_seed,
        power_seed=args.power_seed,
        evaluation_seed=args.evaluation_seed,
        physical_gpu=args.physical_gpu,
        prelaunch_used_mib=args.prelaunch_used_mib,
        prelaunch_free_mib=args.prelaunch_free_mib,
        declared_peak_gib=args.declared_peak_gib,
        reserve_gib=args.reserve_gib,
        shared_gpu=args.shared_gpu,
        direct_operating_target_age=args.direct_operating_target_age,
        task_parameterization=args.task_parameterization,
        task_rank=args.task_rank,
        task_time_weighting=args.task_time_weighting,
        task_weight_decay=args.task_weight_decay,
        task_grad_clip=args.task_grad_clip,
        task_detach_interval=args.task_detach_interval,
        task_train_bias=not args.task_freeze_bias,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
