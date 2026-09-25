from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    _attention_parts,
    _attention_pattern,
    _project_context,
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


@dataclass(frozen=True)
class VectorAffine:
    """A row-vector affine map, x -> x @ weight + bias."""

    weight: torch.Tensor
    bias: torch.Tensor
    update_rank: int
    fit_dimension: int
    retained_fit_energy: float

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.weight.shape[0]:
            raise ValueError("value feature count does not match map")
        return value.float() @ self.weight + self.bias

    def repeated(self, value: torch.Tensor, count: int) -> torch.Tensor:
        if count < 0:
            raise ValueError("count must be nonnegative")
        result = value
        for _ in range(count):
            result = self(result)
        return result

    @property
    def factor_parameter_count(self) -> int:
        dimension = self.weight.shape[0]
        if self.update_rank >= dimension:
            return dimension * dimension + dimension
        return 2 * dimension * self.update_rank + dimension


@dataclass(frozen=True)
class ReducedRankFamily:
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    update_ols: torch.Tensor
    output_right_vectors: torch.Tensor
    fitted_singular_values: torch.Tensor

    @property
    def dimension(self) -> int:
        return self.update_ols.shape[0]

    def map_for_rank(self, rank: int) -> VectorAffine:
        if not 0 <= rank <= self.dimension:
            raise ValueError("rank must be between zero and feature dimension")
        if rank == 0:
            update = torch.zeros_like(self.update_ols)
        else:
            basis = self.output_right_vectors[:, :rank]
            update = self.update_ols @ basis @ basis.transpose(0, 1)
        identity = torch.eye(
            self.dimension,
            device=update.device,
            dtype=update.dtype,
        )
        weight = identity + update
        bias = self.target_mean - self.source_mean @ weight
        energy = self.fitted_singular_values.square()
        retained = (
            0.0
            if rank == 0
            else float(
                energy[:rank].sum()
                / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
            )
        )
        return VectorAffine(
            weight=weight,
            bias=bias,
            update_rank=rank,
            fit_dimension=self.dimension,
            retained_fit_energy=retained,
        )


@dataclass(frozen=True)
class AlignedPairData:
    h3_answer: torch.Tensor
    h2_answer: torch.Tensor
    z3_block2_input: torch.Tensor
    z2_block2_input: torch.Tensor
    z3_block2_full: torch.Tensor
    z2_block2_full: torch.Tensor
    q3_head2: torch.Tensor
    q2_head2: torch.Tensor
    h3_full: torch.Tensor
    h2_full: torch.Tensor
    successors: torch.Tensor
    current: torch.Tensor
    next_two_hop: torch.Tensor


@dataclass(frozen=True)
class LoopStep:
    state: torch.Tensor
    logits: torch.Tensor
    block2_hidden_pre_intervention: torch.Tensor
    block2_hidden_in: torch.Tensor
    block2_q: torch.Tensor
    block2_k: torch.Tensor
    block2_v: torch.Tensor
    block2_pattern: torch.Tensor
    block2_head_context: torch.Tensor
    block2_attention_out: torch.Tensor
    block2_residual_mid: torch.Tensor
    block2_mlp_out: torch.Tensor


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.float() - target.float()).square().mean()
    centered = target.float() - target.float().mean(dim=0, keepdim=True)
    return float(numerator / centered.square().mean().clamp_min(1e-12))


def _plain_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


def _apply_answer_map(
    state: torch.Tensor,
    *,
    answer_position: int,
    age_map: VectorAffine,
    count: int = 1,
) -> torch.Tensor:
    result = state.clone()
    result[:, answer_position] = age_map.repeated(
        result[:, answer_position],
        count,
    ).to(dtype=result.dtype)
    return result


def fit_reduced_rank_update_family(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> ReducedRankFamily:
    """Fit target-source with reduced-rank regression in output space.

    This is a rank constraint on the learned update matrix, not PCA of hidden
    states.  The unconstrained ridge update is fitted once; each rank uses the
    reduced-rank-regression output basis of its fitted update.
    """

    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must share [sample, feature] shape")
    if source.shape[0] <= source.shape[1]:
        raise ValueError("reduced-rank fit needs more samples than features")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    source = source.float()
    target = target.float()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    centered_source = source - source_mean
    centered_update = (target - source) - (target_mean - source_mean)
    gram = centered_source.transpose(0, 1) @ centered_source
    cross = centered_source.transpose(0, 1) @ centered_update
    scale = gram.diagonal().mean().clamp_min(1e-6)
    regularized = gram + ridge * scale * torch.eye(
        source.shape[1],
        device=source.device,
        dtype=source.dtype,
    )
    update_ols = torch.linalg.solve(regularized, cross)
    fitted_update = centered_source @ update_ols
    _, singular_values, right_transpose = torch.linalg.svd(
        fitted_update,
        full_matrices=False,
    )
    return ReducedRankFamily(
        source_mean=source_mean,
        target_mean=target_mean,
        update_ols=update_ols,
        output_right_vectors=right_transpose.transpose(0, 1),
        fitted_singular_values=singular_values,
    )


@torch.no_grad()
def run_one_loop(
    model: LoopedGraphPathTransformer,
    state: torch.Tensor,
    *,
    loop_index: int,
    loop_input_position_override: tuple[
        tuple[int, ...],
        torch.Tensor,
    ]
    | None = None,
    block2_input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    block2_input_override: torch.Tensor | None = None,
    block2_position_transform: tuple[
        tuple[int, ...],
        Callable[[torch.Tensor], torch.Tensor],
    ]
    | None = None,
    block2_position_override: tuple[
        tuple[int, ...],
        torch.Tensor,
    ]
    | None = None,
    query_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    query_transform_head: int = 2,
    query_override: dict[int, torch.Tensor] | None = None,
    key_override: dict[int, torch.Tensor] | None = None,
    value_override: dict[int, torch.Tensor] | None = None,
    pattern_answer_override: dict[int, torch.Tensor] | None = None,
    context_answer_override: dict[int, torch.Tensor] | None = None,
    attention_answer_override: torch.Tensor | None = None,
    mlp_answer_override: torch.Tensor | None = None,
) -> LoopStep:
    """Run one shared stack with an optional Block2 answer-query intervention."""

    if model.block_style != "legacy" or len(model.blocks) != 2:
        raise ValueError("localized query experiment requires two legacy blocks")
    answer_position = state.shape[1] - 1
    x = state
    if loop_input_position_override is not None:
        positions, replacement = loop_input_position_override
        index = list(positions)
        if replacement.shape != x[:, index].shape:
            raise ValueError("loop-input position override has the wrong shape")
        x = x.clone()
        x[:, index] = replacement.to(dtype=x.dtype)
    block2_hidden_in: torch.Tensor | None = None
    block2_hidden_pre_intervention: torch.Tensor | None = None
    block2_q: torch.Tensor | None = None
    block2_k: torch.Tensor | None = None
    block2_v: torch.Tensor | None = None
    block2_pattern: torch.Tensor | None = None
    block2_head_context: torch.Tensor | None = None
    block2_attention_out: torch.Tensor | None = None
    block2_residual_mid: torch.Tensor | None = None
    block2_mlp_out: torch.Tensor | None = None
    for block_index in model.active_block_indices(loop_index):
        block = model.blocks[block_index]
        if not isinstance(block, TransformerBlock):
            raise TypeError("legacy TransformerBlock required")
        if block_index == 1:
            block2_hidden_pre_intervention = x
            x = x.clone()
            if block2_input_transform is not None:
                x[:, answer_position] = block2_input_transform(
                    x[:, answer_position]
                ).to(dtype=x.dtype)
            if block2_input_override is not None:
                if block2_input_override.shape != x[:, answer_position].shape:
                    raise ValueError("Block2 input override has the wrong shape")
                x[:, answer_position] = block2_input_override.to(dtype=x.dtype)
            if block2_position_transform is not None:
                positions, transform = block2_position_transform
                index = list(positions)
                x[:, index] = transform(x[:, index]).to(dtype=x.dtype)
            if block2_position_override is not None:
                positions, replacement = block2_position_override
                index = list(positions)
                if replacement.shape != x[:, index].shape:
                    raise ValueError(
                        "Block2 position override has the wrong shape"
                    )
                x[:, index] = replacement.to(dtype=x.dtype)
            block2_hidden_in = x
        hidden_in = x
        q, k, v = _attention_parts(block.attn, block.ln_1(x))
        if block_index == 1:
            q = q.clone()
            if query_transform is not None:
                live = q[:, query_transform_head, answer_position]
                q[:, query_transform_head, answer_position] = query_transform(
                    live
                ).to(dtype=q.dtype)
            if query_override is not None:
                for head, replacement in query_override.items():
                    if replacement.shape != q[:, head, answer_position].shape:
                        raise ValueError("query override has the wrong shape")
                    q[:, head, answer_position] = replacement.to(dtype=q.dtype)
            if key_override is not None:
                k = k.clone()
                for head, replacement in key_override.items():
                    if replacement.shape != k[:, head].shape:
                        raise ValueError("key override has the wrong shape")
                    k[:, head] = replacement.to(dtype=k.dtype)
            if value_override is not None:
                v = v.clone()
                for head, replacement in value_override.items():
                    if replacement.shape != v[:, head].shape:
                        raise ValueError("value override has the wrong shape")
                    v[:, head] = replacement.to(dtype=v.dtype)
            block2_q = q
            block2_k = k
            block2_v = v
        pattern = _attention_pattern(q, k)
        if block_index == 1:
            if pattern_answer_override is not None:
                pattern = pattern.clone()
                for head, replacement in pattern_answer_override.items():
                    if replacement.shape != pattern[
                        :, head, answer_position
                    ].shape:
                        raise ValueError(
                            "attention-pattern override has the wrong shape"
                        )
                    pattern[:, head, answer_position] = replacement.to(
                        dtype=pattern.dtype
                    )
            block2_pattern = pattern
        context = torch.matmul(pattern, v)
        if block_index == 1:
            if context_answer_override is not None:
                context = context.clone()
                for head, replacement in context_answer_override.items():
                    if replacement.shape != context[
                        :, head, answer_position
                    ].shape:
                        raise ValueError(
                            "head-context override has the wrong shape"
                        )
                    context[:, head, answer_position] = replacement.to(
                        dtype=context.dtype
                    )
            block2_head_context = context
        attention_out = _project_context(block.attn, context)
        if block.inner_norm_style == "ouro_sandwich_rms":
            attention_out = block.attn_out_norm(attention_out)
        if block_index == 1:
            if attention_answer_override is not None:
                if attention_answer_override.shape != attention_out[
                    :, answer_position
                ].shape:
                    raise ValueError(
                        "attention-output override has the wrong shape"
                    )
                attention_out = attention_out.clone()
                attention_out[:, answer_position] = (
                    attention_answer_override.to(dtype=attention_out.dtype)
                )
            block2_attention_out = attention_out
        residual_mid = x + attention_out
        if block_index == 1:
            block2_residual_mid = residual_mid
        normalized = block.ln_2(residual_mid)
        mlp_hidden = block.mlp[1](block.mlp[0](normalized))
        mlp_out = block.mlp[3](block.mlp[2](mlp_hidden))
        if block.inner_norm_style == "ouro_sandwich_rms":
            mlp_out = block.mlp_out_norm(mlp_out)
        if block_index == 1:
            if mlp_answer_override is not None:
                if mlp_answer_override.shape != mlp_out[
                    :, answer_position
                ].shape:
                    raise ValueError("MLP-output override has the wrong shape")
                mlp_out = mlp_out.clone()
                mlp_out[:, answer_position] = mlp_answer_override.to(
                    dtype=mlp_out.dtype
                )
            block2_mlp_out = mlp_out
        x = residual_mid + mlp_out
        if block.residual_projector is not None:
            x = hidden_in + block.residual_projector(
                hidden_in,
                x - hidden_in,
            )
    if model.outer_norm is not None:
        x = model.outer_norm(x)
    if (
        block2_hidden_in is None
        or block2_hidden_pre_intervention is None
        or block2_q is None
        or block2_k is None
        or block2_v is None
        or block2_pattern is None
        or block2_head_context is None
        or block2_attention_out is None
        or block2_residual_mid is None
        or block2_mlp_out is None
    ):
        raise RuntimeError("Block2 trace was not produced")
    return LoopStep(
        state=x,
        logits=logits_from_raw_state(model, x),
        block2_hidden_pre_intervention=block2_hidden_pre_intervention,
        block2_hidden_in=block2_hidden_in,
        block2_q=block2_q,
        block2_k=block2_k,
        block2_v=block2_v,
        block2_pattern=block2_pattern,
        block2_head_context=block2_head_context,
        block2_attention_out=block2_attention_out,
        block2_residual_mid=block2_residual_mid,
        block2_mlp_out=block2_mlp_out,
    )


@torch.no_grad()
def collect_aligned_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    executor_head: int,
) -> AlignedPairData:
    answer_position = explicit_depth_position_groups(cfg.node_count)["answer"][0]
    parts: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "h3_answer",
            "h2_answer",
            "z3_block2_input",
            "z2_block2_input",
            "z3_block2_full",
            "z2_block2_full",
            "q3_head2",
            "q2_head2",
            "h3_full",
            "h2_full",
            "successors",
            "current",
            "next_two_hop",
        )
    }
    jump = phase_positions[3] - phase_positions[2]
    set_seed(seed)
    for _ in range(batches):
        _, targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump,
        )
        current = targets[:, cfg.max_depth - 1]
        next_two_hop = targets[:, cfg.max_depth + jump - 1]
        h3 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=3,
            phase_position=phase_positions[3],
        )
        h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        h3_step = run_one_loop(
            model,
            h3,
            loop_index=cfg.max_loops,
        )
        h2_step = run_one_loop(
            model,
            h2,
            loop_index=cfg.max_loops,
        )
        q3 = h3_step.block2_q[:, executor_head, answer_position]
        q2 = h2_step.block2_q[:, executor_head, answer_position]
        values = {
            "h3_answer": h3[:, answer_position].float(),
            "h2_answer": h2[:, answer_position].float(),
            "z3_block2_input": h3_step.block2_hidden_in[
                :, answer_position
            ].float(),
            "z2_block2_input": h2_step.block2_hidden_in[
                :, answer_position
            ].float(),
            "z3_block2_full": h3_step.block2_hidden_in.float(),
            "z2_block2_full": h2_step.block2_hidden_in.float(),
            "q3_head2": q3.float(),
            "q2_head2": q2.float(),
            "h3_full": h3,
            "h2_full": h2,
            "successors": successors,
            "current": current,
            "next_two_hop": next_two_hop,
        }
        for name, value in values.items():
            parts[name].append(value)
    return AlignedPairData(
        **{name: torch.cat(values) for name, values in parts.items()}
    )


@torch.no_grad()
def evaluate_heldout_fit(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    data: AlignedPairData,
    residual_maps: dict[int, VectorAffine],
    executor_maps: dict[int, VectorAffine],
    query_maps: dict[int, VectorAffine],
    executor_head: int,
) -> list[dict[str, Any]]:
    answer_position = explicit_depth_position_groups(cfg.node_count)["answer"][0]
    oracle_step = run_one_loop(
        model,
        data.h2_full,
        loop_index=cfg.max_loops,
    )
    oracle_metrics = _masked_metrics(
        oracle_step.logits,
        data.next_two_hop,
        endpoint=data.current,
    )
    rows: list[dict[str, Any]] = []
    for space, maps in (
        ("residual", residual_maps),
        ("executor_input", executor_maps),
        ("query", query_maps),
    ):
        for rank, age_map in maps.items():
            if space == "residual":
                predicted = age_map(data.h3_answer)
                intervened_state = data.h3_full.clone()
                intervened_state[:, answer_position] = predicted.to(
                    dtype=intervened_state.dtype
                )
                step = run_one_loop(
                    model,
                    intervened_state,
                    loop_index=cfg.max_loops,
                )
                state_current_accuracy = _plain_accuracy(
                    logits_from_raw_state(model, intervened_state),
                    data.current,
                )
                target = data.h2_answer
            elif space == "executor_input":
                predicted = age_map(data.z3_block2_input)
                step = run_one_loop(
                    model,
                    data.h3_full,
                    loop_index=cfg.max_loops,
                    block2_input_override=predicted,
                )
                state_current_accuracy = float("nan")
                target = data.z2_block2_input
            else:
                predicted = age_map(data.q3_head2)
                step = run_one_loop(
                    model,
                    data.h3_full,
                    loop_index=cfg.max_loops,
                    query_override={executor_head: predicted},
                )
                state_current_accuracy = float("nan")
                target = data.q2_head2
            metrics = _masked_metrics(
                step.logits,
                data.next_two_hop,
                endpoint=data.current,
            )
            rows.append(
                {
                    "space": space,
                    "rank": rank,
                    "factor_parameter_count": age_map.factor_parameter_count,
                    "retained_fit_energy": age_map.retained_fit_energy,
                    "heldout_relative_mse": _relative_mse(predicted, target),
                    "direct_current_accuracy": state_current_accuracy,
                    "next_two_hop_accuracy": metrics["accuracy"],
                    "next_two_hop_margin": metrics["margin"],
                    "valid_count": metrics["valid_count"],
                    "exact_h2_next_accuracy": oracle_metrics["accuracy"],
                    "exact_h2_next_margin": oracle_metrics["margin"],
                }
            )
    return rows


@torch.no_grad()
def evaluate_age_ladder(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    residual_maps: dict[int, VectorAffine],
    executor_maps: dict[int, VectorAffine],
    query_maps: dict[int, VectorAffine],
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    executor_head: int,
) -> list[dict[str, Any]]:
    answer_position = explicit_depth_position_groups(cfg.node_count)["answer"][0]
    jump = phase_positions[3] - phase_positions[2]
    rows: list[dict[str, Any]] = []
    set_seed(seed)
    for batch_index in range(batches):
        _, targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump,
        )
        current = targets[:, cfg.max_depth - 1]
        states = {
            age: _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=age,
                phase_position=phase_positions[age],
            )
            for age in range(2, cfg.max_loops + 1)
        }
        natural_steps = {
            age: run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops,
            )
            for age, state in states.items()
        }
        q_by_age = {
            age: step.block2_q[:, executor_head, answer_position].float()
            for age, step in natural_steps.items()
        }
        z_by_age = {
            age: step.block2_hidden_in[:, answer_position].float()
            for age, step in natural_steps.items()
        }
        for source_age in range(3, cfg.max_loops + 1):
            adjacent_age = source_age - 1
            adjacent_jump = (
                phase_positions[min(adjacent_age + 1, cfg.max_loops)]
                - phase_positions[adjacent_age]
            )
            adjacent_target = advance_nodes(
                successors,
                current,
                steps=adjacent_jump,
            )
            two_hop_target = advance_nodes(
                successors,
                current,
                steps=jump,
            )
            for space, maps in (
                ("residual", residual_maps),
                ("executor_input", executor_maps),
                ("query", query_maps),
            ):
                for rank, age_map in maps.items():
                    for test, count, target_age, behavior_target in (
                        (
                            "one_step",
                            1,
                            adjacent_age,
                            adjacent_target,
                        ),
                        (
                            "power_to_h2",
                            source_age - 2,
                            2,
                            two_hop_target,
                        ),
                    ):
                        if space == "residual":
                            prediction = age_map.repeated(
                                states[source_age][:, answer_position],
                                count,
                            )
                            mapped_state = states[source_age].clone()
                            mapped_state[:, answer_position] = prediction.to(
                                dtype=mapped_state.dtype
                            )
                            step = run_one_loop(
                                model,
                                mapped_state,
                                loop_index=cfg.max_loops,
                            )
                            target_representation = states[target_age][
                                :, answer_position
                            ]
                        elif space == "executor_input":
                            prediction = age_map.repeated(
                                z_by_age[source_age],
                                count,
                            )
                            step = run_one_loop(
                                model,
                                states[source_age],
                                loop_index=cfg.max_loops,
                                block2_input_override=prediction,
                            )
                            target_representation = z_by_age[target_age]
                        else:
                            prediction = age_map.repeated(
                                q_by_age[source_age],
                                count,
                            )
                            step = run_one_loop(
                                model,
                                states[source_age],
                                loop_index=cfg.max_loops,
                                query_override={executor_head: prediction},
                            )
                            target_representation = q_by_age[target_age]
                        rows.append(
                            {
                                "batch": batch_index,
                                "space": space,
                                "rank": rank,
                                "source_age": source_age,
                                "target_age": target_age,
                                "test": test,
                                "applications": count,
                                "representation_relative_mse": _relative_mse(
                                    prediction,
                                    target_representation,
                                ),
                                "behavior_target_jump": (
                                    adjacent_jump
                                    if test == "one_step"
                                    else jump
                                ),
                                "behavior_accuracy": _plain_accuracy(
                                    step.logits,
                                    behavior_target,
                                ),
                            }
                        )
    return rows


def _routing_metrics(
    pattern: torch.Tensor,
    *,
    current: torch.Tensor,
    destination_positions: tuple[int, ...],
    executor_head: int,
    answer_position: int,
) -> tuple[float, float]:
    row = pattern[:, executor_head, answer_position]
    positions = torch.as_tensor(
        destination_positions,
        device=row.device,
    )
    correct = positions[current]
    batch = torch.arange(row.shape[0], device=row.device)
    return (
        float(row[batch, correct].mean()),
        float(row.argmax(dim=-1).eq(correct).float().mean()),
    )


@torch.no_grad()
def evaluate_closed_loop(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    residual_maps: dict[int, VectorAffine],
    executor_maps: dict[int, VectorAffine],
    query_maps: dict[int, VectorAffine],
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
    executor_head: int,
) -> list[dict[str, Any]]:
    position_groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = position_groups["answer"][0]
    destination_positions = position_groups["destination"]
    jump = phase_positions[3] - phase_positions[2]
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
        exact_h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states: dict[str, torch.Tensor] = {
            "no_intervention": exact_h2.clone(),
            "oracle_head2_q": exact_h2.clone(),
            "shuffled_oracle_head2_q": exact_h2.clone(),
            "oracle_head3_q": exact_h2.clone(),
            "oracle_block2_input": exact_h2.clone(),
            "shuffled_oracle_block2_input": exact_h2.clone(),
            **{
                f"residual_r{rank}": exact_h2.clone()
                for rank in residual_maps
            },
            **{
                f"executor_r{rank}": exact_h2.clone()
                for rank in executor_maps
            },
            **{
                f"query_r{rank}": exact_h2.clone()
                for rank in query_maps
            },
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
            condition_steps: dict[str, LoopStep] = {
                "oracle_full_state": oracle_step,
            }
            condition_steps["no_intervention"] = run_one_loop(
                model,
                states["no_intervention"],
                loop_index=cfg.max_loops + cycle - 1,
            )
            if cycle == 1:
                for condition in (
                    "oracle_head2_q",
                    "shuffled_oracle_head2_q",
                    "oracle_head3_q",
                    "oracle_block2_input",
                    "shuffled_oracle_block2_input",
                ):
                    condition_steps[condition] = run_one_loop(
                        model,
                        states[condition],
                        loop_index=cfg.max_loops + cycle - 1,
                    )
            else:
                oracle_head2 = oracle_step.block2_q[
                    :, executor_head, answer_position
                ]
                condition_steps["oracle_head2_q"] = run_one_loop(
                    model,
                    states["oracle_head2_q"],
                    loop_index=cfg.max_loops + cycle - 1,
                    query_override={executor_head: oracle_head2},
                )
                condition_steps["shuffled_oracle_head2_q"] = run_one_loop(
                    model,
                    states["shuffled_oracle_head2_q"],
                    loop_index=cfg.max_loops + cycle - 1,
                    query_override={
                        executor_head: oracle_head2.roll(1, dims=0)
                    },
                )
                control_head = 3
                condition_steps["oracle_head3_q"] = run_one_loop(
                    model,
                    states["oracle_head3_q"],
                    loop_index=cfg.max_loops + cycle - 1,
                    query_override={
                        control_head: oracle_step.block2_q[
                            :, control_head, answer_position
                        ]
                    },
                )
                oracle_block2_input = oracle_step.block2_hidden_in[
                    :, answer_position
                ]
                condition_steps["oracle_block2_input"] = run_one_loop(
                    model,
                    states["oracle_block2_input"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_input_override=oracle_block2_input,
                )
                condition_steps[
                    "shuffled_oracle_block2_input"
                ] = run_one_loop(
                    model,
                    states["shuffled_oracle_block2_input"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_input_override=oracle_block2_input.roll(
                        1,
                        dims=0,
                    ),
                )
            for rank, age_map in residual_maps.items():
                condition = f"residual_r{rank}"
                state = states[condition]
                if cycle > 1:
                    state = _apply_answer_map(
                        state,
                        answer_position=answer_position,
                        age_map=age_map,
                    )
                condition_steps[condition] = run_one_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops + cycle - 1,
                )
            for rank, age_map in executor_maps.items():
                condition = f"executor_r{rank}"
                transform = age_map if cycle > 1 else None
                condition_steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_input_transform=transform,
                )
            for rank, age_map in query_maps.items():
                condition = f"query_r{rank}"
                transform = age_map if cycle > 1 else None
                condition_steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    query_transform=transform,
                    query_transform_head=executor_head,
                )
            for condition, step in condition_steps.items():
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
                    }
                )
            for condition in states:
                states[condition] = condition_steps[condition].state
    return rows


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    weighted_fields: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, parts in groups.items():
        result = dict(zip(keys, group, strict=True))
        numeric_fields = [
            key
            for key, value in parts[0].items()
            if key not in keys
            and key != "batch"
            and isinstance(value, (int, float))
        ]
        for field in numeric_fields:
            if field in weighted_fields:
                count = sum(float(part["valid_count"]) for part in parts)
                result[field] = (
                    sum(
                        float(part[field]) * float(part["valid_count"])
                        for part in parts
                    )
                    / count
                    if count
                    else float("nan")
                )
            else:
                values = [float(part[field]) for part in parts]
                result[field] = float(np.mean(values))
        output.append(result)
    return sorted(output, key=lambda row: tuple(row[key] for key in keys))


def _curve_summary(
    closed_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregated = _aggregate_rows(
        closed_rows,
        keys=("condition", "cycle"),
        weighted_fields=("accuracy", "margin"),
    )
    result: dict[str, dict[str, Any]] = {}
    conditions = sorted({str(row["condition"]) for row in aggregated})
    for condition in conditions:
        parts = [
            row
            for row in aggregated
            if row["condition"] == condition
        ]
        parts.sort(key=lambda row: int(row["cycle"]))
        accuracy = [float(row["accuracy"]) for row in parts]
        result[condition] = {
            "accuracy": accuracy,
            "auc": float(np.mean(accuracy)),
            "routing_mass": [
                float(row["head2_correct_destination_mass"])
                for row in parts
            ],
            "routing_argmax": [
                float(row["head2_correct_destination_argmax"])
                for row in parts
            ],
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
    evaluation_batch_size: int,
    evaluation_batches: int,
    age_ladder_batches: int,
    extra_loops: int,
    calibration_seed: int,
    heldout_seed: int,
    evaluation_seed: int,
    ridge: float,
    residual_ranks: Sequence[int],
    executor_ranks: Sequence[int],
    query_ranks: Sequence[int],
    executor_head: int,
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
    if cfg.n_layers != 2 or cfg.max_loops != 8 or cfg.max_depth != 8:
        raise ValueError("experiment is fixed to the D8L8 two-block model")
    if executor_head >= cfg.n_heads:
        raise ValueError("executor head is out of range")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    if len(phase_positions) != cfg.max_loops + 1:
        raise ValueError("phase summary has the wrong trajectory length")
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
        executor_head=executor_head,
    )
    heldout = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=heldout_seed,
        executor_head=executor_head,
    )
    residual_family = fit_reduced_rank_update_family(
        calibration.h3_answer,
        calibration.h2_answer,
        ridge=ridge,
    )
    executor_family = fit_reduced_rank_update_family(
        calibration.z3_block2_input,
        calibration.z2_block2_input,
        ridge=ridge,
    )
    query_family = fit_reduced_rank_update_family(
        calibration.q3_head2,
        calibration.q2_head2,
        ridge=ridge,
    )
    residual_maps = {
        int(rank): residual_family.map_for_rank(int(rank))
        for rank in residual_ranks
    }
    executor_maps = {
        int(rank): executor_family.map_for_rank(int(rank))
        for rank in executor_ranks
    }
    query_maps = {
        int(rank): query_family.map_for_rank(int(rank))
        for rank in query_ranks
    }
    heldout_rows = evaluate_heldout_fit(
        model=model,
        cfg=cfg,
        data=heldout,
        residual_maps=residual_maps,
        executor_maps=executor_maps,
        query_maps=query_maps,
        executor_head=executor_head,
    )
    age_rows = evaluate_age_ladder(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        residual_maps=residual_maps,
        executor_maps=executor_maps,
        query_maps=query_maps,
        device=device,
        batch_size=evaluation_batch_size,
        batches=age_ladder_batches,
        seed=evaluation_seed + 1,
        executor_head=executor_head,
    )
    closed_rows = evaluate_closed_loop(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        residual_maps=residual_maps,
        executor_maps=executor_maps,
        query_maps=query_maps,
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=executor_head,
    )
    age_aggregate = _aggregate_rows(
        age_rows,
        keys=("space", "rank", "source_age", "target_age", "test"),
    )
    closed_aggregate = _aggregate_rows(
        closed_rows,
        keys=("condition", "cycle"),
        weighted_fields=("accuracy", "margin"),
    )
    curves = _curve_summary(closed_rows)
    best_residual = max(
        (
            (name, data["auc"])
            for name, data in curves.items()
            if name.startswith("residual_")
        ),
        key=lambda item: item[1],
    )
    best_query = max(
        (
            (name, data["auc"])
            for name, data in curves.items()
            if name.startswith("query_")
        ),
        key=lambda item: item[1],
    )
    best_executor = max(
        (
            (name, data["auc"])
            for name, data in curves.items()
            if name.startswith("executor_")
        ),
        key=lambda item: item[1],
    )
    map_payload = {
        "kind": "graph_path_telomere_localized_query",
        "residual": {
            rank: {
                "weight": age_map.weight.cpu(),
                "bias": age_map.bias.cpu(),
                "retained_fit_energy": age_map.retained_fit_energy,
            }
            for rank, age_map in residual_maps.items()
        },
        "query": {
            rank: {
                "weight": age_map.weight.cpu(),
                "bias": age_map.bias.cpu(),
                "retained_fit_energy": age_map.retained_fit_energy,
            }
            for rank, age_map in query_maps.items()
        },
        "executor_input": {
            rank: {
                "weight": age_map.weight.cpu(),
                "bias": age_map.bias.cpu(),
                "retained_fit_energy": age_map.retained_fit_energy,
            }
            for rank, age_map in executor_maps.items()
        },
    }
    torch.save(map_payload, out_dir / "localized_maps.pt")
    _write_csv(out_dir / "heldout_fit_rows.csv", heldout_rows)
    _write_csv(out_dir / "age_ladder_rows.csv", age_rows)
    _write_csv(out_dir / "age_ladder_aggregate.csv", age_aggregate)
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    _write_csv(out_dir / "closed_loop_aggregate.csv", closed_aggregate)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "checkpoint_seed": checkpoint_payload.get("seed"),
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
        "training": {
            "relation": "strict same-current H3 -> H2 only",
            "ridge": ridge,
            "calibration_examples": int(
                calibration_batch_size * calibration_batches
            ),
            "calibration_seed": calibration_seed,
            "excluded": [
                "task CE",
                "other ages",
                "power loss",
                "closed-loop loss",
                "rollout states",
                "DAgger",
            ],
        },
        "heldout_examples": int(heldout_batch_size * heldout_batches),
        "evaluation_graphs": int(
            evaluation_batch_size * evaluation_batches
        ),
        "extra_loops": extra_loops,
        "residual_ranks": [int(rank) for rank in residual_ranks],
        "executor_ranks": [int(rank) for rank in executor_ranks],
        "query_ranks": [int(rank) for rank in query_ranks],
        "heldout_fit": heldout_rows,
        "closed_loop": curves,
        "best_residual_by_eval_auc_descriptive_only": {
            "condition": best_residual[0],
            "auc": best_residual[1],
        },
        "best_query_by_eval_auc_descriptive_only": {
            "condition": best_query[0],
            "auc": best_query[1],
        },
        "best_executor_by_eval_auc_descriptive_only": {
            "condition": best_executor[0],
            "auc": best_executor[1],
        },
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "maps": "localized_maps.pt",
            "heldout_fit": "heldout_fit_rows.csv",
            "age_ladder": "age_ladder_rows.csv",
            "age_ladder_aggregate": "age_ladder_aggregate.csv",
            "closed_loop": "closed_loop_rows.csv",
            "closed_loop_aggregate": "closed_loop_aggregate.csv",
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
            "Test low-rank residual and Block2-head2 query rejuvenation maps "
            "on the fixed D8L8-seed1 Graph checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--heldout-batch-size", type=int, default=256)
    parser.add_argument("--heldout-batches", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--age-ladder-batches", type=int, default=2)
    parser.add_argument("--extra-loops", type=int, default=16)
    parser.add_argument("--calibration-seed", type=int, default=73101)
    parser.add_argument("--heldout-seed", type=int, default=73102)
    parser.add_argument("--evaluation-seed", type=int, default=73103)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument(
        "--residual-ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    parser.add_argument(
        "--executor-ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    parser.add_argument(
        "--query-ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64),
    )
    parser.add_argument("--executor-head", type=int, default=2)
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
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        age_ladder_batches=args.age_ladder_batches,
        extra_loops=args.extra_loops,
        calibration_seed=args.calibration_seed,
        heldout_seed=args.heldout_seed,
        evaluation_seed=args.evaluation_seed,
        ridge=args.ridge,
        residual_ranks=args.residual_ranks,
        executor_ranks=args.executor_ranks,
        query_ranks=args.query_ranks,
        executor_head=args.executor_head,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
