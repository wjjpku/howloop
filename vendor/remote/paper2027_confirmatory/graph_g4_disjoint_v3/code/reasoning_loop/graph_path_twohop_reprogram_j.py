from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


@dataclass(frozen=True)
class ActiveAgeBatch:
    source: torch.Tensor
    current_node: torch.Tensor
    one_node: torch.Tensor
    two_node: torch.Tensor
    age: torch.Tensor


def select_active_age_examples(
    states: torch.Tensor,
    path_targets: torch.Tensor,
    ages: torch.Tensor,
) -> ActiveAgeBatch:
    """Select h_t and semantic targets at t, t+1, and t+2."""

    if states.ndim != 4:
        raise ValueError("states must have [age, batch, position, feature] shape")
    if path_targets.ndim != 2 or path_targets.shape[0] != states.shape[1]:
        raise ValueError("path_targets must have [batch, path] shape")
    if ages.ndim != 1 or ages.shape[0] != states.shape[1]:
        raise ValueError("ages must have one entry per example")
    max_source_age = min(states.shape[0] - 1, path_targets.shape[1] - 2)
    if bool((ages < 1).any()) or bool((ages > max_source_age).any()):
        raise ValueError(f"ages must lie in [1, {max_source_age}]")
    batch_index = torch.arange(states.shape[1], device=states.device)
    target_index = ages - 1
    return ActiveAgeBatch(
        source=states[ages, batch_index],
        current_node=path_targets[batch_index, target_index],
        one_node=path_targets[batch_index, target_index + 1],
        two_node=path_targets[batch_index, target_index + 2],
        age=ages,
    )


def distinct_step_mask(
    current_node: torch.Tensor,
    one_node: torch.Tensor,
    two_node: torch.Tensor,
) -> torch.Tensor:
    if not (
        current_node.shape == one_node.shape == two_node.shape
        and current_node.ndim == 1
    ):
        raise ValueError("node targets must share one-dimensional shape")
    return (
        current_node.ne(one_node)
        & current_node.ne(two_node)
        & one_node.ne(two_node)
    )


class DenseAffineJ(nn.Module):
    """One row-vector affine map shared across all token positions."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.weight = nn.Parameter(torch.eye(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state.float() @ self.weight + self.bias


@dataclass(frozen=True)
class ControllerLosses:
    total: torch.Tensor
    task: torch.Tensor
    preloop: torch.Tensor
    mapped: torch.Tensor
    preloop_logits: torch.Tensor
    output_logits: torch.Tensor


def controller_losses(
    model: nn.Module,
    controller: DenseAffineJ,
    source: torch.Tensor,
    *,
    current_node: torch.Tensor,
    two_node: torch.Tensor,
    preloop_weight: float,
    execute: bool,
) -> ControllerLosses:
    if preloop_weight < 0:
        raise ValueError("preloop_weight must be nonnegative")
    mapped = controller(source)
    preloop_logits = logits_from_raw_state(model, mapped)
    if execute:
        output = model.apply_loop(
            mapped.to(dtype=source.dtype),
            loop_index=model.cfg.max_loops,
        )
        output_logits = logits_from_raw_state(model, output)
    else:
        output_logits = preloop_logits
    task_loss = F.cross_entropy(output_logits, two_node)
    preloop_loss = F.cross_entropy(preloop_logits, current_node)
    return ControllerLosses(
        total=task_loss + preloop_weight * preloop_loss,
        task=task_loss,
        preloop=preloop_loss,
        mapped=mapped,
        preloop_logits=preloop_logits,
        output_logits=output_logits,
    )


def validate_d8l8_onehop_config(cfg: GraphPathConfig) -> None:
    required = (
        cfg.node_count == 8
        and cfg.max_depth == 8
        and cfg.d_model == 256
        and cfg.n_heads == 4
        and cfg.n_layers == 2
        and cfg.max_loops == 8
        and cfg.block_schedule == "all_blocks"
    )
    if not required:
        raise ValueError(
            "two-hop reprogramming requires D8L8, d256, four heads, "
            "two shared blocks, and all_blocks schedule"
        )


def strict_unseen_permutations(
    node_count: int,
    seen: set[tuple[int, ...]],
    *,
    count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    if node_count < 1 or count < 1:
        raise ValueError("node_count and count must be positive")
    candidates = [
        permutation
        for permutation in itertools.permutations(range(node_count))
        if permutation not in seen
    ]
    if len(candidates) < count:
        raise ValueError("not enough unseen permutations for requested count")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def write_summary_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _cat_active_batches(parts: list[ActiveAgeBatch]) -> ActiveAgeBatch:
    if not parts:
        raise ValueError("at least one active-age batch is required")
    return ActiveAgeBatch(
        source=torch.cat([part.source for part in parts]),
        current_node=torch.cat([part.current_node for part in parts]),
        one_node=torch.cat([part.one_node for part in parts]),
        two_node=torch.cat([part.two_node for part in parts]),
        age=torch.cat([part.age for part in parts]),
    )


def _subset_active_batch(
    dataset: ActiveAgeBatch,
    index: torch.Tensor,
) -> ActiveAgeBatch:
    return ActiveAgeBatch(
        source=dataset.source[index],
        current_node=dataset.current_node[index],
        one_node=dataset.one_node[index],
        two_node=dataset.two_node[index],
        age=dataset.age[index],
    )


def _all_permutations_excluding(
    node_count: int,
    excluded: set[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    return [
        permutation
        for permutation in itertools.permutations(range(node_count))
        if permutation not in excluded
    ]


def _dataset_from_explicit_examples(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    starts: list[int],
    ages: list[int],
    device: torch.device,
    collection_batch_size: int,
) -> ActiveAgeBatch:
    if not (len(permutations) == len(starts) == len(ages)):
        raise ValueError("permutations, starts, and ages must have equal length")
    parts: list[ActiveAgeBatch] = []
    for offset in range(0, len(permutations), collection_batch_size):
        stop = min(offset + collection_batch_size, len(permutations))
        successors = torch.tensor(
            permutations[offset:stop],
            dtype=torch.long,
            device=device,
        )
        start = torch.tensor(
            starts[offset:stop],
            dtype=torch.long,
            device=device,
        )
        age = torch.tensor(
            ages[offset:stop],
            dtype=torch.long,
            device=device,
        )
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            stop - offset,
            device,
            path_positions=cfg.max_depth + 2,
            successors=successors,
            start=start,
        )
        states = torch.stack(
            cache_states_with_initial(
                model,
                tokens,
                loops=cfg.max_loops,
            )
        )
        parts.append(select_active_age_examples(states, path_targets, age))
    return _cat_active_batches(parts)


def build_training_dataset(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    excluded_permutations: set[tuple[int, ...]],
    examples: int,
    seed: int,
    device: torch.device,
    collection_batch_size: int,
) -> tuple[ActiveAgeBatch, set[tuple[int, ...]]]:
    if examples < 1:
        raise ValueError("examples must be positive")
    candidates = _all_permutations_excluding(
        cfg.node_count,
        excluded_permutations,
    )
    if not candidates:
        raise ValueError("no training permutations remain")
    generator = random.Random(seed)
    permutations = generator.choices(candidates, k=examples)
    starts = [generator.randrange(cfg.node_count) for _ in range(examples)]
    ages = [generator.randrange(1, 7) for _ in range(examples)]
    dataset = _dataset_from_explicit_examples(
        model=model,
        cfg=cfg,
        permutations=permutations,
        starts=starts,
        ages=ages,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    return dataset, set(permutations)


def build_all_start_age_dataset(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    device: torch.device,
    collection_batch_size: int,
) -> ActiveAgeBatch:
    examples: list[tuple[tuple[int, ...], int, int]] = []
    for permutation in permutations:
        for start in range(cfg.node_count):
            for age in range(1, 7):
                examples.append((permutation, start, age))
    return _dataset_from_explicit_examples(
        model=model,
        cfg=cfg,
        permutations=[row[0] for row in examples],
        starts=[row[1] for row in examples],
        ages=[row[2] for row in examples],
        device=device,
        collection_batch_size=collection_batch_size,
    )


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    learning_rate: float
    preloop_weight: float
    execute: bool


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, Any]]
    validation: dict[str, float]


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


@torch.no_grad()
def evaluate_candidate(
    *,
    model: nn.Module,
    controller: DenseAffineJ,
    dataset: ActiveAgeBatch,
    execute: bool,
    batch_size: int,
) -> dict[str, float]:
    totals = {
        "examples": 0,
        "distinct_examples": 0,
        "pre_current_correct": 0,
        "pre_one_correct": 0,
        "pre_two_correct": 0,
        "post_current_correct": 0,
        "post_one_correct": 0,
        "post_two_correct": 0,
        "distinct_post_two_correct": 0,
    }
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(
            offset,
            min(offset + batch_size, dataset.source.shape[0]),
            device=dataset.source.device,
        )
        batch = _subset_active_batch(dataset, index)
        mapped = controller(batch.source)
        pre_logits = logits_from_raw_state(model, mapped)
        if execute:
            output = model.apply_loop(
                mapped.to(dtype=batch.source.dtype),
                loop_index=model.cfg.max_loops,
            )
            post_logits = logits_from_raw_state(model, output)
        else:
            post_logits = pre_logits
        pre_prediction = pre_logits.argmax(dim=-1)
        post_prediction = post_logits.argmax(dim=-1)
        distinct = distinct_step_mask(
            batch.current_node,
            batch.one_node,
            batch.two_node,
        )
        totals["examples"] += batch.source.shape[0]
        totals["distinct_examples"] += int(distinct.sum())
        for name, prediction in (("pre", pre_prediction), ("post", post_prediction)):
            totals[f"{name}_current_correct"] += int(
                prediction.eq(batch.current_node).sum()
            )
            totals[f"{name}_one_correct"] += int(
                prediction.eq(batch.one_node).sum()
            )
            totals[f"{name}_two_correct"] += int(
                prediction.eq(batch.two_node).sum()
            )
        totals["distinct_post_two_correct"] += int(
            post_prediction[distinct].eq(batch.two_node[distinct]).sum()
        )
    n = max(totals["examples"], 1)
    distinct_n = max(totals["distinct_examples"], 1)
    return {
        "examples": float(totals["examples"]),
        "distinct_examples": float(totals["distinct_examples"]),
        "pre_current_accuracy": totals["pre_current_correct"] / n,
        "pre_one_accuracy": totals["pre_one_correct"] / n,
        "pre_two_accuracy": totals["pre_two_correct"] / n,
        "post_current_accuracy": totals["post_current_correct"] / n,
        "post_one_accuracy": totals["post_one_correct"] / n,
        "post_two_accuracy": totals["post_two_correct"] / n,
        "distinct_post_two_accuracy": (
            totals["distinct_post_two_correct"] / distinct_n
        ),
    }


def train_candidate(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    train_dataset: ActiveAgeBatch,
    validation_dataset: ActiveAgeBatch,
    spec: CandidateSpec,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    weight_decay: float,
    controller_seed: int,
) -> CandidateResult:
    if steps < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError("steps, batch_size, and eval_every must be positive")
    controller = DenseAffineJ(cfg.d_model).to(train_dataset.source.device)
    optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=spec.learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=train_dataset.source.device)
    generator.manual_seed(controller_seed)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float] | None = None
    for step in range(1, steps + 1):
        index = torch.randint(
            0,
            train_dataset.source.shape[0],
            (batch_size,),
            generator=generator,
            device=train_dataset.source.device,
        )
        batch = _subset_active_batch(train_dataset, index)
        losses = controller_losses(
            model,
            controller,
            batch.source,
            current_node=batch.current_node,
            two_node=batch.two_node,
            preloop_weight=spec.preloop_weight,
            execute=spec.execute,
        )
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            controller.parameters(),
            max_norm=1.0,
        )
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone unexpectedly received gradients")
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            validation = evaluate_candidate(
                model=model,
                controller=controller,
                dataset=validation_dataset,
                execute=spec.execute,
                batch_size=eval_batch_size,
            )
            row = {
                "condition": spec.name,
                "step": step,
                "learning_rate": spec.learning_rate,
                "preloop_weight": spec.preloop_weight,
                "execute": spec.execute,
                "loss": float(losses.total.detach()),
                "task_loss": float(losses.task.detach()),
                "preloop_loss": float(losses.preloop.detach()),
                "gradient_norm": float(gradient_norm),
                **validation,
            }
            history.append(row)
            print(
                json.dumps(
                    {"event": "training", **row},
                    allow_nan=True,
                ),
                flush=True,
            )
            score = validation["distinct_post_two_accuracy"]
            if score > best_score:
                best_score = score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in controller.state_dict().items()
                }
                best_validation = dict(validation)
    if best_state is None or best_validation is None:
        raise RuntimeError("candidate training produced no checkpoint")
    return CandidateResult(
        spec=spec,
        state_dict=best_state,
        history=history,
        validation=best_validation,
    )


def _controller_from_result(
    result: CandidateResult,
    *,
    dimension: int,
    device: torch.device,
) -> DenseAffineJ:
    controller = DenseAffineJ(dimension).to(device)
    controller.load_state_dict(
        {name: value.to(device) for name, value in result.state_dict.items()}
    )
    controller.eval()
    return controller


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _MetricTable:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], dict[str, int]] = {}

    def update(
        self,
        condition: str,
        age: torch.Tensor,
        prediction: torch.Tensor,
        batch: ActiveAgeBatch,
    ) -> None:
        distinct = distinct_step_mask(
            batch.current_node,
            batch.one_node,
            batch.two_node,
        )
        for selected_age in torch.unique(age).tolist():
            mask = age.eq(int(selected_age))
            distinct_mask = mask & distinct
            key = (condition, int(selected_age))
            row = self.rows.setdefault(
                key,
                {
                    "n": 0,
                    "distinct_n": 0,
                    "current_correct": 0,
                    "one_correct": 0,
                    "two_correct": 0,
                    "distinct_two_correct": 0,
                },
            )
            row["n"] += int(mask.sum())
            row["distinct_n"] += int(distinct_mask.sum())
            row["current_correct"] += int(
                prediction[mask].eq(batch.current_node[mask]).sum()
            )
            row["one_correct"] += int(
                prediction[mask].eq(batch.one_node[mask]).sum()
            )
            row["two_correct"] += int(
                prediction[mask].eq(batch.two_node[mask]).sum()
            )
            row["distinct_two_correct"] += int(
                prediction[distinct_mask]
                .eq(batch.two_node[distinct_mask])
                .sum()
            )

    def finalized(self) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for (condition, age), row in sorted(self.rows.items()):
            n = max(row["n"], 1)
            distinct_n = max(row["distinct_n"], 1)
            finalized.append(
                {
                    "condition": condition,
                    "age": age,
                    "examples": row["n"],
                    "distinct_examples": row["distinct_n"],
                    "current_accuracy": row["current_correct"] / n,
                    "one_accuracy": row["one_correct"] / n,
                    "two_accuracy": row["two_correct"] / n,
                    "distinct_two_accuracy": (
                        row["distinct_two_correct"] / distinct_n
                    ),
                }
            )
        return finalized


def _aggregate_condition_rows(
    per_age_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accumulators: dict[str, dict[str, float]] = {}
    for row in per_age_rows:
        condition = str(row["condition"])
        accumulator = accumulators.setdefault(
            condition,
            {
                "examples": 0.0,
                "distinct_examples": 0.0,
                "current_correct": 0.0,
                "one_correct": 0.0,
                "two_correct": 0.0,
                "distinct_two_correct": 0.0,
            },
        )
        n = float(row["examples"])
        distinct_n = float(row["distinct_examples"])
        accumulator["examples"] += n
        accumulator["distinct_examples"] += distinct_n
        accumulator["current_correct"] += n * float(row["current_accuracy"])
        accumulator["one_correct"] += n * float(row["one_accuracy"])
        accumulator["two_correct"] += n * float(row["two_accuracy"])
        accumulator["distinct_two_correct"] += (
            distinct_n * float(row["distinct_two_accuracy"])
        )
    rows: list[dict[str, Any]] = []
    for condition, accumulator in sorted(accumulators.items()):
        n = max(accumulator["examples"], 1.0)
        distinct_n = max(accumulator["distinct_examples"], 1.0)
        rows.append(
            {
                "condition": condition,
                "examples": int(accumulator["examples"]),
                "distinct_examples": int(accumulator["distinct_examples"]),
                "current_accuracy": accumulator["current_correct"] / n,
                "one_accuracy": accumulator["one_correct"] / n,
                "two_accuracy": accumulator["two_correct"] / n,
                "distinct_two_accuracy": (
                    accumulator["distinct_two_correct"] / distinct_n
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate_strict_conditions(
    *,
    model: nn.Module,
    exec_controller: DenseAffineJ,
    direct_controller: DenseAffineJ,
    dataset: ActiveAgeBatch,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = _MetricTable()
    generator = torch.Generator(device=dataset.source.device)
    generator.manual_seed(shuffle_seed)
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(
            offset,
            min(offset + batch_size, dataset.source.shape[0]),
            device=dataset.source.device,
        )
        batch = _subset_active_batch(dataset, index)
        raw_output = model.apply_loop(
            batch.source,
            loop_index=model.cfg.max_loops,
        )
        raw_prediction = logits_from_raw_state(model, raw_output).argmax(dim=-1)
        identity_prediction = logits_from_raw_state(model, batch.source).argmax(
            dim=-1
        )
        exec_mapped = exec_controller(batch.source)
        exec_pre_prediction = logits_from_raw_state(model, exec_mapped).argmax(
            dim=-1
        )
        exec_output = model.apply_loop(
            exec_mapped.to(dtype=batch.source.dtype),
            loop_index=model.cfg.max_loops,
        )
        exec_prediction = logits_from_raw_state(model, exec_output).argmax(dim=-1)
        shuffle = torch.randperm(
            batch.source.shape[0],
            generator=generator,
            device=batch.source.device,
        )
        shuffled_output = model.apply_loop(
            exec_mapped[shuffle].to(dtype=batch.source.dtype),
            loop_index=model.cfg.max_loops,
        )
        shuffled_prediction = logits_from_raw_state(
            model,
            shuffled_output,
        ).argmax(dim=-1)
        direct_prediction = logits_from_raw_state(
            model,
            direct_controller(batch.source),
        ).argmax(dim=-1)
        for condition, prediction in (
            ("identity_pre", identity_prediction),
            ("raw_F", raw_prediction),
            ("exec_J_pre_executor_off", exec_pre_prediction),
            ("exec_F_after_J", exec_prediction),
            ("exec_shuffled_F_after_J", shuffled_prediction),
            ("direct_J_only", direct_prediction),
        ):
            table.update(condition, batch.age, prediction, batch)
    per_age = table.finalized()
    return per_age, _aggregate_condition_rows(per_age)


@torch.no_grad()
def evaluate_closure(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    controller: DenseAffineJ,
    permutations: list[tuple[int, ...]],
    device: torch.device,
    batch_size: int,
    cycles: int,
) -> list[dict[str, Any]]:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    rows: list[dict[str, Any]] = []
    examples = [
        (permutation, start)
        for permutation in permutations
        for start in range(cfg.node_count)
    ]
    counts = {
        cycle: {"n": 0, "correct": 0, "distinct_n": 0, "distinct_correct": 0}
        for cycle in range(1, cycles + 1)
    }
    for offset in range(0, len(examples), batch_size):
        chunk = examples[offset : offset + batch_size]
        successors = torch.tensor(
            [row[0] for row in chunk],
            dtype=torch.long,
            device=device,
        )
        starts = torch.tensor(
            [row[1] for row in chunk],
            dtype=torch.long,
            device=device,
        )
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            len(chunk),
            device,
            path_positions=1 + 2 * cycles,
            successors=successors,
            start=starts,
        )
        state = cache_states_with_initial(model, tokens, loops=1)[1]
        for cycle in range(1, cycles + 1):
            mapped = controller(state)
            state = model.apply_loop(
                mapped.to(dtype=state.dtype),
                loop_index=cfg.max_loops,
            )
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            current_index = 2 * cycle - 2
            one_index = 2 * cycle - 1
            two_index = 2 * cycle
            current = path_targets[:, current_index]
            one = path_targets[:, one_index]
            two = path_targets[:, two_index]
            distinct = distinct_step_mask(current, one, two)
            counts[cycle]["n"] += len(chunk)
            counts[cycle]["correct"] += int(prediction.eq(two).sum())
            counts[cycle]["distinct_n"] += int(distinct.sum())
            counts[cycle]["distinct_correct"] += int(
                prediction[distinct].eq(two[distinct]).sum()
            )
    for cycle, row in counts.items():
        rows.append(
            {
                "cycle": cycle,
                "examples": row["n"],
                "accuracy": row["correct"] / max(row["n"], 1),
                "distinct_examples": row["distinct_n"],
                "distinct_accuracy": (
                    row["distinct_correct"] / max(row["distinct_n"], 1)
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate_wrong_ages(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    controller: DenseAffineJ,
    permutations: list[tuple[int, ...]],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    table = _MetricTable()
    examples = [
        (permutation, start)
        for permutation in permutations
        for start in range(cfg.node_count)
    ]
    for offset in range(0, len(examples), batch_size):
        chunk = examples[offset : offset + batch_size]
        successors = torch.tensor(
            [row[0] for row in chunk],
            dtype=torch.long,
            device=device,
        )
        starts = torch.tensor(
            [row[1] for row in chunk],
            dtype=torch.long,
            device=device,
        )
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            len(chunk),
            device,
            path_positions=10,
            successors=successors,
            start=starts,
        )
        states = torch.stack(
            cache_states_with_initial(model, tokens, loops=cfg.max_loops)
        )
        targets_with_start = torch.cat((starts[:, None], path_targets), dim=1)
        for age_value in (0, 7, 8):
            age = torch.full(
                (len(chunk),),
                age_value,
                dtype=torch.long,
                device=device,
            )
            batch = ActiveAgeBatch(
                source=states[age_value],
                current_node=targets_with_start[:, age_value],
                one_node=targets_with_start[:, age_value + 1],
                two_node=targets_with_start[:, age_value + 2],
                age=age,
            )
            raw_prediction = logits_from_raw_state(
                model,
                model.apply_loop(batch.source, loop_index=cfg.max_loops),
            ).argmax(dim=-1)
            mapped = controller(batch.source)
            pre_prediction = logits_from_raw_state(model, mapped).argmax(dim=-1)
            post_prediction = logits_from_raw_state(
                model,
                model.apply_loop(
                    mapped.to(dtype=batch.source.dtype),
                    loop_index=cfg.max_loops,
                ),
            ).argmax(dim=-1)
            table.update("wrong_age_raw_F", age, raw_prediction, batch)
            table.update("wrong_age_exec_J_pre", age, pre_prediction, batch)
            table.update("wrong_age_exec_F_after_J", age, post_prediction, batch)
    return table.finalized()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _permutation_digest(permutations: list[tuple[int, ...]]) -> str:
    payload = "\n".join(",".join(map(str, row)) for row in permutations)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _condition_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["condition"]): row for row in rows}


def decision_from_condition_summary(
    condition_summary: list[dict[str, Any]],
) -> dict[str, bool]:
    lookup = _condition_lookup(condition_summary)
    executed = lookup["exec_F_after_J"]
    raw = lookup["raw_F"]
    shuffled = lookup["exec_shuffled_F_after_J"]
    pre = lookup["exec_J_pre_executor_off"]
    direct = lookup["direct_J_only"]
    no_prewrite_pass = bool(
        float(pre["current_accuracy"]) >= 0.80
        and float(pre["distinct_two_accuracy"]) <= 0.20
    )
    shuffle_gap_pass = bool(
        float(executed["distinct_two_accuracy"])
        - float(shuffled["distinct_two_accuracy"])
        >= 0.50
    )
    reprogramming_positive = bool(
        float(executed["distinct_two_accuracy"]) >= 0.80
        and float(executed["distinct_two_accuracy"])
        - float(raw["distinct_two_accuracy"])
        >= 0.50
        and shuffle_gap_pass
        and no_prewrite_pass
    )
    return {
        "reprogramming_positive": reprogramming_positive,
        "direct_linear_positive": bool(
            float(direct["distinct_two_accuracy"]) >= 0.80
        ),
        "no_prewrite_pass": no_prewrite_pass,
        "shuffle_gap_pass": shuffle_gap_pass,
    }


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
) -> None:
    lookup = _condition_lookup(summary["condition_summary"])
    raw = lookup["raw_F"]
    exec_pre = lookup["exec_J_pre_executor_off"]
    executed = lookup["exec_F_after_J"]
    shuffled = lookup["exec_shuffled_F_after_J"]
    direct = lookup["direct_J_only"]
    decision = summary["decision"]
    lines = [
        "# D8L8 seed3：full-rank J 强制单 loop 两跳",
        "",
        "## 固定协议",
        "",
        "- frozen backbone：final-only D8L8 seed3 step 20000；",
        "- 两个共享物理 block，每个 recurrent loop 自然前进一步；",
        "- J：所有 token 共享一张 256×256 affine weight 和 bias；",
        "- 训练 ages：1--6；目标：同一个 frozen loop 输出两步后节点；",
        "- 正式评估：固定 strict-unseen permutation 集合，枚举全部 start。",
        "",
        "## 严格未见图结果",
        "",
        "| condition | current | one-hop | two-hop | distinct two-hop |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in (
        ("raw F", raw),
        ("J before F / executor-off", exec_pre),
        ("F after J", executed),
        ("shuffled J then F", shuffled),
        ("direct J only", direct),
    ):
        lines.append(
            f"| {name} | {row['current_accuracy']:.4f} | "
            f"{row['one_accuracy']:.4f} | {row['two_accuracy']:.4f} | "
            f"{row['distinct_two_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- F∘J 两跳重编程门：{'通过' if decision['reprogramming_positive'] else '未通过'}；",
            f"- direct token-wise linear readout 门：{'通过' if decision['direct_linear_positive'] else '未通过'}；",
            f"- no-prewrite 门：{'通过' if decision['no_prewrite_pass'] else '未通过'}；",
            f"- shuffled control 门：{'通过' if decision['shuffle_gap_pass'] else '未通过'}。",
            "",
            summary["bounded_conclusion_zh"],
            "",
            "训练失败只约束当前 checkpoint、矩阵族和优化协议，不构成不存在任意线性映射的数学证明。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    controller_seed: int,
    train_examples: int,
    val_permutations: int,
    eval_permutations: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    collection_batch_size: int,
    eval_every: int,
    learning_rates: tuple[float, ...],
    preloop_weights: tuple[float, ...],
    weight_decay: float,
    eval_seed: int,
    closure_cycles: int,
) -> dict[str, Any]:
    if 1.0 not in preloop_weights:
        raise ValueError("primary preloop weight 1 must be included")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "initializing",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "out_dir": str(out_dir),
        "controller_seed": controller_seed,
        "device_request": device_name,
        "train_examples": train_examples,
        "val_permutations": val_permutations,
        "eval_permutations": eval_permutations,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rates": list(learning_rates),
        "preloop_weights": list(preloop_weights),
        "eval_seed": eval_seed,
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
    }
    write_summary_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(os.environ.get("TWOHOP_J_CUDA_MEMORY_FRACTION", "0.30"))
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    if int(checkpoint_payload.get("step", -1)) != 20_000:
        raise ValueError("expected seed3 final checkpoint at step 20000")
    model.requires_grad_(False)
    set_seed(controller_seed)

    formal_eval = strict_unseen_permutations(
        cfg.node_count,
        set(),
        count=eval_permutations,
        seed=eval_seed,
    )
    validation = strict_unseen_permutations(
        cfg.node_count,
        set(formal_eval),
        count=val_permutations,
        seed=eval_seed + 1,
    )
    train_dataset, train_seen = build_training_dataset(
        model=model,
        cfg=cfg,
        excluded_permutations=set(formal_eval) | set(validation),
        examples=train_examples,
        seed=controller_seed + 10_003,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    validation_dataset = build_all_start_age_dataset(
        model=model,
        cfg=cfg,
        permutations=validation,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    if train_seen & (set(formal_eval) | set(validation)):
        raise RuntimeError("reserved evaluation permutation leaked into training")
    print(
        json.dumps(
            {
                "event": "dataset_ready",
                "controller_seed": controller_seed,
                "train_examples": train_dataset.source.shape[0],
                "validation_examples": validation_dataset.source.shape[0],
                "unique_train_permutations": len(train_seen),
            }
        ),
        flush=True,
    )

    candidate_specs = [
        CandidateSpec(
            name=f"exec_lr{learning_rate:g}_pre{preloop_weight:g}",
            learning_rate=learning_rate,
            preloop_weight=preloop_weight,
            execute=True,
        )
        for learning_rate in learning_rates
        for preloop_weight in preloop_weights
    ]
    candidate_specs.extend(
        CandidateSpec(
            name=f"direct_lr{learning_rate:g}",
            learning_rate=learning_rate,
            preloop_weight=0.0,
            execute=False,
        )
        for learning_rate in learning_rates
    )
    candidates: list[CandidateResult] = []
    history_rows: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs):
        print(
            json.dumps(
                {
                    "event": "candidate_start",
                    "candidate_index": candidate_index,
                    **asdict(spec),
                }
            ),
            flush=True,
        )
        result = train_candidate(
            model=model,
            cfg=cfg,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            spec=spec,
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            weight_decay=weight_decay,
            controller_seed=controller_seed * 1009 + candidate_index + 17,
        )
        candidates.append(result)
        history_rows.extend(result.history)
    primary_exec = max(
        (
            result
            for result in candidates
            if result.spec.execute and result.spec.preloop_weight == 1.0
        ),
        key=lambda result: result.validation["distinct_post_two_accuracy"],
    )
    primary_direct = max(
        (result for result in candidates if not result.spec.execute),
        key=lambda result: result.validation["distinct_post_two_accuracy"],
    )
    exec_controller = _controller_from_result(
        primary_exec,
        dimension=cfg.d_model,
        device=device,
    )
    direct_controller = _controller_from_result(
        primary_direct,
        dimension=cfg.d_model,
        device=device,
    )
    torch.save(
        {
            "primary_exec_name": primary_exec.spec.name,
            "primary_exec": primary_exec.state_dict,
            "primary_direct_name": primary_direct.spec.name,
            "primary_direct": primary_direct.state_dict,
            "candidates": {
                result.spec.name: {
                    "spec": asdict(result.spec),
                    "state_dict": result.state_dict,
                    "validation": result.validation,
                }
                for result in candidates
            },
        },
        out_dir / "controllers.pt",
    )

    strict_dataset = build_all_start_age_dataset(
        model=model,
        cfg=cfg,
        permutations=formal_eval,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    per_age_rows, condition_summary = evaluate_strict_conditions(
        model=model,
        exec_controller=exec_controller,
        direct_controller=direct_controller,
        dataset=strict_dataset,
        batch_size=eval_batch_size,
        shuffle_seed=eval_seed + controller_seed,
    )
    wrong_age_rows = evaluate_wrong_ages(
        model=model,
        cfg=cfg,
        controller=exec_controller,
        permutations=formal_eval,
        device=device,
        batch_size=eval_batch_size,
    )
    closure_rows = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=exec_controller,
        permutations=formal_eval,
        device=device,
        batch_size=eval_batch_size,
        cycles=closure_cycles,
    )
    decision = decision_from_condition_summary(condition_summary)
    reprogramming_positive = decision["reprogramming_positive"]
    direct_linear_positive = decision["direct_linear_positive"]
    no_prewrite_pass = decision["no_prewrite_pass"]
    shuffle_gap_pass = decision["shuffle_gap_pass"]
    if reprogramming_positive and not direct_linear_positive:
        bounded_conclusion = (
            "在这个固定 checkpoint 上，full-rank J 本身没有把两跳答案线性写出，"
            "但它成功重编程了冻结 nonlinear loop 所消费的接口。"
        )
    elif reprogramming_positive and direct_linear_positive:
        bounded_conclusion = (
            "在这个固定 checkpoint 上，两跳答案既可由 token-wise affine 直接读出，"
            "也可经 F∘J 得到；本实验不能用来排除线性求解捷径。"
        )
    elif direct_linear_positive:
        bounded_conclusion = (
            "F∘J 没通过预注册重编程门，但 direct affine 能读出两跳答案；"
            "失败不能解释为目标在线性层中不可表示。"
        )
    else:
        bounded_conclusion = (
            "在本 checkpoint、token-wise full-rank affine、学习率扫描和数据协议下，"
            "既未得到可靠 F∘J 两跳重编程，也未得到 direct affine 两跳读出。"
        )
    candidate_summary = [
        {
            "name": result.spec.name,
            **asdict(result.spec),
            **{f"validation_{key}": value for key, value in result.validation.items()},
        }
        for result in candidates
    ]
    summary: dict[str, Any] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "loss_placement": "final_only_ce_at_loop8",
        "config": asdict(cfg),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "natural_semantic_step_per_loop": 1,
        "controller": "tokenwise full-rank affine J(z)=zW+b shared over 29 positions",
        "controller_parameters": cfg.d_model * cfg.d_model + cfg.d_model,
        "controller_seed": controller_seed,
        "train_examples": train_examples,
        "unique_train_permutations": len(train_seen),
        "train_permutation_sha256": _permutation_digest(sorted(train_seen)),
        "validation_permutations": val_permutations,
        "validation_permutation_sha256": _permutation_digest(validation),
        "strict_eval_permutations": eval_permutations,
        "strict_eval_permutation_sha256": _permutation_digest(formal_eval),
        "strict_eval_examples": eval_permutations * cfg.node_count * 6,
        "source_ages": list(range(1, 7)),
        "learning_rates": list(learning_rates),
        "preloop_weights": list(preloop_weights),
        "steps_per_candidate": steps,
        "batch_size": batch_size,
        "primary_exec_candidate": primary_exec.spec.name,
        "primary_direct_candidate": primary_direct.spec.name,
        "candidate_summary": candidate_summary,
        "per_age_rows": per_age_rows,
        "condition_summary": condition_summary,
        "wrong_age_rows": wrong_age_rows,
        "closure_rows": closure_rows,
        "decision": decision,
        "bounded_conclusion_zh": bounded_conclusion,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    _write_csv(out_dir / "training_rows.csv", history_rows)
    _write_csv(out_dir / "candidate_summary.csv", candidate_summary)
    _write_csv(out_dir / "per_age_rows.csv", per_age_rows)
    _write_csv(out_dir / "condition_summary.csv", condition_summary)
    _write_csv(out_dir / "wrong_age_rows.csv", wrong_age_rows)
    _write_csv(out_dir / "closure_rows.csv", closure_rows)
    write_summary_atomic(out_dir / "summary.json", summary)
    _write_report(out_dir / "REPORT_CN.md", summary=summary)
    manifest.update(
        {
            "status": "complete",
            "completed_at": summary["completed_at"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "strict_eval_permutation_sha256": summary[
                "strict_eval_permutation_sha256"
            ],
            "summary": str(out_dir / "summary.json"),
        }
    )
    write_summary_atomic(out_dir / "manifest.json", manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a full-rank token-wise J to make frozen D8L8 seed3 "
            "advance two graph steps in one recurrent loop."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=12_288)
    parser.add_argument("--val-permutations", type=int, default=128)
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--collection-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(1e-5, 3e-5, 1e-4),
    )
    parser.add_argument(
        "--preloop-weights",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 10.0),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    parser.add_argument("--closure-cycles", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device_name=args.device,
        controller_seed=args.controller_seed,
        train_examples=args.train_examples,
        val_permutations=args.val_permutations,
        eval_permutations=args.eval_permutations,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        collection_batch_size=args.collection_batch_size,
        eval_every=args.eval_every,
        learning_rates=tuple(args.learning_rates),
        preloop_weights=tuple(args.preloop_weights),
        weight_decay=args.weight_decay,
        eval_seed=args.eval_seed,
        closure_cycles=args.closure_cycles,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
