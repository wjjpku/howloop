from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_twohop_reprogram_j import (
    DenseAffineJ,
    _permutation_digest,
    _sha256,
    _write_csv,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


@dataclass(frozen=True)
class RateSpec:
    name: str
    display: str
    physical_stride: int
    semantic_stride: int


RATE_SPECS = {
    "zero": RateSpec("zero", "0", physical_stride=1, semantic_stride=0),
    "half": RateSpec("half", "1/2", physical_stride=2, semantic_stride=1),
    "one": RateSpec("one", "1", physical_stride=1, semantic_stride=1),
    "two": RateSpec("two", "2", physical_stride=1, semantic_stride=2),
}


@dataclass(frozen=True)
class HalfStepMultiBatch:
    """Natural h_t states, current node y_t, and y_(t+1)...y_(t+Q)."""

    source: Tensor
    current_node: Tensor
    targets: Tensor
    age: Tensor


@dataclass(frozen=True)
class ControlledUnroll:
    mapped: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]


@dataclass(frozen=True)
class HalfStepMultiLoss:
    total: Tensor
    losses: tuple[Tensor, ...]
    targets: tuple[Tensor, ...]
    supervised_calls: tuple[int, ...]
    unroll: ControlledUnroll


@dataclass(frozen=True)
class CandidateSpec:
    learning_rate: float

    @property
    def name(self) -> str:
        return f"rate_lr{self.learning_rate:g}"


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    state_dict: dict[str, Tensor]
    validation: dict[str, float]
    history: list[dict[str, Any]]


def rate_schedule(rate_name: str, *, logical_steps: int) -> tuple[tuple[int, int], ...]:
    if rate_name not in RATE_SPECS or logical_steps < 1:
        raise ValueError("rate_name and logical_steps must be valid")
    rate = RATE_SPECS[rate_name]
    return tuple(
        (rate.physical_stride * q, rate.semantic_stride * q)
        for q in range(1, logical_steps + 1)
    )


def controlled_call(model: nn.Module, controller: nn.Module, state: Tensor) -> tuple[Tensor, Tensor]:
    mapped = controller(state)
    output = model.apply_loop(mapped.to(dtype=state.dtype), loop_index=model.cfg.max_loops)
    return mapped, output


def unroll_controlled(
    model: nn.Module,
    controller: nn.Module,
    source: Tensor,
    *,
    calls: int,
) -> ControlledUnroll:
    if calls < 1:
        raise ValueError("calls must be positive")
    mapped: list[Tensor] = []
    outputs: list[Tensor] = []
    state = source
    for _ in range(calls):
        mapped_state, state = controlled_call(model, controller, state)
        mapped.append(mapped_state)
        outputs.append(state)
    return ControlledUnroll(tuple(mapped), tuple(outputs))


def halfstep_multi_loss(
    model: nn.Module,
    controller: nn.Module,
    batch: HalfStepMultiBatch,
    *,
    logical_steps: int,
    physical_stride: int = 2,
) -> HalfStepMultiLoss:
    if logical_steps < 1 or physical_stride < 1 or batch.targets.ndim != 2:
        raise ValueError("logical_steps, physical_stride, and target matrix are required")
    if batch.targets.shape[1] < logical_steps or batch.source.shape[0] != batch.targets.shape[0]:
        raise ValueError("batch does not contain all requested targets")
    calls = tuple(physical_stride * step for step in range(1, logical_steps + 1))
    unroll = unroll_controlled(model, controller, batch.source, calls=calls[-1])
    targets = tuple(batch.targets[:, step - 1] for step in range(1, logical_steps + 1))
    losses = tuple(
        F.cross_entropy(logits_from_raw_state(model, unroll.outputs[call - 1]), target)
        for call, target in zip(calls, targets)
    )
    return HalfStepMultiLoss(torch.stack(losses).mean(), losses, targets, calls, unroll)


def _subset(dataset: HalfStepMultiBatch, index: Tensor) -> HalfStepMultiBatch:
    return HalfStepMultiBatch(
        source=dataset.source[index],
        current_node=dataset.current_node[index],
        targets=dataset.targets[index],
        age=dataset.age[index],
    )


def _cat(parts: list[HalfStepMultiBatch]) -> HalfStepMultiBatch:
    if not parts:
        raise ValueError("at least one dataset part is required")
    return HalfStepMultiBatch(
        source=torch.cat([part.source for part in parts]),
        current_node=torch.cat([part.current_node for part in parts]),
        targets=torch.cat([part.targets for part in parts]),
        age=torch.cat([part.age for part in parts]),
    )


@torch.no_grad()
def _dataset_from_examples(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    starts: list[int],
    ages: list[int],
    logical_steps: int,
    semantic_stride: int,
    device: torch.device,
    collection_batch_size: int,
) -> HalfStepMultiBatch:
    if not (len(permutations) == len(starts) == len(ages)):
        raise ValueError("permutations, starts, and ages must match")
    if logical_steps < 1 or semantic_stride < 0 or collection_batch_size < 1:
        raise ValueError("logical_steps, semantic_stride, and collection_batch_size must be valid")
    if not ages or min(ages) < 1 or max(ages) > cfg.max_loops - 2:
        raise ValueError("ages must lie in the natural h1..h6 source band")
    parts: list[HalfStepMultiBatch] = []
    path_positions = max(ages) + max(logical_steps * semantic_stride, 1)
    for offset in range(0, len(permutations), collection_batch_size):
        stop = min(offset + collection_batch_size, len(permutations))
        successors = torch.tensor(permutations[offset:stop], dtype=torch.long, device=device)
        start = torch.tensor(starts[offset:stop], dtype=torch.long, device=device)
        age = torch.tensor(ages[offset:stop], dtype=torch.long, device=device)
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            stop - offset,
            device,
            path_positions=path_positions,
            successors=successors,
            start=start,
        )
        states = torch.stack(cache_states_with_initial(model, tokens, loops=cfg.max_loops))
        index = torch.arange(stop - offset, device=device)
        target_offsets = semantic_stride * torch.arange(1, logical_steps + 1, device=device)[None, :]
        parts.append(
            HalfStepMultiBatch(
                source=states[age, index],
                current_node=path_targets[index, age - 1],
                targets=path_targets[index[:, None], age[:, None] - 1 + target_offsets],
                age=age,
            )
        )
    return _cat(parts)


def build_training_dataset(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    excluded_permutations: set[tuple[int, ...]],
    examples: int,
    logical_steps: int,
    semantic_stride: int,
    seed: int,
    device: torch.device,
    collection_batch_size: int,
) -> tuple[HalfStepMultiBatch, set[tuple[int, ...]]]:
    permutations = strict_unseen_permutations(
        cfg.node_count,
        excluded_permutations,
        count=examples,
        seed=seed,
    )
    generator = random.Random(seed + 1)
    starts = [generator.randrange(cfg.node_count) for _ in permutations]
    ages = [generator.randrange(1, 7) for _ in permutations]
    return (
        _dataset_from_examples(
            model=model,
            cfg=cfg,
            permutations=permutations,
            starts=starts,
            ages=ages,
            logical_steps=logical_steps,
            semantic_stride=semantic_stride,
            device=device,
            collection_batch_size=collection_batch_size,
        ),
        set(permutations),
    )


def build_all_start_age_dataset(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    logical_steps: int,
    semantic_stride: int,
    device: torch.device,
    collection_batch_size: int,
) -> HalfStepMultiBatch:
    examples = [
        (permutation, start, age)
        for permutation in permutations
        for start in range(cfg.node_count)
        for age in range(1, 7)
    ]
    return _dataset_from_examples(
        model=model,
        cfg=cfg,
        permutations=[row[0] for row in examples],
        starts=[row[1] for row in examples],
        ages=[row[2] for row in examples],
        logical_steps=logical_steps,
        semantic_stride=semantic_stride,
        device=device,
        collection_batch_size=collection_batch_size,
    )


class _MetricTable:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, int, int], dict[str, int]] = {}

    def update(
        self,
        *,
        condition: str,
        logical_step: int,
        batch: HalfStepMultiBatch,
        prediction: Tensor,
    ) -> None:
        target = batch.targets[:, logical_step - 1]
        moving = batch.current_node.ne(target)
        for age in torch.unique(batch.age).tolist():
            mask = batch.age.eq(int(age))
            moving_mask = mask & moving
            row = self.rows.setdefault(
                (condition, logical_step, int(age)),
                {"examples": 0, "moving_examples": 0, "current": 0, "target": 0, "moving_current": 0, "moving_target": 0},
            )
            row["examples"] += int(mask.sum())
            row["moving_examples"] += int(moving_mask.sum())
            row["current"] += int(prediction[mask].eq(batch.current_node[mask]).sum())
            row["target"] += int(prediction[mask].eq(target[mask]).sum())
            row["moving_current"] += int(prediction[moving_mask].eq(batch.current_node[moving_mask]).sum())
            row["moving_target"] += int(prediction[moving_mask].eq(target[moving_mask]).sum())

    def finalized(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (condition, logical_step, age), values in sorted(self.rows.items()):
            examples = max(values["examples"], 1)
            moving = max(values["moving_examples"], 1)
            rows.append(
                {
                    "condition": condition,
                    "logical_step": logical_step,
                    "age": age,
                    "examples": values["examples"],
                    "moving_examples": values["moving_examples"],
                    "current_accuracy": values["current"] / examples,
                    "target_accuracy": values["target"] / examples,
                    "moving_current_accuracy": values["moving_current"] / moving,
                    "moving_target_accuracy": values["moving_target"] / moving,
                }
            )
        return rows


def aggregate_rows(per_age_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int], dict[str, float]] = {}
    for row in per_age_rows:
        key = (str(row["condition"]), int(row["logical_step"]))
        values = totals.setdefault(
            key,
            {"examples": 0.0, "moving_examples": 0.0, "current": 0.0, "target": 0.0, "moving_current": 0.0, "moving_target": 0.0},
        )
        examples = float(row["examples"])
        moving = float(row["moving_examples"])
        values["examples"] += examples
        values["moving_examples"] += moving
        values["current"] += examples * float(row["current_accuracy"])
        values["target"] += examples * float(row["target_accuracy"])
        values["moving_current"] += moving * float(row["moving_current_accuracy"])
        values["moving_target"] += moving * float(row["moving_target_accuracy"])
    result: list[dict[str, Any]] = []
    for (condition, logical_step), values in sorted(totals.items()):
        examples = max(values["examples"], 1.0)
        moving = max(values["moving_examples"], 1.0)
        result.append(
            {
                "condition": condition,
                "logical_step": logical_step,
                "examples": int(values["examples"]),
                "moving_examples": int(values["moving_examples"]),
                "current_accuracy": values["current"] / examples,
                "target_accuracy": values["target"] / examples,
                "moving_current_accuracy": values["moving_current"] / moving,
                "moving_target_accuracy": values["moving_target"] / moving,
            }
        )
    return result


@torch.no_grad()
def evaluate_lattice(
    *,
    model: nn.Module,
    controller: nn.Module,
    dataset: HalfStepMultiBatch,
    rate: int,
    logical_steps: int,
    batch_size: int,
    use_current_metric: bool,
) -> dict[str, float]:
    table = _MetricTable()
    max_calls = rate * logical_steps
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        unroll = unroll_controlled(model, controller, batch.source, calls=max_calls)
        for logical_step in range(1, logical_steps + 1):
            prediction = logits_from_raw_state(model, unroll.outputs[rate * logical_step - 1]).argmax(dim=-1)
            table.update(condition="rate_FJ", logical_step=logical_step, batch=batch, prediction=prediction)
    rows = aggregate_rows(table.finalized())
    metric = "current_accuracy" if use_current_metric else "moving_target_accuracy"
    result = {f"q{row['logical_step']}_selection_accuracy": row[metric] for row in rows}
    result["mean_selection_accuracy"] = sum(result.values()) / logical_steps
    return result


def train_candidate(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    train_dataset: HalfStepMultiBatch,
    validation_dataset: HalfStepMultiBatch,
    spec: CandidateSpec,
    rate: int,
    logical_steps: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    controller_seed: int,
    use_current_metric: bool,
) -> CandidateResult:
    controller = DenseAffineJ(cfg.d_model).to(train_dataset.source.device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=spec.learning_rate)
    generator = torch.Generator(device=train_dataset.source.device).manual_seed(controller_seed)
    best_score = -math.inf
    best_state: dict[str, Tensor] | None = None
    best_validation: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        index = torch.randint(train_dataset.source.shape[0], (batch_size,), device=train_dataset.source.device, generator=generator)
        batch = _subset(train_dataset, index)
        loss = halfstep_multi_loss(
            model,
            controller,
            batch,
            logical_steps=logical_steps,
            physical_stride=rate,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone unexpectedly received gradients")
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            validation = evaluate_lattice(
                model=model,
                controller=controller,
                dataset=validation_dataset,
                rate=rate,
                logical_steps=logical_steps,
                batch_size=eval_batch_size,
                use_current_metric=use_current_metric,
            )
            row = {
                "candidate": spec.name,
                "learning_rate": spec.learning_rate,
                "step": step,
                "loss": float(loss.total.detach()),
                "gradient_norm": float(gradient_norm),
                **{f"loss_q{q + 1}": float(value.detach()) for q, value in enumerate(loss.losses)},
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}), flush=True)
            if validation["mean_selection_accuracy"] > best_score:
                best_score = validation["mean_selection_accuracy"]
                best_state = {name: value.detach().cpu().clone() for name, value in controller.state_dict().items()}
                best_validation = dict(validation)
    if best_state is None or best_validation is None:
        raise RuntimeError("candidate training produced no checkpoint")
    return CandidateResult(spec, best_state, best_validation, history)


def select_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda candidate: candidate.validation["mean_selection_accuracy"])


def controller_from_result(result: CandidateResult, *, dimension: int, device: torch.device) -> DenseAffineJ:
    controller = DenseAffineJ(dimension).to(device)
    controller.load_state_dict({name: value.to(device) for name, value in result.state_dict.items()})
    controller.eval()
    return controller


def matched_random_state(state: Tensor, generator: torch.Generator) -> Tensor:
    reference = state.float()
    mean = reference.mean()
    variance = (reference - mean).square().mean()
    noise = torch.randn(reference.shape, device=reference.device, dtype=reference.dtype, generator=generator)
    noise = noise - noise.mean()
    return noise * torch.sqrt(variance / noise.square().mean().clamp_min(1e-12)) + mean


@torch.no_grad()
def evaluate_formal(
    *,
    model: nn.Module,
    controller: nn.Module,
    dataset: HalfStepMultiBatch,
    rate: int,
    logical_steps: int,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = _MetricTable()
    pre_values = {"examples": 0, "moving_examples": 0, "current": 0, "boundary": 0}
    generator = torch.Generator(device=dataset.source.device).manual_seed(shuffle_seed)
    max_calls = rate * logical_steps
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        trained = unroll_controlled(model, controller, batch.source, calls=max_calls)
        raw_state = batch.source
        first_mapped = trained.mapped[0]
        permutation = torch.randperm(batch.source.shape[0], generator=generator, device=batch.source.device)
        shuffled_state = model.apply_loop(first_mapped[permutation].to(dtype=batch.source.dtype), loop_index=model.cfg.max_loops)
        random_state = model.apply_loop(matched_random_state(first_mapped, generator).to(dtype=batch.source.dtype), loop_index=model.cfg.max_loops)
        pre_prediction = logits_from_raw_state(model, first_mapped).argmax(dim=-1)
        boundary_target = batch.targets[:, 0]
        moving = batch.current_node.ne(boundary_target)
        pre_values["examples"] += int(batch.source.shape[0])
        pre_values["moving_examples"] += int(moving.sum())
        pre_values["current"] += int(pre_prediction.eq(batch.current_node).sum())
        pre_values["boundary"] += int(pre_prediction[moving].eq(boundary_target[moving]).sum())
        for call in range(1, max_calls + 1):
            raw_state = model.apply_loop(raw_state, loop_index=model.cfg.max_loops)
            if call > 1:
                _, shuffled_state = controlled_call(model, controller, shuffled_state)
                _, random_state = controlled_call(model, controller, random_state)
            if call % rate:
                continue
            logical_step = call // rate
            for condition, state in (
                ("raw_F", raw_state),
                ("rate_FJ", trained.outputs[call - 1]),
                ("shuffle_first_J", shuffled_state),
                ("random_first_J", random_state),
            ):
                table.update(
                    condition=condition,
                    logical_step=logical_step,
                    batch=batch,
                    prediction=logits_from_raw_state(model, state).argmax(dim=-1),
                )
    pre_rows = [
        {
            "condition": "J_pre_executor_off",
            "examples": pre_values["examples"],
            "moving_examples": pre_values["moving_examples"],
            "current_accuracy": pre_values["current"] / max(pre_values["examples"], 1),
            "moving_boundary_accuracy": pre_values["boundary"] / max(pre_values["moving_examples"], 1),
        }
    ]
    return table.finalized(), pre_rows


def _rows_by_condition(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["condition"]), int(row["logical_step"])): row for row in aggregate_rows(rows)}


def decision(
    *,
    rows: list[dict[str, Any]],
    pre_rows: list[dict[str, Any]],
    rate: RateSpec,
    trained_logical_steps: int,
) -> dict[str, bool]:
    lookup = _rows_by_condition(rows)
    trained = range(1, trained_logical_steps + 1)
    target_metric = "target_accuracy" if rate.name == "zero" else "moving_target_accuracy"
    high = all(lookup[("rate_FJ", q)][target_metric] >= 0.90 for q in trained)
    raw_margin = 0.05 if rate.name == "one" else 0.50
    beats_raw = all(
        lookup[("rate_FJ", q)][target_metric] - lookup[("raw_F", q)][target_metric] >= raw_margin
        for q in trained
    )
    controls = all(
        lookup[("rate_FJ", q)][target_metric] - lookup[("shuffle_first_J", q)][target_metric] >= 0.50
        and lookup[("rate_FJ", q)][target_metric] - lookup[("random_first_J", q)][target_metric] >= 0.50
        for q in trained
    )
    pre = pre_rows[0]
    no_prewrite = (
        True
        if rate.name == "zero"
        else pre["current_accuracy"] >= 0.80 and pre["moving_boundary_accuracy"] <= 0.20
    )
    return {
        "all_trained_lattice_high_accuracy": high,
        "beats_rate_matched_raw_F": beats_raw,
        "shuffle_random_controls": controls,
        "no_prewrite": no_prewrite,
        "finite_rate_program_positive": high and beats_raw and controls and no_prewrite,
        "q5_extrapolation_high_accuracy": lookup[("rate_FJ", 5)][target_metric] >= 0.70,
        "q8_extrapolation_high_accuracy": lookup[("rate_FJ", 8)][target_metric] >= 0.70,
    }


def _report_lines(summary: dict[str, Any]) -> list[str]:
    rows = _rows_by_condition(summary["per_age_rows"])
    pre = summary["pre_rows"][0]
    rate = summary["rate"]
    metric = "target_accuracy" if summary["rate_name"] == "zero" else "moving_target_accuracy"
    backbone_label = summary.get("backbone_label", "D8L8 backbone")
    lines = [
        f"# {backbone_label}：FJ logical rate {rate}",
        "",
        f"- controller-data seed: {summary['controller_seed']}; source ages: h1..h6; numerical rank: {summary['controller_numerical_matrix_rank']}.",
        f"- G=F∘J; lattice (physical call, semantic offset): {summary['loss_lattice']}; no teacher forcing; only half-rate has unsupervised odd calls.",
        "",
        "| lattice index q | status | raw F | G | shuffled first J | random first J |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for q in range(1, 9):
        status = "trained" if q <= 4 else "extrapolation"
        lines.append(
            f"| {q} | {status} | {rows[('raw_F', q)][metric]:.4f} | "
            f"{rows[('rate_FJ', q)][metric]:.4f} | "
            f"{rows[('shuffle_first_J', q)][metric]:.4f} | "
            f"{rows[('random_first_J', q)][metric]:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- J(h_t) executor-off current / boundary-moving: {pre['current_accuracy']:.4f} / {pre['moving_boundary_accuracy']:.4f} (zero rate: no-prewrite gate intentionally N/A).",
            f"- finite rate-program gate: {summary['decision']['finite_rate_program_positive']}; q5 / q8 extrapolation: {summary['decision']['q5_extrapolation_high_accuracy']} / {summary['decision']['q8_extrapolation_high_accuracy']}.",
            "- Passing q1..q4 supports a finite time-rescaled interface program. Passing q>4 is required before calling it a reusable transition; neither result alone identifies F's unique original circuit.",
        ]
    )
    return lines


def run_experiment(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    controller_seed: int,
    rate_name: str,
    train_examples: int,
    val_permutations: int,
    eval_permutations: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    collection_batch_size: int,
    eval_every: int,
    learning_rates: tuple[float, ...],
    eval_seed: int,
) -> dict[str, Any]:
    if rate_name not in RATE_SPECS or not learning_rates:
        raise ValueError("rate_name must be zero, half, one, or two and learning_rates must be nonempty")
    rate = RATE_SPECS[rate_name]
    trained_logical_steps, formal_logical_steps = 4, 8
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "initializing", "pid": os.getpid(), "hostname": os.uname().nodename, "created_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint), "controller_seed": controller_seed, "rate": rate.name}
    write_summary_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("FJ_RATE_CUDA_MEMORY_FRACTION", "0.30")), device=torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    if int(checkpoint_payload.get("step", -1)) != 20_000:
        raise ValueError("expected a D8L8 final checkpoint at step 20000")
    model.requires_grad_(False)
    set_seed(controller_seed)
    formal = strict_unseen_permutations(cfg.node_count, set(), count=eval_permutations, seed=eval_seed)
    validation = strict_unseen_permutations(cfg.node_count, set(formal), count=val_permutations, seed=eval_seed + 1)
    train_dataset, train_seen = build_training_dataset(
        model=model, cfg=cfg, excluded_permutations=set(formal) | set(validation), examples=train_examples,
        logical_steps=trained_logical_steps, semantic_stride=rate.semantic_stride, seed=controller_seed + 47_007, device=device, collection_batch_size=collection_batch_size,
    )
    validation_dataset = build_all_start_age_dataset(
        model=model, cfg=cfg, permutations=validation, logical_steps=trained_logical_steps, semantic_stride=rate.semantic_stride, device=device, collection_batch_size=collection_batch_size,
    )
    if train_seen & (set(formal) | set(validation)):
        raise RuntimeError("reserved permutation leaked into training")
    print(json.dumps({"event": "dataset_ready", "rate": rate.name, "controller_seed": controller_seed, "train_examples": train_dataset.source.shape[0], "validation_examples": validation_dataset.source.shape[0], "unique_train_permutations": len(train_seen)}), flush=True)
    candidates: list[CandidateResult] = []
    history: list[dict[str, Any]] = []
    for candidate_index, learning_rate in enumerate(learning_rates):
        spec = CandidateSpec(learning_rate)
        print(json.dumps({"event": "candidate_start", "rate": rate.name, "candidate": spec.name}), flush=True)
        candidate = train_candidate(
            model=model, cfg=cfg, train_dataset=train_dataset, validation_dataset=validation_dataset, spec=spec, rate=rate.physical_stride,
            logical_steps=trained_logical_steps, steps=steps, batch_size=batch_size, eval_batch_size=eval_batch_size,
            eval_every=eval_every, controller_seed=controller_seed * 1009 + candidate_index + 79,
            use_current_metric=rate.name == "zero",
        )
        candidates.append(candidate)
        history.extend(candidate.history)
    selected = select_best_candidate(candidates)
    controller = controller_from_result(selected, dimension=cfg.d_model, device=device)
    torch.save({"selected": selected.spec.name, "candidates": {candidate.spec.name: {"spec": asdict(candidate.spec), "validation": candidate.validation, "state_dict": candidate.state_dict} for candidate in candidates}}, out_dir / "controller.pt")
    formal_dataset = build_all_start_age_dataset(
        model=model, cfg=cfg, permutations=formal, logical_steps=formal_logical_steps, semantic_stride=rate.semantic_stride, device=device, collection_batch_size=collection_batch_size,
    )
    per_age_rows, pre_rows = evaluate_formal(
        model=model, controller=controller, dataset=formal_dataset, rate=rate.physical_stride, logical_steps=formal_logical_steps,
        batch_size=eval_batch_size, shuffle_seed=eval_seed + controller_seed,
    )
    result_decision = decision(rows=per_age_rows, pre_rows=pre_rows, rate=rate, trained_logical_steps=trained_logical_steps)
    summary: dict[str, Any] = {
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
        "backbone_label": checkpoint.parent.name,
        "checkpoint_sha256": _sha256(checkpoint), "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "source_code_sha256": _sha256(Path(__file__)), "config": asdict(cfg), "rate": rate.display, "rate_name": rate.name,
        "physical_stride": rate.physical_stride, "semantic_stride": rate.semantic_stride,
        "loss_placement": "mean_CE_on_requested_rate_lattice_only",
        "loss_lattice": [list(row) for row in rate_schedule(rate.name, logical_steps=trained_logical_steps)], "runtime_order": "J_then_F_shared_at_every_physical_call_no_teacher_forcing",
        "source_ages": list(range(1, 7)), "trained_logical_steps": [1, 2, 3, 4], "formal_logical_steps": list(range(1, 9)),
        "shared_physical_blocks": cfg.n_layers, "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "controller": "one shared tokenwise dense affine J across all physical calls", "controller_parameters": cfg.d_model * cfg.d_model + cfg.d_model,
        "controller_numerical_matrix_rank": int(torch.linalg.matrix_rank(controller.weight.detach().float()).cpu()), "controller_seed": controller_seed,
        "train_examples": train_examples, "unique_train_permutations": len(train_seen), "validation_permutations": val_permutations,
        "strict_eval_permutations": eval_permutations, "strict_eval_examples": formal_dataset.source.shape[0], "strict_eval_permutation_sha256": _permutation_digest(formal),
        "learning_rates": list(learning_rates), "steps_per_candidate": steps, "batch_size": batch_size, "selected_candidate": selected.spec.name,
        "candidate_summary": [{"name": candidate.spec.name, "learning_rate": candidate.spec.learning_rate, **candidate.validation} for candidate in candidates],
        "per_age_rows": per_age_rows, "condition_summary": aggregate_rows(per_age_rows), "pre_rows": pre_rows, "decision": result_decision,
        "peak_cuda_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else 0.0,
    }
    _write_csv(out_dir / "candidate_rows.csv", summary["candidate_summary"])
    _write_csv(out_dir / "training_rows.csv", history)
    _write_csv(out_dir / "per_age_rows.csv", per_age_rows)
    _write_csv(out_dir / "condition_summary.csv", summary["condition_summary"])
    _write_csv(out_dir / "pre_rows.csv", pre_rows)
    write_summary_atomic(out_dir / "summary.json", summary)
    (out_dir / "REPORT_CN.md").write_text("\n".join(_report_lines(summary)) + "\n", encoding="utf-8")
    manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "summary": str(out_dir / "summary.json")})
    write_summary_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "rate": rate.name, "selected": selected.spec.name, "decision": result_decision}), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--rate", required=True, choices=tuple(RATE_SPECS))
    parser.add_argument("--train-examples", type=int, default=12_288)
    parser.add_argument("--val-permutations", type=int, default=128)
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--collection-batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-5, 3e-5, 1e-4))
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(
        checkpoint=args.checkpoint, out_dir=args.out_dir, device_name=args.device, controller_seed=args.controller_seed,
        rate_name=args.rate, train_examples=args.train_examples, val_permutations=args.val_permutations,
        eval_permutations=args.eval_permutations, steps=args.steps, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, collection_batch_size=args.collection_batch_size,
        eval_every=args.eval_every, learning_rates=tuple(args.learning_rates), eval_seed=args.eval_seed,
    )


if __name__ == "__main__":
    main()
