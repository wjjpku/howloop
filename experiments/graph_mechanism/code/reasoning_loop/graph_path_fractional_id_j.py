from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_twohop_reprogram_j import (
    ActiveAgeBatch,
    DenseAffineJ,
    _permutation_digest,
    _sha256,
    _write_csv,
    build_all_start_age_dataset,
    build_training_dataset,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


Family = Literal["pair", "id"]


@dataclass(frozen=True)
class FamilyCallResult:
    calls: int
    mapped_first: Tensor
    output_first: Tensor
    mapped_second: Tensor | None
    output: Tensor


@dataclass(frozen=True)
class CandidateSpec:
    family: Family
    learning_rate: float

    @property
    def name(self) -> str:
        return f"{self.family}_lr{self.learning_rate:g}"


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    state_dict: dict[str, Tensor]
    validation: dict[str, float]
    history: list[dict[str, Any]]


def moving_edge_mask(current: Tensor, one: Tensor) -> Tensor:
    if current.shape != one.shape or current.ndim != 1:
        raise ValueError("current and one must be matching one-dimensional tensors")
    return current.ne(one)


def closure_moving_mask(
    family: Family,
    current: Tensor,
    natural_one: Tensor,
    target: Tensor,
) -> Tensor:
    if family == "id":
        return moving_edge_mask(current, natural_one)
    if family == "pair":
        return current.ne(target)
    raise ValueError("family must be pair or id")


def controlled_call(
    model: nn.Module,
    controller: nn.Module,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    mapped = controller(state)
    output = model.apply_loop(
        mapped.to(dtype=state.dtype),
        loop_index=model.cfg.max_loops,
    )
    return mapped, output


def apply_family_calls(
    model: nn.Module,
    controller: nn.Module,
    source: Tensor,
    *,
    family: Family,
) -> FamilyCallResult:
    mapped_first, output_first = controlled_call(model, controller, source)
    if family == "id":
        return FamilyCallResult(
            calls=1,
            mapped_first=mapped_first,
            output_first=output_first,
            mapped_second=None,
            output=output_first,
        )
    if family != "pair":
        raise ValueError("family must be pair or id")
    mapped_second, output_second = controlled_call(model, controller, output_first)
    return FamilyCallResult(
        calls=2,
        mapped_first=mapped_first,
        output_first=output_first,
        mapped_second=mapped_second,
        output=output_second,
    )


def select_best_candidate(
    candidates: list[CandidateResult],
    *,
    family: Family,
) -> CandidateResult:
    selected = [candidate for candidate in candidates if candidate.spec.family == family]
    if not selected:
        raise ValueError(f"no candidates for family {family}")
    return max(
        selected,
        key=lambda candidate: candidate.validation["moving_target_accuracy"],
    )


def _condition_lookup(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {str(row["condition"]): row for row in rows}


def pair_decision(
    condition_rows: list[dict[str, float]],
    closure_rows: list[dict[str, float]],
) -> dict[str, bool]:
    rows = _condition_lookup(condition_rows)
    endpoint = rows["pair_after_call2"]["moving_one_accuracy"]
    baselines = (
        rows["raw_F2"]["moving_one_accuracy"],
        rows["pair_shuffle_J1"]["moving_one_accuracy"],
        rows["pair_shuffle_J2"]["moving_one_accuracy"],
    )
    endpoint_positive = endpoint >= 0.80 and all(endpoint - value >= 0.50 for value in baselines)
    ablations_causal = (
        endpoint - rows["pair_first_J_off"]["moving_one_accuracy"] >= 0.30
        and endpoint - rows["pair_second_J_off"]["moving_one_accuracy"] >= 0.30
    )
    closure = {int(row["calls"]): row["moving_target_accuracy"] for row in closure_rows}
    closure_positive = all(closure[calls] >= 0.70 for calls in (4, 6, 8))
    return {
        "endpoint_positive": endpoint_positive,
        "ablations_causal": ablations_causal,
        "closure_positive": closure_positive,
        "half_speed_operator_positive": (
            endpoint_positive and ablations_causal and closure_positive
        ),
    }


def id_decision(
    condition_rows: list[dict[str, float]],
    closure_rows: list[dict[str, float]],
) -> dict[str, bool]:
    rows = _condition_lookup(condition_rows)
    current = rows["id_after_call1"]["moving_current_accuracy"]
    one_shot_positive = (
        current >= 0.90
        and current - rows["raw_F1"]["moving_current_accuracy"] >= 0.60
        and current - rows["id_shuffled_after_F"]["moving_current_accuracy"] >= 0.60
    )
    closure = {int(row["calls"]): row["moving_target_accuracy"] for row in closure_rows}
    reusable_positive = one_shot_positive and closure[8] >= 0.80
    return {
        "one_shot_positive": one_shot_positive,
        "reusable_positive": reusable_positive,
    }


@dataclass(frozen=True)
class FamilyLoss:
    total: Tensor
    target: Tensor
    forward: FamilyCallResult
    logits: Tensor


def family_loss(
    model: nn.Module,
    controller: nn.Module,
    batch: ActiveAgeBatch,
    *,
    family: Family,
) -> FamilyLoss:
    forward = apply_family_calls(model, controller, batch.source, family=family)
    target = batch.one_node if family == "pair" else batch.current_node
    logits = logits_from_raw_state(model, forward.output)
    return FamilyLoss(
        total=F.cross_entropy(logits, target),
        target=target,
        forward=forward,
        logits=logits,
    )


def _subset(dataset: ActiveAgeBatch, index: Tensor) -> ActiveAgeBatch:
    return ActiveAgeBatch(
        source=dataset.source[index],
        current_node=dataset.current_node[index],
        one_node=dataset.one_node[index],
        two_node=dataset.two_node[index],
        age=dataset.age[index],
    )


class _MetricTable:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], dict[str, int]] = {}

    def update(
        self,
        condition: str,
        batch: ActiveAgeBatch,
        prediction: Tensor,
    ) -> None:
        moving = moving_edge_mask(batch.current_node, batch.one_node)
        for age in torch.unique(batch.age).tolist():
            mask = batch.age.eq(int(age))
            moving_mask = mask & moving
            row = self.rows.setdefault(
                (condition, int(age)),
                {
                    "examples": 0,
                    "moving_examples": 0,
                    "current_correct": 0,
                    "one_correct": 0,
                    "two_correct": 0,
                    "moving_current_correct": 0,
                    "moving_one_correct": 0,
                },
            )
            row["examples"] += int(mask.sum())
            row["moving_examples"] += int(moving_mask.sum())
            row["current_correct"] += int(
                prediction[mask].eq(batch.current_node[mask]).sum()
            )
            row["one_correct"] += int(prediction[mask].eq(batch.one_node[mask]).sum())
            row["two_correct"] += int(prediction[mask].eq(batch.two_node[mask]).sum())
            row["moving_current_correct"] += int(
                prediction[moving_mask]
                .eq(batch.current_node[moving_mask])
                .sum()
            )
            row["moving_one_correct"] += int(
                prediction[moving_mask].eq(batch.one_node[moving_mask]).sum()
            )

    def finalized(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (condition, age), row in sorted(self.rows.items()):
            examples = max(row["examples"], 1)
            moving = max(row["moving_examples"], 1)
            rows.append(
                {
                    "condition": condition,
                    "age": age,
                    "examples": row["examples"],
                    "moving_examples": row["moving_examples"],
                    "current_accuracy": row["current_correct"] / examples,
                    "one_accuracy": row["one_correct"] / examples,
                    "two_accuracy": row["two_correct"] / examples,
                    "moving_current_accuracy": (
                        row["moving_current_correct"] / moving
                    ),
                    "moving_one_accuracy": row["moving_one_correct"] / moving,
                }
            )
        return rows


def aggregate_condition_rows(per_age_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "current_accuracy",
        "one_accuracy",
        "two_accuracy",
        "moving_current_accuracy",
        "moving_one_accuracy",
    )
    totals: dict[str, dict[str, float]] = {}
    for row in per_age_rows:
        condition = str(row["condition"])
        accumulator = totals.setdefault(
            condition,
            {"examples": 0.0, "moving_examples": 0.0, **{name: 0.0 for name in metric_names}},
        )
        n = float(row["examples"])
        moving = float(row["moving_examples"])
        accumulator["examples"] += n
        accumulator["moving_examples"] += moving
        for name in ("current_accuracy", "one_accuracy", "two_accuracy"):
            accumulator[name] += n * float(row[name])
        for name in ("moving_current_accuracy", "moving_one_accuracy"):
            accumulator[name] += moving * float(row[name])
    result: list[dict[str, Any]] = []
    for condition, row in sorted(totals.items()):
        n = max(row["examples"], 1.0)
        moving = max(row["moving_examples"], 1.0)
        result.append(
            {
                "condition": condition,
                "examples": int(row["examples"]),
                "moving_examples": int(row["moving_examples"]),
                "current_accuracy": row["current_accuracy"] / n,
                "one_accuracy": row["one_accuracy"] / n,
                "two_accuracy": row["two_accuracy"] / n,
                "moving_current_accuracy": row["moving_current_accuracy"] / moving,
                "moving_one_accuracy": row["moving_one_accuracy"] / moving,
            }
        )
    return result


@torch.no_grad()
def evaluate_primary(
    *,
    model: nn.Module,
    controller: nn.Module,
    dataset: ActiveAgeBatch,
    family: Family,
    batch_size: int,
) -> dict[str, float]:
    total = 0
    moving_total = 0
    current_correct = 0
    one_correct = 0
    moving_target_correct = 0
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(
            offset,
            min(offset + batch_size, dataset.source.shape[0]),
            device=dataset.source.device,
        )
        batch = _subset(dataset, index)
        forward = apply_family_calls(model, controller, batch.source, family=family)
        prediction = logits_from_raw_state(model, forward.output).argmax(dim=-1)
        moving = moving_edge_mask(batch.current_node, batch.one_node)
        target = batch.one_node if family == "pair" else batch.current_node
        total += batch.source.shape[0]
        moving_total += int(moving.sum())
        current_correct += int(prediction.eq(batch.current_node).sum())
        one_correct += int(prediction.eq(batch.one_node).sum())
        moving_target_correct += int(prediction[moving].eq(target[moving]).sum())
    return {
        "examples": float(total),
        "moving_examples": float(moving_total),
        "current_accuracy": current_correct / max(total, 1),
        "one_accuracy": one_correct / max(total, 1),
        "moving_target_accuracy": moving_target_correct / max(moving_total, 1),
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
    controller_seed: int,
) -> CandidateResult:
    if steps < 1 or batch_size < 1 or eval_every < 1:
        raise ValueError("steps, batch_size, and eval_every must be positive")
    controller = DenseAffineJ(cfg.d_model).to(train_dataset.source.device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=spec.learning_rate)
    generator = torch.Generator(device=train_dataset.source.device)
    generator.manual_seed(controller_seed)
    best_score = -math.inf
    best_state: dict[str, Tensor] | None = None
    best_validation: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        index = torch.randint(
            0,
            train_dataset.source.shape[0],
            (batch_size,),
            device=train_dataset.source.device,
            generator=generator,
        )
        batch = _subset(train_dataset, index)
        losses = family_loss(model, controller, batch, family=spec.family)
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone unexpectedly received gradients")
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            validation = evaluate_primary(
                model=model,
                controller=controller,
                dataset=validation_dataset,
                family=spec.family,
                batch_size=eval_batch_size,
            )
            row = {
                "candidate": spec.name,
                "family": spec.family,
                "learning_rate": spec.learning_rate,
                "step": step,
                "loss": float(losses.total.detach()),
                "gradient_norm": float(gradient_norm),
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}), flush=True)
            if validation["moving_target_accuracy"] > best_score:
                best_score = validation["moving_target_accuracy"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in controller.state_dict().items()
                }
                best_validation = dict(validation)
    if best_state is None or best_validation is None:
        raise RuntimeError("candidate training produced no checkpoint")
    return CandidateResult(spec, best_state, best_validation, history)


def controller_from_result(
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


@torch.no_grad()
def evaluate_pair_conditions(
    *,
    model: nn.Module,
    controller: nn.Module,
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
        batch = _subset(dataset, index)
        source = batch.source
        raw_first = model.apply_loop(source, loop_index=model.cfg.max_loops)
        raw_second = model.apply_loop(raw_first, loop_index=model.cfg.max_loops)
        mapped_first, controlled_first = controlled_call(model, controller, source)
        mapped_second, controlled_second = controlled_call(
            model, controller, controlled_first
        )
        first_off_mapped, first_off = controlled_call(model, controller, raw_first)
        second_off = model.apply_loop(
            controlled_first, loop_index=model.cfg.max_loops
        )
        shuffle_first = torch.randperm(
            source.shape[0], generator=generator, device=source.device
        )
        shuffled_first_output = model.apply_loop(
            mapped_first[shuffle_first].to(dtype=source.dtype),
            loop_index=model.cfg.max_loops,
        )
        _, shuffled_first_final = controlled_call(
            model, controller, shuffled_first_output
        )
        shuffle_second = torch.randperm(
            source.shape[0], generator=generator, device=source.device
        )
        shuffled_second_final = model.apply_loop(
            mapped_second[shuffle_second].to(dtype=source.dtype),
            loop_index=model.cfg.max_loops,
        )
        states = (
            ("source", source),
            ("raw_F1", raw_first),
            ("raw_F2", raw_second),
            ("identity_pair", raw_second),
            ("pair_J1_pre", mapped_first),
            ("pair_after_call1", controlled_first),
            ("pair_J2_pre", mapped_second),
            ("pair_after_call2", controlled_second),
            ("pair_first_J_off", first_off),
            ("pair_second_J_off", second_off),
            ("pair_shuffle_J1", shuffled_first_final),
            ("pair_shuffle_J2", shuffled_second_final),
        )
        for condition, state in states:
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            table.update(condition, batch, prediction)
    per_age = table.finalized()
    return per_age, aggregate_condition_rows(per_age)


@torch.no_grad()
def evaluate_id_conditions(
    *,
    model: nn.Module,
    controller: nn.Module,
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
        batch = _subset(dataset, index)
        raw_first = model.apply_loop(
            batch.source, loop_index=model.cfg.max_loops
        )
        mapped, controlled = controlled_call(model, controller, batch.source)
        shuffle = torch.randperm(
            batch.source.shape[0], generator=generator, device=batch.source.device
        )
        shuffled = model.apply_loop(
            mapped[shuffle].to(dtype=batch.source.dtype),
            loop_index=model.cfg.max_loops,
        )
        for condition, state in (
            ("source", batch.source),
            ("raw_F1", raw_first),
            ("id_J_pre", mapped),
            ("id_after_call1", controlled),
            ("id_shuffled_after_F", shuffled),
        ):
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            table.update(condition, batch, prediction)
    per_age = table.finalized()
    return per_age, aggregate_condition_rows(per_age)


@torch.no_grad()
def evaluate_closure(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    controller: nn.Module,
    permutations: list[tuple[int, ...]],
    family: Family,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    requested = (2, 4, 6, 8) if family == "pair" else (1, 2, 4, 8)
    max_target_step = 4 if family == "pair" else 1
    counts = {
        calls: {"examples": 0, "moving_examples": 0, "correct": 0, "moving_correct": 0}
        for calls in requested
    }
    examples = [
        (permutation, start)
        for permutation in permutations
        for start in range(cfg.node_count)
    ]
    for offset in range(0, len(examples), batch_size):
        chunk = examples[offset : offset + batch_size]
        successors = torch.tensor(
            [row[0] for row in chunk], dtype=torch.long, device=device
        )
        starts = torch.tensor(
            [row[1] for row in chunk], dtype=torch.long, device=device
        )
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            len(chunk),
            device,
            path_positions=max_target_step + 1,
            successors=successors,
            start=starts,
        )
        state = cache_states_with_initial(model, tokens, loops=1)[1]
        current = path_targets[:, 0]
        for calls in range(1, max(requested) + 1):
            _, state = controlled_call(model, controller, state)
            if calls not in counts:
                continue
            target = path_targets[:, calls // 2] if family == "pair" else current
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            moving = closure_moving_mask(
                family,
                current,
                path_targets[:, 1],
                target,
            )
            counts[calls]["examples"] += len(chunk)
            counts[calls]["moving_examples"] += int(moving.sum())
            counts[calls]["correct"] += int(prediction.eq(target).sum())
            counts[calls]["moving_correct"] += int(
                prediction[moving].eq(target[moving]).sum()
            )
    return [
        {
            "family": family,
            "calls": calls,
            "target_step": calls // 2 if family == "pair" else 0,
            "examples": values["examples"],
            "moving_examples": values["moving_examples"],
            "target_accuracy": values["correct"] / max(values["examples"], 1),
            "moving_target_accuracy": values["moving_correct"]
            / max(values["moving_examples"], 1),
        }
        for calls, values in sorted(counts.items())
    ]


@torch.no_grad()
def evaluate_id_wrong_ages(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    controller: nn.Module,
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
            [row[0] for row in chunk], dtype=torch.long, device=device
        )
        starts = torch.tensor(
            [row[1] for row in chunk], dtype=torch.long, device=device
        )
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            len(chunk),
            device,
            path_positions=cfg.max_depth + 2,
            successors=successors,
            start=starts,
        )
        states = torch.stack(cache_states_with_initial(model, tokens, loops=cfg.max_loops))
        targets_with_start = torch.cat((starts[:, None], path_targets), dim=1)
        for age_value in (0, 7, 8):
            age = torch.full(
                (len(chunk),), age_value, dtype=torch.long, device=device
            )
            batch = ActiveAgeBatch(
                source=states[age_value],
                current_node=targets_with_start[:, age_value],
                one_node=targets_with_start[:, age_value + 1],
                two_node=targets_with_start[:, age_value + 2],
                age=age,
            )
            mapped, output = controlled_call(model, controller, batch.source)
            table.update(
                "id_wrong_age_pre", batch, logits_from_raw_state(model, mapped).argmax(dim=-1)
            )
            table.update(
                "id_wrong_age_after_F", batch, logits_from_raw_state(model, output).argmax(dim=-1)
            )
    return table.finalized()


def _candidate_payload(result: CandidateResult) -> dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "validation": result.validation,
        "state_dict": result.state_dict,
    }


def _report_lines(summary: dict[str, Any]) -> list[str]:
    pair_rows = {
        row["condition"]: row for row in summary["pair_condition_summary"]
    }
    id_rows = {row["condition"]: row for row in summary["id_condition_summary"]}
    pair = pair_rows["pair_after_call2"]
    identity = pair_rows["identity_pair"]
    first_off = pair_rows["pair_first_J_off"]
    second_off = pair_rows["pair_second_J_off"]
    identity_result = id_rows["id_after_call1"]
    raw = id_rows["raw_F1"]
    lines = [
        "# D8L8 seed3 shared-J half-speed and ID experiment",
        "",
        f"- controller-data seed: {summary['controller_seed']}; checkpoint step: {summary['checkpoint_step']};",
        "- pair loss: CE(readout(J→F→J→F), one-hop); intermediate controlled call unsupervised;",
        "- ID loss: CE(readout(K→F), current);",
        "",
        "## Pair endpoint",
        "",
        f"- pair J→F→J→F moving one-hop: {pair['moving_one_accuracy']:.4f};",
        f"- identity/raw F² moving one-hop: {identity['moving_one_accuracy']:.4f};",
        f"- first/second J-off moving one-hop: {first_off['moving_one_accuracy']:.4f} / {second_off['moving_one_accuracy']:.4f};",
        f"- endpoint gate: {summary['pair_decision']['endpoint_positive']}; reusable-half-speed gate: {summary['pair_decision']['half_speed_operator_positive']};",
        "",
        "## ID endpoint",
        "",
        f"- K→F moving current: {identity_result['moving_current_accuracy']:.4f};",
        f"- raw F moving current: {raw['moving_current_accuracy']:.4f};",
        f"- one-shot / reusable ID gates: {summary['id_decision']['one_shot_positive']} / {summary['id_decision']['reusable_positive']};",
        "",
        "Endpoint behavior and reusable fractional dynamics are reported separately.",
    ]
    return lines


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
    eval_seed: int,
) -> dict[str, Any]:
    if not learning_rates:
        raise ValueError("at least one learning rate is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "initializing",
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "controller_seed": controller_seed,
    }
    write_summary_atomic(out_dir / "manifest.json", manifest)
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(os.environ.get("FRACTIONAL_ID_CUDA_MEMORY_FRACTION", "0.30"))
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    if int(checkpoint_payload.get("step", -1)) != 20_000:
        raise ValueError("expected D8L8 seed3 final checkpoint at step 20000")
    model.requires_grad_(False)
    set_seed(controller_seed)
    formal_eval = strict_unseen_permutations(
        cfg.node_count, set(), count=eval_permutations, seed=eval_seed
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
        seed=controller_seed + 42_007,
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
        raise RuntimeError("reserved permutation leaked into training")
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
        CandidateSpec(family, learning_rate)
        for family in ("pair", "id")
        for learning_rate in learning_rates
    ]
    candidates: list[CandidateResult] = []
    history: list[dict[str, Any]] = []
    for candidate_index, spec in enumerate(candidate_specs):
        print(
            json.dumps(
                {
                    "event": "candidate_start",
                    "candidate_index": candidate_index,
                    "family": spec.family,
                    "learning_rate": spec.learning_rate,
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
            controller_seed=controller_seed * 1009 + candidate_index + 17,
        )
        candidates.append(result)
        history.extend(result.history)
    pair_result = select_best_candidate(candidates, family="pair")
    id_result = select_best_candidate(candidates, family="id")
    pair_controller = controller_from_result(
        pair_result, dimension=cfg.d_model, device=device
    )
    id_controller = controller_from_result(id_result, dimension=cfg.d_model, device=device)
    torch.save(
        {
            "selected_pair": pair_result.spec.name,
            "selected_id": id_result.spec.name,
            "candidates": {
                result.spec.name: _candidate_payload(result) for result in candidates
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
    pair_per_age, pair_summary = evaluate_pair_conditions(
        model=model,
        controller=pair_controller,
        dataset=strict_dataset,
        batch_size=eval_batch_size,
        shuffle_seed=eval_seed + controller_seed,
    )
    id_per_age, id_summary = evaluate_id_conditions(
        model=model,
        controller=id_controller,
        dataset=strict_dataset,
        batch_size=eval_batch_size,
        shuffle_seed=eval_seed + controller_seed,
    )
    pair_closure = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=pair_controller,
        permutations=formal_eval,
        family="pair",
        device=device,
        batch_size=eval_batch_size,
    )
    id_closure = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=id_controller,
        permutations=formal_eval,
        family="id",
        device=device,
        batch_size=eval_batch_size,
    )
    id_wrong_ages = evaluate_id_wrong_ages(
        model=model,
        cfg=cfg,
        controller=id_controller,
        permutations=formal_eval,
        device=device,
        batch_size=eval_batch_size,
    )
    pair_outcome = pair_decision(pair_summary, pair_closure)
    id_outcome = id_decision(id_summary, id_closure)
    summary: dict[str, Any] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "loss_placement": {
            "pair": "final_CE_after_J_F_J_F_to_one_hop_only",
            "id": "final_CE_after_J_F_to_current_only",
        },
        "config": asdict(cfg),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "natural_semantic_step_per_loop": 1,
        "controller": "separate tokenwise full-rank affine maps shared over 29 positions",
        "controller_parameters": cfg.d_model * cfg.d_model + cfg.d_model,
        "controller_seed": controller_seed,
        "train_examples": train_examples,
        "unique_train_permutations": len(train_seen),
        "validation_permutations": val_permutations,
        "strict_eval_permutations": eval_permutations,
        "strict_eval_examples": strict_dataset.source.shape[0],
        "strict_eval_permutation_sha256": _permutation_digest(formal_eval),
        "source_ages": list(range(1, 7)),
        "learning_rates": list(learning_rates),
        "steps_per_candidate": steps,
        "batch_size": batch_size,
        "selected_pair_candidate": pair_result.spec.name,
        "selected_id_candidate": id_result.spec.name,
        "candidate_summary": [
            {
                "name": result.spec.name,
                "family": result.spec.family,
                "learning_rate": result.spec.learning_rate,
                **result.validation,
            }
            for result in candidates
        ],
        "pair_per_age_rows": pair_per_age,
        "pair_condition_summary": pair_summary,
        "id_per_age_rows": id_per_age,
        "id_condition_summary": id_summary,
        "pair_closure_rows": pair_closure,
        "id_closure_rows": id_closure,
        "id_wrong_age_rows": id_wrong_ages,
        "pair_decision": pair_outcome,
        "id_decision": id_outcome,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    _write_csv(out_dir / "candidate_rows.csv", summary["candidate_summary"])
    _write_csv(out_dir / "training_rows.csv", history)
    _write_csv(out_dir / "pair_per_age_rows.csv", pair_per_age)
    _write_csv(out_dir / "pair_condition_summary.csv", pair_summary)
    _write_csv(out_dir / "id_per_age_rows.csv", id_per_age)
    _write_csv(out_dir / "id_condition_summary.csv", id_summary)
    _write_csv(out_dir / "pair_closure_rows.csv", pair_closure)
    _write_csv(out_dir / "id_closure_rows.csv", id_closure)
    _write_csv(out_dir / "id_wrong_age_rows.csv", id_wrong_ages)
    write_summary_atomic(out_dir / "summary.json", summary)
    (out_dir / "REPORT_CN.md").write_text(
        "\n".join(_report_lines(summary)) + "\n", encoding="utf-8"
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "summary": str(out_dir / "summary.json"),
        }
    )
    write_summary_atomic(out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "complete",
                "selected_pair": pair_result.spec.name,
                "selected_id": id_result.spec.name,
                "pair_decision": pair_outcome,
                "id_decision": id_outcome,
            }
        ),
        flush=True,
    )
    return summary


def recheck_closures(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    eval_permutations: int,
    eval_seed: int,
    batch_size: int,
) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("only complete runs can receive a closure recheck")
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("FRACTIONAL_ID_CUDA_MEMORY_FRACTION", "0.30")),
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, _ = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    model.requires_grad_(False)
    formal_eval = strict_unseen_permutations(
        cfg.node_count, set(), count=eval_permutations, seed=eval_seed
    )
    if _permutation_digest(formal_eval) != summary["strict_eval_permutation_sha256"]:
        raise ValueError("formal permutation hash mismatch during closure recheck")
    payload = torch.load(out_dir / "controllers.pt", map_location=device, weights_only=False)
    pair_state = payload["candidates"][summary["selected_pair_candidate"]]["state_dict"]
    id_state = payload["candidates"][summary["selected_id_candidate"]]["state_dict"]
    pair_controller = DenseAffineJ(cfg.d_model).to(device)
    id_controller = DenseAffineJ(cfg.d_model).to(device)
    pair_controller.load_state_dict({key: value.to(device) for key, value in pair_state.items()})
    id_controller.load_state_dict({key: value.to(device) for key, value in id_state.items()})
    pair_controller.eval()
    id_controller.eval()
    summary["pair_closure_rows"] = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=pair_controller,
        permutations=formal_eval,
        family="pair",
        device=device,
        batch_size=batch_size,
    )
    summary["id_closure_rows"] = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=id_controller,
        permutations=formal_eval,
        family="id",
        device=device,
        batch_size=batch_size,
    )
    summary["pair_decision"] = pair_decision(
        summary["pair_condition_summary"], summary["pair_closure_rows"]
    )
    summary["id_decision"] = id_decision(
        summary["id_condition_summary"], summary["id_closure_rows"]
    )
    summary["closure_rechecked_at"] = datetime.now(timezone.utc).isoformat()
    summary["closure_recheck_peak_cuda_memory_gib"] = (
        float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0
    )
    _write_csv(out_dir / "pair_closure_rows.csv", summary["pair_closure_rows"])
    _write_csv(out_dir / "id_closure_rows.csv", summary["id_closure_rows"])
    write_summary_atomic(summary_path, summary)
    (out_dir / "REPORT_CN.md").write_text(
        "\n".join(_report_lines(summary)) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "closure_rechecked",
                "pair_decision": summary["pair_decision"],
                "id_decision": summary["id_decision"],
            }
        ),
        flush=True,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(1e-5, 3e-5, 1e-4))
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    parser.add_argument("--recheck-closures", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.recheck_closures:
        recheck_closures(
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            device_name=args.device,
            eval_permutations=args.eval_permutations,
            eval_seed=args.eval_seed,
            batch_size=args.eval_batch_size,
        )
        return
    run_experiment(
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
        eval_seed=args.eval_seed,
    )


if __name__ == "__main__":
    main()
