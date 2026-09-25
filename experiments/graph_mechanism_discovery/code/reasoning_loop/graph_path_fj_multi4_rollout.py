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
class MultiHorizonBatch:
    """Natural h1 states and semantic targets f^1 through f^H from that state."""

    source: Tensor
    current_node: Tensor
    targets: Tensor


@dataclass(frozen=True)
class ControlledUnroll:
    mapped: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    calls: int


@dataclass(frozen=True)
class MultiHorizonLoss:
    total: Tensor
    losses: tuple[Tensor, ...]
    targets: tuple[Tensor, ...]
    logits: tuple[Tensor, ...]
    unroll: ControlledUnroll
    calls: int


@dataclass(frozen=True)
class CandidateSpec:
    learning_rate: float

    @property
    def name(self) -> str:
        return f"multi4_lr{self.learning_rate:g}"


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    state_dict: dict[str, Tensor]
    validation: dict[str, float]
    history: list[dict[str, Any]]


def horizon_target_steps(calls: tuple[int, ...]) -> dict[int, int]:
    if not calls or any(call < 1 for call in calls):
        raise ValueError("controlled call counts must be positive")
    return {call: call for call in calls}


def moving_target_mask(current: Tensor, target: Tensor) -> Tensor:
    if current.ndim != 1 or current.shape != target.shape:
        raise ValueError("current and target must be matching one-dimensional tensors")
    return current.ne(target)


def matched_random_state(state: Tensor, generator: torch.Generator) -> Tensor:
    """Match global mean and second moment while destroying example-specific code."""

    reference = state.float()
    mean = reference.mean()
    variance = (reference - mean).square().mean()
    noise = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    noise = noise - noise.mean()
    noise = noise * torch.sqrt(variance / noise.square().mean().clamp_min(1e-12))
    return noise + mean


def controlled_call(
    model: nn.Module,
    controller: nn.Module,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    mapped = controller(state)
    output = model.apply_loop(mapped.to(dtype=state.dtype), loop_index=model.cfg.max_loops)
    return mapped, output


def unroll_controlled(
    model: nn.Module,
    controller: nn.Module,
    source: Tensor,
    *,
    steps: int,
) -> ControlledUnroll:
    if steps < 1:
        raise ValueError("steps must be positive")
    mapped_states: list[Tensor] = []
    outputs: list[Tensor] = []
    state = source
    for _ in range(steps):
        mapped, state = controlled_call(model, controller, state)
        mapped_states.append(mapped)
        outputs.append(state)
    return ControlledUnroll(tuple(mapped_states), tuple(outputs), calls=steps)


def multi_horizon_loss(
    model: nn.Module,
    controller: nn.Module,
    batch: MultiHorizonBatch,
    *,
    horizons: int,
) -> MultiHorizonLoss:
    if horizons < 1 or batch.targets.ndim != 2 or batch.targets.shape[1] < horizons:
        raise ValueError("batch must contain one target for every requested horizon")
    if batch.source.shape[0] != batch.targets.shape[0]:
        raise ValueError("source and targets must have the same batch dimension")
    unroll = unroll_controlled(model, controller, batch.source, steps=horizons)
    targets = tuple(batch.targets[:, horizon] for horizon in range(horizons))
    logits = tuple(logits_from_raw_state(model, output) for output in unroll.outputs)
    losses = tuple(F.cross_entropy(logit, target) for logit, target in zip(logits, targets))
    return MultiHorizonLoss(
        total=torch.stack(losses).mean(),
        losses=losses,
        targets=targets,
        logits=logits,
        unroll=unroll,
        calls=horizons,
    )


def _subset(dataset: MultiHorizonBatch, index: Tensor) -> MultiHorizonBatch:
    return MultiHorizonBatch(
        source=dataset.source[index],
        current_node=dataset.current_node[index],
        targets=dataset.targets[index],
    )


def _cat(parts: list[MultiHorizonBatch]) -> MultiHorizonBatch:
    if not parts:
        raise ValueError("at least one part is required")
    return MultiHorizonBatch(
        source=torch.cat([part.source for part in parts]),
        current_node=torch.cat([part.current_node for part in parts]),
        targets=torch.cat([part.targets for part in parts]),
    )


@torch.no_grad()
def _dataset_from_examples(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    starts: list[int],
    horizons: int,
    device: torch.device,
    collection_batch_size: int,
) -> MultiHorizonBatch:
    if horizons < 1 or collection_batch_size < 1:
        raise ValueError("horizons and collection_batch_size must be positive")
    if len(permutations) != len(starts):
        raise ValueError("permutations and starts must have the same length")
    parts: list[MultiHorizonBatch] = []
    for offset in range(0, len(permutations), collection_batch_size):
        stop = min(offset + collection_batch_size, len(permutations))
        successors = torch.tensor(permutations[offset:stop], dtype=torch.long, device=device)
        start = torch.tensor(starts[offset:stop], dtype=torch.long, device=device)
        tokens, path_targets, _, _ = fixed_depth_batch(
            cfg,
            stop - offset,
            device,
            path_positions=horizons + 1,
            successors=successors,
            start=start,
        )
        source = cache_states_with_initial(model, tokens, loops=1)[1]
        parts.append(
            MultiHorizonBatch(
                source=source,
                current_node=path_targets[:, 0],
                targets=path_targets[:, 1 : horizons + 1],
            )
        )
    return _cat(parts)


def build_training_dataset(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    excluded_permutations: set[tuple[int, ...]],
    examples: int,
    horizons: int,
    seed: int,
    device: torch.device,
    collection_batch_size: int,
) -> tuple[MultiHorizonBatch, set[tuple[int, ...]]]:
    if examples < 1:
        raise ValueError("examples must be positive")
    permutations = strict_unseen_permutations(
        cfg.node_count,
        excluded_permutations,
        count=examples,
        seed=seed,
    )
    generator = random.Random(seed + 1)
    starts = [generator.randrange(cfg.node_count) for _ in permutations]
    dataset = _dataset_from_examples(
        model=model,
        cfg=cfg,
        permutations=permutations,
        starts=starts,
        horizons=horizons,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    return dataset, set(permutations)


def build_all_start_dataset(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    permutations: list[tuple[int, ...]],
    horizons: int,
    device: torch.device,
    collection_batch_size: int,
) -> MultiHorizonBatch:
    examples = [(permutation, start) for permutation in permutations for start in range(cfg.node_count)]
    return _dataset_from_examples(
        model=model,
        cfg=cfg,
        permutations=[permutation for permutation, _ in examples],
        starts=[start for _, start in examples],
        horizons=horizons,
        device=device,
        collection_batch_size=collection_batch_size,
    )


def _metric_row(
    *,
    condition: str,
    horizon: int,
    prediction: Tensor,
    current: Tensor,
    target: Tensor,
) -> dict[str, Any]:
    moving = moving_target_mask(current, target)
    examples = int(target.shape[0])
    moving_examples = int(moving.sum())
    return {
        "condition": condition,
        "horizon": horizon,
        "examples": examples,
        "moving_examples": moving_examples,
        "current_accuracy": float(prediction.eq(current).float().mean()),
        "target_accuracy": float(prediction.eq(target).float().mean()),
        "moving_current_accuracy": float(prediction[moving].eq(current[moving]).float().mean())
        if moving_examples
        else float("nan"),
        "moving_target_accuracy": float(prediction[moving].eq(target[moving]).float().mean())
        if moving_examples
        else float("nan"),
    }


def _merge_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["horizon"]))
        values = merged.setdefault(
            key,
            {
                "examples": 0.0,
                "moving_examples": 0.0,
                "current_correct": 0.0,
                "target_correct": 0.0,
                "moving_current_correct": 0.0,
                "moving_target_correct": 0.0,
            },
        )
        examples = float(row["examples"])
        moving = float(row["moving_examples"])
        values["examples"] += examples
        values["moving_examples"] += moving
        values["current_correct"] += examples * float(row["current_accuracy"])
        values["target_correct"] += examples * float(row["target_accuracy"])
        if moving:
            values["moving_current_correct"] += moving * float(row["moving_current_accuracy"])
            values["moving_target_correct"] += moving * float(row["moving_target_accuracy"])
    result: list[dict[str, Any]] = []
    for (condition, horizon), values in sorted(merged.items()):
        examples = max(values["examples"], 1.0)
        moving = max(values["moving_examples"], 1.0)
        result.append(
            {
                "condition": condition,
                "horizon": horizon,
                "examples": int(values["examples"]),
                "moving_examples": int(values["moving_examples"]),
                "current_accuracy": values["current_correct"] / examples,
                "target_accuracy": values["target_correct"] / examples,
                "moving_current_accuracy": values["moving_current_correct"] / moving,
                "moving_target_accuracy": values["moving_target_correct"] / moving,
            }
        )
    return result


@torch.no_grad()
def evaluate_multi_horizon(
    *,
    model: nn.Module,
    controller: nn.Module,
    dataset: MultiHorizonBatch,
    horizons: int,
    batch_size: int,
) -> dict[str, float]:
    if horizons > dataset.targets.shape[1]:
        raise ValueError("dataset has too few targets")
    rows: list[dict[str, Any]] = []
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        unroll = unroll_controlled(model, controller, batch.source, steps=horizons)
        for horizon, output in enumerate(unroll.outputs, start=1):
            prediction = logits_from_raw_state(model, output).argmax(dim=-1)
            rows.append(
                _metric_row(
                    condition="FJ_multi4",
                    horizon=horizon,
                    prediction=prediction,
                    current=batch.current_node,
                    target=batch.targets[:, horizon - 1],
                )
            )
    result = {f"h{row['horizon']}_moving_target_accuracy": float(row["moving_target_accuracy"]) for row in _merge_metric_rows(rows)}
    result["mean_moving_target_accuracy"] = sum(result.values()) / horizons
    return result


def train_candidate(
    *,
    model: nn.Module,
    cfg: GraphPathConfig,
    train_dataset: MultiHorizonBatch,
    validation_dataset: MultiHorizonBatch,
    spec: CandidateSpec,
    horizons: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_every: int,
    controller_seed: int,
) -> CandidateResult:
    if min(horizons, steps, batch_size, eval_batch_size, eval_every) < 1:
        raise ValueError("all training sizes must be positive")
    controller = DenseAffineJ(cfg.d_model).to(train_dataset.source.device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=spec.learning_rate)
    generator = torch.Generator(device=train_dataset.source.device).manual_seed(controller_seed)
    best_score = -math.inf
    best_state: dict[str, Tensor] | None = None
    best_validation: dict[str, float] | None = None
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        index = torch.randint(
            train_dataset.source.shape[0],
            (batch_size,),
            device=train_dataset.source.device,
            generator=generator,
        )
        batch = _subset(train_dataset, index)
        losses = multi_horizon_loss(model, controller, batch, horizons=horizons)
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone unexpectedly received gradients")
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            validation = evaluate_multi_horizon(
                model=model,
                controller=controller,
                dataset=validation_dataset,
                horizons=horizons,
                batch_size=eval_batch_size,
            )
            row = {
                "candidate": spec.name,
                "learning_rate": spec.learning_rate,
                "step": step,
                "loss": float(losses.total.detach()),
                "gradient_norm": float(gradient_norm),
                **{f"loss_h{index + 1}": float(value.detach()) for index, value in enumerate(losses.losses)},
                **validation,
            }
            history.append(row)
            print(json.dumps({"event": "training", **row}), flush=True)
            if validation["mean_moving_target_accuracy"] > best_score:
                best_score = validation["mean_moving_target_accuracy"]
                best_state = {name: value.detach().cpu().clone() for name, value in controller.state_dict().items()}
                best_validation = dict(validation)
    if best_state is None or best_validation is None:
        raise RuntimeError("candidate training produced no checkpoint")
    return CandidateResult(spec, best_state, best_validation, history)


def select_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("at least one candidate is required")
    return max(candidates, key=lambda candidate: candidate.validation["mean_moving_target_accuracy"])


def controller_from_result(
    result: CandidateResult, *, dimension: int, device: torch.device
) -> DenseAffineJ:
    controller = DenseAffineJ(dimension).to(device)
    controller.load_state_dict({name: value.to(device) for name, value in result.state_dict.items()})
    controller.eval()
    return controller


@torch.no_grad()
def evaluate_formal(
    *,
    model: nn.Module,
    controller: nn.Module,
    dataset: MultiHorizonBatch,
    horizons: int,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pre_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=dataset.source.device).manual_seed(shuffle_seed)
    for offset in range(0, dataset.source.shape[0], batch_size):
        index = torch.arange(offset, min(offset + batch_size, dataset.source.shape[0]), device=dataset.source.device)
        batch = _subset(dataset, index)
        raw_state = batch.source
        identity_state = batch.source
        trained = unroll_controlled(model, controller, batch.source, steps=horizons)
        first_mapped = trained.mapped[0]
        shuffle = torch.randperm(batch.source.shape[0], generator=generator, device=batch.source.device)
        shuffled_state = model.apply_loop(first_mapped[shuffle].to(dtype=batch.source.dtype), loop_index=model.cfg.max_loops)
        random_state = model.apply_loop(
            matched_random_state(first_mapped, generator).to(dtype=batch.source.dtype),
            loop_index=model.cfg.max_loops,
        )
        pre_prediction = logits_from_raw_state(model, first_mapped).argmax(dim=-1)
        moving_next = moving_target_mask(batch.current_node, batch.targets[:, 0])
        pre_rows.append(
            {
                "condition": "J_pre_executor_off",
                "examples": int(batch.current_node.shape[0]),
                "moving_examples": int(moving_next.sum()),
                "current_accuracy": float(pre_prediction.eq(batch.current_node).float().mean()),
                "moving_next_accuracy": float(pre_prediction[moving_next].eq(batch.targets[:, 0][moving_next]).float().mean()) if int(moving_next.sum()) else float("nan"),
            }
        )
        for horizon in range(1, horizons + 1):
            raw_state = model.apply_loop(raw_state, loop_index=model.cfg.max_loops)
            identity_state = model.apply_loop(identity_state, loop_index=model.cfg.max_loops)
            if horizon > 1:
                _, shuffled_state = controlled_call(model, controller, shuffled_state)
                _, random_state = controlled_call(model, controller, random_state)
            target = batch.targets[:, horizon - 1]
            for condition, state in (
                ("raw_F", raw_state),
                ("identity_J_then_F", identity_state),
                ("FJ_multi4", trained.outputs[horizon - 1]),
                ("shuffle_first_J_then_FJ", shuffled_state),
                ("random_first_J_then_FJ", random_state),
            ):
                prediction = logits_from_raw_state(model, state).argmax(dim=-1)
                rows.append(
                    _metric_row(
                        condition=condition,
                        horizon=horizon,
                        prediction=prediction,
                        current=batch.current_node,
                        target=target,
                    )
                )
    return _merge_metric_rows(rows), _merge_pre_rows(pre_rows)


def _merge_pre_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    examples = sum(int(row["examples"]) for row in rows)
    moving = sum(int(row["moving_examples"]) for row in rows)
    return [
        {
            "condition": "J_pre_executor_off",
            "examples": examples,
            "moving_examples": moving,
            "current_accuracy": sum(float(row["current_accuracy"]) * int(row["examples"]) for row in rows) / max(examples, 1),
            "moving_next_accuracy": sum(float(row["moving_next_accuracy"]) * int(row["moving_examples"]) for row in rows) / max(moving, 1),
        }
    ]


def _row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["condition"]), int(row["horizon"])): row for row in rows}


def multi4_decision(
    horizon_rows: list[dict[str, Any]], pre_rows: list[dict[str, Any]]
) -> dict[str, bool]:
    rows = _row_lookup(horizon_rows)
    supervised = tuple(range(1, 5))
    high_accuracy = all(rows[("FJ_multi4", horizon)]["moving_target_accuracy"] >= 0.90 for horizon in supervised)
    improves_raw = all(
        rows[("FJ_multi4", horizon)]["moving_target_accuracy"]
        - rows[("raw_F", horizon)]["moving_target_accuracy"]
        >= 0.05
        for horizon in supervised
    )
    controls_causal = all(
        rows[("FJ_multi4", horizon)]["moving_target_accuracy"]
        - rows[("shuffle_first_J_then_FJ", horizon)]["moving_target_accuracy"]
        >= 0.50
        and rows[("FJ_multi4", horizon)]["moving_target_accuracy"]
        - rows[("random_first_J_then_FJ", horizon)]["moving_target_accuracy"]
        >= 0.50
        for horizon in supervised
    )
    pre = pre_rows[0]
    no_prewrite = pre["current_accuracy"] >= 0.80 and pre["moving_next_accuracy"] <= 0.20
    supervised_positive = high_accuracy and improves_raw and controls_causal and no_prewrite
    return {
        "all_supervised_horizons_high_accuracy": high_accuracy,
        "improves_raw_each_supervised_horizon": improves_raw,
        "controls_causal_each_supervised_horizon": controls_causal,
        "no_prewrite": no_prewrite,
        "supervised_multi4_positive": supervised_positive,
        "h5_extrapolation_high_accuracy": rows[("FJ_multi4", 5)]["moving_target_accuracy"] >= 0.70,
        "h6_extrapolation_high_accuracy": rows[("FJ_multi4", 6)]["moving_target_accuracy"] >= 0.70,
    }


def _report_lines(summary: dict[str, Any]) -> list[str]:
    rows = _row_lookup(summary["horizon_rows"])
    pre = summary["pre_rows"][0]
    decision = summary["decision"]
    lines = [
        "# D8L8 seed3：FJ 四步联合监督",
        "",
        f"- controller-data seed: {summary['controller_seed']}; checkpoint step: {summary['checkpoint_step']};",
        "- source: frozen natural h1. One trainable tokenwise dense affine J is reused at every call; runtime is J→F.",
        "- loss: mean[CE(readout((FJ)^k(h1)), successor^k(readout(h1))) for k=1..4]; no teacher forcing.",
        "",
        "## 严格未见图正式集",
        "",
        "| horizon | supervision | raw F^k moving | (FJ)^k moving | shuffled-first-J | random-first-J |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for horizon in range(1, 7):
        supervision = "trained" if horizon <= 4 else "extrapolation"
        lines.append(
            f"| {horizon} | {supervision} | "
            f"{rows[('raw_F', horizon)]['moving_target_accuracy']:.4f} | "
            f"{rows[('FJ_multi4', horizon)]['moving_target_accuracy']:.4f} | "
            f"{rows[('shuffle_first_J_then_FJ', horizon)]['moving_target_accuracy']:.4f} | "
            f"{rows[('random_first_J_then_FJ', horizon)]['moving_target_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## executor-off 控制",
            "",
            f"- J(h1) readout current / next-moving: {pre['current_accuracy']:.4f} / {pre['moving_next_accuracy']:.4f}.",
            "",
            "## 判定",
            "",
            f"- all four supervised horizons gate: {decision['supervised_multi4_positive']}; h5/h6 extrapolation gates: {decision['h5_extrapolation_high_accuracy']} / {decision['h6_extrapolation_high_accuracy']}.",
            "- A four-step success establishes a finite learned FJ rollout under this fixed backbone and interface. It does not alone identify F's original or unique circuit.",
        ]
    )
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
    trained_horizons = 4
    formal_horizons = 6
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
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
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("FJ_MULTI4_CUDA_MEMORY_FRACTION", "0.30")),
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    if int(checkpoint_payload.get("step", -1)) != 20_000:
        raise ValueError("expected D8L8 seed3 final checkpoint at step 20000")
    model.requires_grad_(False)
    set_seed(controller_seed)
    formal = strict_unseen_permutations(cfg.node_count, set(), count=eval_permutations, seed=eval_seed)
    validation = strict_unseen_permutations(
        cfg.node_count,
        set(formal),
        count=val_permutations,
        seed=eval_seed + 1,
    )
    train_dataset, train_seen = build_training_dataset(
        model=model,
        cfg=cfg,
        excluded_permutations=set(formal) | set(validation),
        examples=train_examples,
        horizons=trained_horizons,
        seed=controller_seed + 43_007,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    validation_dataset = build_all_start_dataset(
        model=model,
        cfg=cfg,
        permutations=validation,
        horizons=trained_horizons,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    if train_seen & (set(formal) | set(validation)):
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
    candidates: list[CandidateResult] = []
    history: list[dict[str, Any]] = []
    for candidate_index, learning_rate in enumerate(learning_rates):
        spec = CandidateSpec(learning_rate)
        print(json.dumps({"event": "candidate_start", "candidate": spec.name}), flush=True)
        candidate = train_candidate(
            model=model,
            cfg=cfg,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            spec=spec,
            horizons=trained_horizons,
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_every=eval_every,
            controller_seed=controller_seed * 1009 + candidate_index + 29,
        )
        candidates.append(candidate)
        history.extend(candidate.history)
    selected = select_best_candidate(candidates)
    controller = controller_from_result(selected, dimension=cfg.d_model, device=device)
    torch.save(
        {
            "selected": selected.spec.name,
            "candidates": {
                candidate.spec.name: {
                    "spec": asdict(candidate.spec),
                    "validation": candidate.validation,
                    "state_dict": candidate.state_dict,
                }
                for candidate in candidates
            },
        },
        out_dir / "controller.pt",
    )
    strict_dataset = build_all_start_dataset(
        model=model,
        cfg=cfg,
        permutations=formal,
        horizons=formal_horizons,
        device=device,
        collection_batch_size=collection_batch_size,
    )
    horizon_rows, pre_rows = evaluate_formal(
        model=model,
        controller=controller,
        dataset=strict_dataset,
        horizons=formal_horizons,
        batch_size=eval_batch_size,
        shuffle_seed=eval_seed + controller_seed,
    )
    decision = multi4_decision(horizon_rows, pre_rows)
    matrix_rank = int(torch.linalg.matrix_rank(controller.weight.detach().float()).cpu())
    summary: dict[str, Any] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "source_code_sha256": _sha256(Path(__file__)),
        "loss_placement": "mean_final_CE_after_each_FJ_call_horizons_1_to_4",
        "runtime_order": "J_then_F_reused_four_times_without_teacher_forcing",
        "config": asdict(cfg),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "source_state": "natural_h1_only",
        "natural_semantic_step_per_loop": 1,
        "controller": "one shared tokenwise dense affine J across all positions and calls",
        "controller_parameters": cfg.d_model * cfg.d_model + cfg.d_model,
        "controller_numerical_matrix_rank": matrix_rank,
        "controller_seed": controller_seed,
        "train_examples": train_examples,
        "unique_train_permutations": len(train_seen),
        "validation_permutations": val_permutations,
        "strict_eval_permutations": eval_permutations,
        "strict_eval_examples": strict_dataset.source.shape[0],
        "strict_eval_permutation_sha256": _permutation_digest(formal),
        "trained_horizons": list(range(1, trained_horizons + 1)),
        "formal_horizons": list(range(1, formal_horizons + 1)),
        "learning_rates": list(learning_rates),
        "steps_per_candidate": steps,
        "batch_size": batch_size,
        "selected_candidate": selected.spec.name,
        "candidate_summary": [
            {"name": candidate.spec.name, "learning_rate": candidate.spec.learning_rate, **candidate.validation}
            for candidate in candidates
        ],
        "horizon_rows": horizon_rows,
        "pre_rows": pre_rows,
        "decision": decision,
        "peak_cuda_memory_gib": float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else 0.0,
    }
    _write_csv(out_dir / "candidate_rows.csv", summary["candidate_summary"])
    _write_csv(out_dir / "training_rows.csv", history)
    _write_csv(out_dir / "horizon_rows.csv", horizon_rows)
    _write_csv(out_dir / "pre_rows.csv", pre_rows)
    write_summary_atomic(out_dir / "summary.json", summary)
    (out_dir / "REPORT_CN.md").write_text("\n".join(_report_lines(summary)) + "\n", encoding="utf-8")
    manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "summary": str(out_dir / "summary.json")})
    write_summary_atomic(out_dir / "manifest.json", manifest)
    print(json.dumps({"event": "complete", "selected": selected.spec.name, "decision": decision}), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
