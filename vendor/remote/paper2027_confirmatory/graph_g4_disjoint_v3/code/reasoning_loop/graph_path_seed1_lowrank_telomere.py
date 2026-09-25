from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import run_instrumented_state
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_rejuvenation_circuit import behavior_metrics
from reasoning_loop.graph_path_telomere_overloop import (
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


TRANSITIONS = ("h8_to_h2", "h3_to_h2")


@dataclass(frozen=True)
class TransitionData:
    h8: torch.Tensor
    h2_for_h8: torch.Tensor
    h3: torch.Tensor
    h2_for_h3: torch.Tensor
    h8_full: torch.Tensor
    h3_full: torch.Tensor
    h8_current: torch.Tensor
    h8_next: torch.Tensor
    h3_current: torch.Tensor
    h3_next: torch.Tensor

    def pair(self, transition: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        if transition == "h8_to_h2":
            return self.h8, self.h2_for_h8, 6
        if transition == "h3_to_h2":
            return self.h3, self.h2_for_h3, 1
        raise ValueError(f"unknown transition: {transition}")

    def full_and_labels(
        self,
        transition: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if transition == "h8_to_h2":
            return self.h8_full, self.h8_current, self.h8_next
        if transition == "h3_to_h2":
            return self.h3_full, self.h3_current, self.h3_next
        raise ValueError(f"unknown transition: {transition}")


class LowRankRejuvenator(nn.Module):
    def __init__(
        self,
        feature_count: int,
        rank: int,
        *,
        use_bias: bool,
        initial_bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= rank <= feature_count:
            raise ValueError("rank must be between zero and feature_count")
        self.feature_count = feature_count
        self.rank = rank
        self.use_bias = use_bias
        if rank:
            scale = 0.02 / rank**0.5
            self.left = nn.Parameter(scale * torch.randn(feature_count, rank))
            self.right = nn.Parameter(scale * torch.randn(rank, feature_count))
        else:
            self.register_parameter("left", None)
            self.register_parameter("right", None)
        bias = (
            torch.zeros(feature_count)
            if initial_bias is None
            else initial_bias.detach().float().clone()
        )
        if use_bias:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("bias", torch.zeros_like(bias))

    def unit_update(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.feature_count:
            raise ValueError("state feature count does not match rejuvenator")
        update = torch.zeros_like(state)
        if self.rank:
            assert self.left is not None and self.right is not None
            update = (state @ self.left) @ self.right
        return update + self.bias

    def forward(
        self,
        state: torch.Tensor,
        *,
        dose: float = 1.0,
    ) -> torch.Tensor:
        return state + float(dose) * self.unit_update(state)

    def repeated(self, state: torch.Tensor, *, count: int) -> torch.Tensor:
        result = state
        for _ in range(count):
            result = self(result)
        return result

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.float() - target.float()).square().mean()
    denominator = (
        target.float() - target.float().mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            first.float().flatten(1),
            second.float().flatten(1),
            dim=1,
        ).mean()
    )


@torch.no_grad()
def collect_transition_data(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> TransitionData:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    if reference_age != 2 or jump != 2:
        raise ValueError("expected the seed1 age2/two-hop candidate")
    set_seed(seed)
    tokens, targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + 2 * jump,
    )
    h8_full = cache_states_with_initial(
        model,
        tokens,
        loops=cfg.max_loops,
    )[-1]
    reference_start = advance_nodes(
        successors,
        start,
        steps=cfg.max_depth - reference_position,
    )
    next_reference_start = advance_nodes(
        successors,
        reference_start,
        steps=jump,
    )
    reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=reference_start,
    )
    next_reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=next_reference_start,
    )
    reference_states = cache_states_with_initial(
        model,
        reference_tokens,
        loops=reference_age + 1,
    )
    h2_for_h3_full = cache_states_with_initial(
        model,
        next_reference_tokens,
        loops=reference_age,
    )[reference_age]
    return TransitionData(
        h8=h8_full[:, -1].float(),
        h2_for_h8=reference_states[reference_age][:, -1].float(),
        h3=reference_states[reference_age + 1][:, -1].float(),
        h2_for_h3=h2_for_h3_full[:, -1].float(),
        h8_full=h8_full,
        h3_full=reference_states[reference_age + 1],
        h8_current=targets[:, cfg.max_depth - 1],
        h8_next=targets[:, cfg.max_depth + jump - 1],
        h3_current=targets[:, cfg.max_depth + jump - 1],
        h3_next=targets[:, cfg.max_depth + 2 * jump - 1],
    )


def concatenate_data(parts: Sequence[TransitionData]) -> TransitionData:
    names = TransitionData.__dataclass_fields__
    return TransitionData(
        **{
            name: torch.cat([getattr(part, name) for part in parts])
            for name in names
        }
    )


def training_transitions(
    data: TransitionData,
    family: str,
) -> tuple[str, ...]:
    if family in {
        "shared_bias",
        "shared_no_bias",
        "reusable_bias",
        "reusable_no_bias",
    }:
        return TRANSITIONS
    if family == "init_bias":
        return ("h8_to_h2",)
    if family == "cycle_bias":
        return ("h3_to_h2",)
    raise ValueError(f"unknown family: {family}")


def mean_unit_bias(
    data: TransitionData,
    transitions: Sequence[str],
) -> torch.Tensor:
    updates = []
    for transition in transitions:
        source, target, gap = data.pair(transition)
        updates.append((target - source) / gap)
    return torch.cat(updates).mean(dim=0)


def normalized_training_loss(
    model: LowRankRejuvenator,
    batches: Sequence[tuple[str, torch.Tensor, torch.Tensor, int]],
    denominators: Sequence[torch.Tensor],
    *,
    family: str,
) -> torch.Tensor:
    losses = []
    for (transition, source, target, gap), denominator in zip(
        batches,
        denominators,
        strict=True,
    ):
        losses.append(
            (
                apply_training_map(
                    model,
                    source,
                    transition=transition,
                    gap=gap,
                    family=family,
                )
                - target
            )
            .square()
            .mean()
            / denominator
        )
    return torch.stack(losses).mean()


def apply_training_map(
    model: LowRankRejuvenator,
    source: torch.Tensor,
    *,
    transition: str,
    gap: int,
    family: str,
) -> torch.Tensor:
    if family.startswith("reusable_") and transition == "h8_to_h2":
        return model.repeated(source, count=gap)
    return model(source, dose=gap)


def _sample_indices(
    sample_count: int,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        sample_count,
        (batch_size,),
        generator=generator,
        device=device,
    )


def train_lowrank_model(
    *,
    calibration: TransitionData,
    validation: TransitionData,
    family: str,
    rank: int,
    use_bias: bool,
    factor_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_interval: int,
) -> tuple[LowRankRejuvenator, dict[str, Any]]:
    transitions = training_transitions(calibration, family)
    device = calibration.h8.device
    torch.manual_seed(factor_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(factor_seed)
    model = LowRankRejuvenator(
        calibration.h8.shape[-1],
        rank,
        use_bias=use_bias,
        initial_bias=(
            mean_unit_bias(calibration, transitions)
            if use_bias
            else None
        ),
    ).to(device)
    if model.parameter_count() == 0:
        metrics = validation_metrics(
            model=model,
            data=validation,
            family=family,
        )
        return model, {
            "best_step": 0,
            "best_validation_objective": metrics["objective"],
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    denominators = []
    for transition in transitions:
        _, target, _ = calibration.pair(transition)
        denominators.append(
            (
                target - target.mean(dim=0, keepdim=True)
            ).square().mean().clamp_min(1e-12)
        )
    generator = torch.Generator(device=device).manual_seed(factor_seed + 17)
    best_objective = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, steps + 1):
        batches = []
        for transition in transitions:
            source, target, gap = calibration.pair(transition)
            indices = _sample_indices(
                source.shape[0],
                min(batch_size, source.shape[0]),
                generator=generator,
                device=device,
            )
            batches.append(
                (transition, source[indices], target[indices], gap)
            )
        loss = normalized_training_loss(
            model,
            batches,
            denominators,
            family=family,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % validation_interval == 0 or step == steps:
            metrics = validation_metrics(
                model=model,
                data=validation,
                family=family,
            )
            objective = float(metrics["objective"])
            if objective < best_objective:
                best_objective = objective
                best_step = step
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_step": best_step,
        "best_validation_objective": best_objective,
    }


@torch.no_grad()
def validation_metrics(
    *,
    model: LowRankRejuvenator,
    data: TransitionData,
    family: str,
) -> dict[str, float]:
    trained = training_transitions(data, family)
    errors = {}
    for transition in TRANSITIONS:
        source, target, gap = data.pair(transition)
        errors[transition] = _relative_mse(
            apply_training_map(
                model,
                source,
                transition=transition,
                gap=gap,
                family=family,
            ),
            target,
        )
    return {
        **{f"{transition}_relative_mse": value for transition, value in errors.items()},
        "objective": sum(errors[name] for name in trained) / len(trained),
    }


def _patch_answer(
    full_state: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    state = full_state.clone()
    state[:, -1] = answer.to(state.dtype)
    return state


@torch.no_grad()
def evaluate_transition(
    *,
    transformer: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    model: LowRankRejuvenator,
    data: TransitionData,
    transition: str,
    application: str,
) -> dict[str, Any]:
    source, target, gap = data.pair(transition)
    full_state, endpoint, next_target = data.full_and_labels(transition)
    if application == "direct_dose":
        prediction = model(source, dose=gap)
    elif application == "repeated_unit":
        prediction = model.repeated(source, count=gap)
    else:
        raise ValueError(f"unknown application: {application}")
    state = _patch_answer(full_state, prediction)
    loop_index = cfg.max_loops if transition == "h8_to_h2" else cfg.max_loops + 1
    pre = behavior_metrics(
        logits_from_raw_state(transformer, state),
        target=next_target,
        endpoint=endpoint,
    )
    logits, trace = run_instrumented_state(
        transformer,
        state,
        loop_indices=(loop_index,),
    )
    post = behavior_metrics(
        logits,
        target=next_target,
        endpoint=endpoint,
    )
    oracle_state = _patch_answer(full_state, target)
    _, oracle_trace = run_instrumented_state(
        transformer,
        oracle_state,
        loop_indices=(loop_index,),
    )
    answer_position = cfg.seq_len - 1
    q = trace.sites[1].q[:, 2, answer_position]
    oracle_q = oracle_trace.sites[1].q[:, 2, answer_position]
    context = trace.sites[1].head_context[:, 2, answer_position]
    oracle_context = oracle_trace.sites[1].head_context[:, 2, answer_position]
    return {
        "transition": transition,
        "application": application,
        "state_relative_mse": _relative_mse(prediction, target),
        "state_cosine": _cosine(prediction, target),
        "update_cosine": _cosine(prediction - source, target - source),
        "pre_stack_accuracy": pre["accuracy"],
        "post_stack_accuracy": post["accuracy"],
        "B2_H2_q_cosine_to_oracle": _cosine(q, oracle_q),
        "B2_H2_context_cosine_to_oracle": _cosine(context, oracle_context),
    }


@torch.no_grad()
def evaluate_oracle_and_source(
    *,
    transformer: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    data: TransitionData,
) -> list[dict[str, Any]]:
    rows = []
    for transition in TRANSITIONS:
        source, target, _ = data.pair(transition)
        full_state, endpoint, next_target = data.full_and_labels(transition)
        loop_index = (
            cfg.max_loops if transition == "h8_to_h2" else cfg.max_loops + 1
        )
        for condition, answer in (("source", source), ("oracle", target)):
            state = _patch_answer(full_state, answer)
            pre = behavior_metrics(
                logits_from_raw_state(transformer, state),
                target=next_target,
                endpoint=endpoint,
            )
            logits, _ = run_instrumented_state(
                transformer,
                state,
                loop_indices=(loop_index,),
            )
            post = behavior_metrics(
                logits,
                target=next_target,
                endpoint=endpoint,
            )
            rows.append(
                {
                    "transition": transition,
                    "family": condition,
                    "rank": -1,
                    "factor_seed": -1,
                    "application": "direct_dose",
                    "state_relative_mse": (
                        _relative_mse(answer, target)
                        if condition == "source"
                        else 0.0
                    ),
                    "state_cosine": _cosine(answer, target),
                    "update_cosine": float("nan"),
                    "pre_stack_accuracy": pre["accuracy"],
                    "post_stack_accuracy": post["accuracy"],
                    "B2_H2_q_cosine_to_oracle": (
                        float("nan") if condition == "source" else 1.0
                    ),
                    "B2_H2_context_cosine_to_oracle": (
                        float("nan") if condition == "source" else 1.0
                    ),
                }
            )
    return rows


def _model_payload(model: LowRankRejuvenator) -> dict[str, Any]:
    return {
        "feature_count": model.feature_count,
        "rank": model.rank,
        "use_bias": model.use_bias,
        "parameter_count": model.parameter_count(),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
    }


def raw_trend_similarity(data: TransitionData) -> dict[str, float]:
    initial_unit = (data.h2_for_h8 - data.h8) / 6.0
    cycle_unit = data.h2_for_h3 - data.h3
    mean_initial = initial_unit.mean(dim=0)
    mean_cycle = cycle_unit.mean(dim=0)
    flat_initial = initial_unit.flatten(1)
    flat_cycle = cycle_unit.flatten(1)
    scalar = (
        (flat_initial * flat_cycle).sum()
        / flat_cycle.square().sum().clamp_min(1e-12)
    )
    scalar_prediction = scalar * flat_cycle
    centered = flat_initial - flat_initial.mean(dim=0, keepdim=True)
    scalar_r2 = 1.0 - (
        (flat_initial - scalar_prediction).square().sum()
        / centered.square().sum().clamp_min(1e-12)
    )
    return {
        "mean_unit_update_cosine": float(
            F.cosine_similarity(mean_initial, mean_cycle, dim=0)
        ),
        "samplewise_unit_update_cosine": float(
            F.cosine_similarity(flat_initial, flat_cycle, dim=1).mean()
        ),
        "mean_unit_update_norm_ratio_h8_over_h3": float(
            mean_initial.norm() / mean_cycle.norm().clamp_min(1e-12)
        ),
        "best_samplewise_scalar_h8_from_h3": float(scalar),
        "samplewise_scalar_fit_r2": float(scalar_r2),
    }


def _find_min_rank(
    rows: Sequence[dict[str, Any]],
    *,
    family: str,
    application: str,
    threshold: float,
) -> int | None:
    ranks = sorted(
        {
            int(row["rank"])
            for row in rows
            if row["family"] == family
            and row["application"] == application
            and all(
                float(candidate["post_stack_accuracy"]) >= threshold
                for candidate in rows
                if candidate["family"] == family
                and int(candidate["rank"]) == int(row["rank"])
                and candidate["application"] == application
            )
        }
    )
    return ranks[0] if ranks else None


def _find_min_reusable_rank(
    rows: Sequence[dict[str, Any]],
    *,
    family: str,
    threshold: float,
) -> int | None:
    ranks = sorted(
        {
            int(row["rank"])
            for row in rows
            if row["family"] == family and int(row["rank"]) >= 0
        }
    )
    for rank in ranks:
        h8 = next(
            row
            for row in rows
            if row["family"] == family
            and int(row["rank"]) == rank
            and row["transition"] == "h8_to_h2"
            and row["application"] == "repeated_unit"
        )
        h3 = next(
            row
            for row in rows
            if row["family"] == family
            and int(row["rank"]) == rank
            and row["transition"] == "h3_to_h2"
            and row["application"] == "direct_dose"
        )
        if (
            float(h8["post_stack_accuracy"]) >= threshold
            and float(h3["post_stack_accuracy"]) >= threshold
        ):
            return rank
    return None


def run_experiment(
    *,
    checkpoint: Path,
    lifespan_summary_path: Path,
    out_dir: Path,
    device: torch.device,
    requested_families: Sequence[str],
    ranks: Sequence[int],
    factor_seeds: Sequence[int],
    calibration_batch_size: int,
    calibration_batches: int,
    validation_size: int,
    evaluation_size: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    validation_interval: int,
    seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    transformer, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    lifespan_summary = json.loads(
        lifespan_summary_path.read_text(encoding="utf-8")
    )
    candidate = lifespan_summary["candidate"]
    calibration = concatenate_data(
        [
            collect_transition_data(
                model=transformer,
                cfg=cfg,
                candidate=candidate,
                batch_size=calibration_batch_size,
                device=device,
                seed=seed + index,
            )
            for index in range(calibration_batches)
        ]
    )
    validation = collect_transition_data(
        model=transformer,
        cfg=cfg,
        candidate=candidate,
        batch_size=validation_size,
        device=device,
        seed=seed + 100,
    )
    evaluation = collect_transition_data(
        model=transformer,
        cfg=cfg,
        candidate=candidate,
        batch_size=evaluation_size,
        device=device,
        seed=seed + 200,
    )
    family_bias = {
        "shared_bias": True,
        "shared_no_bias": False,
        "reusable_bias": True,
        "reusable_no_bias": False,
        "init_bias": True,
        "cycle_bias": True,
    }
    unknown = set(requested_families) - set(family_bias)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    families = tuple(
        (family, family_bias[family])
        for family in requested_families
    )
    training_rows = []
    evaluation_rows = evaluate_oracle_and_source(
        transformer=transformer,
        cfg=cfg,
        data=evaluation,
    )
    selected_models: dict[str, dict[int, LowRankRejuvenator]] = {
        family: {} for family, _ in families
    }
    saved_models: dict[str, Any] = {}
    for family, use_bias in families:
        for rank in ranks:
            candidates = []
            for factor_seed in factor_seeds:
                model, training = train_lowrank_model(
                    calibration=calibration,
                    validation=validation,
                    family=family,
                    rank=int(rank),
                    use_bias=use_bias,
                    factor_seed=int(factor_seed),
                    steps=steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    validation_interval=validation_interval,
                )
                metrics = validation_metrics(
                    model=model,
                    data=validation,
                    family=family,
                )
                row = {
                    "family": family,
                    "rank": int(rank),
                    "factor_seed": int(factor_seed),
                    "use_bias": use_bias,
                    "parameter_count": model.parameter_count(),
                    **training,
                    **metrics,
                }
                training_rows.append(row)
                candidates.append((float(metrics["objective"]), factor_seed, model))
            _, selected_seed, selected = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            selected_models[family][int(rank)] = selected
            saved_models[f"{family}_rank{rank}"] = {
                "selected_factor_seed": int(selected_seed),
                **_model_payload(selected),
            }
            for transition in TRANSITIONS:
                for application in (
                    ("direct_dose", "repeated_unit")
                    if transition == "h8_to_h2"
                    else ("direct_dose",)
                ):
                    evaluation_rows.append(
                        {
                            "family": family,
                            "rank": int(rank),
                            "factor_seed": int(selected_seed),
                            **evaluate_transition(
                                transformer=transformer,
                                cfg=cfg,
                                model=selected,
                                data=evaluation,
                                transition=transition,
                                application=application,
                            ),
                        }
                    )
            print(f"done {family} rank {rank}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "training_selection_rows.csv", training_rows)
    _write_csv(out_dir / "rank_sweep_evaluation_rows.csv", evaluation_rows)
    torch.save(
        {
            "format_version": 1,
            "checkpoint": str(checkpoint),
            "candidate": candidate,
            "ranks": tuple(int(rank) for rank in ranks),
            "factor_seeds": tuple(int(value) for value in factor_seeds),
            "models": saved_models,
        },
        out_dir / "lowrank_telomere_models.pt",
    )
    thresholds = (0.50, 0.75, 0.85, 0.90)
    summary = {
        "model": "D8L8-seed1",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": asdict(cfg),
        "candidate": candidate,
        "definition_under_test": (
            "one factorized R(h)=h+(hU)V+b with "
            "R^6(h8) and R(h3) both returning to aligned h2"
        ),
        "families": list(requested_families),
        "ranks": [int(rank) for rank in ranks],
        "factor_seeds": [int(value) for value in factor_seeds],
        "calibration_samples": calibration.h8.shape[0],
        "validation_samples": validation.h8.shape[0],
        "evaluation_samples": evaluation.h8.shape[0],
        "raw_unit_trend_similarity": raw_trend_similarity(evaluation),
        "minimum_shared_bias_rank_by_accuracy_threshold": {
            str(threshold): (
                _find_min_rank(
                    evaluation_rows,
                    family="shared_bias",
                    application="direct_dose",
                    threshold=threshold,
                )
                if "shared_bias" in requested_families
                else None
            )
            for threshold in thresholds
        },
        "minimum_shared_bias_repeated_rank_by_accuracy_threshold": {
            str(threshold): (
                _find_min_reusable_rank(
                    evaluation_rows,
                    family="reusable_bias",
                    threshold=threshold,
                )
                if "reusable_bias" in requested_families
                else None
            )
            for threshold in thresholds
        },
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train low-rank factorized rejuvenation functions for the "
            "D8L8-seed1 h8->h2 and h3->h2 telomere hypothesis."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lifespan-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    parser.add_argument(
        "--factor-seeds",
        type=int,
        nargs="+",
        default=(0, 1),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=(
            "shared_bias",
            "shared_no_bias",
            "reusable_bias",
            "reusable_no_bias",
            "init_bias",
            "cycle_bias",
        ),
    )
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--evaluation-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026076101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("cuda memory fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    run_experiment(
        checkpoint=args.checkpoint,
        lifespan_summary_path=args.lifespan_summary,
        out_dir=args.out_dir,
        device=device,
        requested_families=args.families,
        ranks=args.ranks,
        factor_seeds=args.factor_seeds,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        validation_size=args.validation_size,
        evaluation_size=args.evaluation_size,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_interval=args.validation_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
