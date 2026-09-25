from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_rejuvenator_commutator import (
    AffineAnswerMap,
    _cosine,
    _relative_mse,
    _relative_pair_rms,
    apply_answer_map,
    load_rejuvenator,
    random_orientation_control,
    rejuvenator_affine_parts,
    spectral_metrics,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


TRAIN_SOURCE_AGES = tuple(range(3, 9))
COMMUTATOR_BOTH_INPUTS_TRAINED_AGES = tuple(range(3, 8))
COMMUTATOR_SAME_STEP_REGIME_AGES = (3, 5, 6, 7)


@dataclass(frozen=True)
class PairDataset:
    source: torch.Tensor
    target: torch.Tensor
    source_age: torch.Tensor


@dataclass(frozen=True)
class EvaluationBatch:
    base_states: dict[int, torch.Tensor]
    states_by_start_shift: dict[int, dict[int, torch.Tensor]]
    reset_oracles: dict[int, torch.Tensor]
    successors: torch.Tensor
    start: torch.Tensor
    jump: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plain_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


def _masked_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    endpoint: torch.Tensor | None,
) -> float:
    if endpoint is None:
        return _plain_accuracy(logits, target)
    valid = target.ne(endpoint)
    if not bool(valid.any()):
        return float("nan")
    return float(logits[valid].argmax(dim=-1).eq(target[valid]).float().mean())


def _labels_after_steps(
    successors: torch.Tensor,
    start: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    return advance_nodes(successors, start, steps=steps)


def age_path_position(
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    age: int,
) -> int:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    return min(
        cfg.max_depth,
        max(0, reference_position + jump * (int(age) - reference_age)),
    )


def adjacent_alignment_shift(
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    source_age: int,
) -> int:
    if source_age < 1:
        raise ValueError("source_age must be positive")
    return age_path_position(
        cfg, candidate, source_age
    ) - age_path_position(cfg, candidate, source_age - 1)


@torch.no_grad()
def collect_states_at_ages(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    ages: Iterable[int],
) -> dict[int, torch.Tensor]:
    selected = frozenset(int(age) for age in ages)
    if not selected or min(selected) < 0:
        raise ValueError("ages must be a non-empty collection of non-negative integers")
    state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    states: dict[int, torch.Tensor] = {}
    if 0 in selected:
        states[0] = state
    for loop_index in range(max(selected)):
        state = apply_shared_stack(model, state, loop_index=loop_index)
        age = loop_index + 1
        if age in selected:
            states[age] = state
    return states


def concatenate_pair_datasets(parts: Sequence[PairDataset]) -> PairDataset:
    if not parts:
        raise ValueError("at least one pair dataset is required")
    return PairDataset(
        source=torch.cat([part.source for part in parts]),
        target=torch.cat([part.target for part in parts]),
        source_age=torch.cat([part.source_age for part in parts]),
    )


@torch.no_grad()
def collect_pair_dataset(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    sample_count: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    source_ages: Sequence[int] = TRAIN_SOURCE_AGES,
) -> PairDataset:
    if sample_count < 1 or batch_size < 1:
        raise ValueError("sample_count and batch_size must be positive")
    jump = int(candidate["programmed_jump"])
    if jump < 1:
        raise ValueError("programmed_jump must be positive")
    source_ages = tuple(int(age) for age in source_ages)
    if not source_ages or min(source_ages) < 1:
        raise ValueError("source ages must be positive")
    set_seed(seed)
    parts: list[PairDataset] = []
    remaining = sample_count
    while remaining:
        current_batch = min(batch_size, remaining)
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            current_batch,
            device,
            path_positions=cfg.max_depth + 2 * jump,
        )
        required_source_ages = set(source_ages)
        required_source_ages.update(age - 1 for age in source_ages)
        source_states = collect_states_at_ages(
            model, tokens, required_source_ages
        )
        target_ages = tuple(age - 1 for age in source_ages)
        target_states_by_shift = {0: source_states}
        nonzero_shifts = sorted(
            {
                adjacent_alignment_shift(cfg, candidate, age)
                for age in source_ages
                if adjacent_alignment_shift(cfg, candidate, age) != 0
            }
        )
        for shift in nonzero_shifts:
            shifted_start = advance_nodes(successors, start, steps=shift)
            shifted_tokens, _, _, _ = fixed_depth_batch(
                cfg,
                current_batch,
                device,
                path_positions=cfg.max_depth + 2 * jump,
                successors=successors,
                start=shifted_start,
            )
            required_target_ages = tuple(
                age - 1
                for age in source_ages
                if adjacent_alignment_shift(cfg, candidate, age) == shift
            )
            target_states_by_shift[shift] = collect_states_at_ages(
                model,
                shifted_tokens,
                required_target_ages,
            )
        source_rows = []
        target_rows = []
        age_rows = []
        for age in source_ages:
            shift = adjacent_alignment_shift(cfg, candidate, age)
            source_rows.append(source_states[age][:, -1].detach().float().cpu())
            target_rows.append(
                target_states_by_shift[shift][age - 1][
                    :, -1
                ].detach().float().cpu()
            )
            age_rows.append(torch.full((current_batch,), age, dtype=torch.long))
        parts.append(
            PairDataset(
                source=torch.cat(source_rows),
                target=torch.cat(target_rows),
                source_age=torch.cat(age_rows),
            )
        )
        remaining -= current_batch
    return concatenate_pair_datasets(parts)


def age_balanced_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ages: torch.Tensor,
) -> tuple[float, dict[int, float]]:
    per_age = {}
    for age in sorted(int(value) for value in ages.unique()):
        mask = ages == age
        per_age[age] = _relative_mse(prediction[mask], target[mask])
    return sum(per_age.values()) / len(per_age), per_age


def _age_weights(data: PairDataset) -> torch.Tensor:
    weights = torch.empty(data.source.shape[0], dtype=torch.float64)
    target = data.target.double()
    for age in sorted(int(value) for value in data.source_age.unique()):
        mask = data.source_age == age
        selected = target[mask]
        denominator = (
            selected - selected.mean(dim=0, keepdim=True)
        ).square().mean().clamp_min(1e-12)
        weights[mask] = denominator.rsqrt()
    return weights


def fit_weighted_ridge(
    data: PairDataset,
    *,
    use_bias: bool,
    ridge_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ridge_multiplier < 0:
        raise ValueError("ridge_multiplier must be non-negative")
    source = data.source.double()
    target = data.target.double()
    if use_bias:
        source = torch.cat(
            [source, torch.ones(source.shape[0], 1, dtype=source.dtype)],
            dim=1,
        )
    weights = _age_weights(data)
    weighted_source = source * weights[:, None]
    weighted_target = target * weights[:, None]
    gram = weighted_source.T @ weighted_source
    cross = weighted_source.T @ weighted_target
    feature_count = data.source.shape[1]
    scale = (
        torch.diagonal(gram[:feature_count, :feature_count]).mean().clamp_min(1e-12)
    )
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype)
    if use_bias:
        penalty[-1, -1] = 0
    regularized = gram + float(ridge_multiplier) * scale * penalty
    try:
        coefficients = torch.linalg.solve(regularized, cross)
    except torch.linalg.LinAlgError:
        coefficients = torch.linalg.pinv(regularized) @ cross
    if use_bias:
        matrix = coefficients[:-1]
        bias = coefficients[-1]
    else:
        matrix = coefficients
        bias = torch.zeros(feature_count, dtype=matrix.dtype)
    return matrix.float(), bias.float()


def to_answer_map(
    matrix: torch.Tensor,
    bias: torch.Tensor,
    *,
    device: torch.device,
) -> AffineAnswerMap:
    matrix = matrix.to(device=device, dtype=torch.float32)
    bias = bias.to(device=device, dtype=torch.float32)
    identity = torch.eye(matrix.shape[0], device=device, dtype=matrix.dtype)
    return AffineAnswerMap(update_matrix=matrix - identity, bias=bias)


def predict_pair_dataset(
    data: PairDataset,
    matrix: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return data.source @ matrix + bias


def select_and_refit_map(
    *,
    calibration: PairDataset,
    validation: PairDataset,
    use_bias: bool,
    ridge_grid: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], float]:
    rows = []
    best_objective = float("inf")
    best_ridge = float(ridge_grid[0])
    for ridge in ridge_grid:
        matrix, bias = fit_weighted_ridge(
            calibration,
            use_bias=use_bias,
            ridge_multiplier=float(ridge),
        )
        prediction = predict_pair_dataset(validation, matrix, bias)
        objective, per_age = age_balanced_relative_mse(
            prediction,
            validation.target,
            validation.source_age,
        )
        row: dict[str, Any] = {
            "use_bias": use_bias,
            "ridge_multiplier": float(ridge),
            "validation_age_balanced_relative_mse": objective,
        }
        for age, value in per_age.items():
            row[f"validation_h{age}_to_h{age - 1}_relative_mse"] = value
        rows.append(row)
        if objective < best_objective:
            best_objective = objective
            best_ridge = float(ridge)
    combined = concatenate_pair_datasets((calibration, validation))
    matrix, bias = fit_weighted_ridge(
        combined,
        use_bias=use_bias,
        ridge_multiplier=best_ridge,
    )
    return matrix, bias, rows, best_ridge


@torch.no_grad()
def collect_evaluation_batch(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int,
    maximum_commutator_age: int,
    reset_source_ages: Sequence[int],
) -> EvaluationBatch:
    jump = int(candidate["programmed_jump"])
    set_seed(seed)
    tokens, _, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + 4 * jump,
    )
    base_ages = set(range(3, maximum_commutator_age + 2))
    base_ages.update(range(2, maximum_commutator_age + 1))
    base_ages.update(int(age) for age in reset_source_ages)
    base_states = collect_states_at_ages(model, tokens, base_ages)
    states_by_start_shift = {0: base_states}
    required_shifts = {
        adjacent_alignment_shift(cfg, candidate, age)
        for age in range(3, maximum_commutator_age + 2)
    }
    for shift in sorted(required_shifts - {0}):
        shifted_start = advance_nodes(successors, start, steps=shift)
        shifted_tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 4 * jump,
            successors=successors,
            start=shifted_start,
        )
        states_by_start_shift[shift] = collect_states_at_ages(
            model,
            shifted_tokens,
            range(2, maximum_commutator_age + 1),
        )
    reset_oracles = {}
    for source_age in reset_source_ages:
        total_shift = age_path_position(
            cfg, candidate, int(source_age)
        ) - age_path_position(cfg, candidate, 2)
        oracle_start = advance_nodes(
            successors,
            start,
            steps=total_shift,
        )
        oracle_tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + 2 * jump,
            successors=successors,
            start=oracle_start,
        )
        reset_oracles[int(source_age)] = collect_states_at_ages(
            model,
            oracle_tokens,
            (2,),
        )[2]
    return EvaluationBatch(
        base_states=base_states,
        states_by_start_shift=states_by_start_shift,
        reset_oracles=reset_oracles,
        successors=successors,
        start=start,
        jump=jump,
    )


def _is_global_learned_condition(condition: str) -> bool:
    return condition in {
        "global_linear_256x256",
        "global_affine_256x256_plus_bias",
    }


def _support_label_for_map(condition: str, source_age: int) -> str:
    if not _is_global_learned_condition(condition):
        return "comparison_control"
    return "joint_train_ages" if source_age in TRAIN_SOURCE_AGES else "age_ood"


def _support_label_for_commutator(condition: str, source_age: int) -> str:
    if not _is_global_learned_condition(condition):
        return "comparison_control"
    if source_age == 4:
        return "joint_train_structural_halting_boundary"
    if source_age in COMMUTATOR_SAME_STEP_REGIME_AGES:
        return "both_J_inputs_joint_train_same_step_regime"
    if source_age == 8:
        return "boundary_JF_uses_h9"
    return "overloop_age_ood"


@torch.no_grad()
def evaluate_mapping(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch: EvaluationBatch,
    answer_map: AffineAnswerMap,
    condition: str,
    source_age: int,
) -> dict[str, Any]:
    source = batch.base_states[source_age]
    shift = adjacent_alignment_shift(cfg, candidate, source_age)
    aligned_states = batch.states_by_start_shift[shift]
    oracle = aligned_states[source_age - 1]
    mapped = apply_answer_map(source, answer_map)
    post = apply_shared_stack(model, mapped, loop_index=cfg.max_loops)
    oracle_post = aligned_states[source_age]
    source_position = age_path_position(cfg, candidate, source_age)
    post_position = shift + source_position
    current = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=source_position,
    )
    post_target = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=post_position,
    )
    post_endpoint = current if post_position != source_position else None
    return {
        "condition": condition,
        "source_age": source_age,
        "support": _support_label_for_map(condition, source_age),
        "sample_count": source.shape[0],
        "mapped_answer_relative_mse_to_oracle": _relative_mse(
            mapped[:, -1], oracle[:, -1]
        ),
        "mapped_answer_cosine_to_oracle": _cosine(
            mapped[:, -1], oracle[:, -1]
        ),
        "mapped_pre_current_accuracy": _plain_accuracy(
            logits_from_raw_state(model, mapped), current
        ),
        "alignment_start_shift": shift,
        "source_path_position": source_position,
        "post_oracle_path_position": post_position,
        "mapped_pre_post_target_accuracy": _masked_accuracy(
            logits_from_raw_state(model, mapped),
            post_target,
            endpoint=post_endpoint,
        ),
        "mapped_post_oracle_accuracy": _masked_accuracy(
            logits_from_raw_state(model, post),
            post_target,
            endpoint=post_endpoint,
        ),
        "oracle_pre_current_accuracy": _plain_accuracy(
            logits_from_raw_state(model, oracle), current
        ),
        "oracle_post_accuracy": _masked_accuracy(
            logits_from_raw_state(model, oracle_post),
            post_target,
            endpoint=post_endpoint,
        ),
    }


@torch.no_grad()
def evaluate_commutator(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch: EvaluationBatch,
    answer_map: AffineAnswerMap,
    condition: str,
    source_age: int,
) -> dict[str, Any]:
    source = batch.base_states[source_age]
    after_f = apply_shared_stack(model, source, loop_index=cfg.max_loops)
    jf = apply_answer_map(after_f, answer_map)
    fj = apply_shared_stack(
        model,
        apply_answer_map(source, answer_map),
        loop_index=cfg.max_loops,
    )
    jf_shift = adjacent_alignment_shift(cfg, candidate, source_age + 1)
    fj_shift = adjacent_alignment_shift(cfg, candidate, source_age)
    jf_oracle = batch.states_by_start_shift[jf_shift][source_age]
    fj_oracle = batch.states_by_start_shift[fj_shift][source_age]
    jf_future = apply_shared_stack(model, jf, loop_index=cfg.max_loops)
    fj_future = apply_shared_stack(model, fj, loop_index=cfg.max_loops)
    jf_oracle_future = apply_shared_stack(
        model, jf_oracle, loop_index=cfg.max_loops
    )
    fj_oracle_future = apply_shared_stack(
        model, fj_oracle, loop_index=cfg.max_loops
    )
    source_position = age_path_position(cfg, candidate, source_age)
    next_position = age_path_position(cfg, candidate, source_age + 1)
    jf_position = next_position
    fj_position = fj_shift + source_position
    jf_target = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=jf_position,
    )
    fj_target = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=fj_position,
    )
    jf_future_target = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=jf_shift + next_position,
    )
    fj_future_target = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=fj_shift + next_position,
    )
    source_current = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=source_position,
    )
    jf_direct_endpoint = source_current if jf_position != source_position else None
    fj_direct_endpoint = source_current if fj_position != source_position else None
    jf_future_position = jf_shift + next_position
    fj_future_position = fj_shift + next_position
    jf_future_endpoint = jf_target if jf_future_position != jf_position else None
    fj_future_endpoint = fj_target if fj_future_position != fj_position else None
    return {
        "condition": condition,
        "source_age": source_age,
        "support": _support_label_for_commutator(condition, source_age),
        "sample_count": source.shape[0],
        "jf_alignment_start_shift": jf_shift,
        "fj_alignment_start_shift": fj_shift,
        "paths_share_algorithmic_position": jf_position == fj_position,
        "jf_path_position": jf_position,
        "fj_path_position": fj_position,
        "answer_commutator_relative_rms": _relative_pair_rms(
            jf[:, -1], fj[:, -1]
        ),
        "oracle_answer_commutator_relative_rms": _relative_pair_rms(
            jf_oracle[:, -1], fj_oracle[:, -1]
        ),
        "answer_commutator_cosine": _cosine(jf[:, -1], fj[:, -1]),
        "jf_answer_relative_mse_to_own_oracle": _relative_mse(
            jf[:, -1], jf_oracle[:, -1]
        ),
        "fj_answer_relative_mse_to_own_oracle": _relative_mse(
            fj[:, -1], fj_oracle[:, -1]
        ),
        "jf_answer_cosine_to_own_oracle": _cosine(
            jf[:, -1], jf_oracle[:, -1]
        ),
        "fj_answer_cosine_to_own_oracle": _cosine(
            fj[:, -1], fj_oracle[:, -1]
        ),
        "jf_direct_accuracy": _masked_accuracy(
            logits_from_raw_state(model, jf),
            jf_target,
            endpoint=jf_direct_endpoint,
        ),
        "fj_direct_accuracy": _masked_accuracy(
            logits_from_raw_state(model, fj),
            fj_target,
            endpoint=fj_direct_endpoint,
        ),
        "jf_oracle_direct_accuracy": _masked_accuracy(
            logits_from_raw_state(model, jf_oracle),
            jf_target,
            endpoint=jf_direct_endpoint,
        ),
        "fj_oracle_direct_accuracy": _masked_accuracy(
            logits_from_raw_state(model, fj_oracle),
            fj_target,
            endpoint=fj_direct_endpoint,
        ),
        "jf_future_accuracy": _masked_accuracy(
            logits_from_raw_state(model, jf_future),
            jf_future_target,
            endpoint=jf_future_endpoint,
        ),
        "fj_future_accuracy": _masked_accuracy(
            logits_from_raw_state(model, fj_future),
            fj_future_target,
            endpoint=fj_future_endpoint,
        ),
        "jf_oracle_future_accuracy": _masked_accuracy(
            logits_from_raw_state(model, jf_oracle_future),
            jf_future_target,
            endpoint=jf_future_endpoint,
        ),
        "fj_oracle_future_accuracy": _masked_accuracy(
            logits_from_raw_state(model, fj_oracle_future),
            fj_future_target,
            endpoint=fj_future_endpoint,
        ),
    }


@torch.no_grad()
def evaluate_repeated_reset(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    batch: EvaluationBatch,
    answer_map: AffineAnswerMap,
    condition: str,
    source_age: int,
) -> dict[str, Any]:
    source = batch.base_states[source_age]
    oracle = batch.reset_oracles[source_age]
    reset = source
    for _ in range(source_age - 2):
        reset = apply_answer_map(reset, answer_map)
    post = apply_shared_stack(model, reset, loop_index=cfg.max_loops)
    source_position = age_path_position(cfg, candidate, source_age)
    h2_position = age_path_position(cfg, candidate, 2)
    h3_position = age_path_position(cfg, candidate, 3)
    total_shift = source_position - h2_position
    post_position = total_shift + h3_position
    current = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=source_position,
    )
    post_target = _labels_after_steps(
        batch.successors,
        batch.start,
        steps=post_position,
    )
    post_endpoint = current if post_position != source_position else None
    return {
        "condition": condition,
        "source_age": source_age,
        "application_count": source_age - 2,
        "support": (
            (
                "composition_within_train_horizon"
                if source_age == 8
                else "overloop_composition_age_ood"
            )
            if _is_global_learned_condition(condition)
            else "comparison_control"
        ),
        "sample_count": source.shape[0],
        "source_path_position": source_position,
        "reset_oracle_start_shift": total_shift,
        "post_reset_path_position": post_position,
        "reset_answer_relative_mse_to_oracle_h2": _relative_mse(
            reset[:, -1], oracle[:, -1]
        ),
        "reset_answer_cosine_to_oracle_h2": _cosine(
            reset[:, -1], oracle[:, -1]
        ),
        "reset_pre_current_accuracy": _plain_accuracy(
            logits_from_raw_state(model, reset), current
        ),
        "reset_pre_post_target_accuracy": _masked_accuracy(
            logits_from_raw_state(model, reset),
            post_target,
            endpoint=post_endpoint,
        ),
        "reset_post_target_accuracy": _masked_accuracy(
            logits_from_raw_state(model, post),
            post_target,
            endpoint=post_endpoint,
        ),
    }


def _aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    keys: Sequence[str],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregates = []
    for group_key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        aggregate = {key: value for key, value in zip(keys, group_key, strict=True)}
        aggregate["replications"] = len(selected)
        for field in fields:
            values = torch.tensor(
                [float(row[field]) for row in selected], dtype=torch.float64
            )
            aggregate[f"{field}_mean"] = float(values.mean())
            aggregate[f"{field}_min"] = float(values.min())
            aggregate[f"{field}_max"] = float(values.max())
        aggregates.append(aggregate)
    return aggregates


def run_experiment(
    *,
    checkpoint: Path,
    local_rejuvenator_artifact: Path,
    out_dir: Path,
    device: torch.device,
    calibration_samples: int,
    validation_samples: int,
    collection_batch_size: int,
    evaluation_batch_size: int,
    calibration_seed: int,
    validation_seed: int,
    evaluation_seeds: Sequence[int],
    ridge_grid: Sequence[float],
    maximum_commutator_age: int,
    reset_source_ages: Sequence[int],
    control_seed: int,
) -> dict[str, Any]:
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("expected the D8L8 two-block checkpoint")
    if any(
        model.active_block_indices(index) != tuple(range(cfg.n_layers))
        for index in range(max(reset_source_ages) + 1)
    ):
        raise ValueError("F changes its physical block set across recurrent cycles")
    local_model, local_payload = load_rejuvenator(
        local_rejuvenator_artifact,
        model_name="reusable_bias_rank15",
        device=device,
    )
    candidate = local_payload["candidate"]

    calibration = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=candidate,
        sample_count=calibration_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=calibration_seed,
    )
    print("collected calibration pairs", flush=True)
    validation = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=candidate,
        sample_count=validation_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=validation_seed,
    )
    print("collected validation pairs", flush=True)

    linear_matrix, linear_bias, linear_rows, linear_ridge = select_and_refit_map(
        calibration=calibration,
        validation=validation,
        use_bias=False,
        ridge_grid=ridge_grid,
    )
    affine_matrix, affine_bias, affine_rows, affine_ridge = select_and_refit_map(
        calibration=calibration,
        validation=validation,
        use_bias=True,
        ridge_grid=ridge_grid,
    )
    print("fit global linear and affine maps", flush=True)

    linear_map = to_answer_map(linear_matrix, linear_bias, device=device)
    affine_map = to_answer_map(affine_matrix, affine_bias, device=device)
    local_update, local_bias = rejuvenator_affine_parts(local_model)
    local_map = AffineAnswerMap(local_update, local_bias)
    identity = AffineAnswerMap(
        torch.zeros_like(linear_map.update_matrix),
        torch.zeros_like(linear_map.bias),
    )
    random_linear = random_orientation_control(linear_map, seed=control_seed)
    conditions = (
        ("global_linear_256x256", linear_map),
        ("global_affine_256x256_plus_bias", affine_map),
        ("previous_endpoint_rank15", local_map),
        ("random_orientation_global_linear", random_linear),
        ("identity", identity),
    )

    mapping_rows = []
    commutator_rows = []
    reset_rows = []
    mapping_ages = tuple(range(3, maximum_commutator_age + 1))
    commutator_ages = tuple(range(3, maximum_commutator_age + 1))
    for replication, evaluation_seed in enumerate(evaluation_seeds):
        batch = collect_evaluation_batch(
            model=model,
            cfg=cfg,
            candidate=candidate,
            batch_size=evaluation_batch_size,
            device=device,
            seed=int(evaluation_seed),
            maximum_commutator_age=maximum_commutator_age,
            reset_source_ages=reset_source_ages,
        )
        for condition, answer_map in conditions:
            for source_age in mapping_ages:
                mapping_rows.append(
                    {
                        "replication": replication,
                        "data_seed": int(evaluation_seed),
                        **evaluate_mapping(
                            model=model,
                            cfg=cfg,
                            candidate=candidate,
                            batch=batch,
                            answer_map=answer_map,
                            condition=condition,
                            source_age=source_age,
                        ),
                    }
                )
            for source_age in commutator_ages:
                commutator_rows.append(
                    {
                        "replication": replication,
                        "data_seed": int(evaluation_seed),
                        **evaluate_commutator(
                            model=model,
                            cfg=cfg,
                            candidate=candidate,
                            batch=batch,
                            answer_map=answer_map,
                            condition=condition,
                            source_age=source_age,
                        ),
                    }
                )
            for source_age in reset_source_ages:
                reset_rows.append(
                    {
                        "replication": replication,
                        "data_seed": int(evaluation_seed),
                        **evaluate_repeated_reset(
                            model=model,
                            cfg=cfg,
                            candidate=candidate,
                            batch=batch,
                            answer_map=answer_map,
                            condition=condition,
                            source_age=int(source_age),
                        ),
                    }
                )
        print(f"completed evaluation replication {replication}", flush=True)

    mapping_fields = (
        "mapped_answer_relative_mse_to_oracle",
        "mapped_answer_cosine_to_oracle",
        "mapped_pre_current_accuracy",
        "mapped_pre_post_target_accuracy",
        "mapped_post_oracle_accuracy",
        "oracle_pre_current_accuracy",
        "oracle_post_accuracy",
    )
    commutator_fields = (
        "answer_commutator_relative_rms",
        "oracle_answer_commutator_relative_rms",
        "answer_commutator_cosine",
        "jf_answer_relative_mse_to_own_oracle",
        "fj_answer_relative_mse_to_own_oracle",
        "jf_answer_cosine_to_own_oracle",
        "fj_answer_cosine_to_own_oracle",
        "jf_direct_accuracy",
        "fj_direct_accuracy",
        "jf_oracle_direct_accuracy",
        "fj_oracle_direct_accuracy",
        "jf_future_accuracy",
        "fj_future_accuracy",
        "jf_oracle_future_accuracy",
        "fj_oracle_future_accuracy",
    )
    reset_fields = (
        "reset_answer_relative_mse_to_oracle_h2",
        "reset_answer_cosine_to_oracle_h2",
        "reset_pre_current_accuracy",
        "reset_pre_post_target_accuracy",
        "reset_post_target_accuracy",
    )
    mapping_aggregate = _aggregate(
        mapping_rows,
        keys=(
            "condition",
            "source_age",
            "support",
            "alignment_start_shift",
            "source_path_position",
            "post_oracle_path_position",
        ),
        fields=mapping_fields,
    )
    commutator_aggregate = _aggregate(
        commutator_rows,
        keys=(
            "condition",
            "source_age",
            "support",
            "jf_alignment_start_shift",
            "fj_alignment_start_shift",
            "paths_share_algorithmic_position",
            "jf_path_position",
            "fj_path_position",
        ),
        fields=commutator_fields,
    )
    reset_aggregate = _aggregate(
        reset_rows,
        keys=(
            "condition",
            "source_age",
            "application_count",
            "support",
            "source_path_position",
            "reset_oracle_start_shift",
            "post_reset_path_position",
        ),
        fields=reset_fields,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fit_rows = linear_rows + affine_rows
    _write_csv(out_dir / "fit_selection_rows.csv", fit_rows)
    _write_csv(out_dir / "mapping_rows.csv", mapping_rows)
    _write_csv(out_dir / "mapping_aggregate.csv", mapping_aggregate)
    _write_csv(out_dir / "commutator_rows.csv", commutator_rows)
    _write_csv(out_dir / "commutator_aggregate.csv", commutator_aggregate)
    _write_csv(out_dir / "repeated_reset_rows.csv", reset_rows)
    _write_csv(out_dir / "repeated_reset_aggregate.csv", reset_aggregate)

    spectra = {
        "global_linear_256x256": spectral_metrics(linear_map),
        "global_affine_256x256_plus_bias": spectral_metrics(affine_map),
        "previous_endpoint_rank15": spectral_metrics(local_map),
        "random_orientation_global_linear": spectral_metrics(random_linear),
    }
    (out_dir / "spectral_summary.json").write_text(
        json.dumps(spectra, indent=2),
        encoding="utf-8",
    )
    artifact = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "candidate": candidate,
        "train_source_ages": list(TRAIN_SOURCE_AGES),
        "linear": {
            "matrix": linear_matrix,
            "bias": linear_bias,
            "selected_ridge_multiplier": linear_ridge,
        },
        "affine": {
            "matrix": affine_matrix,
            "bias": affine_bias,
            "selected_ridge_multiplier": affine_ridge,
        },
    }
    torch.save(artifact, out_dir / "global_rejuvenator.pt")
    summary = {
        "model": "D8L8-seed1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": asdict(cfg),
        "loss_placement": "final CE at recurrent loop 8 only",
        "shared_unit": "two physical transformer blocks per recurrent cycle",
        "trained_recurrent_cycles": cfg.max_loops,
        "trained_effective_block_applications": cfg.max_loops * cfg.n_layers,
        "F_definition": "one recurrent cycle containing both shared blocks",
        "J_definition": (
            "one global answer-token map jointly fit on every position-aligned "
            "adjacent age pair h_k(start)->h_{k-1}(f^delta_k(start)), "
            "k=3,...,8, delta_k=p_k-p_{k-1}"
        ),
        "primary_J": "pure 256x256 linear matrix without bias",
        "secondary_J": "full affine 256x256 matrix plus bias",
        "train_source_ages": list(TRAIN_SOURCE_AGES),
        "commutator_both_J_inputs_trained_ages": list(
            COMMUTATOR_BOTH_INPUTS_TRAINED_AGES
        ),
        "commutator_same_step_regime_ages": list(
            COMMUTATOR_SAME_STEP_REGIME_AGES
        ),
        "trajectory_positions_including_initial": [
            age_path_position(cfg, candidate, age)
            for age in range(cfg.max_loops + 1)
        ],
        "calibration_samples_per_age": calibration_samples,
        "validation_samples_per_age": validation_samples,
        "evaluation_batch_size": evaluation_batch_size,
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "linear_selected_ridge_multiplier": linear_ridge,
        "affine_selected_ridge_multiplier": affine_ridge,
        "device": str(device),
        "mapping_aggregate": mapping_aggregate,
        "commutator_aggregate": commutator_aggregate,
        "repeated_reset_aggregate": reset_aggregate,
        "spectra": spectra,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the best global linear/affine rejuvenator across every adjacent "
            "D8L8 age pair, then test functional commutation with F."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--local-rejuvenator-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--collection-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--calibration-seed", type=int, default=2026077001)
    parser.add_argument("--validation-seed", type=int, default=2026077101)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=(2026077201, 2026077301),
    )
    parser.add_argument(
        "--ridge-grid",
        type=float,
        nargs="+",
        default=(
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0,
        ),
    )
    parser.add_argument("--maximum-commutator-age", type=int, default=12)
    parser.add_argument(
        "--reset-source-ages",
        type=int,
        nargs="+",
        default=(8, 16, 32, 64),
    )
    parser.add_argument("--control-seed", type=int, default=2026077401)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        local_rejuvenator_artifact=args.local_rejuvenator_artifact,
        out_dir=args.out_dir,
        device=pick_device(args.device),
        calibration_samples=args.calibration_samples,
        validation_samples=args.validation_samples,
        collection_batch_size=args.collection_batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        calibration_seed=args.calibration_seed,
        validation_seed=args.validation_seed,
        evaluation_seeds=args.evaluation_seeds,
        ridge_grid=args.ridge_grid,
        maximum_commutator_age=args.maximum_commutator_age,
        reset_source_ages=args.reset_source_ages,
        control_seed=args.control_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
