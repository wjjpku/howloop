from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    LoopStep,
    ReducedRankFamily,
    VectorAffine,
    fit_reduced_rank_update_family,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_telomere_rejuvenator import (
    PositionwiseAffine,
    fit_positionwise_affine,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


@dataclass(frozen=True)
class JumpMode:
    name: str
    reference_age: int
    reference_path_before: int
    programmed_jump: int


@dataclass(frozen=True)
class JumpPairBatch:
    terminal: torch.Tensor
    one_target_state: torch.Tensor
    two_target_state: torch.Tensor
    trajectory_states: tuple[torch.Tensor, ...]
    all_targets: torch.Tensor


ONE_MODE = JumpMode(
    name="one",
    reference_age=5,
    reference_path_before=7,
    programmed_jump=1,
)
TWO_MODE = JumpMode(
    name="two",
    reference_age=0,
    reference_path_before=0,
    programmed_jump=2,
)


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


def _validate_modes(
    cfg: GraphPathConfig,
    *,
    two_mode: JumpMode = TWO_MODE,
) -> None:
    if cfg.max_depth != 8 or cfg.max_loops != 6:
        raise ValueError("jump-controller experiment requires D8L6")
    if cfg.d_model != 256 or cfg.n_layers != 2 or cfg.n_heads != 4:
        raise ValueError(
            "jump-controller experiment requires d256, two blocks, four heads"
        )
    if cfg.block_schedule != "all_blocks":
        raise ValueError("the compared transition requires all_blocks schedule")
    if (
        two_mode.reference_age,
        two_mode.reference_path_before,
    ) not in {(0, 0), (1, 2)}:
        raise ValueError(
            "two-hop phase must be age/path (0,0) or (1,2) for D8L6 seed6"
        )
    for mode in (ONE_MODE, two_mode):
        if mode.reference_path_before + mode.programmed_jump > cfg.max_depth + 2:
            raise ValueError("mode target lies outside the generated path")


def controller_position_groups(
    cfg: GraphPathConfig,
) -> dict[str, tuple[int, ...]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    query_work = groups["query_metadata"] + groups["answer"]
    registers = groups["start"] + groups["depth"] + groups["answer"]
    return {
        "answer": groups["answer"],
        "registers": tuple(sorted(registers)),
        "query_work": tuple(sorted(query_work)),
        "graph_answer": tuple(sorted(groups["graph"] + groups["answer"])),
        "all": tuple(range(cfg.seq_len)),
    }


@torch.no_grad()
def _matched_target_state(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    successors: torch.Tensor,
    start: torch.Tensor,
    mode: JumpMode,
) -> torch.Tensor:
    reference_start = advance_nodes(
        successors,
        start,
        steps=cfg.max_depth - mode.reference_path_before,
    )
    reference_tokens, _, _, _ = fixed_depth_batch(
        cfg,
        start.shape[0],
        start.device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=reference_start,
    )
    states = cache_states_with_initial(
        model,
        reference_tokens,
        loops=max(1, mode.reference_age),
    )
    return states[mode.reference_age]


@torch.no_grad()
def collect_jump_pair_batch(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch_size: int,
    device: torch.device,
    two_mode: JumpMode = TWO_MODE,
) -> JumpPairBatch:
    tokens, path_targets, successors, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth + two_mode.programmed_jump,
    )
    trajectory = tuple(
        cache_states_with_initial(model, tokens, loops=cfg.max_loops)
    )
    return JumpPairBatch(
        terminal=trajectory[-1],
        one_target_state=_matched_target_state(
            model=model,
            cfg=cfg,
            successors=successors,
            start=start,
            mode=ONE_MODE,
        ),
        two_target_state=_matched_target_state(
            model=model,
            cfg=cfg,
            successors=successors,
            start=start,
            mode=two_mode,
        ),
        trajectory_states=trajectory,
        all_targets=_all_targets(start, path_targets),
    )


def _flatten_positions(
    state: torch.Tensor,
    positions: tuple[int, ...],
) -> torch.Tensor:
    return state[:, list(positions)].reshape(-1, state.shape[-1])


def fit_full_linear_no_bias(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> VectorAffine:
    """Fit a residual linear map with the affine bias fixed to zero."""

    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must share [sample, feature] shape")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    source = source.float()
    target = target.float()
    gram = source.transpose(0, 1) @ source
    cross = source.transpose(0, 1) @ (target - source)
    scale = gram.diagonal().mean().clamp_min(1e-6)
    update = torch.linalg.solve(
        gram
        + ridge
        * scale
        * torch.eye(
            source.shape[1],
            device=source.device,
            dtype=source.dtype,
        ),
        cross,
    )
    weight = (
        torch.eye(
            source.shape[1],
            device=source.device,
            dtype=source.dtype,
        )
        + update
    )
    return VectorAffine(
        weight=weight,
        bias=torch.zeros(
            source.shape[1],
            device=source.device,
            dtype=source.dtype,
        ),
        update_rank=source.shape[1],
        fit_dimension=source.shape[1],
        retained_fit_energy=1.0,
    )


def apply_vector_map(
    state: torch.Tensor,
    *,
    positions: tuple[int, ...],
    controller: VectorAffine,
) -> torch.Tensor:
    if not positions:
        raise ValueError("positions must not be empty")
    result = state.clone()
    index = list(positions)
    result[:, index] = controller(state[:, index]).to(dtype=state.dtype)
    return result


def apply_positionwise_map(
    state: torch.Tensor,
    *,
    positions: tuple[int, ...],
    controller: PositionwiseAffine,
) -> torch.Tensor:
    if not positions:
        raise ValueError("positions must not be empty")
    result = state.clone()
    index = list(positions)
    result[:, index] = controller(state[:, index]).to(dtype=state.dtype)
    return result


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.float() - target.float()).square().mean()
    centered = target.float() - target.float().mean(dim=0, keepdim=True)
    denominator = centered.square().mean().clamp_min(1e-12)
    return float(numerator / denominator)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().flatten(1),
            right.float().flatten(1),
            dim=-1,
        ).mean()
    )


def behavior_metrics(
    logits: torch.Tensor,
    all_targets: torch.Tensor,
    *,
    endpoint_position: int,
) -> dict[str, float | int]:
    endpoint = all_targets[:, endpoint_position]
    one = all_targets[:, endpoint_position + 1]
    two = all_targets[:, endpoint_position + 2]
    result: dict[str, float | int] = {}
    for name, target in (
        ("endpoint", endpoint),
        ("one", one),
        ("two", two),
    ):
        metrics = _masked_metrics(
            logits,
            target,
            endpoint=None if name == "endpoint" else endpoint,
        )
        for metric, value in metrics.items():
            result[f"{name}_{metric}"] = value

    distinct = endpoint.ne(one) & endpoint.ne(two) & one.ne(two)
    result["distinct_count"] = int(distinct.sum())
    if bool(distinct.any()):
        chosen = logits[distinct]
        one_target = one[distinct]
        two_target = two[distinct]
        prediction = chosen.argmax(dim=-1)
        probability = chosen.softmax(dim=-1)
        one_logits = chosen.gather(1, one_target[:, None]).squeeze(1)
        two_logits = chosen.gather(1, two_target[:, None]).squeeze(1)
        result.update(
            {
                "distinct_one_accuracy": float(
                    prediction.eq(one_target).float().mean()
                ),
                "distinct_two_accuracy": float(
                    prediction.eq(two_target).float().mean()
                ),
                "distinct_one_probability": float(
                    probability.gather(1, one_target[:, None]).mean()
                ),
                "distinct_two_probability": float(
                    probability.gather(1, two_target[:, None]).mean()
                ),
                "distinct_one_minus_two_logit": float(
                    (one_logits - two_logits).mean()
                ),
            }
        )
    else:
        result.update(
            {
                "distinct_one_accuracy": float("nan"),
                "distinct_two_accuracy": float("nan"),
                "distinct_one_probability": float("nan"),
                "distinct_two_probability": float("nan"),
                "distinct_one_minus_two_logit": float("nan"),
            }
        )
    return result


def component_similarity(
    step: LoopStep,
    oracle: LoopStep,
    *,
    cfg: GraphPathConfig,
    prefix: str,
) -> dict[str, float]:
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"][0]
    destinations = list(groups["destination"])
    result: dict[str, float] = {}
    for head in range(cfg.n_heads):
        result.update(
            {
                f"{prefix}_h{head}_q_answer_cosine": _cosine(
                    step.block2_q[:, head, answer],
                    oracle.block2_q[:, head, answer],
                ),
                f"{prefix}_h{head}_k_destination_cosine": _cosine(
                    step.block2_k[:, head, destinations],
                    oracle.block2_k[:, head, destinations],
                ),
                f"{prefix}_h{head}_v_destination_cosine": _cosine(
                    step.block2_v[:, head, destinations],
                    oracle.block2_v[:, head, destinations],
                ),
                f"{prefix}_h{head}_pattern_answer_cosine": _cosine(
                    step.block2_pattern[:, head, answer],
                    oracle.block2_pattern[:, head, answer],
                ),
                f"{prefix}_h{head}_context_answer_cosine": _cosine(
                    step.block2_head_context[:, head, answer],
                    oracle.block2_head_context[:, head, answer],
                ),
            }
        )
    result.update(
        {
            f"{prefix}_attention_answer_cosine": _cosine(
                step.block2_attention_out[:, answer],
                oracle.block2_attention_out[:, answer],
            ),
            f"{prefix}_mlp_answer_cosine": _cosine(
                step.block2_mlp_out[:, answer],
                oracle.block2_mlp_out[:, answer],
            ),
            f"{prefix}_output_answer_cosine": _cosine(
                step.state[:, answer],
                oracle.state[:, answer],
            ),
        }
    )
    return result


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for name, group in groups.items():
        summary: dict[str, Any] = {key: name, "batches": len(group)}
        for field in group[0]:
            if field in {key, "batch"}:
                continue
            values = [row[field] for row in group]
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                if field.endswith("_count"):
                    summary[field] = int(sum(int(value) for value in values))
                else:
                    tensor = torch.tensor(values, dtype=torch.float64)
                    finite = torch.isfinite(tensor)
                    summary[field] = (
                        float(tensor[finite].mean())
                        if bool(finite.any())
                        else float("nan")
                    )
            elif all(value == values[0] for value in values):
                summary[field] = values[0]
        summaries.append(summary)
    return summaries


def _trajectory_rows(
    *,
    batch: JumpPairBatch,
    model: LoopedGraphPathTransformer,
    batch_index: int,
    endpoint_position: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for age, state in enumerate(batch.trajectory_states):
        logits = logits_from_raw_state(model, state)
        accuracies = []
        prediction = logits.argmax(dim=-1)
        for path_position in range(batch.all_targets.shape[1]):
            accuracies.append(
                float(
                    prediction.eq(batch.all_targets[:, path_position])
                    .float()
                    .mean()
                )
            )
        best = max(range(len(accuracies)), key=accuracies.__getitem__)
        row: dict[str, Any] = {
            "batch": batch_index,
            "age": age,
            "best_path_position": best,
            "best_raw_accuracy": accuracies[best],
        }
        row.update(
            behavior_metrics(
                logits,
                batch.all_targets,
                endpoint_position=endpoint_position,
            )
        )
        rows.append(row)
    return rows


def _map_payload(controller: VectorAffine) -> dict[str, Any]:
    return {
        "weight": controller.weight.detach().cpu(),
        "bias": controller.bias.detach().cpu(),
        "update_rank": controller.update_rank,
        "fit_dimension": controller.fit_dimension,
        "retained_fit_energy": controller.retained_fit_energy,
    }


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    eval_batch_size: int,
    eval_batches: int,
    ranks: tuple[int, ...],
    ridge: float,
    seed: int,
    eval_seed: int,
    two_reference_age: int = TWO_MODE.reference_age,
    two_reference_path_before: int = TWO_MODE.reference_path_before,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("JUMP_CONTROLLER_CUDA_MEMORY_FRACTION", "0.06")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    two_mode = JumpMode(
        name="two",
        reference_age=two_reference_age,
        reference_path_before=two_reference_path_before,
        programmed_jump=2,
    )
    _validate_modes(cfg, two_mode=two_mode)
    if not ranks or min(ranks) < 0 or max(ranks) > cfg.d_model:
        raise ValueError("ranks must lie between zero and d_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = controller_position_groups(cfg)["all"]

    set_seed(seed)
    calibration: dict[str, list[torch.Tensor]] = {
        "source": [],
        "one": [],
        "two": [],
    }
    for _ in range(calibration_batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=calibration_batch_size,
            device=device,
            two_mode=two_mode,
        )
        calibration["source"].append(pair.terminal)
        calibration["one"].append(pair.one_target_state)
        calibration["two"].append(pair.two_target_state)
    source_state = torch.cat(calibration["source"])
    target_states = {
        "one": torch.cat(calibration["one"]),
        "two": torch.cat(calibration["two"]),
    }
    source_flat = _flatten_positions(source_state, positions)

    families: dict[str, ReducedRankFamily] = {}
    rank_maps: dict[str, dict[int, VectorAffine]] = {}
    no_bias_maps: dict[str, VectorAffine] = {}
    positionwise_maps: dict[str, PositionwiseAffine] = {}
    fit_rows: list[dict[str, Any]] = []
    for mode in ("one", "two"):
        target_flat = _flatten_positions(target_states[mode], positions)
        family = fit_reduced_rank_update_family(
            source_flat,
            target_flat,
            ridge=ridge,
        )
        families[mode] = family
        rank_maps[mode] = {}
        for rank in ranks:
            controller = family.map_for_rank(rank)
            rank_maps[mode][rank] = controller
            fit_rows.append(
                {
                    "mode": mode,
                    "family": "shared_affine",
                    "rank": rank,
                    "bias": True,
                    "retained_fit_energy": controller.retained_fit_energy,
                    "calibration_relative_mse": _relative_mse(
                        controller(source_flat),
                        target_flat,
                    ),
                    "parameter_count": controller.factor_parameter_count,
                }
            )
        no_bias = fit_full_linear_no_bias(
            source_flat,
            target_flat,
            ridge=ridge,
        )
        no_bias_maps[mode] = no_bias
        fit_rows.append(
            {
                "mode": mode,
                "family": "shared_linear_no_bias",
                "rank": cfg.d_model,
                "bias": False,
                "retained_fit_energy": 1.0,
                "calibration_relative_mse": _relative_mse(
                    no_bias(source_flat),
                    target_flat,
                ),
                "parameter_count": cfg.d_model * cfg.d_model,
            }
        )
        positionwise = fit_positionwise_affine(
            source_state[:, list(positions)],
            target_states[mode][:, list(positions)],
            ridge=ridge,
        )
        positionwise_maps[mode] = positionwise
        fit_rows.append(
            {
                "mode": mode,
                "family": "positionwise_affine_upper_bound",
                "rank": cfg.d_model,
                "bias": True,
                "retained_fit_energy": 1.0,
                "calibration_relative_mse": _relative_mse(
                    positionwise(source_state[:, list(positions)]),
                    target_states[mode][:, list(positions)],
                ),
                "parameter_count": (
                    len(positions)
                    * (cfg.d_model * cfg.d_model + cfg.d_model)
                ),
            }
        )

    saved_maps: dict[str, Any] = {
        "shared": {
            mode: {
                str(rank): _map_payload(controller)
                for rank, controller in maps.items()
            }
            for mode, maps in rank_maps.items()
        },
        "no_bias": {
            mode: _map_payload(controller)
            for mode, controller in no_bias_maps.items()
        },
        "positionwise": {
            mode: {
                "weight": controller.weight.detach().cpu(),
                "bias": controller.bias.detach().cpu(),
            }
            for mode, controller in positionwise_maps.items()
        },
    }
    torch.save(saved_maps, out_dir / "controllers.pt")

    set_seed(eval_seed)
    behavior_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    position_groups = controller_position_groups(cfg)
    for batch_index in range(eval_batches):
        pair = collect_jump_pair_batch(
            model=model,
            cfg=cfg,
            batch_size=eval_batch_size,
            device=device,
            two_mode=two_mode,
        )
        trajectory_rows.extend(
            _trajectory_rows(
                batch=pair,
                model=model,
                batch_index=batch_index,
                endpoint_position=cfg.max_depth,
            )
        )

        one_oracle = run_one_loop(
            model,
            pair.one_target_state,
            loop_index=cfg.max_loops,
        )
        two_oracle = run_one_loop(
            model,
            pair.two_target_state,
            loop_index=cfg.max_loops,
        )

        condition_states: list[tuple[str, str, int, torch.Tensor]] = [
            ("identity", "control", -1, pair.terminal),
        ]
        for group_name, group_positions in position_groups.items():
            one_exact = pair.terminal.clone()
            two_exact = pair.terminal.clone()
            one_exact[:, list(group_positions)] = pair.one_target_state[
                :, list(group_positions)
            ]
            two_exact[:, list(group_positions)] = pair.two_target_state[
                :, list(group_positions)
            ]
            condition_states.extend(
                [
                    (
                        f"exact_{group_name}_one",
                        "exact_one",
                        -1,
                        one_exact,
                    ),
                    (
                        f"exact_{group_name}_two",
                        "exact_two",
                        -1,
                        two_exact,
                    ),
                ]
            )
        for mode in ("one", "two"):
            for rank in ranks:
                condition_states.append(
                    (
                        f"J_{mode}_rank{rank}",
                        f"learned_{mode}",
                        rank,
                        apply_vector_map(
                            pair.terminal,
                            positions=positions,
                            controller=rank_maps[mode][rank],
                        ),
                    )
                )
            condition_states.extend(
                [
                    (
                        f"J_{mode}_no_bias",
                        f"no_bias_{mode}",
                        cfg.d_model,
                        apply_vector_map(
                            pair.terminal,
                            positions=positions,
                            controller=no_bias_maps[mode],
                        ),
                    ),
                    (
                        f"J_{mode}_positionwise",
                        f"positionwise_{mode}",
                        cfg.d_model,
                        apply_positionwise_map(
                            pair.terminal,
                            positions=positions,
                            controller=positionwise_maps[mode],
                        ),
                    ),
                ]
            )
            mapped = apply_vector_map(
                pair.terminal,
                positions=positions,
                controller=rank_maps[mode][max(ranks)],
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(eval_seed + 1009 * batch_index + (0 if mode == "one" else 1))
            permutation = torch.randperm(
                mapped.shape[0],
                generator=generator,
                device=device,
            )
            shuffled = pair.terminal.clone()
            shuffled[:, list(positions)] = mapped[
                permutation
            ][:, list(positions)]
            condition_states.append(
                (
                    f"J_{mode}_rank{max(ranks)}_batch_shuffled",
                    f"shuffled_{mode}",
                    max(ranks),
                    shuffled,
                )
            )

        for condition, family_name, rank, state in condition_states:
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops,
            )
            row: dict[str, Any] = {
                "batch": batch_index,
                "condition": condition,
                "family": family_name,
                "rank": rank,
            }
            row.update(
                behavior_metrics(
                    step.logits,
                    pair.all_targets,
                    endpoint_position=cfg.max_depth,
                )
            )
            row.update(
                component_similarity(
                    step,
                    one_oracle,
                    cfg=cfg,
                    prefix="to_one_oracle",
                )
            )
            row.update(
                component_similarity(
                    step,
                    two_oracle,
                    cfg=cfg,
                    prefix="to_two_oracle",
                )
            )
            behavior_rows.append(row)

    condition_summary = _aggregate_rows(behavior_rows, key="condition")
    trajectory_summary = _aggregate_rows(trajectory_rows, key="age")
    condition_lookup = {
        str(row["condition"]): row for row in condition_summary
    }
    oracle_accuracy = {
        "one": float(condition_lookup["exact_all_one"]["one_accuracy"]),
        "two": float(condition_lookup["exact_all_two"]["two_accuracy"]),
    }
    lowest_successful_rank: dict[str, int | None] = {}
    for mode in ("one", "two"):
        intended = f"{mode}_accuracy"
        other = "two_accuracy" if mode == "one" else "one_accuracy"
        distinct_intended = f"distinct_{mode}_accuracy"
        distinct_other = (
            "distinct_two_accuracy"
            if mode == "one"
            else "distinct_one_accuracy"
        )
        successes = []
        for rank in ranks:
            row = condition_lookup[f"J_{mode}_rank{rank}"]
            row["intended_oracle_fraction"] = (
                float(row[intended]) / max(oracle_accuracy[mode], 1e-12)
            )
            row["selects_intended_on_distinct"] = bool(
                float(row[distinct_intended]) > float(row[distinct_other])
            )
            if (
                float(row["intended_oracle_fraction"]) >= 0.8
                and bool(row["selects_intended_on_distinct"])
                and float(row[intended]) > float(row[other])
            ):
                successes.append(rank)
        lowest_successful_rank[mode] = min(successes) if successes else None

    primary_conditions = {
        name: condition_lookup[name]
        for name in (
            "identity",
            "exact_all_one",
            "exact_all_two",
            f"J_one_rank{max(ranks)}",
            f"J_two_rank{max(ranks)}",
            "J_one_no_bias",
            "J_two_no_bias",
            "J_one_positionwise",
            "J_two_positionwise",
            f"J_one_rank{max(ranks)}_batch_shuffled",
            f"J_two_rank{max(ranks)}_batch_shuffled",
        )
    }
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "config": asdict(cfg),
        "loss_placement": "final_only",
        "trained_loop_count": cfg.max_loops,
        "evaluated_receiver_age": cfg.max_loops,
        "evaluated_extra_loops": 1,
        "shared_physical_blocks": cfg.n_layers,
        "effective_depth_at_training_horizon": cfg.n_layers * cfg.max_loops,
        "tested_transition_active_blocks": model.active_block_indices(
            cfg.max_loops
        ),
        "modes": {
            "one": asdict(ONE_MODE),
            "two": asdict(two_mode),
        },
        "positions": list(positions),
        "calibration_examples": calibration_batch_size * calibration_batches,
        "evaluation_examples": eval_batch_size * eval_batches,
        "calibration_seed": seed,
        "evaluation_seed": eval_seed,
        "ridge": ridge,
        "ranks": list(ranks),
        "oracle_accuracy": oracle_accuracy,
        "lowest_successful_shared_rank": lowest_successful_rank,
        "strong_same_source_switch": all(
            value is not None for value in lowest_successful_rank.values()
        ),
        "primary_conditions": primary_conditions,
        "fit_rows": fit_rows,
        "trajectory_summary": trajectory_summary,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    _write_csv(out_dir / "fit_rows.csv", fit_rows)
    _write_csv(out_dir / "behavior_per_batch.csv", behavior_rows)
    _write_csv(out_dir / "condition_summary.csv", condition_summary)
    _write_csv(out_dir / "trajectory_per_batch.csv", trajectory_rows)
    _write_csv(out_dir / "trajectory_summary.csv", trajectory_summary)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit two affine controllers that select one-hop versus two-hop "
            "behavior from the same frozen D8L6 terminal state."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[0, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6301)
    parser.add_argument("--eval-seed", type=int, default=6401)
    parser.add_argument("--two-reference-age", type=int, default=0)
    parser.add_argument("--two-reference-path-before", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        eval_batch_size=args.eval_batch_size,
        eval_batches=args.eval_batches,
        ranks=tuple(sorted(set(args.ranks))),
        ridge=args.ridge,
        seed=args.seed,
        eval_seed=args.eval_seed,
        two_reference_age=args.two_reference_age,
        two_reference_path_before=args.two_reference_path_before,
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
