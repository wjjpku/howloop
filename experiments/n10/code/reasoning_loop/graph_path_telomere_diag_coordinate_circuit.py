from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import (
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_diag_rank48_circuit import _metrics
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


Mode = Literal["corrupt", "restore"]


@dataclass(frozen=True)
class ScaleCondition:
    label: str
    family: str
    mode: str
    coordinate_count: int
    coordinates: tuple[int, ...]
    scale: torch.Tensor
    random_draw: int | None = None


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _coordinate_scale_bank(
    *,
    full: torch.Tensor,
    damaged: torch.Tensor,
    coordinates: torch.Tensor,
    mode: Mode,
) -> torch.Tensor:
    if full.ndim != 1 or damaged.shape != full.shape:
        raise ValueError("full and damaged scales must be matching vectors")
    coordinates = coordinates.to(device=full.device, dtype=torch.long)
    if coordinates.ndim != 1:
        raise ValueError("coordinates must be a vector")
    if mode == "corrupt":
        result = full[None].repeat(coordinates.numel(), 1)
        result[torch.arange(coordinates.numel(), device=full.device), coordinates] = (
            damaged[coordinates]
        )
        return result
    if mode == "restore":
        result = damaged[None].repeat(coordinates.numel(), 1)
        result[torch.arange(coordinates.numel(), device=full.device), coordinates] = (
            full[coordinates]
        )
        return result
    raise ValueError(f"unsupported coordinate mode: {mode}")


def _position_restricted_scale(
    *,
    full: torch.Tensor,
    damaged: torch.Tensor,
    position_count: int,
    positions: Sequence[int],
    coordinates: torch.Tensor,
    mode: Mode,
) -> torch.Tensor:
    if position_count < 1:
        raise ValueError("position_count must be positive")
    position_index = torch.as_tensor(positions, device=full.device, dtype=torch.long)
    coordinate_index = coordinates.to(device=full.device, dtype=torch.long)
    if bool((position_index < 0).any()) or bool((position_index >= position_count).any()):
        raise ValueError("position index out of range")
    base, replacement = (
        (full, damaged) if mode == "corrupt" else (damaged, full)
    )
    result = base[None].repeat(position_count, 1)
    result[position_index[:, None], coordinate_index[None, :]] = replacement[
        coordinate_index
    ]
    return result


def _apply_scale_bank(
    value: torch.Tensor,
    *,
    scale_bank: torch.Tensor,
    examples_per_condition: int,
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    condition_count = scale_bank.shape[0]
    if value.shape[0] != condition_count * examples_per_condition:
        raise ValueError("value batch does not match condition bank")
    reshaped = value.float().reshape(
        condition_count, examples_per_condition, *value.shape[1:]
    )
    if scale_bank.ndim == 2:
        scale = scale_bank[:, None, None, :]
    elif scale_bank.ndim == 3:
        if scale_bank.shape[1] != value.shape[1]:
            raise ValueError("position-specific scale has wrong position count")
        scale = scale_bank[:, None, :, :]
    else:
        raise ValueError("scale bank must be [condition, d] or [condition, position, d]")
    output = reshaped * scale + (reshaped @ left) @ right + bias
    return output.reshape_as(value)


def _shuffled_diagonal(
    diagonal: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=diagonal.device).manual_seed(seed)
    permutation = torch.randperm(
        diagonal.numel(), generator=generator, device=diagonal.device
    )
    return diagonal[permutation], permutation


def _scale_operator(
    *,
    operator: DiagonalIdentityLoRAJ,
    scale_bank: torch.Tensor,
    examples_per_condition: int,
):
    left = operator.A.detach().float()
    right = operator.B.detach().float()
    bias = operator.bias.detach().float()

    def apply(value: torch.Tensor) -> torch.Tensor:
        return _apply_scale_bank(
            value,
            scale_bank=scale_bank,
            examples_per_condition=examples_per_condition,
            left=left,
            right=right,
            bias=bias,
        )

    return apply


def _trajectory_bank(
    *,
    model,
    cfg,
    initial: torch.Tensor,
    endpoint: torch.Tensor,
    successors: torch.Tensor,
    jump: int,
    positions: tuple[int, ...],
    operator: DiagonalIdentityLoRAJ,
    scale_bank: torch.Tensor,
    requested_cycles: Sequence[int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    requested = set(int(cycle) for cycle in requested_cycles)
    if not requested or min(requested) < 1:
        raise ValueError("requested cycles must be positive")
    condition_count = scale_bank.shape[0]
    batch = initial.shape[0]
    trajectory = initial[None].expand(condition_count, -1, -1, -1).reshape(
        condition_count * batch, *initial.shape[1:]
    ).clone()
    apply = _scale_operator(
        operator=operator,
        scale_bank=scale_bank,
        examples_per_condition=batch,
    )
    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for cycle in range(1, max(requested) + 1):
        current_target = advance_nodes(successors, endpoint, steps=jump * cycle)
        repeated_target = current_target[None].expand(condition_count, -1).reshape(-1)
        step = _controlled_loop(
            loop_runner=run_one_loop,
            model=model,
            state=trajectory,
            loop_index=cfg.max_loops + cycle - 1,
            positions=positions,
            operator=apply,
            placement="loop_boundary",
        )
        trajectory = step.state
        if cycle in requested:
            result[cycle] = (
                step.logits.reshape(condition_count, batch, -1),
                repeated_target.reshape(condition_count, batch),
            )
    return result


def _rows_for_bank(
    *,
    outputs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    conditions: Sequence[ScaleCondition],
    split: str,
    successors: torch.Tensor,
    endpoint: torch.Tensor,
    jump: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    for cycle, (logits, target) in outputs.items():
        current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
        for index, condition in enumerate(conditions):
            rows.append(
                {
                    "cycle": cycle,
                    "effective_loop": cycle + 8,
                    "condition": condition.label,
                    "family": condition.family,
                    "mode": condition.mode,
                    "coordinate_count": condition.coordinate_count,
                    "coordinates": " ".join(str(v) for v in condition.coordinates),
                    "random_draw": condition.random_draw,
                    **_metrics(logits[index], target[index]),
                }
            )
            prediction = logits[index].argmax(dim=-1)
            margin = target_margin(logits[index].float(), target[index])
            for sample in range(target.shape[1]):
                per_example.append(
                    {
                        "split": split,
                        "cycle": cycle,
                        "effective_loop": cycle + 8,
                        "sample": sample,
                        "condition": condition.label,
                        "family": condition.family,
                        "mode": condition.mode,
                        "coordinate_count": condition.coordinate_count,
                        "coordinates": " ".join(
                            str(value) for value in condition.coordinates
                        ),
                        "random_draw": condition.random_draw,
                        "successors": " ".join(
                            str(int(value))
                            for value in successors[sample].tolist()
                        ),
                        "current": int(current[sample]),
                        "target": int(target[index, sample]),
                        "prediction": int(prediction[sample]),
                        "correct": int(prediction[sample].eq(target[index, sample])),
                        "target_margin": float(margin[sample]),
                    }
                )
    return rows, per_example


def _weight_loading_rows(model, operator: DiagonalIdentityLoRAJ, shuffled: torch.Tensor, permutation: torch.Tensor) -> list[dict[str, Any]]:
    dimension = operator.diagonal_scale.numel()
    rows: list[dict[str, Any]] = [
        {
            "coordinate": index,
            "D": float(operator.diagonal_scale[index]),
            "shuffled_D": float(shuffled[index]),
            "shuffled_source_coordinate": int(permutation[index]),
            "abs_D_minus_1": float((operator.diagonal_scale[index] - 1.0).abs()),
            "abs_D_minus_shuffled": float((operator.diagonal_scale[index] - shuffled[index]).abs()),
            "bias_abs": float(operator.bias[index].abs()),
            "AB_output_l2": float((operator.A @ operator.B)[:, index].float().norm()),
        }
        for index in range(dimension)
    ]
    for block_index, block in enumerate(model.blocks):
        if not isinstance(block, TransformerBlock):
            raise TypeError("coordinate audit requires legacy TransformerBlock")
        qkv = block.attn.qkv.weight.detach().float()
        gamma1 = block.ln_1.weight.detach().float()
        gamma2 = block.ln_2.weight.detach().float()
        for head in range(block.attn.n_heads):
            start = head * block.attn.d_head
            stop = start + block.attn.d_head
            for part_index, part in enumerate(("q", "k", "v")):
                offset = part_index * dimension
                loading = (qkv[offset + start : offset + stop] * gamma1[None]).norm(dim=0)
                for coordinate in range(dimension):
                    rows[coordinate][f"B{block_index + 1}H{head}_{part}_input_l2"] = float(loading[coordinate])
        mlp_loading = (block.mlp[0].weight.detach().float() * gamma2[None]).norm(dim=0)
        for coordinate in range(dimension):
            rows[coordinate][f"B{block_index + 1}_mlp_input_l2"] = float(mlp_loading[coordinate])
    final_gamma = getattr(model.ln_final, "weight", torch.ones(dimension, device=shuffled.device)).detach().float()
    readout_loading = (model.unembed.weight[: model.cfg.node_count].detach().float() * final_gamma[None]).norm(dim=0)
    for coordinate in range(dimension):
        rows[coordinate]["readout_input_l2"] = float(readout_loading[coordinate])
    return rows


def _initialization_audit(
    *,
    artifact_payload: dict[str, Any],
    operator: DiagonalIdentityLoRAJ,
    device: torch.device,
) -> tuple[dict[str, float | str], torch.Tensor]:
    path = Path(str(artifact_payload["reference_affine_artifact"]))
    label = str(artifact_payload["reference_affine_label"])
    payload = torch.load(path, map_location=device, weights_only=False)
    reference = payload["maps"][label]
    weight = reference["weight"].detach().float()
    initial_bias = reference["bias"].detach().float()
    initial_diagonal = weight.diag()
    initial_off_diagonal = weight - torch.diag(initial_diagonal)
    left, singular_values, right_t = torch.linalg.svd(
        initial_off_diagonal, full_matrices=False
    )
    initial_rank48 = (
        left[:, : operator.rank] * singular_values[: operator.rank]
    ) @ right_t[: operator.rank]
    final_low_rank = operator.A.detach().float() @ operator.B.detach().float()
    final_diagonal = operator.diagonal_scale.detach().float()
    final_bias = operator.bias.detach().float()
    return (
        {
            "reference_affine_artifact": str(path),
            "reference_affine_label": label,
            "D_initial_mean": float(initial_diagonal.mean()),
            "D_initial_std": float(initial_diagonal.std()),
            "D_final_mean": float(final_diagonal.mean()),
            "D_final_std": float(final_diagonal.std()),
            "D_training_delta_l2": float((final_diagonal - initial_diagonal).norm()),
            "D_training_delta_max_abs": float(
                (final_diagonal - initial_diagonal).abs().max()
            ),
            "D_training_delta_relative_l2": float(
                (final_diagonal - initial_diagonal).norm()
                / initial_diagonal.norm().clamp_min(1e-12)
            ),
            "D_initial_final_cosine": float(
                torch.nn.functional.cosine_similarity(
                    initial_diagonal, final_diagonal, dim=0
                )
            ),
            "bias_training_delta_relative_l2": float(
                (final_bias - initial_bias).norm()
                / initial_bias.norm().clamp_min(1e-12)
            ),
            "initial_offdiagonal_energy_retained_rank48": float(
                initial_rank48.square().sum()
                / initial_off_diagonal.square().sum().clamp_min(1e-12)
            ),
            "AB_training_delta_from_initial_rank48_relative_l2": float(
                (final_low_rank - initial_rank48).norm()
                / initial_rank48.norm().clamp_min(1e-12)
            ),
            "final_J_delta_from_reference_affine_relative_l2": float(
                (torch.diag(final_diagonal) + final_low_rank - weight).norm()
                / weight.norm().clamp_min(1e-12)
            ),
        },
        initial_diagonal,
    )


@torch.no_grad()
def _trajectory_coordinate_signal(
    *,
    model,
    cfg,
    initial: torch.Tensor,
    endpoint: torch.Tensor,
    successors: torch.Tensor,
    jump: int,
    positions: tuple[int, ...],
    operator: DiagonalIdentityLoRAJ,
    shuffled: torch.Tensor,
    max_cycle: int,
) -> dict[str, torch.Tensor]:
    trajectory = initial.clone()
    diagonal_delta = (operator.diagonal_scale.detach().float() - shuffled).abs()
    answer_sum = torch.zeros_like(diagonal_delta)
    graph_sum = torch.zeros_like(diagonal_delta)
    all_sum = torch.zeros_like(diagonal_delta)
    count = 0
    full_apply = lambda value: operator(value)
    graph_positions = explicit_depth_position_groups(cfg.node_count)["graph"]
    graph_relative = [positions.index(position) for position in graph_positions]
    answer_relative = positions.index(cfg.seq_len - 1)
    for cycle in range(1, max_cycle + 1):
        selected = trajectory[:, list(positions)].float().abs() * diagonal_delta
        answer_sum += selected[:, answer_relative].mean(dim=0)
        graph_sum += selected[:, graph_relative].mean(dim=(0, 1))
        all_sum += selected.mean(dim=(0, 1))
        count += 1
        step = _controlled_loop(
            loop_runner=run_one_loop,
            model=model,
            state=trajectory,
            loop_index=cfg.max_loops + cycle - 1,
            positions=positions,
            operator=full_apply,
            placement="loop_boundary",
        )
        trajectory = step.state
    return {
        "trajectory_D_signal_answer": answer_sum / count,
        "trajectory_D_signal_graph": graph_sum / count,
        "trajectory_D_signal_all": all_sum / count,
    }


def _rankings(
    *,
    static_rows: list[dict[str, Any]],
    single_rows: list[dict[str, Any]],
    lookup_head: int,
    screen_cycle: int,
) -> dict[str, list[int]]:
    by_coordinate = {int(row["coordinate"]): row for row in static_rows}
    full_row = next(
        row
        for row in single_rows
        if row["cycle"] == screen_cycle and row["condition"] == "full"
    )
    single = {
        int(row["coordinates"]): float(full_row["target_margin"]) - float(row["target_margin"])
        for row in single_rows
        if row["cycle"] == screen_cycle and row["family"] == "single_coordinate"
    }

    def ordered(score) -> list[int]:
        return sorted(by_coordinate, key=lambda coordinate: score(by_coordinate[coordinate]), reverse=True)

    delta_key = "abs_D_minus_shuffled"
    q_key = f"B2H{lookup_head}_q_input_l2"
    k_key = f"B2H{lookup_head}_k_input_l2"
    v_key = f"B2H{lookup_head}_v_input_l2"
    return {
        "causal_single_margin": sorted(single, key=single.get, reverse=True),
        "abs_D_minus_1": ordered(lambda row: float(row["abs_D_minus_1"])),
        "abs_D_minus_shuffled": ordered(lambda row: float(row[delta_key])),
        "lookup_qk_weighted": ordered(
            lambda row: float(row[delta_key])
            * math.sqrt(float(row[q_key]) ** 2 + float(row[k_key]) ** 2)
        ),
        "lookup_v_weighted": ordered(
            lambda row: float(row[delta_key]) * float(row[v_key])
        ),
        "B2_mlp_weighted": ordered(
            lambda row: float(row[delta_key]) * float(row["B2_mlp_input_l2"])
        ),
        "trajectory_answer_signal": ordered(
            lambda row: float(row["trajectory_D_signal_answer"])
        ),
        "trajectory_graph_signal": ordered(
            lambda row: float(row["trajectory_D_signal_graph"])
        ),
    }


def _group_conditions(
    *,
    full: torch.Tensor,
    damaged: torch.Tensor,
    rankings: dict[str, list[int]],
    counts: Sequence[int],
    random_draws: int,
    random_seed: int,
) -> list[ScaleCondition]:
    result: list[ScaleCondition] = []
    for family, ranking in rankings.items():
        for requested in counts:
            count = min(int(requested), full.numel())
            coordinates = tuple(int(value) for value in ranking[:count])
            index = torch.as_tensor(coordinates, device=full.device)
            for mode in ("restore", "corrupt"):
                scale = _position_restricted_scale(
                    full=full,
                    damaged=damaged,
                    position_count=1,
                    positions=(0,),
                    coordinates=index,
                    mode=mode,
                )[0]
                result.append(
                    ScaleCondition(
                        label=f"{mode}.{family}.k{count}",
                        family=family,
                        mode=mode,
                        coordinate_count=count,
                        coordinates=coordinates,
                        scale=scale,
                    )
                )
    generator = torch.Generator(device=full.device).manual_seed(random_seed)
    for draw in range(random_draws):
        ranking = torch.randperm(full.numel(), generator=generator, device=full.device)
        for requested in counts:
            count = min(int(requested), full.numel())
            coordinates = tuple(int(value) for value in ranking[:count])
            index = torch.as_tensor(coordinates, device=full.device)
            for mode in ("restore", "corrupt"):
                scale = _position_restricted_scale(
                    full=full,
                    damaged=damaged,
                    position_count=1,
                    positions=(0,),
                    coordinates=index,
                    mode=mode,
                )[0]
                result.append(
                    ScaleCondition(
                        label=f"{mode}.random.k{count}.draw{draw}",
                        family="random",
                        mode=mode,
                        coordinate_count=count,
                        coordinates=coordinates,
                        scale=scale,
                        random_draw=draw,
                    )
                )
    return result


def _position_conditions(
    *,
    cfg,
    controlled_positions: tuple[int, ...],
    full: torch.Tensor,
    damaged: torch.Tensor,
    ranking: Sequence[int],
    counts: Sequence[int],
) -> list[ScaleCondition]:
    groups = explicit_depth_position_groups(cfg.node_count)
    group_positions = {
        "answer": groups["answer"],
        "graph": groups["graph"],
        "metadata": tuple(
            position
            for position in controlled_positions
            if position not in groups["answer"] and position not in groups["graph"]
        ),
        "answer_plus_graph": groups["answer"] + groups["graph"],
        "all": controlled_positions,
    }
    relative = {
        name: tuple(controlled_positions.index(position) for position in positions)
        for name, positions in group_positions.items()
    }
    result: list[ScaleCondition] = []
    for requested in counts:
        count = min(int(requested), full.numel())
        coordinates = tuple(int(value) for value in ranking[:count])
        index = torch.as_tensor(coordinates, device=full.device)
        for token_group, positions in relative.items():
            for mode in ("restore", "corrupt"):
                scale = _position_restricted_scale(
                    full=full,
                    damaged=damaged,
                    position_count=len(controlled_positions),
                    positions=positions,
                    coordinates=index,
                    mode=mode,
                )
                result.append(
                    ScaleCondition(
                        label=f"{mode}.{token_group}.k{count}",
                        family=token_group,
                        mode=mode,
                        coordinate_count=count,
                        coordinates=coordinates,
                        scale=scale,
                    )
                )
    return result


def _run_condition_chunks(
    *,
    conditions: Sequence[ScaleCondition],
    chunk_size: int,
    model,
    cfg,
    initial: torch.Tensor,
    endpoint: torch.Tensor,
    successors: torch.Tensor,
    jump: int,
    positions: tuple[int, ...],
    operator: DiagonalIdentityLoRAJ,
    cycles: Sequence[int],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    for offset in range(0, len(conditions), chunk_size):
        current = conditions[offset : offset + chunk_size]
        scales = torch.stack([condition.scale for condition in current])
        outputs = _trajectory_bank(
            model=model,
            cfg=cfg,
            initial=initial,
            endpoint=endpoint,
            successors=successors,
            jump=jump,
            positions=positions,
            operator=operator,
            scale_bank=scales,
            requested_cycles=cycles,
        )
        current_rows, current_examples = _rows_for_bank(
            outputs=outputs,
            conditions=current,
            split=split,
            successors=successors,
            endpoint=endpoint,
            jump=jump,
        )
        rows.extend(current_rows)
        per_example.extend(current_examples)
        print(
            json.dumps(
                {
                    "event": "condition_chunk_complete",
                    "split": split,
                    "conditions_complete": min(offset + chunk_size, len(conditions)),
                    "conditions_total": len(conditions),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return rows, per_example


def _make_batch(*, model, cfg, batch_size: int, device: torch.device, seed: int, phase_positions: Sequence[int]):
    set_seed(seed)
    _, _, successors, _ = fixed_depth_batch(
        cfg, batch_size, device, path_positions=cfg.max_depth
    )
    current = torch.randint(0, cfg.node_count, (batch_size,), device=device)
    endpoint = current.clone()
    initial = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=endpoint,
        age=8,
        phase_position=int(phase_positions[8]),
    )
    return initial, endpoint, successors


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    manifest = {
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "checkpoint": str(args.checkpoint),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "physical_gpu": args.physical_gpu,
        "prelaunch_used_mib": args.prelaunch_used_mib,
        "prelaunch_free_mib": args.prelaunch_free_mib,
        "declared_peak_gib": args.declared_peak_gib,
        "reserve_gib": args.reserve_gib,
        "shared_gpu": args.shared_gpu,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, loaded, payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if payload.get("placement") != "loop_boundary":
        raise ValueError("coordinate audit requires loop-boundary J")
    operator = loaded[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("coordinate audit requires diagonal rank-48 J")
    if tuple(positions) != tuple(range(cfg.seq_len)):
        raise ValueError("coordinate audit currently requires all-position J")
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase["trajectory_positions_including_initial"]]
    jump = phase_positions[3] - phase_positions[2]
    full = operator.diagonal_scale.detach().float()
    damaged, permutation = _shuffled_diagonal(full, seed=args.shuffle_seed)
    initialization_audit, initial_diagonal = _initialization_audit(
        artifact_payload=payload,
        operator=operator,
        device=device,
    )

    screen_initial, screen_endpoint, screen_successors = _make_batch(
        model=model,
        cfg=cfg,
        batch_size=args.screen_batch_size,
        device=device,
        seed=args.screen_seed,
        phase_positions=phase_positions,
    )
    static_rows = _weight_loading_rows(model, operator, damaged, permutation)
    signals = _trajectory_coordinate_signal(
        model=model,
        cfg=cfg,
        initial=screen_initial,
        endpoint=screen_endpoint,
        successors=screen_successors,
        jump=jump,
        positions=positions,
        operator=operator,
        shuffled=damaged,
        max_cycle=args.screen_cycle,
    )
    for row in static_rows:
        coordinate = int(row["coordinate"])
        row["initial_D"] = float(initial_diagonal[coordinate])
        row["D_training_delta"] = float(full[coordinate] - initial_diagonal[coordinate])
        for name, value in signals.items():
            row[name] = float(value[coordinate])

    baseline_conditions = [
        ScaleCondition("full", "baseline", "full", full.numel(), tuple(range(full.numel())), full),
        ScaleCondition("shuffled", "baseline", "shuffled", 0, (), damaged),
    ]
    single_conditions: list[ScaleCondition] = []
    for start in range(0, full.numel(), args.condition_chunk):
        coordinates = torch.arange(
            start, min(start + args.condition_chunk, full.numel()), device=device
        )
        scale_bank = _coordinate_scale_bank(
            full=full, damaged=damaged, coordinates=coordinates, mode="corrupt"
        )
        for local, coordinate in enumerate(coordinates.tolist()):
            single_conditions.append(
                ScaleCondition(
                    label=f"corrupt.coordinate.{coordinate}",
                    family="single_coordinate",
                    mode="corrupt",
                    coordinate_count=1,
                    coordinates=(int(coordinate),),
                    scale=scale_bank[local],
                )
            )
    single_rows, single_examples = _run_condition_chunks(
        conditions=baseline_conditions + single_conditions,
        chunk_size=args.condition_chunk,
        model=model,
        cfg=cfg,
        initial=screen_initial,
        endpoint=screen_endpoint,
        successors=screen_successors,
        jump=jump,
        positions=positions,
        operator=operator,
        cycles=args.screen_cycles,
        split="discovery",
    )
    rankings = _rankings(
        static_rows=static_rows,
        single_rows=single_rows,
        lookup_head=args.lookup_head,
        screen_cycle=args.screen_cycle,
    )

    eval_initial, eval_endpoint, eval_successors = _make_batch(
        model=model,
        cfg=cfg,
        batch_size=args.eval_batch_size,
        device=device,
        seed=args.eval_seed,
        phase_positions=phase_positions,
    )
    group_conditions = baseline_conditions + _group_conditions(
        full=full,
        damaged=damaged,
        rankings=rankings,
        counts=args.counts,
        random_draws=args.random_draws,
        random_seed=args.random_seed,
    )
    group_rows, group_examples = _run_condition_chunks(
        conditions=group_conditions,
        chunk_size=args.eval_condition_chunk,
        model=model,
        cfg=cfg,
        initial=eval_initial,
        endpoint=eval_endpoint,
        successors=eval_successors,
        jump=jump,
        positions=positions,
        operator=operator,
        cycles=args.eval_cycles,
        split="validation",
    )
    position_family = args.position_family
    if position_family not in rankings:
        raise ValueError(f"unknown position family: {position_family}")
    position_baselines = [
        ScaleCondition(
            "full",
            "baseline",
            "full",
            full.numel(),
            tuple(range(full.numel())),
            full[None].repeat(len(positions), 1),
        ),
        ScaleCondition(
            "shuffled",
            "baseline",
            "shuffled",
            0,
            (),
            damaged[None].repeat(len(positions), 1),
        ),
    ]
    position_conditions = position_baselines + _position_conditions(
        cfg=cfg,
        controlled_positions=positions,
        full=full,
        damaged=damaged,
        ranking=rankings[position_family],
        counts=args.position_counts,
    )
    position_rows, position_examples = _run_condition_chunks(
        conditions=position_conditions,
        chunk_size=args.eval_condition_chunk,
        model=model,
        cfg=cfg,
        initial=eval_initial,
        endpoint=eval_endpoint,
        successors=eval_successors,
        jump=jump,
        positions=positions,
        operator=operator,
        cycles=args.eval_cycles,
        split="validation",
    )

    for family, ranking in rankings.items():
        for rank_index, coordinate in enumerate(ranking):
            static_rows[coordinate][f"rank_{family}"] = rank_index + 1
    _write(args.out_dir / "coordinate_static.csv", static_rows)
    _write(args.out_dir / "single_coordinate.csv", single_rows)
    _write(args.out_dir / "single_coordinate_per_example.csv", single_examples)
    _write(args.out_dir / "group_curves.csv", group_rows)
    _write(args.out_dir / "group_curves_per_example.csv", group_examples)
    _write(args.out_dir / "position_curves.csv", position_rows)
    _write(args.out_dir / "position_curves_per_example.csv", position_examples)
    ranking_payload = {family: ranking for family, ranking in rankings.items()}
    (args.out_dir / "rankings.json").write_text(
        json.dumps(ranking_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "initialization_audit.json").write_text(
        json.dumps(initialization_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_cycles": list(args.eval_cycles),
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": operator.rank,
        "controller_parameters": operator.parameter_count,
        "controller_placement": "loop boundary after Block2 FFN",
        "controller_training_loss": "successor CE at every controlled continuation loop; no hidden MSE",
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "lookup_head": args.lookup_head,
        "screen_examples": args.screen_batch_size,
        "screen_seed": args.screen_seed,
        "evaluation_examples": args.eval_batch_size,
        "evaluation_seed": args.eval_seed,
        "shuffle_seed": args.shuffle_seed,
        "random_draws": args.random_draws,
        "selection_families": list(rankings),
        "position_family": position_family,
        "initialization_audit": initialization_audit,
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "declared_peak_gib": args.declared_peak_gib,
            "reserve_gib": args.reserve_gib,
            "shared_gpu": args.shared_gpu,
            "cuda_memory_fraction": args.cuda_memory_fraction,
        },
        "gpu_peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_peak_allocated_gib": result["gpu_peak_allocated_gib"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal coordinate audit of the diagonal branch in rank-48 loop-boundary J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lookup-head", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screen-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--screen-cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument("--screen-cycle", type=int, default=64)
    parser.add_argument("--eval-cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument("--counts", type=int, nargs="+", default=(1, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256))
    parser.add_argument("--position-counts", type=int, nargs="+", default=(16, 32, 48, 64, 96, 128, 192, 256))
    parser.add_argument("--position-family", default="causal_single_margin")
    parser.add_argument("--condition-chunk", type=int, default=8)
    parser.add_argument("--eval-condition-chunk", type=int, default=2)
    parser.add_argument("--random-draws", type=int, default=5)
    parser.add_argument("--screen-seed", type=int, default=20260811)
    parser.add_argument("--eval-seed", type=int, default=20260812)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
    parser.add_argument("--random-seed", type=int, default=20260813)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.045)
    parser.add_argument("--physical-gpu", type=int, default=None)
    parser.add_argument("--prelaunch-used-mib", type=int, default=None)
    parser.add_argument("--prelaunch-free-mib", type=int, default=None)
    parser.add_argument("--declared-peak-gib", type=float, default=None)
    parser.add_argument("--reserve-gib", type=float, default=None)
    parser.add_argument(
        "--shared-gpu", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))
