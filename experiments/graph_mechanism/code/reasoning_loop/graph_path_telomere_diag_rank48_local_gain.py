from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_diag_rank48_branch_mediation import (
    _variants,
)
from reasoning_loop.graph_path_telomere_diag_rank48_circuit import _metrics
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


Operator = Callable[[torch.Tensor], torch.Tensor]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _direction_families(
    *,
    operator: DiagonalIdentityLoRAJ,
    state: torch.Tensor,
    exact_current: torch.Tensor,
    exact_next: torch.Tensor,
    random_seed: int,
) -> dict[str, tuple[torch.Tensor, list[str]]]:
    batch, dimension = state.shape[0], state.shape[-1]
    correction = operator.A.float() @ operator.B.float()
    left, _, _ = torch.linalg.svd(correction, full_matrices=False)
    j_modes = left[:, : operator.rank].T[:, None, :].expand(-1, batch, -1)

    generator = torch.Generator(device=state.device).manual_seed(random_seed)
    random = torch.randn(
        dimension,
        operator.rank,
        generator=generator,
        device=state.device,
        dtype=torch.float32,
    )
    random_basis, _ = torch.linalg.qr(random, mode="reduced")
    random_modes = random_basis.T[:, None, :].expand(-1, batch, -1)

    diagonal_order = (operator.diagonal_scale.float() - 1.0).abs().argsort(
        descending=True
    )[: operator.rank]
    coordinate_basis = torch.eye(
        dimension, device=state.device, dtype=torch.float32
    )[diagonal_order]
    diagonal_coordinates = coordinate_basis[:, None, :].expand(-1, batch, -1)

    controlled = operator(state)[:, -1].float()
    interface_delta = _normalize(exact_current[:, -1].float() - controlled)[None]
    task_tangent = _normalize(
        exact_next[:, -1].float() - exact_current[:, -1].float()
    )[None]
    bias_direction = _normalize(operator.bias.float())[None, None, :].expand(
        1, batch, -1
    )
    return {
        "J_left_singular": (
            j_modes,
            [str(index) for index in range(operator.rank)],
        ),
        "random_orthonormal": (
            random_modes,
            [str(index) for index in range(operator.rank)],
        ),
        "top_abs_D_coordinate": (
            diagonal_coordinates,
            [str(int(index)) for index in diagonal_order],
        ),
        "bias_direction": (bias_direction, ["bias"]),
        "exact_interface_delta": (interface_delta, ["exact_H7_minus_J"]),
        "successor_task_tangent": (task_tangent, ["current_to_next_H7"]),
    }


def _map(
    *,
    model,
    state: torch.Tensor,
    loop_index: int,
    positions: tuple[int, ...],
    operator: Operator,
) -> tuple[torch.Tensor, torch.Tensor]:
    step = _controlled_loop(
        loop_runner=run_one_loop,
        model=model,
        state=state,
        loop_index=loop_index,
        positions=positions,
        operator=operator,
        placement="loop_boundary",
    )
    return step.state, step.logits


def _strongest_incorrect(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    masked = logits.float().clone()
    masked.scatter_(1, target[:, None], float("-inf"))
    return masked.argmax(dim=-1)


@torch.no_grad()
def _gain_rows(
    *,
    model,
    base_state: torch.Tensor,
    loop_index: int,
    positions: tuple[int, ...],
    operator: Operator,
    operator_name: str,
    target: torch.Tensor,
    cycle: int,
    family: str,
    directions: torch.Tensor,
    labels: Sequence[str],
    epsilon: float,
    direction_chunk: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_output, base_logits = _map(
        model=model,
        state=base_state,
        loop_index=loop_index,
        positions=positions,
        operator=operator,
    )
    competitor = _strongest_incorrect(base_logits, target)
    batch = base_state.shape[0]
    summary_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    for offset in range(0, directions.shape[0], direction_chunk):
        current = directions[offset : offset + direction_chunk].float()
        count = current.shape[0]
        perturbation = torch.zeros(
            count,
            batch,
            *base_state.shape[1:],
            device=base_state.device,
            dtype=torch.float32,
        )
        perturbation[:, :, -1, :] = current
        expanded = base_state.float()[None].expand(count, -1, -1, -1)
        plus = (expanded + epsilon * perturbation).reshape(
            count * batch, *base_state.shape[1:]
        )
        minus = (expanded - epsilon * perturbation).reshape(
            count * batch, *base_state.shape[1:]
        )
        packed = torch.cat((plus, minus), dim=0)
        output, logits = _map(
            model=model,
            state=packed,
            loop_index=loop_index,
            positions=positions,
            operator=operator,
        )
        output_plus, output_minus = output.split(count * batch, dim=0)
        logits_plus, logits_minus = logits.split(count * batch, dim=0)
        output_derivative = (
            output_plus.reshape(count, batch, *base_output.shape[1:])
            - output_minus.reshape(count, batch, *base_output.shape[1:])
        ) / (2.0 * epsilon)
        logit_derivative = (
            logits_plus.reshape(count, batch, -1)
            - logits_minus.reshape(count, batch, -1)
        ) / (2.0 * epsilon)
        input_norm = current.norm(dim=-1)
        valid = input_norm.gt(1e-8)
        answer_gain = output_derivative[:, :, -1].norm(dim=-1) / input_norm.clamp_min(
            1e-12
        )
        full_gain = output_derivative.flatten(2).norm(dim=-1) / input_norm.clamp_min(
            1e-12
        )
        graph_gain = output_derivative[:, :, :-1].flatten(2).norm(dim=-1) / input_norm.clamp_min(
            1e-12
        )
        logit_gain = logit_derivative.norm(dim=-1) / input_norm.clamp_min(1e-12)
        repeated_target = target[None].expand(count, -1)
        repeated_competitor = competitor[None].expand(count, -1)
        contrast_derivative = (
            logit_derivative.gather(2, repeated_target[:, :, None]).squeeze(-1)
            - logit_derivative.gather(
                2, repeated_competitor[:, :, None]
            ).squeeze(-1)
        ).abs() / input_norm.clamp_min(1e-12)
        for local_index in range(count):
            direction_index = offset + local_index
            mask = valid[local_index]
            values = {
                "answer_state_gain": answer_gain[local_index, mask],
                "full_state_gain": full_gain[local_index, mask],
                "graph_state_gain": graph_gain[local_index, mask],
                "logit_l2_gain": logit_gain[local_index, mask],
                "target_contrast_gain": contrast_derivative[local_index, mask],
            }
            row: dict[str, Any] = {
                "cycle": cycle,
                "effective_loop": cycle + 8,
                "operator": operator_name,
                "direction_family": family,
                "direction": labels[direction_index],
                "epsilon": epsilon,
                "valid_examples": int(mask.sum()),
                "base_accuracy": float(
                    base_logits.argmax(dim=-1).eq(target).float().mean()
                ),
            }
            for name, value in values.items():
                row[f"{name}_mean"] = float(value.mean())
                row[f"{name}_median"] = float(value.median())
                row[f"{name}_max"] = float(value.max())
            summary_rows.append(row)
            valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
            for sample in valid_indices.tolist():
                per_example_rows.append(
                    {
                        "cycle": cycle,
                        "operator": operator_name,
                        "direction_family": family,
                        "direction": labels[direction_index],
                        "epsilon": epsilon,
                        "sample": sample,
                        "base_correct": int(
                            base_logits[sample].argmax().eq(target[sample])
                        ),
                        "answer_state_gain": float(
                            answer_gain[local_index, sample]
                        ),
                        "full_state_gain": float(full_gain[local_index, sample]),
                        "graph_state_gain": float(graph_gain[local_index, sample]),
                        "logit_l2_gain": float(logit_gain[local_index, sample]),
                        "target_contrast_gain": float(
                            contrast_derivative[local_index, sample]
                        ),
                    }
                )
    return summary_rows, per_example_rows


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
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
        raise ValueError("local gain audit requires loop-boundary J")
    learned = loaded[args.operator_label]
    if not isinstance(learned, DiagonalIdentityLoRAJ) or learned.rank != 48:
        raise ValueError("local gain audit requires diagonal rank-48 J")
    all_variants = _variants(learned, shuffle_seed=args.shuffle_seed)
    operators: dict[str, Operator] = {
        "full": all_variants["full"].operator,
        "no_control": lambda value: value.float(),
        "no_bias": all_variants["no_bias"].operator,
        "shuffled_D": all_variants["shuffled_D"].operator,
    }
    unknown = set(args.operators) - set(operators)
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")

    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    set_seed(args.seed)
    _, path_targets, successors, _ = fixed_depth_batch(
        cfg, args.batch_size, device, path_positions=cfg.max_depth
    )
    endpoint = path_targets[:, cfg.max_depth - 1]
    state = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=endpoint,
        age=8,
        phase_position=phase_positions[8],
    )
    requested = set(args.cycles)
    baseline_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    per_example_rows: list[dict[str, Any]] = []
    for cycle in range(1, max(requested) + 1):
        current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
        target = advance_nodes(successors, endpoint, steps=jump * cycle)
        loop_index = cfg.max_loops + cycle - 1
        if cycle in requested:
            exact_current = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=7,
                phase_position=phase_positions[7],
            )
            exact_next = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=target,
                age=7,
                phase_position=phase_positions[7],
            )
            families = _direction_families(
                operator=learned,
                state=state,
                exact_current=exact_current,
                exact_next=exact_next,
                random_seed=args.random_seed + cycle,
            )
            for operator_name in args.operators:
                base_output, base_logits = _map(
                    model=model,
                    state=state,
                    loop_index=loop_index,
                    positions=positions,
                    operator=operators[operator_name],
                )
                baseline_rows.append(
                    {
                        "cycle": cycle,
                        "effective_loop": cfg.max_loops + cycle,
                        "operator": operator_name,
                        **_metrics(base_logits, target),
                        "output_answer_norm": float(
                            base_output[:, -1].float().norm(dim=-1).mean()
                        ),
                    }
                )
                for family, (directions, labels) in families.items():
                    for epsilon in args.epsilons:
                        summary, raw = _gain_rows(
                            model=model,
                            base_state=state,
                            loop_index=loop_index,
                            positions=positions,
                            operator=operators[operator_name],
                            operator_name=operator_name,
                            target=target,
                            cycle=cycle,
                            family=family,
                            directions=directions,
                            labels=labels,
                            epsilon=epsilon,
                            direction_chunk=args.direction_chunk,
                        )
                        gain_rows.extend(summary)
                        per_example_rows.extend(raw)
        step = _controlled_loop(
            loop_runner=run_one_loop,
            model=model,
            state=state,
            loop_index=loop_index,
            positions=positions,
            operator=operators["full"],
            placement="loop_boundary",
        )
        state = step.state

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "baseline.csv", baseline_rows)
    _write(args.out_dir / "directional_gain.csv", gain_rows)
    _write(args.out_dir / "per_example_gain.csv", per_example_rows)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": learned.rank,
        "controller_parameters": learned.parameter_count,
        "controller_placement": "loop boundary after Block2 FFN",
        "examples": args.batch_size,
        "cycles": sorted(requested),
        "operators": list(args.operators),
        "epsilons": list(args.epsilons),
        "direction_families": [
            "J_left_singular",
            "random_orthonormal",
            "top_abs_D_coordinate",
            "bias_direction",
            "exact_interface_delta",
            "successor_task_tangent",
        ],
        "interpretation_boundary": (
            "finite-difference local directional gains are dynamics evidence, "
            "not a component circuit claim"
        ),
        "gpu_peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-difference local gain audit for full J and controls."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument(
        "--operators",
        nargs="+",
        default=("full", "no_control", "no_bias", "shuffled_D"),
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=(0.01, 0.03))
    parser.add_argument("--direction-chunk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--random-seed", type=int, default=20260802)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))
