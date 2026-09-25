"""Train a CE-only long-rollout J_7->6 and test composition with J_8->7.

J_8->7 is the existing canonical closed-loop controller.  J_7->6 starts from
that trained matrix but is independently optimized in the recurrent cycle

    H7 --J_7->6--> interface --F--> H7(next current),

with task CE at every long-rollout step and no hidden-state regression loss.
After training, both matrices are frozen and composed zero-shot.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.analyze_graph_path_j_transition_matrices import affine_metrics
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import (
    GOOD_TASK_STAGES,
    _train_stage,
)
from reasoning_loop.graph_path_telomere_two_matrix_cycle import (
    compose,
    draw,
    evaluate,
    write_csv,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train J76 and compose it with canonical J87.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--canonical-artifact", type=Path, required=True)
    parser.add_argument("--canonical-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate-multiplier", type=float, default=30.0)
    parser.add_argument("--scale-learning-rate-multiplier", type=float, default=0.1)
    parser.add_argument("--stage-round-limit", type=int)
    parser.add_argument("--evaluation-examples", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--evaluation-loops", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.05)
    return parser.parse_args(argv)


def affine(module: DiagonalIdentityLoRAJ) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.diag(module.diagonal_scale.float())
        + module.A.float() @ module.B.float(),
        module.bias.float(),
    )


def homogeneous(affine_map: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    weight, bias = affine_map
    dimension = weight.shape[0]
    result = torch.zeros(
        dimension + 1, dimension + 1, device=weight.device, dtype=weight.dtype
    )
    result[:dimension, :dimension] = weight
    result[dimension, :dimension] = bias
    result[dimension, dimension] = 1.0
    return result


def save_checkpoint(
    path: Path,
    *,
    checkpoint: Path,
    positions: tuple[int, ...],
    canonical_label: str,
    j87: DiagonalIdentityLoRAJ,
    j76: DiagonalIdentityLoRAJ,
    stages: list[dict[str, Any]],
) -> None:
    j87_affine, j76_affine = affine(j87), affine(j76)
    product = compose(j87_affine, j76_affine)
    payload = {
        "kind": "graph_path_stage_specific_J87_J76",
        "checkpoint": str(checkpoint),
        "positions": positions,
        "canonical_J87_label": canonical_label,
        "J87_role": "CE-only long-rollout H8 recurrent controller",
        "J76_role": "independently CE-only long-rollout H7 recurrent controller",
        "training_hidden_state_loss": 0.0,
        "stages": stages,
        "J87_state_dict": {key: value.detach().cpu() for key, value in j87.state_dict().items()},
        "J76_state_dict": {key: value.detach().cpu() for key, value in j76.state_dict().items()},
        "product_H8_to_H6_weight": product[0].detach().cpu(),
        "product_H8_to_H6_bias": product[1].detach().cpu(),
        "product_H8_to_H6_homogeneous": (
            homogeneous(j87_affine) @ homogeneous(j76_affine)
        ).detach().cpu(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def heldout_composition_metrics(
    *,
    model: torch.nn.Module,
    cfg: Any,
    phase_positions: list[int],
    j87: DiagonalIdentityLoRAJ,
    j76: DiagonalIdentityLoRAJ,
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    sequential: list[torch.Tensor] = []
    product_values: list[torch.Tensor] = []
    product = compose(affine(j87), affine(j76))
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        current = path_targets[:, cfg.max_depth - 1]
        h8 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=8,
            phase_position=phase_positions[8],
        )
        h6 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=6,
            phase_position=phase_positions[6],
        )
        sources.append(h8)
        targets.append(h6)
        sequential.append(j76(j87(h8)))
        product_values.append(h8.float() @ product[0] + product[1])
    target = torch.cat(targets)
    seq = torch.cat(sequential)
    one = torch.cat(product_values)
    metrics = affine_metrics(torch.cat(sources), target, *product)
    return {
        **metrics,
        "sequential_vs_product_relative_error": float(
            (seq - one).norm() / seq.norm().clamp_min(1e-12)
        ),
    }


def main(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    artifact_checkpoint, positions, modules, artifact_payload = load_task_lora_modules(
        args.canonical_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("canonical J and backbone checkpoint differ")
    j87 = modules[args.canonical_label]
    if not isinstance(j87, DiagonalIdentityLoRAJ):
        raise TypeError("expected canonical diagonal plus low-rank J87")
    j76 = copy.deepcopy(j87).train()
    for parameter in j76.parameters():
        parameter.requires_grad_(True)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    stages = []
    for original in GOOD_TASK_STAGES:
        rounds = (
            original.rounds
            if args.stage_round_limit is None
            else min(original.rounds, args.stage_round_limit)
        )
        stages.append(
            replace(
                original,
                rounds=rounds,
                learning_rate=original.learning_rate * args.learning_rate_multiplier,
                data_seed=original.data_seed + 100000,
            )
        )
    training_rows: list[dict[str, Any]] = []
    recorded_stages: list[dict[str, Any]] = []
    for stage in stages:
        rows = _train_stage(
            module=j76,
            stage=stage,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            state_loss_weight=0.0,
            grad_clip=1.0,
            placement="loop_boundary",
            optimizer_parameter_groups=[
                {"params": [j76.A, j76.B, j76.bias]},
                {
                    "params": [j76.diagonal_scale],
                    "lr": stage.learning_rate * args.scale_learning_rate_multiplier,
                },
            ],
            closed_loop_age=7,
        )
        training_rows.extend(rows)
        recorded_stages.append(asdict(stage))
        save_checkpoint(
            args.out_dir / "J87_J76.pt",
            checkpoint=args.checkpoint,
            positions=positions,
            canonical_label=args.canonical_label,
            j87=j87,
            j76=j76,
            stages=recorded_stages,
        )
        write_csv(args.out_dir / "training_rounds.csv", training_rows)
    j76 = j76.frozen()
    product = compose(affine(j87), affine(j76))
    resets = {
        "no_reset_start_H6": (None, 6),
        "exact_H6_every2": (None, 6),
        "J87_then_J76": (lambda state: j76(j87(state)), 6),
        "product_J76xJ87_one_shot": (
            lambda state: state.float() @ product[0] + product[1],
            6,
        ),
        "J76_then_J87_wrong_order": (lambda state: j87(j76(state)), 6),
        "J87_then_J87": (lambda state: j87(j87(state)), 6),
        "J76_then_J76": (lambda state: j76(j76(state)), 6),
    }
    curve_rows, closed_loop_rows = evaluate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        resets=resets,
        examples=args.evaluation_examples,
        batch_size=args.evaluation_batch_size,
        continuation_loops=args.evaluation_loops,
        seed=args.seed + 1,
    )
    composition_metrics = heldout_composition_metrics(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        j87=j87,
        j76=j76,
        device=device,
        examples=args.evaluation_examples,
        batch_size=args.evaluation_batch_size,
        seed=args.seed + 2,
    )
    write_csv(args.out_dir / "closed_loop_curves.csv", curve_rows)
    write_csv(args.out_dir / "closed_loop_summary.csv", closed_loop_rows)
    draw(args.out_dir, curve_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": artifact_payload.get("backbone_loss_description", "not recorded"),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "J87_training": "existing canonical CE-only long-rollout controller",
        "J76_training": "independent CE-only H7 closed-loop controller; hidden loss 0",
        "composition_test": "start H6; F,F,J87,J76; repeat",
        "stages": recorded_stages,
        "composition_metrics": composition_metrics,
        "closed_loop_metrics": closed_loop_rows,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
