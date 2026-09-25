"""Evaluate every ordered pair of independently CE-trained Js in F,F,J1,J2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_two_matrix_cycle import draw, evaluate, write_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ordered CE-trained J pairs in F,F,J1,J2.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args(argv)


def affine(module: DiagonalIdentityLoRAJ) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.diag(module.diagonal_scale.float())
        + module.A.float() @ module.B.float(),
        module.bias.float(),
    )


def matrix_rows(
    modules: dict[str, torch.nn.Module]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_label, left in modules.items():
        left_weight, left_bias = affine(left)  # type: ignore[arg-type]
        for right_label, right in modules.items():
            right_weight, right_bias = affine(right)  # type: ignore[arg-type]
            rows.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "weight_relative_difference": float(
                        (left_weight - right_weight).norm()
                        / left_weight.norm().clamp_min(1e-12)
                    ),
                    "bias_relative_difference": float(
                        (left_bias - right_bias).norm()
                        / left_bias.norm().clamp_min(1e-12)
                    ),
                    "delta_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            (left_weight - torch.eye(left_weight.shape[0])).reshape(1, -1),
                            (right_weight - torch.eye(right_weight.shape[0])).reshape(1, -1),
                        ).item()
                    ),
                }
            )
    return rows


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, modules, payload = load_task_lora_modules(
        args.artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("artifact and checkpoint differ")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("expected all-position Js")
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(x) for x in phase["trajectory_positions_including_initial"]]
    resets: dict[str, tuple[Any, int]] = {
        "no_reset_start_H6": (None, 6),
        "exact_H6_every2": (None, 6),
    }
    for first_label, first in modules.items():
        for second_label, second in modules.items():
            first_seed = first_label.rsplit("seed", 1)[-1]
            second_seed = second_label.rsplit("seed", 1)[-1]
            resets[f"J{first_seed}_then_J{second_seed}"] = (
                lambda state, first=first, second=second: second(first(state)),
                6,
            )
    curve_rows, summary_rows = evaluate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        resets=resets,
        examples=args.evaluation_examples,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        seed=args.seed,
    )
    geometry_rows = matrix_rows(modules)
    write_csv(args.out_dir / "pair_curves.csv", curve_rows)
    write_csv(args.out_dir / "pair_summary.csv", summary_rows)
    write_csv(args.out_dir / "pair_matrix_relations.csv", geometry_rows)
    draw(args.out_dir, curve_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": payload.get("backbone_loss_description", "not recorded"),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "cycle": "start H6; F,F,J_first,J_second; repeat",
        "important_boundary": "the three Js are optimization replicas, not phase-specialist training",
        "evaluation_examples": args.evaluation_examples,
        "closed_loop_metrics": summary_rows,
        "matrix_relations": geometry_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
