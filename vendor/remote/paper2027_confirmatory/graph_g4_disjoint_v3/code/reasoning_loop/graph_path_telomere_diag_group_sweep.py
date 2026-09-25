from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_diag_coordinate_circuit import (
    ScaleCondition,
    _make_batch,
    _run_condition_chunks,
    _write,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


def coarse_diagonal_groups(
    diagonal: torch.Tensor, *, group_size: int, random_seed: int
) -> dict[str, torch.Tensor]:
    if diagonal.ndim != 1:
        raise ValueError("diagonal must be a vector")
    if not 1 <= group_size <= diagonal.numel():
        raise ValueError("group size is out of range")
    order = diagonal.argsort()
    median = diagonal.median()
    generator = torch.Generator(device=diagonal.device).manual_seed(random_seed)
    return {
        "strongly_damped": order[:group_size],
        "high_retention": order[-group_size:],
        "middle": (diagonal - median).abs().argsort()[:group_size],
        "random": torch.randperm(
            diagonal.numel(), generator=generator, device=diagonal.device
        )[:group_size],
    }


def _conditions(
    full: torch.Tensor,
    *,
    groups: dict[str, torch.Tensor],
    deltas: Sequence[float],
) -> tuple[list[ScaleCondition], dict[str, dict[str, Any]]]:
    conditions = [
        ScaleCondition(
            "full", "baseline", "full", full.numel(), tuple(range(full.numel())), full
        )
    ]
    metadata: dict[str, dict[str, Any]] = {
        "full": {"group": "all", "signed_delta": 0.0, "delta_sign": "zero"}
    }
    for group_name, coordinates in groups.items():
        for delta in deltas:
            if delta <= 0:
                raise ValueError("deltas must be positive")
            for sign, sign_label in ((-1, "minus"), (1, "plus")):
                scale = full.clone()
                scale[coordinates] += sign * delta
                label = f"{group_name}.{sign_label}.{delta:.6f}"
                conditions.append(
                    ScaleCondition(
                        label,
                        "coarse_D_group",
                        "corrupt",
                        int(coordinates.numel()),
                        tuple(int(value) for value in coordinates.tolist()),
                        scale,
                    )
                )
                metadata[label] = {
                    "group": group_name,
                    "signed_delta": float(sign * delta),
                    "delta_sign": sign_label,
                }
    return conditions, metadata


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
        raise ValueError("coarse D sweep requires loop-boundary J")
    if tuple(positions) != tuple(range(cfg.seq_len)):
        raise ValueError("coarse D sweep currently requires all-position J")
    operator = loaded[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("coarse D sweep requires diagonal rank-48 J")
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase["trajectory_positions_including_initial"]]
    jump = phase_positions[3] - phase_positions[2]
    full = operator.diagonal_scale.detach().float()
    groups = coarse_diagonal_groups(
        full, group_size=args.group_size, random_seed=args.group_seed
    )
    conditions, metadata = _conditions(full, groups=groups, deltas=args.deltas)

    all_rows: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []
    for data_seed in args.eval_seeds:
        initial, endpoint, successors = _make_batch(
            model=model,
            cfg=cfg,
            batch_size=args.eval_batch_size,
            device=device,
            seed=data_seed,
            phase_positions=phase_positions,
        )
        rows, examples = _run_condition_chunks(
            conditions=conditions,
            chunk_size=args.condition_chunk,
            model=model,
            cfg=cfg,
            initial=initial,
            endpoint=endpoint,
            successors=successors,
            jump=jump,
            positions=positions,
            operator=operator,
            cycles=args.cycles,
            split=f"coarse_D_seed_{data_seed}",
        )
        for row in rows:
            row.update(metadata[row["condition"]])
            row["data_seed"] = data_seed
        for row in examples:
            row.update(metadata[row["condition"]])
            row["data_seed"] = data_seed
        all_rows.extend(rows)
        all_examples.extend(examples)
    _write(args.out_dir / "coarse_D_group_sweep.csv", all_rows)
    _write(args.out_dir / "coarse_D_group_sweep_per_example.csv", all_examples)
    group_payload = {
        name: {
            "coordinates": [int(value) for value in coordinates.tolist()],
            "D_mean": float(full[coordinates].mean()),
            "D_min": float(full[coordinates].min()),
            "D_max": float(full[coordinates].max()),
        }
        for name, coordinates in groups.items()
    }
    (args.out_dir / "group_definitions.json").write_text(
        json.dumps(group_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    peak = (
        float(torch.cuda.max_memory_allocated(device) / 2**30)
        if device.type == "cuda"
        else None
    )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_cycles": list(args.cycles),
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": operator.rank,
        "controller_placement": "loop boundary after Block2 FFN",
        "controller_training_loss": "successor CE at every controlled continuation loop; no hidden MSE",
        "evaluation_examples_per_seed": args.eval_batch_size,
        "evaluation_data_seeds": list(args.eval_seeds),
        "group_size": args.group_size,
        "deltas": list(args.deltas),
        "groups": group_payload,
        "condition_count": len(conditions),
        "gpu_peak_allocated_gib": peak,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "gpu_peak_allocated_gib": peak,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=(20260812, 20260814, 20260815)
    )
    parser.add_argument(
        "--cycles", type=int, nargs="+", default=(1, 8, 16, 32, 48, 64)
    )
    parser.add_argument("--group-size", type=int, default=48)
    parser.add_argument("--deltas", type=float, nargs="+", default=(0.005, 0.01, 0.02, 0.05))
    parser.add_argument("--group-seed", type=int, default=20260816)
    parser.add_argument("--condition-chunk", type=int, default=4)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.025)
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
