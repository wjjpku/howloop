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
    _coordinate_scale_bank,
    _make_batch,
    _run_condition_chunks,
    _shuffled_diagonal,
    _write,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


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
        raise ValueError("validation requires loop-boundary J")
    operator = loaded[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("validation requires diagonal rank-48 J")
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase["trajectory_positions_including_initial"]]
    jump = phase_positions[3] - phase_positions[2]
    full = operator.diagonal_scale.detach().float()
    damaged, _ = _shuffled_diagonal(full, seed=args.shuffle_seed)
    coordinates = torch.arange(full.numel(), device=device)
    bank = _coordinate_scale_bank(
        full=full, damaged=damaged, coordinates=coordinates, mode="corrupt"
    )
    conditions = [
        ScaleCondition(
            "full", "baseline", "full", full.numel(), tuple(range(full.numel())), full
        ),
        ScaleCondition("shuffled", "baseline", "shuffled", 0, (), damaged),
    ] + [
        ScaleCondition(
            f"corrupt.coordinate.{coordinate}",
            "single_coordinate",
            "corrupt",
            1,
            (coordinate,),
            bank[coordinate],
        )
        for coordinate in range(full.numel())
    ]
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
            split=f"validation_seed_{data_seed}",
        )
        for row in rows:
            row["data_seed"] = data_seed
        for row in examples:
            row["data_seed"] = data_seed
        all_rows.extend(rows)
        all_examples.extend(examples)
    _write(args.out_dir / "single_coordinate_validation.csv", all_rows)
    _write(
        args.out_dir / "single_coordinate_validation_per_example.csv", all_examples
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
        "coordinate_count": full.numel(),
        "shuffle_seed": args.shuffle_seed,
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
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=(20260812, 20260814, 20260815))
    parser.add_argument("--cycles", type=int, nargs="+", default=(1, 32, 64))
    parser.add_argument("--condition-chunk", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
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
