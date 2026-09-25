from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_oracle_position_circuit import intervention_groups
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    IdentityJ,
    _curve_summary,
    _scale_statistics,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import (
    TaskMLPStage,
    _train_stage,
    _write_csv,
    evaluate_random_unit_every,
)
from reasoning_loop.graph_path_telomere_unit_j import _training_case, load_unit_j_map


def _balanced_horizon_counts(stage: TaskMLPStage) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for round_index in range(1, stage.rounds + 1):
        round_seed = stage.data_seed + 1000 * round_index
        sampled = []
        for batch_index in range(stage.batches_per_round):
            _, _, horizon = _training_case(
                batch_index=batch_index,
                rollout_horizons=stage.horizons,
                policies=("unit_every",),
                start_ages=(8,),
                seed=round_seed,
            )
            sampled.append(horizon)
            counts[horizon] += 1
        if sorted(sampled) != list(range(1, 9)):
            raise RuntimeError(
                f"round {round_index} is not an exact 1..8 horizon cover: {sampled}"
            )
    return dict(sorted(counts.items()))


def _wsd_learning_rate(
    *,
    update: int,
    total_updates: int,
    peak: float,
    warmup_updates: int,
    stable_updates: int,
    final_ratio: float,
) -> float:
    if not 1 <= update <= total_updates:
        raise ValueError("update lies outside the WSD schedule")
    if peak <= 0 or warmup_updates < 0 or stable_updates < 0:
        raise ValueError("invalid WSD schedule")
    if not 0.0 < final_ratio <= 1.0:
        raise ValueError("final ratio must lie in (0, 1]")
    decay_updates = total_updates - warmup_updates - stable_updates
    if decay_updates < 1:
        raise ValueError("WSD schedule requires at least one decay update")
    if warmup_updates and update <= warmup_updates:
        return peak * update / warmup_updates
    if update <= warmup_updates + stable_updates:
        return peak
    decay_index = update - warmup_updates - stable_updates
    progress = decay_index / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (final_ratio + (1.0 - final_ratio) * cosine)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.rounds <= 0 or args.batch_size <= 0:
        raise ValueError("rounds and batch_size must be positive")
    if args.learning_rate <= 0 or args.diagonal_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    total_updates = args.rounds * 8
    if args.schedule == "wsd":
        _wsd_learning_rate(
            update=1,
            total_updates=total_updates,
            peak=args.learning_rate,
            warmup_updates=args.warmup_updates,
            stable_updates=args.stable_updates,
            final_ratio=args.final_lr_ratio,
        )
        schedule_metadata: dict[str, Any] = {
            "name": "wsd",
            "total_updates": total_updates,
            "warmup_updates": args.warmup_updates,
            "stable_updates": args.stable_updates,
            "decay_updates": (
                total_updates - args.warmup_updates - args.stable_updates
            ),
            "peak_learning_rate": args.learning_rate,
            "diagonal_peak_learning_rate": args.diagonal_learning_rate,
            "final_lr_ratio": args.final_lr_ratio,
        }
    else:
        schedule_metadata = {
            "name": "constant",
            "total_updates": total_updates,
            "learning_rate": args.learning_rate,
            "diagonal_learning_rate": args.diagonal_learning_rate,
        }
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if (cfg.node_count, cfg.max_depth, cfg.max_loops, cfg.n_layers, cfg.d_model) != (
        8,
        8,
        8,
        2,
        256,
    ):
        raise ValueError("experiment requires the frozen D8L8 N8 d256 backbone")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    source_checkpoint, source_positions, modules, source_payload = (
        load_task_lora_modules(args.source_artifact, device=device)
    )
    if source_checkpoint != str(args.checkpoint):
        raise ValueError("source controller belongs to another backbone")
    if source_positions != positions:
        raise ValueError("source controller uses different positions")
    if source_payload.get("placement") != "loop_boundary":
        raise ValueError("dense continuation requires a loop-boundary controller")
    if int(source_payload.get("max_training_horizon", -1)) != 8:
        raise ValueError("source controller was not trained with maximum horizon 8")
    if any(
        item.get("parameterization") != "diagonal_low_rank"
        or int(item.get("rank", -1)) != 48
        for item in source_payload["modules"].values()
    ):
        raise ValueError("source artifact must contain only rank-48 diagonal LoRA J")

    stage = TaskMLPStage(
        name=args.stage_name,
        rounds=args.rounds,
        batch_size=args.batch_size,
        batches_per_round=8,
        horizons=tuple(range(1, 9)),
        learning_rate=args.learning_rate,
        data_seed=args.data_seed,
    )
    horizon_counts = _balanced_horizon_counts(stage)
    if set(horizon_counts) != set(range(1, 9)):
        raise RuntimeError("dense continuation did not sample every horizon 1..8")

    output_payload = copy.deepcopy(source_payload)
    output_payload["kind"] = "graph_path_telomere_task_lora_j"
    output_payload["parent_artifact"] = str(args.source_artifact)
    output_payload["curriculum"] = args.curriculum_name
    output_payload["max_training_horizon"] = 8
    stage_record = asdict(stage)
    stage_record["learning_rate_schedule"] = schedule_metadata
    output_payload["stages"] = list(output_payload["stages"]) + [stage_record]
    output_payload["dense_horizon_counts_per_variant"] = horizon_counts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.out_dir / "task_lora_j.pt"
    training_rows: list[dict[str, Any]] = []
    trained_modules: dict[str, torch.nn.Module] = {}

    for label in sorted(modules):
        module = modules[label]
        if not isinstance(module, DiagonalIdentityLoRAJ):
            raise TypeError(f"{label} is not a DiagonalIdentityLoRAJ")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        module.train()
        optimizer_parameter_groups = [
            {"params": [module.A, module.B, module.bias]},
            {"params": [module.diagonal_scale], "lr": args.diagonal_learning_rate},
        ]
        set_seed(int(source_payload["modules"][label]["initialization_seed"]))
        learning_rate_schedule = None
        if args.schedule == "wsd":
            learning_rate_schedule = lambda update, total: _wsd_learning_rate(
                update=update,
                total_updates=total,
                peak=args.learning_rate,
                warmup_updates=args.warmup_updates,
                stable_updates=args.stable_updates,
                final_ratio=args.final_lr_ratio,
            )
        rows = _train_stage(
            module=module,
            stage=stage,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            device=device,
            state_loss_weight=0.0,
            grad_clip=args.grad_clip,
            placement="loop_boundary",
            optimizer_parameter_groups=optimizer_parameter_groups,
            learning_rate_schedule=learning_rate_schedule,
        )
        for row in rows:
            row.update(
                {
                    "variant": label,
                    "rank": module.rank,
                    "initialization_seed": int(
                        source_payload["modules"][label]["initialization_seed"]
                    ),
                    "controller_parameter_count": module.parameter_count,
                    "parent_artifact": str(args.source_artifact),
                }
            )
        training_rows.extend(rows)
        item = output_payload["modules"][label]
        item["state_dict"] = {
            key: value.detach().cpu() for key, value in module.state_dict().items()
        }
        item["final_scale_statistics"] = _scale_statistics(module.diagonal_scale)
        snapshots = dict(item.get("stage_snapshots", {}))
        snapshots[stage.name] = {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        item["stage_snapshots"] = snapshots
        trained_modules[label] = module.frozen()
        temporary = artifact.with_suffix(".pt.tmp")
        torch.save(output_payload, temporary)
        os.replace(temporary, artifact)
        _write_csv(args.out_dir / "training_rounds.csv", training_rows)

    reference, reference_checkpoint = load_unit_j_map(
        Path(source_payload["reference_affine_artifact"]),
        label=str(source_payload["reference_affine_label"]),
        device=device,
    )
    if reference_checkpoint != str(args.checkpoint):
        raise ValueError("reference affine belongs to another backbone")
    maps: dict[str, Any] = {
        "identity_no_J": IdentityJ().to(device).eval(),
        "reference_full_affine": reference,
        **trained_modules,
    }
    random_rows = evaluate_random_unit_every(
        maps=maps,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.evaluation_batch_size,
        batches=args.evaluation_batches,
        continuation_loops=args.evaluation_loops,
        seed=args.evaluation_seed,
        placement="loop_boundary",
    )
    _write_csv(args.out_dir / "random_graph_closed_loop.csv", random_rows)
    curves = {
        label: _curve_summary(
            [float(row["accuracy"]) for row in random_rows if row["variant"] == label]
        )
        for label in maps
    }
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "parent_artifact": str(args.source_artifact),
        "controller": "J(h)=h*diag(alpha)+(hA)B+b; shared rank-48 token-wise affine",
        "controller_placement": "loop_boundary",
        "loss_placement": (
            "frozen backbone: final-only CE at loop 8; dense continuation uses "
            "successor CE at every controlled step; no hidden-state loss"
        ),
        "curriculum": args.curriculum_name,
        "max_training_horizon": 8,
        "dense_stage": asdict(stage),
        "learning_rate_schedule": schedule_metadata,
        "dense_horizon_counts_per_variant": horizon_counts,
        "dense_graph_draws_per_variant": stage.graphs,
        "controller_variants": sorted(trained_modules),
        "random_graph_evaluation": {
            "examples": args.evaluation_batch_size * args.evaluation_batches,
            "loops": args.evaluation_loops,
            "seed": args.evaluation_seed,
            "curves": curves,
        },
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "artifact": "task_lora_j.pt",
            "training": "training_rounds.csv",
            "random_evaluation": "random_graph_closed_loop.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--diagonal-learning-rate", type=float, default=3e-6)
    parser.add_argument(
        "--schedule", choices=("constant", "wsd"), default="constant"
    )
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--stable-updates", type=int, default=0)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--stage-name", default="strict_dense_h8_continue")
    parser.add_argument(
        "--curriculum-name", default="strict_dense_h8_continuation"
    )
    parser.add_argument("--data-seed", type=int, default=189003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--evaluation-loops", type=int, default=128)
    parser.add_argument("--evaluation-seed", type=int, default=212004)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.06)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_experiment(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
