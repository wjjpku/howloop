"""Pure-CE on-policy training for a reusable D8L6 two-hop boundary controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.evaluate_d8l6_jump_controller_closure import (
    load_controller,
    strict_counts,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes, cache_states_with_initial
from reasoning_loop.graph_path_telomere_task_lora_j import DiagonalIdentityLoRAJ
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


@dataclass(frozen=True)
class Stage:
    name: str
    horizons: tuple[int, ...]
    rounds: int
    batch_size: int
    batches_per_round: int
    learning_rate: float
    data_seed: int


CURRICULUM = (
    Stage("h1", (1,), 8, 64, 8, 1e-5, 51001),
    Stage("h2", (1, 2), 8, 64, 8, 1e-5, 52001),
    Stage("h4", (1, 2, 4), 12, 64, 8, 8e-6, 54001),
    Stage("h8", (2, 4, 6, 8), 16, 48, 8, 5e-6, 58001),
    Stage("h16", (4, 8, 12, 16), 16, 32, 8, 3e-6, 516001),
    Stage("h32", (8, 16, 24, 32), 16, 16, 8, 2e-6, 532001),
)


def selected_stages(
    max_horizon: int,
    round_limit: int | None,
    *,
    round_multiplier: int = 1,
    learning_rate_multiplier: float = 1.0,
) -> tuple[Stage, ...]:
    if round_multiplier < 1:
        raise ValueError("round multiplier must be positive")
    if learning_rate_multiplier <= 0:
        raise ValueError("learning-rate multiplier must be positive")
    stages = []
    for stage in CURRICULUM:
        if max(stage.horizons) > max_horizon:
            continue
        stages.append(
            Stage(
                name=stage.name,
                horizons=stage.horizons,
                rounds=(
                    min(stage.rounds, round_limit)
                    if round_limit
                    else stage.rounds * round_multiplier
                ),
                batch_size=stage.batch_size,
                batches_per_round=stage.batches_per_round,
                learning_rate=stage.learning_rate * learning_rate_multiplier,
                data_seed=stage.data_seed,
            )
        )
    if not stages:
        raise ValueError("max horizon must select at least the h1 stage")
    return tuple(stages)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def train_stage(
    *,
    module: DiagonalIdentityLoRAJ,
    stage: Stage,
    model,
    cfg,
    device: torch.device,
    initialization_seed: int,
    grad_clip: float,
    optimization_mode: str,
    loss_weighting: str,
) -> list[dict[str, Any]]:
    module.train()
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=stage.learning_rate,
        weight_decay=0.0,
    )
    differentiable_loop = run_one_loop.__wrapped__
    positions = tuple(range(cfg.seq_len))
    rows: list[dict[str, Any]] = []
    for round_index in range(1, stage.rounds + 1):
        set_seed(stage.data_seed + 1009 * initialization_seed + 1000 * round_index)
        totals = {
            "loss": 0.0,
            "correct": 0,
            "examples": 0,
            "grad_norm": 0.0,
        }
        started = time.monotonic()
        for batch_index in range(stage.batches_per_round):
            horizon = stage.horizons[
                ((round_index - 1) * stage.batches_per_round + batch_index)
                % len(stage.horizons)
            ]
            with torch.no_grad():
                tokens, _, successors, start = fixed_depth_batch(
                    cfg,
                    stage.batch_size,
                    device,
                    path_positions=cfg.max_depth,
                )
                endpoint = advance_nodes(successors, start, steps=cfg.max_depth)
                state = cache_states_with_initial(
                    model,
                    tokens,
                    loops=cfg.max_loops,
                )[-1]

            optimizer.zero_grad(set_to_none=True)
            if optimization_mode == "dagger_detached":
                detached_states = []
                targets = []
                with torch.no_grad():
                    rollout_state = state
                    for cycle in range(1, horizon + 1):
                        detached_states.append(rollout_state.detach())
                        targets.append(
                            advance_nodes(successors, endpoint, steps=2 * cycle)
                        )
                        rollout_step = _controlled_loop(
                            loop_runner=run_one_loop,
                            model=model,
                            state=rollout_state,
                            loop_index=cfg.max_loops + cycle - 1,
                            positions=positions,
                            operator=module,
                            placement="loop_boundary",
                        )
                        rollout_state = rollout_step.state
            elif optimization_mode == "bptt":
                detached_states = []
                targets = []
            else:
                raise ValueError(f"unknown optimization mode: {optimization_mode}")

            losses = []
            for cycle in range(1, horizon + 1):
                target = (
                    targets[cycle - 1]
                    if optimization_mode == "dagger_detached"
                    else advance_nodes(successors, endpoint, steps=2 * cycle)
                )
                live_state = (
                    detached_states[cycle - 1]
                    if optimization_mode == "dagger_detached"
                    else state
                )
                step = _controlled_loop(
                    loop_runner=differentiable_loop,
                    model=model,
                    state=live_state,
                    loop_index=cfg.max_loops + cycle - 1,
                    positions=positions,
                    operator=module,
                    placement="loop_boundary",
                )
                loss = F.cross_entropy(step.logits.float(), target)
                losses.append(loss)
                totals["loss"] += float(loss.detach())
                totals["correct"] += int(step.logits.argmax(dim=-1).eq(target).sum())
                totals["examples"] += stage.batch_size
                if optimization_mode == "bptt":
                    state = step.state
            stacked = torch.stack(losses)
            if loss_weighting == "uniform":
                objective = stacked.mean()
            elif loss_weighting == "late_linear":
                weights = torch.arange(
                    1,
                    horizon + 1,
                    device=stacked.device,
                    dtype=stacked.dtype,
                )
                objective = (stacked * weights).sum() / weights.sum()
            elif loss_weighting == "final_only":
                objective = stacked[-1]
            else:
                raise ValueError(f"unknown loss weighting: {loss_weighting}")
            objective.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
            optimizer.step()
            totals["grad_norm"] += float(grad_norm)

        rows.append(
            {
                "stage": stage.name,
                "round": round_index,
                "horizons": "/".join(map(str, stage.horizons)),
                "graphs": stage.batch_size * stage.batches_per_round,
                "supervised_steps": totals["examples"],
                "composition_accuracy": totals["correct"] / totals["examples"],
                "pure_task_ce": totals["loss"] / (totals["examples"] / stage.batch_size),
                "mean_preclip_gradient_norm": totals["grad_norm"] / stage.batches_per_round,
                "learning_rate": stage.learning_rate,
                "optimization_mode": optimization_mode,
                "loss_weighting": loss_weighting,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    module.eval()
    return rows


@torch.no_grad()
def evaluate(
    *,
    module: DiagonalIdentityLoRAJ,
    model,
    cfg,
    device: torch.device,
    examples: int,
    batch_size: int,
    cycles: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("evaluation examples must be divisible by batch size")
    set_seed(seed)
    positions = tuple(range(cfg.seq_len))
    totals: dict[int, dict[str, int]] = {
        cycle: {
            "post_correct": 0,
            "post_count": 0,
            "prewrite_correct": 0,
            "prewrite_count": 0,
            "precurrent_correct": 0,
            "precurrent_count": 0,
        }
        for cycle in range(1, cycles + 1)
    }
    for _ in range(examples // batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = advance_nodes(successors, start, steps=cfg.max_depth)
        state = cache_states_with_initial(model, tokens, loops=cfg.max_loops)[-1]
        for cycle in range(1, cycles + 1):
            current = advance_nodes(successors, endpoint, steps=2 * (cycle - 1))
            target = advance_nodes(successors, endpoint, steps=2 * cycle)
            controlled = state.clone()
            controlled[:, list(positions)] = module(controlled[:, list(positions)])
            pre_logits = logits_from_raw_state(model, controlled)
            prewrite = strict_counts(pre_logits, target, current)
            precurrent = strict_counts(pre_logits, current, target)
            step = run_one_loop(
                model,
                controlled,
                loop_index=cfg.max_loops + cycle - 1,
            )
            post = strict_counts(step.logits, target, current)
            bucket = totals[cycle]
            for prefix, counts in (
                ("post", post),
                ("prewrite", prewrite),
                ("precurrent", precurrent),
            ):
                bucket[f"{prefix}_correct"] += counts[0]
                bucket[f"{prefix}_count"] += counts[1]
            state = step.state
    return [
        {
            "cycle": cycle,
            "strict_twohop_accuracy": bucket["post_correct"] / bucket["post_count"],
            "strict_prewrite_accuracy": bucket["prewrite_correct"]
            / bucket["prewrite_count"],
            "strict_precurrent_accuracy": bucket["precurrent_correct"]
            / bucket["precurrent_count"],
        }
        for cycle, bucket in totals.items()
    ]


def curve_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["strict_twohop_accuracy"] for row in rows]
    return {
        "accuracy_by_cycle": values,
        "auc_1_8": sum(values[:8]) / min(8, len(values)),
        "auc_1_16": sum(values[:16]) / min(16, len(values)),
        "auc_1_32": sum(values[:32]) / min(32, len(values)),
        "auc_all": sum(values) / len(values),
        "final_accuracy": values[-1],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-controllers", type=Path, required=True)
    parser.add_argument("--initial-rank", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-training-horizon", type=int, default=32)
    parser.add_argument("--round-limit", type=int)
    parser.add_argument("--round-multiplier", type=int, default=1)
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--optimization-mode",
        choices=("bptt", "dagger_detached"),
        default="dagger_detached",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("uniform", "late_linear", "final_only"),
        default="uniform",
    )
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-examples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-cycles", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=202608095)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.12)
    return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    if device.type == "cuda": 
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    if (cfg.max_depth, cfg.max_loops, cfg.n_layers) != (8, 6, 2):
        raise ValueError("trainer requires D8L6 with two shared blocks")
    model.eval()
    model.requires_grad_(False)
    initial = load_controller(
        args.initial_controllers,
        mode="two",
        rank=args.initial_rank,
        device=device,
    )
    module = DiagonalIdentityLoRAJ(dimension=cfg.d_model, rank=args.rank).to(device)
    retained = module.initialize_from_affine_svd(
        initial,
        gauge_seed=7001 + args.initialization_seed,
        diagonal_scale_init="one",
    )
    stages = selected_stages(
        args.max_training_horizon,
        args.round_limit,
        round_multiplier=args.round_multiplier,
        learning_rate_multiplier=args.learning_rate_multiplier,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    stage_curves: dict[str, dict[str, Any]] = {}
    for stage in stages:
        training_rows.extend(
            train_stage(
                module=module,
                stage=stage,
                model=model,
                cfg=cfg,
                device=device,
                initialization_seed=args.initialization_seed,
                grad_clip=args.grad_clip,
                optimization_mode=args.optimization_mode,
                loss_weighting=args.loss_weighting,
            )
        )
        stage_rows = evaluate(
            module=module,
            model=model,
            cfg=cfg,
            device=device,
            examples=min(args.eval_examples, 256),
            batch_size=args.eval_batch_size,
            cycles=args.eval_cycles,
            seed=args.eval_seed + 1000 * len(stage_curves),
        )
        stage_curves[stage.name] = curve_summary(stage_rows)
        torch.save(
            {
                "kind": "d8l6_twohop_diagonal_lora_boundary_j",
                "checkpoint": str(args.checkpoint),
                "stage": stage.name,
                "rank": args.rank,
                "state_dict": {
                    key: value.detach().cpu() for key, value in module.state_dict().items()
                },
            },
            args.out_dir / f"controller_{stage.name}.pt",
        )

    final_rows = evaluate(
        module=module,
        model=model,
        cfg=cfg,
        device=device,
        examples=args.eval_examples,
        batch_size=args.eval_batch_size,
        cycles=args.eval_cycles,
        seed=args.eval_seed,
    )
    final_curve = curve_summary(final_rows)
    artifact = {
        "kind": "d8l6_twohop_diagonal_lora_boundary_j",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "initial_controllers": str(args.initial_controllers),
        "initial_controllers_sha256": file_sha256(args.initial_controllers),
        "initial_rank": args.initial_rank,
        "rank": args.rank,
        "parameterization": "J(h)=hD+(hA)B+b with diagonal D",
        "placement": "loop_boundary",
        "loss": "pure successor cross-entropy at every on-policy composition step",
        "optimization_mode": args.optimization_mode,
        "loss_weighting": args.loss_weighting,
        "state_loss_weight": 0.0,
        "programmed_jump": 2,
        "state_dict": {
            key: value.detach().cpu() for key, value in module.state_dict().items()
        },
    }
    artifact_path = args.out_dir / "controller_final.pt"
    torch.save(artifact, artifact_path)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": checkpoint_payload.get("loss_mode", "final_only"),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller_artifact": str(artifact_path),
        "controller_parameterization": artifact["parameterization"],
        "controller_rank": args.rank,
        "controller_parameters": sum(p.numel() for p in module.parameters()),
        "controller_placement": artifact["placement"],
        "loss": artifact["loss"],
        "optimization_mode": args.optimization_mode,
        "loss_weighting": args.loss_weighting,
        "state_loss_weight": 0.0,
        "initialization_retained_energy": retained,
        "initialization_seed": args.initialization_seed,
        "stages": [asdict(stage) for stage in stages],
        "training_rows": training_rows,
        "stage_curves": stage_curves,
        "final_curve": final_curve,
        "evaluation_examples": args.eval_examples,
        "evaluation_seed": args.eval_seed,
        "evaluation_cycles": args.eval_cycles,
        "interpretation_boundary": (
            "All training targets are two-hop successor labels on the controller's "
            "own rollout. No hidden-state or oracle-interface loss is used."
        ),
    }
    write_csv(args.out_dir / "training.csv", training_rows)
    write_csv(args.out_dir / "final_curve.csv", final_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_experiment(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
