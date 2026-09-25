from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    _relative_mse,
    collect_aligned_pairs,
    evaluate_closed_loop,
    fit_reduced_rank_update_family,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_executor_rollout_pairs(
    *,
    model,
    cfg,
    phase_positions: list[int],
    age_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    rollout_cycles: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect only the map's own off-manifold executor-input states."""

    if rollout_cycles < 2:
        raise ValueError("rollout_cycles must be at least two")
    jump = phase_positions[3] - phase_positions[2]
    answer_position = explicit_depth_position_groups(cfg.node_count)["answer"][0]
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * rollout_cycles,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        for cycle in range(1, rollout_cycles + 1):
            current = all_targets[
                :, cfg.max_depth + jump * (cycle - 1)
            ]
            oracle_input = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=2,
                phase_position=phase_positions[2],
            )
            oracle_step = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_input_transform=(age_map if cycle > 1 else None),
            )
            if cycle > 1:
                sources.append(
                    step.block2_hidden_pre_intervention[
                        :, answer_position
                    ].float()
                )
                targets.append(
                    oracle_step.block2_hidden_in[:, answer_position].float()
                )
            state = step.state
    return torch.cat(sources), torch.cat(targets)


@torch.no_grad()
def heldout_map_metrics(
    *,
    model,
    cfg,
    heldout,
    age_map: VectorAffine,
) -> dict[str, float]:
    prediction = age_map(heldout.z3_block2_input)
    step = run_one_loop(
        model,
        heldout.h3_full,
        loop_index=cfg.max_loops,
        block2_input_override=prediction,
    )
    metrics = _masked_metrics(
        step.logits,
        heldout.next_two_hop,
        endpoint=heldout.current,
    )
    return {
        "heldout_natural_relative_mse": _relative_mse(
            prediction,
            heldout.z2_block2_input,
        ),
        "heldout_natural_next_accuracy": float(metrics["accuracy"]),
        "heldout_natural_next_margin": float(metrics["margin"]),
    }


def _closed_curve_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["cycle"]))
        groups.setdefault(key, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for condition in sorted({key[0] for key in groups}):
        accuracy: list[float] = []
        routing_mass: list[float] = []
        routing_hit: list[float] = []
        for cycle in sorted(key[1] for key in groups if key[0] == condition):
            parts = groups[(condition, cycle)]
            count = sum(float(part["valid_count"]) for part in parts)
            accuracy.append(
                sum(
                    float(part["accuracy"]) * float(part["valid_count"])
                    for part in parts
                )
                / count
            )
            routing_mass.append(
                float(
                    np.mean(
                        [
                            float(part["head2_correct_destination_mass"])
                            for part in parts
                        ]
                    )
                )
            )
            routing_hit.append(
                float(
                    np.mean(
                        [
                            float(part["head2_correct_destination_argmax"])
                            for part in parts
                        ]
                    )
                )
            )
        result[condition] = {
            "accuracy": accuracy,
            "auc": float(np.mean(accuracy)),
            "routing_mass": routing_mass,
            "routing_argmax": routing_hit,
        }
    return result


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    heldout_batches: int,
    dagger_batch_size: int,
    dagger_batches: int,
    dagger_rounds: int,
    dagger_rollout_cycles: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    ranks: Sequence[int],
    ridge: float,
    calibration_seed: int,
    heldout_seed: int,
    dagger_seed: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("experiment is fixed to the D8L8 two-block model")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=calibration_seed,
        executor_head=2,
    )
    heldout = collect_aligned_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=heldout_seed,
        executor_head=2,
    )

    snapshots: dict[str, VectorAffine] = {}
    training_rows: list[dict[str, Any]] = []
    for rank in ranks:
        train_source = calibration.z3_block2_input
        train_target = calibration.z2_block2_input
        family = fit_reduced_rank_update_family(
            train_source,
            train_target,
            ridge=ridge,
        )
        age_map = family.map_for_rank(int(rank))
        snapshots[f"{rank}_round0"] = age_map
        metrics = heldout_map_metrics(
            model=model,
            cfg=cfg,
            heldout=heldout,
            age_map=age_map,
        )
        training_rows.append(
            {
                "rank": rank,
                "round": 0,
                "training_pairs": int(train_source.shape[0]),
                "new_rollout_pairs": 0,
                "rollout_pair_relative_mse_before_refit": float("nan"),
                **metrics,
            }
        )
        for round_index in range(1, dagger_rounds + 1):
            rollout_source, rollout_target = collect_executor_rollout_pairs(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                age_map=age_map,
                device=device,
                batch_size=dagger_batch_size,
                batches=dagger_batches,
                rollout_cycles=dagger_rollout_cycles,
                seed=dagger_seed + 1000 * int(rank) + round_index,
            )
            rollout_error = _relative_mse(
                age_map(rollout_source),
                rollout_target,
            )
            train_source = torch.cat((train_source, rollout_source))
            train_target = torch.cat((train_target, rollout_target))
            family = fit_reduced_rank_update_family(
                train_source,
                train_target,
                ridge=ridge,
            )
            age_map = family.map_for_rank(int(rank))
            snapshots[f"{rank}_round{round_index}"] = age_map
            metrics = heldout_map_metrics(
                model=model,
                cfg=cfg,
                heldout=heldout,
                age_map=age_map,
            )
            training_rows.append(
                {
                    "rank": rank,
                    "round": round_index,
                    "training_pairs": int(train_source.shape[0]),
                    "new_rollout_pairs": int(rollout_source.shape[0]),
                    "rollout_pair_relative_mse_before_refit": rollout_error,
                    **metrics,
                }
            )

    closed_rows = evaluate_closed_loop(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        residual_maps={},
        executor_maps=snapshots,  # type: ignore[arg-type]
        query_maps={},
        device=device,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=evaluation_seed,
        executor_head=2,
    )
    curves = _closed_curve_summary(closed_rows)
    map_curves = {
        condition: data
        for condition, data in curves.items()
        if condition.startswith("executor_r")
    }
    best = max(map_curves.items(), key=lambda item: item[1]["auc"])
    torch.save(
        {
            "kind": "graph_path_telomere_executor_dagger",
            "maps": {
                label: {
                    "weight": age_map.weight.cpu(),
                    "bias": age_map.bias.cpu(),
                    "rank": age_map.update_rank,
                    "retained_fit_energy": age_map.retained_fit_energy,
                }
                for label, age_map in snapshots.items()
            },
        },
        out_dir / "executor_dagger_maps.pt",
    )
    _write_csv(out_dir / "training_rows.csv", training_rows)
    _write_csv(out_dir / "closed_loop_rows.csv", closed_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": {
            "node_count": cfg.node_count,
            "max_depth": cfg.max_depth,
            "max_loops": cfg.max_loops,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
        },
        "device": str(device),
        "phase_positions": phase_positions,
        "training": {
            "natural_relation": "strict same-current Block2-input H3 -> H2",
            "natural_pairs": int(
                calibration_batch_size * calibration_batches
            ),
            "dagger_rounds": dagger_rounds,
            "rollout_cycles_per_round": dagger_rollout_cycles,
            "rollout_graphs_per_round": dagger_batch_size * dagger_batches,
            "ranks": [int(rank) for rank in ranks],
            "ridge": ridge,
            "loss": "closed-form executor-input state MSE only",
            "excluded": [
                "task CE",
                "long-horizon loss",
                "attention loss",
                "task labels in regression",
            ],
        },
        "heldout_examples": int(heldout_batch_size * heldout_batches),
        "evaluation_graphs": int(
            evaluation_batch_size * evaluation_batches
        ),
        "extra_loops": extra_loops,
        "training_rows": training_rows,
        "closed_loop": curves,
        "best_map_by_eval_auc_descriptive_only": {
            "condition": best[0],
            "auc": best[1]["auc"],
        },
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "maps": "executor_dagger_maps.pt",
            "training": "training_rows.csv",
            "closed_loop": "closed_loop_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one checkpoint-specific affine rejuvenator at the Block2 "
            "executor boundary using short-horizon DAgger closure."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=16)
    parser.add_argument("--heldout-batch-size", type=int, default=256)
    parser.add_argument("--heldout-batches", type=int, default=8)
    parser.add_argument("--dagger-batch-size", type=int, default=256)
    parser.add_argument("--dagger-batches", type=int, default=4)
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--dagger-rollout-cycles", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=32)
    parser.add_argument("--ranks", type=int, nargs="+", default=(32, 256))
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--calibration-seed", type=int, default=76101)
    parser.add_argument("--heldout-seed", type=int, default=76102)
    parser.add_argument("--dagger-seed", type=int, default=76103)
    parser.add_argument("--evaluation-seed", type=int, default=76104)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        heldout_batch_size=args.heldout_batch_size,
        heldout_batches=args.heldout_batches,
        dagger_batch_size=args.dagger_batch_size,
        dagger_batches=args.dagger_batches,
        dagger_rounds=args.dagger_rounds,
        dagger_rollout_cycles=args.dagger_rollout_cycles,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        ranks=args.ranks,
        ridge=args.ridge,
        calibration_seed=args.calibration_seed,
        heldout_seed=args.heldout_seed,
        dagger_seed=args.dagger_seed,
        evaluation_seed=args.evaluation_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
