from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

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
    _routing_metrics,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    _component_similarity,
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
)
from reasoning_loop.graph_path_telomere_probe_dose import (
    _aggregate_rows,
    _validate_probe_cycles,
)
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _write_csv,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _load_booster(
    path: Path,
    *,
    device: torch.device,
) -> tuple[tuple[int, ...], VectorAffine]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_periodic_booster":
        raise ValueError("unexpected periodic-booster artifact kind")
    values = payload["map"]
    weight = values["weight"].to(device)
    return (
        tuple(int(position) for position in payload["positions"]),
        VectorAffine(
            weight=weight,
            bias=values["bias"].to(device),
            update_rank=int(values["rank"]),
            fit_dimension=int(weight.shape[0]),
            retained_fit_energy=float(values["retained_fit_energy"]),
        ),
    )


def _map_geometry(
    first: VectorAffine,
    second: VectorAffine,
) -> dict[str, float]:
    identity = torch.eye(
        first.weight.shape[0],
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    first_update = first.weight - identity
    second_update = second.weight - identity
    return {
        "update_cosine": float(
            torch.nn.functional.cosine_similarity(
                first_update.flatten(),
                second_update.flatten(),
                dim=0,
            )
        ),
        "bias_cosine": float(
            torch.nn.functional.cosine_similarity(
                first.bias,
                second.bias,
                dim=0,
            )
        ),
        "first_update_norm": float(torch.linalg.norm(first_update)),
        "second_update_norm": float(torch.linalg.norm(second_update)),
        "first_bias_norm": float(torch.linalg.norm(first.bias)),
        "second_bias_norm": float(torch.linalg.norm(second.bias)),
    }


def _interpolate_affine(
    first: VectorAffine,
    second: VectorAffine,
    fraction: float,
) -> VectorAffine:
    weight = first.weight + fraction * (second.weight - first.weight)
    bias = first.bias + fraction * (second.bias - first.bias)
    return VectorAffine(
        weight=weight,
        bias=bias,
        update_rank=int(weight.shape[0]),
        fit_dimension=int(weight.shape[0]),
        retained_fit_energy=float("nan"),
    )


def _mean_flat_cosine(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    if first.shape != second.shape:
        raise ValueError("correction tensors must have matching shapes")
    return float(
        torch.nn.functional.cosine_similarity(
            first.float().flatten(start_dim=1),
            second.float().flatten(start_dim=1),
            dim=1,
        ).mean()
    )


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    feedback_artifact: Path,
    feedback_label: str,
    booster24_artifact: Path,
    booster32_artifact: Path,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    extra_loops: int,
    probe_cycles: Sequence[int],
    seed: int,
    executor_head: int,
) -> dict[str, Any]:
    probes = _validate_probe_cycles(
        probe_cycles,
        extra_loops=extra_loops,
    )
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")),
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
    jump = phase_positions[3] - phase_positions[2]
    feedback, interface = _load_map(
        feedback_artifact,
        label=feedback_label,
        device=device,
    )
    booster24_positions, booster24 = _load_booster(
        booster24_artifact,
        device=device,
    )
    booster32_positions, booster32 = _load_booster(
        booster32_artifact,
        device=device,
    )
    expected = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if (
        interface != expected
        or booster24_positions != expected
        or booster32_positions != expected
    ):
        raise ValueError("map positions do not match the Block2 interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    role_names = (
        "edge_marker",
        "source",
        "destination",
        "query_metadata",
        "answer",
    )
    global_to_local = {
        position: index for index, position in enumerate(interface)
    }
    role_local_positions = {
        role: tuple(
            global_to_local[position] for position in groups[role]
        )
        for role in role_names
    }
    rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    set_seed(seed)
    for batch_index in range(batches):
        _, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * extra_loops,
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
        for cycle in range(1, extra_loops + 1):
            current = all_targets[
                :, cfg.max_depth + jump * (cycle - 1)
            ]
            target = all_targets[:, cfg.max_depth + jump * cycle]
            oracle_input = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=2,
                phase_position=phase_positions[2],
            )
            oracle = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            feedback_transform = (
                None if cycle == 1 else (interface, feedback)
            )
            standard = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=feedback_transform,
            )
            if cycle in probes:
                age_fraction = (cycle - 24) / (32 - 24)
                age_linear_reset = _interpolate_affine(
                    booster24,
                    booster32,
                    age_fraction,
                )
                live_interface = standard.block2_hidden_pre_intervention[
                    :, list(interface)
                ].float()
                reset24_update = booster24(live_interface) - live_interface
                reset32_update = booster32(live_interface) - live_interface
                age_linear_update = (
                    age_linear_reset(live_interface) - live_interface
                )
                exact_update = (
                    oracle.block2_hidden_in[:, list(interface)].float()
                    - live_interface
                )
                trend_row = {
                        "batch": batch_index,
                        "cycle": cycle,
                        "reset24_to_reset32_update_cosine": (
                            _mean_flat_cosine(
                                reset24_update,
                                reset32_update,
                            )
                        ),
                        "reset24_to_exact_update_cosine": (
                            _mean_flat_cosine(
                                reset24_update,
                                exact_update,
                            )
                        ),
                        "reset32_to_exact_update_cosine": (
                            _mean_flat_cosine(
                                reset32_update,
                                exact_update,
                            )
                        ),
                        "age_linear_to_exact_update_cosine": (
                            _mean_flat_cosine(
                                age_linear_update,
                                exact_update,
                            )
                        ),
                        "reset24_update_norm": float(
                            torch.linalg.vector_norm(
                                reset24_update.flatten(start_dim=1),
                                dim=1,
                            ).mean()
                        ),
                        "reset32_update_norm": float(
                            torch.linalg.vector_norm(
                                reset32_update.flatten(start_dim=1),
                                dim=1,
                            ).mean()
                        ),
                        "age_linear_update_norm": float(
                            torch.linalg.vector_norm(
                                age_linear_update.flatten(start_dim=1),
                                dim=1,
                            ).mean()
                        ),
                        "exact_update_norm": float(
                            torch.linalg.vector_norm(
                                exact_update.flatten(start_dim=1),
                                dim=1,
                            ).mean()
                        ),
                    }
                for role, local_positions in role_local_positions.items():
                    local = list(local_positions)
                    trend_row[
                        f"reset32_to_exact_{role}_update_cosine"
                    ] = _mean_flat_cosine(
                        reset32_update[:, local],
                        exact_update[:, local],
                    )
                    trend_row[
                        f"age_linear_to_exact_{role}_update_cosine"
                    ] = _mean_flat_cosine(
                        age_linear_update[:, local],
                        exact_update[:, local],
                    )
                trend_rows.append(trend_row)
                steps = {
                    "feedback": standard,
                    "booster_trained_cycle24": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(interface, booster24),
                    ),
                    "booster_trained_cycle32": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(interface, booster32),
                    ),
                    "linear_age_extrapolation": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            interface,
                            age_linear_reset,
                        ),
                    ),
                    "exact_interface": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_override=(
                            interface,
                            oracle.block2_hidden_in[:, list(interface)],
                        ),
                    ),
                }
                for condition, step in steps.items():
                    metrics = _masked_metrics(
                        step.logits,
                        target,
                        endpoint=endpoint,
                    )
                    mass, hit = _routing_metrics(
                        step.block2_pattern,
                        current=current,
                        destination_positions=destination_positions,
                        executor_head=executor_head,
                        answer_position=answer_position,
                    )
                    rows.append(
                        {
                            "batch": batch_index,
                            "cycle": cycle,
                            "condition": condition,
                            "strength": None,
                            "valid_count": metrics["valid_count"],
                            "accuracy": metrics["accuracy"],
                            "margin": metrics["margin"],
                            "head2_correct_destination_mass": mass,
                            "head2_correct_destination_argmax": hit,
                            **_component_similarity(
                                step,
                                oracle,
                                answer_position=answer_position,
                                destination_positions=destination_positions,
                                executor_head=executor_head,
                            ),
                        }
                    )
            state = standard.state
    aggregate = _aggregate_rows(rows)
    trend_summary = []
    for cycle in probes:
        parts = [
            row for row in trend_rows if int(row["cycle"]) == cycle
        ]
        trend_summary.append(
            {
                "cycle": cycle,
                **{
                    key: sum(float(row[key]) for row in parts) / len(parts)
                    for key in trend_rows[0]
                    if key not in {"batch", "cycle"}
                },
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "reset_age_cross_rows.csv", rows)
    _write_csv(out_dir / "reset_age_cross_summary.csv", aggregate)
    _write_csv(out_dir / "reset_age_trend_rows.csv", trend_rows)
    _write_csv(out_dir / "reset_age_trend_summary.csv", trend_summary)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "feedback_artifact": str(feedback_artifact),
        "booster24_artifact": str(booster24_artifact),
        "booster32_artifact": str(booster32_artifact),
        "graphs": batch_size * batches,
        "probe_cycles": list(probes),
        "seed": seed,
        "map_geometry": _map_geometry(booster24, booster32),
        "reset_age_cross": aggregate,
        "functional_trend": trend_summary,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate cycle-24 and cycle-32 reset maps."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--booster24-artifact", type=Path, required=True)
    parser.add_argument("--booster32-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument(
        "--probe-cycles",
        type=int,
        nargs="+",
        default=(24, 32, 48, 64),
    )
    parser.add_argument("--seed", type=int, default=154101)
    parser.add_argument("--executor-head", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        feedback_artifact=args.feedback_artifact,
        feedback_label=args.feedback_label,
        booster24_artifact=args.booster24_artifact,
        booster32_artifact=args.booster32_artifact,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        probe_cycles=args.probe_cycles,
        seed=args.seed,
        executor_head=args.executor_head,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
