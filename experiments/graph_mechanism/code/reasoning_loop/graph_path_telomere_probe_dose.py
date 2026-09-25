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
    _routing_metrics,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_map_ablation import (
    _scaled_map,
    _strength_label,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    _component_similarity,
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
)
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _write_csv,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import (
    _load_map,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _validate_probe_cycles(
    probe_cycles: Sequence[int],
    *,
    extra_loops: int,
) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in probe_cycles}))
    if not result:
        raise ValueError("probe_cycles must not be empty")
    if result[0] < 2:
        raise ValueError("probe cycles start at 2 because cycle 1 is exact H2")
    if result[-1] > extra_loops:
        raise ValueError("probe cycle exceeds extra_loops")
    return result


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["cycle"]),
            str(row["condition"]),
            (
                None
                if row["strength"] is None
                else float(row["strength"])
            ),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    metric_keys = (
        "accuracy",
        "margin",
        "head2_correct_destination_mass",
        "head2_correct_destination_argmax",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    for (cycle, condition, strength), parts in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            -99.0 if item[0][2] is None else item[0][2],
            item[0][1],
        ),
    ):
        weights = [int(part["valid_count"]) for part in parts]
        total = sum(weights)
        row: dict[str, Any] = {
            "cycle": cycle,
            "condition": condition,
            "strength": strength,
            "valid_count": total,
        }
        for key in metric_keys:
            row[key] = (
                sum(float(part[key]) * weight for part, weight in zip(parts, weights))
                / max(total, 1)
            )
        output.append(row)
    return output


@torch.no_grad()
def evaluate_probe_dose(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    map_artifact: Path,
    map_label: str,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    extra_loops: int,
    probe_cycles: Sequence[int],
    strengths: Sequence[float],
    seed: int,
    executor_head: int,
) -> dict[str, Any]:
    probes = _validate_probe_cycles(
        probe_cycles,
        extra_loops=extra_loops,
    )
    if not strengths:
        raise ValueError("strengths must not be empty")
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
    jump = phase_positions[3] - phase_positions[2]
    age_map, artifact_positions = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if artifact_positions != interface:
        raise ValueError("map positions do not match the Block2 interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    transforms = {
        float(strength): _scaled_map(age_map, float(strength))
        for strength in strengths
    }
    standard_map = _scaled_map(age_map, 1.0)
    rows: list[dict[str, Any]] = []
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
            oracle_step = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            standard_step = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    None
                    if cycle == 1
                    else (interface, standard_map)
                ),
            )
            if cycle in probes:
                steps = {
                    f"strength_{_strength_label(strength)}": run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + cycle - 1,
                        block2_position_transform=(
                            interface,
                            transform,
                        ),
                    )
                    for strength, transform in transforms.items()
                }
                steps["oracle_interface"] = run_one_loop(
                    model,
                    state,
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=(
                        interface,
                        oracle_step.block2_hidden_in[:, list(interface)],
                    ),
                )
                for condition, step in steps.items():
                    strength = (
                        None
                        if condition == "oracle_interface"
                        else float(
                            next(
                                value
                                for value in transforms
                                if condition
                                == f"strength_{_strength_label(value)}"
                            )
                        )
                    )
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
                            "strength": strength,
                            "valid_count": metrics["valid_count"],
                            "accuracy": metrics["accuracy"],
                            "margin": metrics["margin"],
                            "head2_correct_destination_mass": mass,
                            "head2_correct_destination_argmax": hit,
                            **_component_similarity(
                                step,
                                oracle_step,
                                answer_position=answer_position,
                                destination_positions=destination_positions,
                                executor_head=executor_head,
                            ),
                        }
                    )
            state = standard_step.state
    aggregate = _aggregate_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "probe_dose_rows.csv", rows)
    _write_csv(out_dir / "probe_dose_summary.csv", aggregate)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "probe_cycles": list(probes),
        "strengths": [float(value) for value in strengths],
        "seed": seed,
        "probe_dose": aggregate,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "rows": "probe_dose_rows.csv",
            "summary": "probe_dose_summary.csv",
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
            "Probe one-step controller dose on states generated by the "
            "standard alpha=1 D8L8 feedback trajectory."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--map-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=64)
    parser.add_argument(
        "--probe-cycles",
        type=int,
        nargs="+",
        default=(2, 8, 16, 24, 32, 48, 64),
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=(-1, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2),
    )
    parser.add_argument("--seed", type=int, default=130101)
    parser.add_argument("--executor-head", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_probe_dose(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        map_artifact=args.map_artifact,
        map_label=args.map_label,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        probe_cycles=args.probe_cycles,
        strengths=args.strengths,
        seed=args.seed,
        executor_head=args.executor_head,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
