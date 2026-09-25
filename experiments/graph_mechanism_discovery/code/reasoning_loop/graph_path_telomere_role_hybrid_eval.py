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
from reasoning_loop.graph_path_telomere_learned_initializer import (
    _curve_summary,
    _write_csv,
)
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
from reasoning_loop.graph_path_telomere_periodic_booster import (
    _booster_cycles,
)
from reasoning_loop.graph_path_telomere_role_conditioned_booster import (
    ROLE_NAMES,
    RoleConditionedAffine,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _load_role_artifact(
    path: Path,
    *,
    device: torch.device,
) -> tuple[
    tuple[int, ...],
    dict[str, tuple[int, ...]],
    dict[str, VectorAffine],
    VectorAffine,
]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_role_conditioned_booster":
        raise ValueError("unexpected role-booster artifact kind")
    interface = tuple(int(value) for value in payload["positions"])
    role_positions = {
        role: tuple(int(value) for value in positions)
        for role, positions in payload["role_positions"].items()
    }

    def vector(values: dict[str, Any]) -> VectorAffine:
        return VectorAffine(
            weight=values["weight"].to(device),
            bias=values["bias"].to(device),
            update_rank=int(values["rank"]),
            fit_dimension=int(values["weight"].shape[0]),
            retained_fit_energy=float(values["retained_fit_energy"]),
        )

    role_maps = {
        role: vector(values)
        for role, values in payload["maps"].items()
    }
    shared = vector(payload["matched_shared_map"])
    return interface, role_positions, role_maps, shared


def _hybrid_transforms(
    *,
    interface: tuple[int, ...],
    role_positions: dict[str, tuple[int, ...]],
    role_maps: dict[str, VectorAffine],
    shared: VectorAffine,
    all_subsets: bool = False,
) -> dict[str, RoleConditionedAffine]:
    if all_subsets:
        selections = {
            f"subset_{mask:02d}": {
                role
                for index, role in enumerate(ROLE_NAMES)
                if mask & (1 << index)
            }
            for mask in range(1 << len(ROLE_NAMES))
        }
    else:
        selections = {
            "shared_all": set(),
            "answer_only": {"answer"},
            "metadata_only": {"query_metadata"},
            "edge_only": {"edge_marker"},
            "source_only": {"source"},
            "destination_only": {"destination"},
            "graph_roles": {"edge_marker", "source", "destination"},
            "answer_metadata": {"answer", "query_metadata"},
            "all_roles": set(ROLE_NAMES),
        }
    return {
        name: RoleConditionedAffine(
            interface=interface,
            role_positions=role_positions,
            maps={
                role: role_maps[role] if role in selected else shared
                for role in ROLE_NAMES
            },
        )
        for name, selected in selections.items()
    }


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    feedback_artifact: Path,
    feedback_label: str,
    role_artifact: Path,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    extra_loops: int,
    first_cycle: int,
    period: int,
    seed: int,
    executor_head: int,
    all_subsets: bool,
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
    jump = phase_positions[3] - phase_positions[2]
    interface, role_positions, role_maps, shared = _load_role_artifact(
        role_artifact,
        device=device,
    )
    expected = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if interface != expected:
        raise ValueError("role artifact positions do not match interface")
    feedback, feedback_positions = _load_map(
        feedback_artifact,
        label=feedback_label,
        device=device,
    )
    if feedback_positions != interface:
        raise ValueError("feedback positions do not match interface")
    hybrids = _hybrid_transforms(
        interface=interface,
        role_positions=role_positions,
        role_maps=role_maps,
        shared=shared,
        all_subsets=all_subsets,
    )
    scheduled = set(
        _booster_cycles(
            first_cycle=first_cycle,
            period=period,
            extra_loops=extra_loops,
        )
    )
    conditions = ("feedback_only", *hybrids, "exact_interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
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
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        states = {condition: initial.clone() for condition in conditions}
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
            oracle_values = oracle.block2_hidden_in[:, list(interface)]
            steps = {}
            for condition in conditions:
                transform = (
                    None
                    if cycle == 1
                    else (interface, feedback)
                )
                override = None
                if condition in hybrids and cycle in scheduled:
                    transform = (interface, hybrids[condition])
                elif condition == "exact_interface" and cycle in scheduled:
                    transform = None
                    override = (interface, oracle_values)
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=transform,
                    block2_position_override=override,
                )
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
                        "accuracy": metrics["accuracy"],
                        "margin": metrics["margin"],
                        "valid_count": metrics["valid_count"],
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
            states = {
                condition: steps[condition].state
                for condition in conditions
            }
    curves = _curve_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "role_hybrid_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "feedback_artifact": str(feedback_artifact),
        "feedback_label": feedback_label,
        "role_artifact": str(role_artifact),
        "hybrids": list(hybrids),
        "hybrid_roles": {
            name: [
                role
                for role in ROLE_NAMES
                if (
                    role_maps[role]
                    is hybrids[name].maps[role]
                )
            ]
            for name in hybrids
        },
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "first_cycle": first_cycle,
        "period": period,
        "seed": seed,
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "role_hybrid_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causally decompose which token-role-specific affine maps "
            "produce the periodic renewal gain."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--role-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=128)
    parser.add_argument("--first-cycle", type=int, default=24)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--seed", type=int, default=146101)
    parser.add_argument("--executor-head", type=int, default=2)
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Evaluate the full 2^5 factorial of role-specific maps.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        feedback_artifact=args.feedback_artifact,
        feedback_label=args.feedback_label,
        role_artifact=args.role_artifact,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        first_cycle=args.first_cycle,
        period=args.period,
        seed=args.seed,
        executor_head=args.executor_head,
        all_subsets=args.all_subsets,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
