from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import (
    _relative_mse,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import _all_targets
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map
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


def _flatten_role(
    value: torch.Tensor,
    positions: tuple[int, ...],
) -> torch.Tensor:
    return value[:, list(positions)].reshape(value.shape[0], -1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().flatten(1),
            right.float().flatten(1),
            dim=-1,
        ).mean()
    )


def _norm_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = left.float().flatten(1).norm(dim=-1)
    right_norm = right.float().flatten(1).norm(dim=-1).clamp_min(1e-12)
    return float((left_norm / right_norm).mean())


@torch.no_grad()
def evaluate_role_action(
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
    seed: int,
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
    probe_cycles = tuple(sorted({int(value) for value in probe_cycles}))
    if not probe_cycles or probe_cycles[0] < 2:
        raise ValueError("probe cycles must start after the first map use")
    if probe_cycles[-1] > extra_loops:
        raise ValueError("probe cycle exceeds rollout horizon")
    age_map, interface = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    expected_interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if interface != expected_interface:
        raise ValueError("map positions do not match the Block2 interface")
    groups = explicit_depth_position_groups(cfg.node_count)
    roles = {
        "edge_marker": groups["edge_marker"],
        "source": groups["source"],
        "destination": groups["destination"],
        "metadata": groups["query_metadata"],
        "answer": groups["answer"],
    }
    identity = torch.eye(
        age_map.weight.shape[0],
        device=device,
        dtype=age_map.weight.dtype,
    )
    update = age_map.weight - identity
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
            mapped = run_one_loop(
                model,
                state,
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(
                    (interface, age_map) if cycle > 1 else None
                ),
            )
            if cycle in probe_cycles:
                before = mapped.block2_hidden_pre_intervention
                after = mapped.block2_hidden_in
                target = oracle.block2_hidden_in
                update_part = before.float() @ update
                bias_part = age_map.bias.view(1, 1, -1).expand_as(before)
                for role, positions in roles.items():
                    before_role = _flatten_role(before, positions)
                    after_role = _flatten_role(after, positions)
                    target_role = _flatten_role(target, positions)
                    desired = target_role - before_role
                    correction = after_role - before_role
                    update_role = _flatten_role(update_part, positions)
                    bias_role = _flatten_role(bias_part, positions)
                    rows.append(
                        {
                            "batch": batch_index,
                            "cycle": cycle,
                            "role": role,
                            "positions": len(positions),
                            "pre_relative_mse": _relative_mse(
                                before_role,
                                target_role,
                            ),
                            "post_relative_mse": _relative_mse(
                                after_role,
                                target_role,
                            ),
                            "correction_to_desired_cosine": _cosine(
                                correction,
                                desired,
                            ),
                            "update_to_desired_cosine": _cosine(
                                update_role,
                                desired,
                            ),
                            "bias_to_desired_cosine": _cosine(
                                bias_role,
                                desired,
                            ),
                            "correction_to_desired_norm_ratio": _norm_ratio(
                                correction,
                                desired,
                            ),
                            "update_to_correction_norm_ratio": _norm_ratio(
                                update_role,
                                correction,
                            ),
                            "bias_to_correction_norm_ratio": _norm_ratio(
                                bias_role,
                                correction,
                            ),
                        }
                    )
            state = mapped.state

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    fields = (
        "pre_relative_mse",
        "post_relative_mse",
        "correction_to_desired_cosine",
        "update_to_desired_cosine",
        "bias_to_desired_cosine",
        "correction_to_desired_norm_ratio",
        "update_to_correction_norm_ratio",
        "bias_to_correction_norm_ratio",
    )
    for role in roles:
        aggregate[role] = {}
        for cycle in probe_cycles:
            parts = [
                row
                for row in rows
                if row["role"] == role and row["cycle"] == cycle
            ]
            aggregate[role][str(cycle)] = {
                field: float(
                    np.mean([float(part[field]) for part in parts])
                )
                for field in fields
            }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "role_action_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "map_rank": age_map.update_rank,
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "probe_cycles": list(probe_cycles),
        "roles": {name: list(value) for name, value in roles.items()},
        "seed": seed,
        "aggregate": aggregate,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "role_action_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose one shared affine map into xDelta and bias actions on "
            "answer, graph, and metadata role distributions during rollout."
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
    parser.add_argument("--seed", type=int, default=98101)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_role_action(
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
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
