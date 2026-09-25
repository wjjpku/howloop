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
    _routing_metrics,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    _component_similarity,
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _masked_metrics,
    advance_nodes,
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


def _load_map(
    artifact: Path,
    *,
    label: str,
    device: torch.device,
) -> tuple[VectorAffine, tuple[int, ...]]:
    payload = torch.load(artifact, map_location=device, weights_only=False)
    if payload.get("kind") not in {
        "graph_path_telomere_shared_position_dagger",
        "graph_path_telomere_adjacent_age",
    }:
        raise ValueError("unexpected shared-map artifact kind")
    if label not in payload["maps"]:
        available = ", ".join(sorted(payload["maps"]))
        raise KeyError(f"map {label!r} not found; available: {available}")
    item = payload["maps"][label]
    weight = item["weight"].to(device=device, dtype=torch.float32)
    bias = item["bias"].to(device=device, dtype=torch.float32)
    age_map = VectorAffine(
        weight=weight,
        bias=bias,
        update_rank=int(item["rank"]),
        fit_dimension=int(weight.shape[0]),
        retained_fit_energy=float(item["retained_fit_energy"]),
    )
    return age_map, tuple(int(value) for value in payload["positions"])


def _group_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    positions: tuple[int, ...],
) -> float:
    return _relative_mse(
        prediction[:, list(positions)],
        target[:, list(positions)],
    )


@torch.no_grad()
def evaluate_power_law(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    map_artifact: Path,
    map_label: str,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    min_age: int,
    max_age: int,
    seed: int,
    executor_head: int,
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
    if not 3 <= min_age <= max_age < len(phase_positions):
        raise ValueError("ages must lie between 3 and the cached maximum age")
    age_map, positions = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    expected_positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if positions != expected_positions:
        raise ValueError("map positions do not match the Block2 interface")

    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    destination_positions = groups["destination"]
    metric_groups = {
        "interface": positions,
        "answer": groups["answer"],
        "graph": groups["graph"],
        "metadata": groups["query_metadata"],
    }
    conditions = (
        "natural_aged",
        "map_once",
        "map_power",
        "oracle_age2",
        "oracle_interface_on_aged",
        "shuffled_map_power",
    )
    rows: list[dict[str, Any]] = []
    set_seed(seed)
    jump = phase_positions[3] - phase_positions[2]
    for batch_index in range(batches):
        _, targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump,
        )
        current = targets[:, cfg.max_depth - 1]
        target = advance_nodes(successors, current, steps=jump)
        h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        oracle = run_one_loop(model, h2, loop_index=cfg.max_loops)
        oracle_values = oracle.block2_hidden_in[:, list(positions)]
        for age in range(min_age, max_age + 1):
            aged = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=age,
                phase_position=phase_positions[age],
            )
            natural = run_one_loop(
                model,
                aged,
                loop_index=cfg.max_loops,
            )
            aged_values = natural.block2_hidden_pre_intervention[
                :, list(positions)
            ]
            once_values = age_map(aged_values)
            power_values = age_map.repeated(
                aged_values,
                count=age - 2,
            )
            steps = {
                "natural_aged": natural,
                "map_once": run_one_loop(
                    model,
                    aged,
                    loop_index=cfg.max_loops,
                    block2_position_override=(positions, once_values),
                ),
                "map_power": run_one_loop(
                    model,
                    aged,
                    loop_index=cfg.max_loops,
                    block2_position_override=(positions, power_values),
                ),
                "oracle_age2": oracle,
                "oracle_interface_on_aged": run_one_loop(
                    model,
                    aged,
                    loop_index=cfg.max_loops,
                    block2_position_override=(positions, oracle_values),
                ),
                "shuffled_map_power": run_one_loop(
                    model,
                    aged,
                    loop_index=cfg.max_loops,
                    block2_position_override=(
                        positions,
                        power_values.roll(1, dims=0),
                    ),
                ),
            }
            predictions = {
                "natural_aged": aged_values,
                "map_once": once_values,
                "map_power": power_values,
                "oracle_age2": oracle_values,
                "oracle_interface_on_aged": oracle_values,
                "shuffled_map_power": power_values.roll(1, dims=0),
            }
            for condition in conditions:
                step = steps[condition]
                metrics = _masked_metrics(
                    step.logits,
                    target,
                    endpoint=current,
                )
                mass, hit = _routing_metrics(
                    step.block2_pattern,
                    current=current,
                    destination_positions=destination_positions,
                    executor_head=executor_head,
                    answer_position=answer_position,
                )
                prediction = predictions[condition]
                group_errors = {
                    f"{name}_relative_mse": _group_relative_mse(
                        prediction,
                        oracle_values,
                        tuple(
                            positions.index(position)
                            for position in group_positions
                        ),
                    )
                    for name, group_positions in metric_groups.items()
                }
                rows.append(
                    {
                        "batch": batch_index,
                        "age": age,
                        "applications": {
                            "natural_aged": 0,
                            "map_once": 1,
                            "map_power": age - 2,
                            "oracle_age2": 0,
                            "oracle_interface_on_aged": 0,
                            "shuffled_map_power": age - 2,
                        }[condition],
                        "condition": condition,
                        "accuracy": metrics["accuracy"],
                        "margin": metrics["margin"],
                        "valid_count": metrics["valid_count"],
                        "head2_correct_destination_mass": mass,
                        "head2_correct_destination_argmax": hit,
                        **group_errors,
                        **_component_similarity(
                            step,
                            oracle,
                            answer_position=answer_position,
                            destination_positions=destination_positions,
                            executor_head=executor_head,
                        ),
                    }
                )

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    numeric_fields = (
        "accuracy",
        "margin",
        "head2_correct_destination_mass",
        "head2_correct_destination_argmax",
        "interface_relative_mse",
        "answer_relative_mse",
        "graph_relative_mse",
        "metadata_relative_mse",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    for condition in conditions:
        aggregate[condition] = {}
        for age in range(min_age, max_age + 1):
            parts = [
                row
                for row in rows
                if row["condition"] == condition and row["age"] == age
            ]
            aggregate[condition][str(age)] = {
                field: float(
                    np.average(
                        [float(part[field]) for part in parts],
                        weights=(
                            [float(part["valid_count"]) for part in parts]
                            if field in {"accuracy", "margin"}
                            else None
                        ),
                    )
                )
                for field in numeric_fields
            }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "power_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "map_rank": age_map.update_rank,
        "map_positions": list(positions),
        "ages": list(range(min_age, max_age + 1)),
        "graphs": batch_size * batches,
        "seed": seed,
        "aggregate": aggregate,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "power_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether one shared Block2-interface map obeys the "
            "compositional law R^(age-2)(z_age) approximately equals z_2."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--map-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--min-age", type=int, default=3)
    parser.add_argument("--max-age", type=int, default=8)
    parser.add_argument("--seed", type=int, default=90101)
    parser.add_argument("--executor-head", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_power_law(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        map_artifact=args.map_artifact,
        map_label=args.map_label,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        min_age=args.min_age,
        max_age=args.max_age,
        seed=args.seed,
        executor_head=args.executor_head,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
