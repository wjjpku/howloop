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
from reasoning_loop.graph_path_telomere_shared_position_dagger import (
    _curve_summary,
    _write_csv,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import (
    _load_map,
)
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)


def _union(*parts: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({value for part in parts for value in part}))


def _map_variant(
    age_map: VectorAffine,
    variant: str,
) -> VectorAffine:
    identity = torch.eye(
        age_map.weight.shape[0],
        device=age_map.weight.device,
        dtype=age_map.weight.dtype,
    )
    if variant == "full":
        weight, bias = age_map.weight, age_map.bias
    elif variant == "no_bias":
        weight, bias = age_map.weight, torch.zeros_like(age_map.bias)
    elif variant == "bias_only":
        weight, bias = identity, age_map.bias
    elif variant == "reverse":
        weight = 2 * identity - age_map.weight
        bias = -age_map.bias
    else:
        raise ValueError(f"unknown map variant: {variant}")
    return VectorAffine(
        weight=weight,
        bias=bias,
        update_rank=age_map.update_rank,
        fit_dimension=age_map.fit_dimension,
        retained_fit_energy=age_map.retained_fit_energy,
    )


def _scaled_map(
    age_map: VectorAffine,
    strength: float,
) -> VectorAffine:
    identity = torch.eye(
        age_map.weight.shape[0],
        device=age_map.weight.device,
        dtype=age_map.weight.dtype,
    )
    return VectorAffine(
        weight=identity + strength * (age_map.weight - identity),
        bias=strength * age_map.bias,
        update_rank=age_map.update_rank,
        fit_dimension=age_map.fit_dimension,
        retained_fit_energy=age_map.retained_fit_energy,
    )


def _strength_label(strength: float) -> str:
    return (
        f"{strength:g}"
        .replace("-", "neg")
        .replace(".", "p")
    )


def _linear_diagnostics(age_map: VectorAffine) -> dict[str, float]:
    weight = age_map.weight.detach().float().cpu()
    bias = age_map.bias.detach().float().cpu()
    identity = torch.eye(weight.shape[0])
    update = weight - identity
    symmetric = (update + update.transpose(0, 1)) / 2
    skew = (update - update.transpose(0, 1)) / 2
    update_norm = torch.linalg.norm(update).clamp_min(1e-12)
    singular_values = torch.linalg.svdvals(weight)
    eigenvalues = torch.linalg.eigvals(weight)
    return {
        "update_to_identity_frobenius": float(
            update_norm / torch.linalg.norm(identity)
        ),
        "symmetric_update_fraction": float(
            torch.linalg.norm(symmetric) / update_norm
        ),
        "skew_update_fraction": float(
            torch.linalg.norm(skew) / update_norm
        ),
        "bias_l2": float(torch.linalg.norm(bias)),
        "weight_min_singular": float(singular_values.min()),
        "weight_max_singular": float(singular_values.max()),
        "weight_spectral_radius": float(eigenvalues.abs().max()),
    }


@torch.no_grad()
def evaluate_map_ablation(
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
    seed: int,
    executor_head: int,
    strengths: Sequence[float] = (),
    strength_only: bool = False,
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
    age_map, artifact_positions = _load_map(
        map_artifact,
        label=map_label,
        device=device,
    )
    groups = explicit_depth_position_groups(cfg.node_count)
    answer = groups["answer"]
    graph = groups["graph"]
    metadata = groups["query_metadata"]
    interface = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if artifact_positions != interface:
        raise ValueError("map positions do not match the Block2 interface")
    answer_position = answer[0]
    destination_positions = groups["destination"]
    full = _map_variant(age_map, "full")
    structural_transforms = {
        "map_full": (interface, full),
        "map_answer": (answer, full),
        "map_graph": (graph, full),
        "map_metadata": (metadata, full),
        "map_answer_graph": (_union(answer, graph), full),
        "map_answer_metadata": (_union(answer, metadata), full),
        "map_graph_metadata": (_union(graph, metadata), full),
        "map_full_no_bias": (
            interface,
            _map_variant(age_map, "no_bias"),
        ),
        "map_full_bias_only": (
            interface,
            _map_variant(age_map, "bias_only"),
        ),
        "map_full_reverse": (
            interface,
            _map_variant(age_map, "reverse"),
        ),
    }
    transforms = {} if strength_only else structural_transforms
    for strength in strengths:
        transforms[f"map_strength_{_strength_label(strength)}"] = (
            interface,
            _scaled_map(age_map, float(strength)),
        )
    conditions = (
        "no_intervention",
        "oracle_interface",
        "shuffled_oracle_interface",
        *transforms,
    )
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
            oracle_step = run_one_loop(
                model,
                oracle_input,
                loop_index=cfg.max_loops + cycle - 1,
            )
            oracle_values = oracle_step.block2_hidden_in[
                :, list(interface)
            ]
            steps = {
                "no_intervention": run_one_loop(
                    model,
                    states["no_intervention"],
                    loop_index=cfg.max_loops + cycle - 1,
                ),
                "oracle_interface": run_one_loop(
                    model,
                    states["oracle_interface"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=(
                        None
                        if cycle == 1
                        else (interface, oracle_values)
                    ),
                ),
                "shuffled_oracle_interface": run_one_loop(
                    model,
                    states["shuffled_oracle_interface"],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_override=(
                        None
                        if cycle == 1
                        else (interface, oracle_values.roll(1, dims=0))
                    ),
                ),
            }
            for condition, transform in transforms.items():
                steps[condition] = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(
                        transform if cycle > 1 else None
                    ),
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
                            oracle_step,
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
    _write_csv(out_dir / "map_ablation_rows.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "map_artifact": str(map_artifact),
        "map_label": map_label,
        "map_rank": age_map.update_rank,
        "map_positions": list(interface),
        "graphs": batch_size * batches,
        "extra_loops": extra_loops,
        "seed": seed,
        "strengths": [float(value) for value in strengths],
        "strength_only": strength_only,
        "linear_diagnostics": _linear_diagnostics(age_map),
        "closed_loop": curves,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {"rows": "map_ablation_rows.csv"},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate position groups, bias, and update terms of one learned "
            "shared rejuvenation map during long closed-loop execution."
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
    parser.add_argument("--seed", type=int, default=94101)
    parser.add_argument("--executor-head", type=int, default=2)
    parser.add_argument("--strengths", type=float, nargs="*", default=())
    parser.add_argument("--strength-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = evaluate_map_ablation(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        map_artifact=args.map_artifact,
        map_label=args.map_label,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        extra_loops=args.extra_loops,
        seed=args.seed,
        executor_head=args.executor_head,
        strengths=args.strengths,
        strength_only=args.strength_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
