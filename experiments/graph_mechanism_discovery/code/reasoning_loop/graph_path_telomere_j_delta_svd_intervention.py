from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_j_schur_intervention import (
    Map,
    _load_training_streams,
    _sha256,
    evaluate_partition,
)
from reasoning_loop.graph_path_telomere_localized_query import (
    VectorAffine,
    run_one_loop,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import (
    exact_interfaces,
    load_unit_j_map,
)


def _affine(
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> Map:
    return lambda value: value.float() @ weight + bias


def _truncated_delta(
    left: torch.Tensor,
    singular: torch.Tensor,
    right: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank == 0:
        return torch.zeros(
            (left.shape[0], right.shape[1]),
            device=left.device,
            dtype=left.dtype,
        )
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def _random_rank_delta(
    *,
    singular: torch.Tensor,
    rank: int,
    dimension: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left_raw = torch.randn(
        dimension,
        rank,
        generator=generator,
        dtype=torch.float32,
    )
    right_raw = torch.randn(
        dimension,
        rank,
        generator=generator,
        dtype=torch.float32,
    )
    left = torch.linalg.qr(left_raw, mode="reduced").Q.to(device)
    right = torch.linalg.qr(right_raw, mode="reduced").Q.to(device)
    return (left * singular[:rank]) @ right.T


def build_delta_maps(
    *,
    base_map: VectorAffine,
    target_map: VectorAffine,
    ranks: Sequence[int],
    random_ranks: Sequence[int],
    random_seeds: Sequence[int],
    device: torch.device,
) -> tuple[dict[str, Map], dict]:
    base_weight = base_map.weight.float()
    base_bias = base_map.bias.float()
    target_weight = target_map.weight.float()
    target_bias = target_map.bias.float()
    delta_weight = target_weight - base_weight
    delta_bias = target_bias - base_bias
    left, singular, right = torch.linalg.svd(
        delta_weight,
        full_matrices=False,
    )
    dimension = delta_weight.shape[0]
    rank_set = sorted(set(int(rank) for rank in ranks))
    if not rank_set or rank_set[0] < 0 or rank_set[-1] > dimension:
        raise ValueError("ranks must lie in [0, d_model]")

    maps: dict[str, Map] = {
        "base_J": base_map,
        "full_J": target_map,
        "bias_only": _affine(base_weight, target_bias),
        "weight_only": _affine(target_weight, base_bias),
    }
    energy = singular.square()
    total_energy = energy.sum().clamp_min(1e-20)
    rank_metadata = {}
    for rank in rank_set:
        learned = _truncated_delta(left, singular, right, rank)
        maps[f"top{rank}"] = _affine(
            base_weight + learned,
            target_bias,
        )
        maps[f"tail_after{rank}"] = _affine(
            target_weight - learned,
            target_bias,
        )
        rank_metadata[str(rank)] = {
            "retained_frobenius_energy": float(
                (energy[:rank].sum() / total_energy).item()
            ),
            "delta_weight_frobenius_norm": float(
                torch.linalg.matrix_norm(learned).item()
            ),
        }

    random_rank_set = sorted(set(int(rank) for rank in random_ranks))
    for rank in random_rank_set:
        if rank <= 0 or rank > dimension:
            raise ValueError("random ranks must lie in [1, d_model]")
        for seed in random_seeds:
            random_delta = _random_rank_delta(
                singular=singular,
                rank=rank,
                dimension=dimension,
                seed=int(seed),
                device=device,
            )
            maps[f"random{rank}s{seed}"] = _affine(
                base_weight + random_delta,
                target_bias,
            )

    metadata = {
        "base_weight_frobenius_norm": float(
            torch.linalg.matrix_norm(base_weight).item()
        ),
        "delta_weight_frobenius_norm": float(
            torch.linalg.matrix_norm(delta_weight).item()
        ),
        "relative_delta_weight_frobenius_norm": float(
            (
                torch.linalg.matrix_norm(delta_weight)
                / torch.linalg.matrix_norm(base_weight)
            ).item()
        ),
        "delta_weight_operator_norm": float(singular[0].item()),
        "delta_bias_norm": float(torch.linalg.vector_norm(delta_bias).item()),
        "relative_delta_bias_norm": float(
            (
                torch.linalg.vector_norm(delta_bias)
                / torch.linalg.vector_norm(base_bias)
            ).item()
        ),
        "singular_values": singular.detach().cpu().tolist(),
        "ranks": rank_metadata,
        "random_ranks": random_rank_set,
        "random_seeds": [int(seed) for seed in random_seeds],
    }
    return maps, metadata


def _relative_mse_per_example(
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    numerator = (value.float() - target.float()).square().mean(dim=(1, 2))
    centered = target.float() - target.float().mean(
        dim=(0, 1),
        keepdim=True,
    )
    denominator = centered.square().mean().clamp_min(1e-12)
    return numerator / denominator


@torch.no_grad()
def evaluate_trajectory_geometry(
    *,
    permutations,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    base_map: VectorAffine,
    target_map: VectorAffine,
    device: torch.device,
    batch_size: int,
    continuation_loops: int,
    selected_cycles: set[int],
    operating_age: int,
) -> dict:
    jump = phase_positions[3] - phase_positions[2]
    successors_all, starts_all = _expand_all_starts(
        permutations,
        device=device,
    )
    metric_names = (
        "base_relative_mse_to_H3",
        "target_relative_mse_to_H3",
        "fraction_H3_error_removed",
        "correction_cosine_with_base_residual",
        "correction_norm_ratio_to_base_residual",
        "correction_norm_ratio_to_target_update",
    )
    totals = {
        cycle: {name: 0.0 for name in metric_names}
        for cycle in sorted(selected_cycles)
    }
    counts = {cycle: 0 for cycle in selected_cycles}

    for offset in range(0, successors_all.shape[0], batch_size):
        successors = successors_all[offset : offset + batch_size]
        starts = starts_all[offset : offset + batch_size]
        current_batch = successors.shape[0]
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        states = {
            "base": initial.clone(),
            "target": initial.clone(),
        }
        for cycle in range(1, continuation_loops + 1):
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            oracle = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=successors,
                current=current,
                ages=(operating_age,),
                loop_index=cfg.max_loops + cycle - 1,
            )[operating_age]
            next_states = {}
            for label, age_map in (
                ("base", base_map),
                ("target", target_map),
            ):
                source_holder: dict[str, torch.Tensor] = {}

                def transform(
                    value: torch.Tensor,
                    *,
                    selected_map=age_map,
                ) -> torch.Tensor:
                    source_holder["value"] = value.detach()
                    return selected_map(value)

                step = run_one_loop(
                    model,
                    states[label],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=(positions, transform),
                )
                next_states[label] = step.state
                if cycle in selected_cycles and label == "target":
                    source = source_holder["value"]
                    base_output = base_map(source)
                    target_output = target_map(source)
                    correction = target_output - base_output
                    residual = oracle - base_output
                    base_error = _relative_mse_per_example(
                        base_output,
                        oracle,
                    )
                    target_error = _relative_mse_per_example(
                        target_output,
                        oracle,
                    )
                    correction_flat = correction.flatten(1)
                    residual_flat = residual.flatten(1)
                    target_update_flat = (
                        target_output - source.float()
                    ).flatten(1)
                    correction_norm = torch.linalg.vector_norm(
                        correction_flat,
                        dim=1,
                    )
                    residual_norm = torch.linalg.vector_norm(
                        residual_flat,
                        dim=1,
                    )
                    update_norm = torch.linalg.vector_norm(
                        target_update_flat,
                        dim=1,
                    )
                    cosine = (
                        (correction_flat * residual_flat).sum(dim=1)
                        / (
                            correction_norm
                            * residual_norm
                        ).clamp_min(1e-12)
                    )
                    row = totals[cycle]
                    row["base_relative_mse_to_H3"] += float(
                        base_error.sum().item()
                    )
                    row["target_relative_mse_to_H3"] += float(
                        target_error.sum().item()
                    )
                    row["fraction_H3_error_removed"] += float(
                        (
                            (base_error - target_error)
                            / base_error.clamp_min(1e-12)
                        )
                        .sum()
                        .item()
                    )
                    row[
                        "correction_cosine_with_base_residual"
                    ] += float(cosine.sum().item())
                    row[
                        "correction_norm_ratio_to_base_residual"
                    ] += float(
                        (
                            correction_norm
                            / residual_norm.clamp_min(1e-12)
                        )
                        .sum()
                        .item()
                    )
                    row[
                        "correction_norm_ratio_to_target_update"
                    ] += float(
                        (
                            correction_norm
                            / update_norm.clamp_min(1e-12)
                        )
                        .sum()
                        .item()
                    )
                    counts[cycle] += current_batch
            states = next_states

    return {
        str(cycle): {
            name: totals[cycle][name] / counts[cycle]
            for name in metric_names
        }
        for cycle in sorted(selected_cycles)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--base-j-artifact", type=Path, required=True)
    parser.add_argument("--target-j-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=731003)
    parser.add_argument("--operating-age", type=int, default=3)
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=(0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256),
    )
    parser.add_argument(
        "--random-ranks",
        nargs="+",
        type=int,
        default=(8, 16, 32, 64),
    )
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=(314159, 271828, 161803),
    )
    parser.add_argument(
        "--geometry-cycles",
        nargs="+",
        type=int,
        default=(1, 8, 16, 24, 32, 48, 64),
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("seen", "unseen"),
        default=("unseen",),
    )
    parser.add_argument("--training-streams-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if cfg.node_count != 8 or cfg.max_depth != 8 or cfg.max_loops != 8:
        raise ValueError("audit is fixed to the D8L8 N8 experiment")
    phase_summary = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_summary["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    base_map, base_checkpoint = load_unit_j_map(
        args.base_j_artifact,
        label=args.j_label,
        device=device,
    )
    target_map, target_checkpoint = load_unit_j_map(
        args.target_j_artifact,
        label=args.j_label,
        device=device,
    )
    if base_checkpoint != str(args.checkpoint):
        raise ValueError("base J and frozen model checkpoints differ")
    if target_checkpoint != str(args.checkpoint):
        raise ValueError("target J and frozen model checkpoints differ")
    maps, matrix_metadata = build_delta_maps(
        base_map=base_map,
        target_map=target_map,
        ranks=args.ranks,
        random_ranks=args.random_ranks,
        random_seeds=args.random_seeds,
        device=device,
    )

    training_streams = _load_training_streams(args.training_streams_json)
    seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
        device=device,
        node_count=cfg.node_count,
        streams=training_streams,
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    sampled_seen, sampled_unseen, distribution = stratified_samples(
        seen,
        unseen,
        count=args.sample_per_partition,
        seed=args.sample_seed,
    )
    requested = []
    if "seen" in args.partitions:
        requested.append(("seen_during_target_J_training", sampled_seen))
    if "unseen" in args.partitions:
        requested.append(
            ("strictly_unseen_by_target_J_training", sampled_unseen)
        )
    results = [
        evaluate_partition(
            label=label,
            permutations=sample,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            maps=maps,
            device=device,
            batch_size=args.batch_size,
            continuation_loops=args.continuation_loops,
            operating_age=args.operating_age,
        )
        for label, sample in requested
    ]
    geometry_sample = (
        sampled_unseen
        if "unseen" in args.partitions
        else sampled_seen
    )
    geometry = evaluate_trajectory_geometry(
        permutations=geometry_sample,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        base_map=base_map,
        target_map=target_map,
        device=device,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        selected_cycles={
            cycle
            for cycle in args.geometry_cycles
            if 1 <= cycle <= args.continuation_loops
        },
        operating_age=args.operating_age,
    )
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "loss_placement": (
            "loop8 final CE plus loop1-7 intermediate CE on p_min(2t,D)"
        ),
        "trained_macro_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_loops": args.continuation_loops,
        "base_J_artifact": str(args.base_j_artifact),
        "target_J_artifact": str(args.target_j_artifact),
        "J_label": args.j_label,
        "base_J_sha256": _sha256(args.base_j_artifact),
        "target_J_sha256": _sha256(args.target_j_artifact),
        "intervention": (
            "target J = base J + delta; top-k uses the first k singular "
            "modes of delta W and the full learned delta bias; tail-after-k "
            "removes those k modes from target J; random controls preserve "
            "the learned top-k singular values but randomize both subspaces"
        ),
        "matrix_delta": matrix_metadata,
        "graph_universe": len(universe),
        "training_graph_draws": total_draws,
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs": len(unseen),
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "per_partition": args.sample_per_partition,
            "all_starts_per_graph": cfg.node_count,
            "matched_cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "graph_steps_per_continuation_loop": (
            phase_positions[3] - phase_positions[2]
        ),
        "results": results,
        "trajectory_geometry_on_selected_partition": geometry,
        "gpu_runtime": {
            "observed_peak_reserved_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            )
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.out_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out_dir / "summary.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
