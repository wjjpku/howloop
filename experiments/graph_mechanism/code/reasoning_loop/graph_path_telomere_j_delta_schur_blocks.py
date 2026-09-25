from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_j_delta_svd_intervention import (
    _affine,
    _random_rank_delta,
)
from reasoning_loop.graph_path_telomere_j_schur_intervention import (
    _load_training_streams,
    _sha256,
    evaluate_partition,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map


def build_block_maps(
    *,
    base_map,
    target_map,
    schur_payload: dict,
    random_seeds: tuple[int, ...],
    device: torch.device,
) -> tuple[dict, dict]:
    basis = schur_payload["basis"].to(device=device, dtype=torch.float32)
    identity = torch.eye(basis.shape[0], device=device)
    orthogonality_error = torch.linalg.matrix_norm(
        basis.T @ basis - identity
    )
    if float(orthogonality_error.item()) > 1e-4:
        raise ValueError("Schur basis is not orthogonal")
    band_bases = {}
    for label in ("zero", "one", "middle"):
        item = schur_payload["bands"][label]
        band_bases[label] = basis[:, int(item["start"]) : int(item["stop"])]

    base_weight = base_map.weight.float()
    base_bias = base_map.bias.float()
    target_weight = target_map.weight.float()
    target_bias = target_map.bias.float()
    delta = target_weight - base_weight
    blocks = {}
    block_energy = {}
    total_energy = torch.linalg.matrix_norm(delta).square().clamp_min(1e-20)
    for input_label, input_basis in band_bases.items():
        input_projector = input_basis @ input_basis.T
        for output_label, output_basis in band_bases.items():
            output_projector = output_basis @ output_basis.T
            label = f"{input_label}_to_{output_label}"
            block = input_projector @ delta @ output_projector
            blocks[label] = block
            block_energy[label] = float(
                (
                    torch.linalg.matrix_norm(block).square() / total_energy
                ).item()
            )
    reconstruction = sum(blocks.values())
    reconstruction_error = torch.linalg.matrix_norm(reconstruction - delta)
    if float(reconstruction_error.item()) > 1e-5:
        raise ValueError("Schur blocks do not reconstruct delta W")

    ordered = sorted(blocks, key=block_energy.get, reverse=True)
    maps = {
        "base_J": base_map,
        "full_J": target_map,
        "weight_only": _affine(target_weight, base_bias),
        "bias_only": _affine(base_weight, target_bias),
    }
    for label, block in blocks.items():
        maps[f"only_{label}"] = _affine(
            base_weight + block,
            base_bias,
        )
        maps[f"without_{label}"] = _affine(
            target_weight - block,
            base_bias,
        )

    cumulative = torch.zeros_like(delta)
    cumulative_labels = {}
    for index, label in enumerate(ordered, start=1):
        cumulative = cumulative + blocks[label]
        if index in (1, 2, 3, 4, 6, 9):
            cumulative_labels[str(index)] = ordered[:index]
            maps[f"cumulative{index}"] = _affine(
                base_weight + cumulative,
                base_bias,
            )
            maps[f"without_cumulative{index}"] = _affine(
                target_weight - cumulative,
                base_bias,
            )

    leading_block = blocks[ordered[0]]
    _, leading_singular, _ = torch.linalg.svd(
        leading_block,
        full_matrices=False,
    )
    leading_rank = int(
        (
            leading_singular
            > leading_singular[0] * 1e-6
        ).sum().item()
    )
    for seed in random_seeds:
        random_delta = _random_rank_delta(
            singular=leading_singular,
            rank=leading_rank,
            dimension=delta.shape[0],
            seed=seed,
            device=device,
        )
        maps[f"random_leading_s{seed}"] = _affine(
            base_weight + random_delta,
            base_bias,
        )

    metadata = {
        "bands": schur_payload["bands"],
        "block_energy_fraction": block_energy,
        "blocks_descending_energy": ordered,
        "cumulative_block_labels": cumulative_labels,
        "leading_block": ordered[0],
        "leading_block_numerical_rank": leading_rank,
        "random_seeds": list(random_seeds),
        "reconstruction_error": float(reconstruction_error.item()),
    }
    return maps, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--base-j-artifact", type=Path, required=True)
    parser.add_argument("--target-j-artifact", type=Path, required=True)
    parser.add_argument("--schur-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=731003)
    parser.add_argument("--operating-age", type=int, default=3)
    parser.add_argument(
        "--random-seeds",
        nargs="+",
        type=int,
        default=(314159, 271828, 161803),
    )
    parser.add_argument("--training-streams-json", type=Path, required=True)
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
        raise ValueError("base J and model checkpoints differ")
    if target_checkpoint != str(args.checkpoint):
        raise ValueError("target J and model checkpoints differ")
    schur_payload = torch.load(
        args.schur_artifact,
        map_location="cpu",
        weights_only=False,
    )
    maps, block_metadata = build_block_maps(
        base_map=base_map,
        target_map=target_map,
        schur_payload=schur_payload,
        random_seeds=tuple(args.random_seeds),
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
    _, sampled_unseen, distribution = stratified_samples(
        seen,
        unseen,
        count=args.sample_per_partition,
        seed=args.sample_seed,
    )
    result = evaluate_partition(
        label="strictly_unseen_by_target_J_training",
        permutations=sampled_unseen,
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
        "schur_artifact": str(args.schur_artifact),
        "base_J_sha256": _sha256(args.base_j_artifact),
        "target_J_sha256": _sha256(args.target_j_artifact),
        "intervention": (
            "delta W is decomposed as P_input delta W P_output in the "
            "original J ordered real-Schur basis; only-block adds one block "
            "to base J, without-block removes one block from target J"
        ),
        "delta_schur_blocks": block_metadata,
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
        "result": result,
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
