from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
    evaluate_partition,
)


def _affine(weight: torch.Tensor, bias: torch.Tensor) -> VectorAffine:
    dimension = int(weight.shape[0])
    return VectorAffine(
        weight=weight.float(),
        bias=bias.float(),
        update_rank=dimension,
        fit_dimension=dimension,
        retained_fit_energy=1.0,
    )


def run_experiment(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]

    payload = torch.load(args.artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_task_lora_j":
        raise ValueError("unexpected controller artifact kind")
    if payload["checkpoint"] != str(args.checkpoint):
        raise ValueError("controller and backbone checkpoints differ")
    item = payload["modules"][args.label]
    if item.get("parameterization") != "diagonal_low_rank":
        raise ValueError("ablation requires a diagonal_low_rank controller")
    state = item["state_dict"]
    left = state["A"].to(device=device, dtype=torch.float32)
    right = state["B"].to(device=device, dtype=torch.float32)
    diagonal = state["diagonal_scale"].to(device=device, dtype=torch.float32)
    bias = state["bias"].to(device=device, dtype=torch.float32)
    dimension = int(item["dimension"])
    identity = torch.eye(dimension, device=device)
    zero_bias = torch.zeros_like(bias)
    correction = left @ right
    diagonal_weight = torch.diag(diagonal)
    mean_weight = diagonal.mean() * identity
    generator = torch.Generator(device=device)
    generator.manual_seed(args.shuffle_seed)
    permutation = torch.randperm(
        dimension, generator=generator, device=device
    )
    shuffled_diagonal = torch.diag(diagonal[permutation])

    operators = {
        "identity_no_J": lambda value: value,
        "full_D_plus_AB_plus_b": _affine(
            diagonal_weight + correction, bias
        ),
        "D_plus_b": _affine(diagonal_weight, bias),
        "I_plus_AB_plus_b": _affine(identity + correction, bias),
        "meanD_I_plus_AB_plus_b": _affine(mean_weight + correction, bias),
        "shuffledD_plus_AB_plus_b": _affine(
            shuffled_diagonal + correction, bias
        ),
        "D_plus_AB_no_bias": _affine(
            diagonal_weight + correction, zero_bias
        ),
        "D_no_bias": _affine(diagonal_weight, zero_bias),
        "I_plus_AB_no_bias": _affine(identity + correction, zero_bias),
        "AB_plus_b_no_base": _affine(correction, bias),
    }

    streams = _canonical_training_streams(payload)
    seen, unique_after_stage, training_draws = (
        reconstruct_primary_training_graphs(
            device=device,
            node_count=cfg.node_count,
            streams=streams,
        )
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    _, sampled_unseen, distribution = stratified_samples(
        seen,
        unseen,
        count=args.sample_per_partition,
        seed=args.sample_seed,
    )
    evaluation = evaluate_partition(
        permutations=sampled_unseen,
        operators=operators,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        placement="loop_boundary",
    )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            f"frozen backbone: {payload.get('backbone_loss_description', 'not recorded')}; "
            "controller loss as recorded in source artifact"
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller_placement": "loop_boundary",
        "source_artifact": str(args.artifact),
        "source_label": args.label,
        "rank": int(item["rank"]),
        "component_definition": {
            "D_mean": float(diagonal.mean()),
            "D_std": float(diagonal.std(unbiased=False)),
            "AB_frobenius_norm": float(correction.norm()),
            "bias_norm": float(bias.norm()),
            "shuffle_seed": args.shuffle_seed,
        },
        "graph_universe": len(universe),
        "training_graph_draws": training_draws,
        "training_graph_streams": streams,
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs": len(unseen),
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "strictly_unseen_permutations": args.sample_per_partition,
            "all_starts_per_permutation": cfg.node_count,
            "examples": args.sample_per_partition * cfg.node_count,
            "matched_cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "strictly_unseen": evaluation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal component ablations for diagonal plus low-rank J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--shuffle-seed", type=int, default=20260801)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_experiment(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
