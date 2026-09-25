from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    cycle_type,
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import intervention_groups
from reasoning_loop.graph_path_telomere_task_lora_j import load_task_lora_modules
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
    evaluate_partition,
)
from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(v) for v in phase["trajectory_positions_including_initial"]]
    positions = intervention_groups(cfg.node_count)["all"]
    old_checkpoint, old_positions, old_modules, old_payload = load_task_lora_modules(
        args.before_artifact, device=device
    )
    new_checkpoint, new_positions, new_modules, new_payload = load_task_lora_modules(
        args.after_artifact, device=device
    )
    if old_checkpoint != str(args.checkpoint) or new_checkpoint != str(args.checkpoint):
        raise ValueError("controller/backbone checkpoint mismatch")
    if old_positions != positions or new_positions != positions:
        raise ValueError("controller position mismatch")
    if set(old_modules) != set(new_modules):
        raise ValueError("before/after controller labels differ")
    if old_payload.get("placement") != "loop_boundary" or new_payload.get("placement") != "loop_boundary":
        raise ValueError("matched audit requires loop-boundary controllers")

    streams = _canonical_training_streams(new_payload)
    seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
        device=device, node_count=cfg.node_count, streams=streams
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    if args.sampling_scope == "strict_unseen":
        _, sampled, distribution = stratified_samples(
            seen, unseen, count=args.permutations, seed=args.sample_seed
        )
        evaluation_definition = (
            "before and after controllers evaluated on identical permutations; "
            "every permutation is unseen by the union of all after-training streams"
        )
    else:
        rng = random.Random(args.sample_seed)
        sampled = rng.sample(sorted(universe), args.permutations)
        counts = Counter(cycle_type(item) for item in sampled)
        distribution = {
            "+".join(map(str, key)): value
            for key, value in sorted(counts.items())
        }
        evaluation_definition = (
            "before and after controllers evaluated on identical permutations "
            "sampled from the full 8! population; training overlap is allowed, "
            "so this measures continuation rather than graph generalization"
        )
    reference, reference_checkpoint = load_unit_j_map(
        Path(new_payload["reference_affine_artifact"]),
        label=str(new_payload["reference_affine_label"]),
        device=device,
    )
    if reference_checkpoint != str(args.checkpoint):
        raise ValueError("reference/backbone checkpoint mismatch")
    operators: dict[str, Any] = {
        "identity_no_J": lambda value: value,
        "reference_affine": reference,
    }
    operators.update({f"before__{label}": module for label, module in old_modules.items()})
    operators.update({f"after__{label}": module for label, module in new_modules.items()})
    result = evaluate_partition(
        permutations=sampled,
        operators=operators,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.batch_size,
        continuation_loops=args.loops,
        placement="loop_boundary",
    )
    old_streams = _canonical_training_streams(old_payload)
    old_seen, _, old_draws = reconstruct_primary_training_graphs(
        device=device, node_count=cfg.node_count, streams=old_streams
    )
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "before_artifact": str(args.before_artifact),
        "after_artifact": str(args.after_artifact),
        "evaluation_definition": evaluation_definition,
        "old_training_graph_draws": old_draws,
        "new_training_graph_draws": total_draws,
        "old_unique_training_graphs": len(old_seen),
        "new_unique_training_graphs": len(seen),
        "strictly_unseen_graphs_after_training": len(unseen),
        "new_training_streams": streams,
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "scope": args.sampling_scope,
            "strictly_unseen_permutations": args.permutations,
            "all_starts_per_permutation": cfg.node_count,
            "examples": args.permutations * cfg.node_count,
            "matched_cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "continuation_loops": args.loops,
        "strictly_unseen": result,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--before-artifact", type=Path, required=True)
    parser.add_argument("--after-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loops", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=20260806)
    parser.add_argument(
        "--sampling-scope",
        choices=("strict_unseen", "all_permutations"),
        default="strict_unseen",
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
