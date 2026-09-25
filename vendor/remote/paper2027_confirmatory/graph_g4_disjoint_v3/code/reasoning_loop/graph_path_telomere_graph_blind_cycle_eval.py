from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_blind_j import (
    curve_windows,
    stratified_permutation_split,
)
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    audit_partition,
    cycle_type,
)
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=190001)
    parser.add_argument("--evaluation-seed", type=int, default=191004)
    parser.add_argument("--heldout-graphs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=2.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            0.03,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    _, heldout, _ = stratified_permutation_split(
        node_count=cfg.node_count,
        seed=args.split_seed,
    )
    heldout_single_cycles = [
        permutation
        for permutation in heldout
        if cycle_type(permutation) == (cfg.node_count,)
    ]
    rng = random.Random(args.evaluation_seed)
    rng.shuffle(heldout_single_cycles)
    sample = heldout_single_cycles[: args.heldout_graphs]
    if len(sample) != args.heldout_graphs:
        raise ValueError("not enough held-out single-cycle permutations")

    payload = torch.load(
        args.map_artifact,
        map_location=device,
        weights_only=False,
    )
    if payload.get("kind") != "graph_path_telomere_graph_blind_j":
        raise ValueError("unexpected graph-blind J artifact kind")
    if payload.get("checkpoint") != str(args.checkpoint):
        raise ValueError("J artifact and frozen checkpoint differ")
    results = []
    for label, item in payload["maps"].items():
        weight = item["weight"].to(device=device, dtype=torch.float32)
        age_map = VectorAffine(
            weight=weight,
            bias=item["bias"].to(device=device, dtype=torch.float32),
            update_rank=(
                0 if label == "bias_only" else weight.shape[0]
            ),
            fit_dimension=weight.shape[0],
            retained_fit_energy=1.0,
        )
        result = audit_partition(
            label=label,
            permutations=sample,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            age_map=age_map,
            device=device,
            batch_size=args.batch_size,
            continuation_loops=args.continuation_loops,
        )
        for condition, curve in result["curves"].items():
            curve["windows"] = curve_windows(curve["accuracy_by_cycle"])
        results.append(result)

    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "map_artifact": str(args.map_artifact),
        "loss_placement": (
            "evaluation only; loaded J was trained with hidden-state age loss"
            " and no successor labels/logits/graph CE"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": args.continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "split": {
            "seed": args.split_seed,
            "heldout_single_cycle_pool": len(heldout_single_cycles),
            "sampled_heldout_graphs": len(sample),
            "all_currents_per_graph": cfg.node_count,
            "evaluation_examples": len(sample) * cfg.node_count,
            "cycle_type": [cfg.node_count],
            "evaluation_seed": args.evaluation_seed,
        },
        "results": results,
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "prelaunch_used_mib": args.prelaunch_used_mib,
            "prelaunch_free_mib": args.prelaunch_free_mib,
            "declared_peak_gib": args.declared_peak_gib,
            "reserve_gib": args.reserve_gib,
            "shared_gpu": args.shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    compact = {}
    for result in results:
        compact[result["partition"]] = {
            condition: curve["windows"]
            for condition, curve in result["curves"].items()
        }
    print(
        json.dumps(
            {
                "status": "complete",
                "split": summary["split"],
                "curves": compact,
                "gpu_runtime": summary["gpu_runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
