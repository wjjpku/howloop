from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_graph_blind_j import (
    curve_windows,
    stratified_permutation_split,
    write_csv,
)
from reasoning_loop.graph_path_telomere_graph_blind_mlp_j import (
    GraphBlindResidualMLP,
)
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    audit_partition,
    cycle_type,
)
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)


def load_mlp_modules(
    path: Path,
    *,
    device: torch.device,
) -> tuple[str, dict[str, GraphBlindResidualMLP]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_graph_blind_mlp_j":
        raise ValueError("unexpected graph-blind MLP J artifact kind")
    modules: dict[str, GraphBlindResidualMLP] = {}
    for label, values in payload["modules"].items():
        state = {
            name: value.to(device)
            for name, value in values["state_dict"].items()
        }
        affine_weight = state["affine_weight"]
        affine_bias = state["affine_bias"]
        initial = VectorAffine(
            weight=affine_weight,
            bias=affine_bias,
            update_rank=affine_weight.shape[0],
            fit_dimension=affine_weight.shape[0],
            retained_fit_energy=1.0,
        )
        module = GraphBlindResidualMLP(
            initial=initial,
            hidden_width=int(values["hidden_width"]),
        ).to(device)
        module.load_state_dict(state)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        modules[label] = module
    return str(payload["checkpoint"]), modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--mlp-artifact", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-graphs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=190104)
    parser.add_argument("--physical-gpu", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            0.05,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    artifact_checkpoint, modules = load_mlp_modules(
        args.mlp_artifact,
        device=device,
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("MLP artifact was trained for another checkpoint")
    training_summary = json.loads(
        args.training_summary.read_text(encoding="utf-8")
    )
    split_seed = int(training_summary["split"]["seed"])
    _, heldout_permutations, _ = stratified_permutation_split(
        node_count=cfg.node_count,
        seed=split_seed,
    )
    heldout_cycles = [
        permutation
        for permutation in heldout_permutations
        if cycle_type(permutation) == (cfg.node_count,)
    ]
    if args.sample_graphs > len(heldout_cycles):
        raise ValueError("requested more held-out cycles than available")
    rng = random.Random(args.sample_seed)
    sample = rng.sample(heldout_cycles, args.sample_graphs)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]

    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for label, module in modules.items():
        result = audit_partition(
            label=f"{label}:strict_heldout_single_{cfg.node_count}_cycle",
            permutations=sample,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            age_map=module,
            device=device,
            batch_size=args.batch_size,
            continuation_loops=args.continuation_loops,
        )
        learned = result["curves"]["learned_J_plus_full_Block2"][
            "accuracy_by_cycle"
        ]
        no_j = result["curves"]["no_J_plus_full_Block2"][
            "accuracy_by_cycle"
        ]
        oracle = result["curves"]["exact_H7_plus_full_Block2"][
            "accuracy_by_cycle"
        ]
        row = {
            "variant": label,
            **{
                f"learned_{key}": value
                for key, value in curve_windows(learned).items()
            },
            **{
                f"no_J_{key}": value
                for key, value in curve_windows(no_j).items()
            },
            **{
                f"oracle_{key}": value
                for key, value in curve_windows(oracle).items()
            },
            "block2_off_accuracy": result[
                "one_step_executor_blocked_accuracy"
            ]["learned_J_Block2_answer_updates_zero"],
            **result["interface_alignment"],
        }
        rows.append(row)
        result["primary_windows"] = row
        results.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "single_cycle_summary.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "mlp_artifact": str(args.mlp_artifact),
        "loss_placement": (
            "frozen backbone final-only; evaluated J was trained with"
            " hidden-state age loss only and no graph CE"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": args.continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "split_seed": split_seed,
        "sample_seed": args.sample_seed,
        "strict_heldout_single_cycle_graphs": args.sample_graphs,
        "all_currents_per_graph": cfg.node_count,
        "random_baseline": 1.0 / cfg.node_count,
        "results": results,
        "gpu_runtime": {
            "physical_gpu": args.physical_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    set_seed(0)
    main()
