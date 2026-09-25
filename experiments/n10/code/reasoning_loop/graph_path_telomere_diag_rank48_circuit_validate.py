from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import run_instrumented_state
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_diag_rank48_circuit import (
    _candidate_circuit_rows,
    _metrics,
)
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    PRIMARY_J_TRAINING_STREAMS,
    _expand_all_starts,
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in key_fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, items in groups.items():
        examples = sum(int(item["examples"]) for item in items)
        output = dict(zip(key_fields, key, strict=True))
        output["examples"] = examples
        for metric in (
            "accuracy",
            "target_probability",
            "target_margin",
            "recovery_all",
        ):
            output[metric] = sum(
                float(item[metric]) * int(item["examples"]) for item in items
            ) / examples
        controlled = sum(int(item.get("controlled_count", 0)) for item in items)
        output["controlled_count"] = controlled
        output["recovery_controlled"] = (
            sum(
                float(item["recovery_controlled"])
                * int(item.get("controlled_count", 0))
                for item in items
                if int(item.get("controlled_count", 0))
            )
            / controlled
            if controlled
            else float("nan")
        )
        result.append(output)
    return result


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_cycle < 1:
        raise ValueError("target_cycle must be positive")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, operators, payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if payload.get("placement") != "loop_boundary":
        raise ValueError("validation requires a loop-boundary controller")
    operator = operators[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ) or operator.rank != 48:
        raise ValueError("validation requires the preregistered rank-48 J")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("validation requires J at every token position")

    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    try:
        streams = _canonical_training_streams(payload)
        training_graph_protocol = "canonical_artifact"
    except ValueError as error:
        if "initial_affine_artifact metadata" not in str(error):
            raise
        streams = PRIMARY_J_TRAINING_STREAMS
        training_graph_protocol = "legacy_primary_exact_replay"
    seen, unique_after_stage, training_draws = reconstruct_primary_training_graphs(
        device=device,
        node_count=cfg.node_count,
        streams=streams,
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    _, sampled_unseen, cycle_distribution = stratified_samples(
        seen,
        unseen,
        count=args.sample_per_partition,
        seed=args.sample_seed,
    )
    successors_all, starts_all = _expand_all_starts(
        sampled_unseen, device=device
    )

    baseline_parts: list[dict[str, Any]] = []
    candidate_parts: list[dict[str, Any]] = []
    for batch_index, offset in enumerate(
        range(0, successors_all.shape[0], args.batch_size)
    ):
        successors = successors_all[offset : offset + args.batch_size]
        starts = starts_all[offset : offset + args.batch_size]
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        for cycle in range(1, args.target_cycle + 1):
            current = advance_nodes(
                successors, endpoint, steps=jump * (cycle - 1)
            )
            target = advance_nodes(successors, endpoint, steps=jump * cycle)
            loop_index = cfg.max_loops + cycle - 1
            if cycle == args.target_cycle:
                states = {
                    "J": operator(state),
                    "exact_H7": _aligned_state_at_age(
                        model=model,
                        cfg=cfg,
                        successors=successors,
                        current=current,
                        age=7,
                        phase_position=phase_positions[7],
                    ),
                }
                logits: dict[str, torch.Tensor] = {}
                traces = {}
                for label, initial in states.items():
                    output, trace = run_instrumented_state(
                        model, initial, loop_indices=(loop_index,)
                    )
                    logits[label] = output
                    traces[label] = trace
                    baseline_parts.append(
                        {
                            "batch": batch_index,
                            "examples": successors.shape[0],
                            "run": label,
                            **_metrics(output, target),
                            "recovery_all": 0.0,
                            "recovery_controlled": 0.0,
                            "controlled_count": 0,
                        }
                    )
                for row in _candidate_circuit_rows(
                    model=model,
                    cfg=cfg,
                    loop_index=loop_index,
                    cycle=cycle,
                    target=target,
                    states=states,
                    logits=logits,
                    traces=traces,
                    neurons=args.fixed_mlp_neurons,
                    seed=args.circuit_seed,
                    random_draws=args.random_draws,
                ):
                    candidate_parts.append(
                        {
                            "batch": batch_index,
                            "examples": successors.shape[0],
                            **row,
                        }
                    )
                break
            step = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=state,
                loop_index=loop_index,
                positions=positions,
                operator=operator,
                placement="loop_boundary",
            )
            state = step.state

    baseline = _aggregate(baseline_parts, key_fields=("run",))
    candidate = _aggregate(
        candidate_parts,
        key_fields=("candidate", "condition", "random_draw"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "baseline_batches.csv", baseline_parts)
    _write_csv(args.out_dir / "candidate_batches.csv", candidate_parts)
    _write_csv(args.out_dir / "baseline.csv", baseline)
    _write_csv(args.out_dir / "candidate_circuit.csv", candidate)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller": "J(h)=h*D+(hA)B+b",
        "controller_rank": operator.rank,
        "controller_parameters": operator.parameter_count,
        "controller_placement": "loop boundary after Block2 FFN",
        "controller_training_loss": (
            "successor CE at every controlled continuation loop; no hidden MSE"
        ),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "target_cycle": args.target_cycle,
        "strict_unseen_permutations": len(sampled_unseen),
        "examples_all_starts": int(successors_all.shape[0]),
        "training_graph_draws": training_draws,
        "training_graph_protocol": training_graph_protocol,
        "training_graph_streams": streams,
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs_available": len(unseen),
        "unique_after_stage": unique_after_stage,
        "cycle_type_distribution": cycle_distribution,
        "sample_seed": args.sample_seed,
        "circuit_seed": args.circuit_seed,
        "fixed_mlp_neurons": list(args.fixed_mlp_neurons),
        "random_draws": args.random_draws,
        "baseline": baseline,
        "candidate_circuit": candidate,
        "gpu_peak_allocated_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict-unseen validation for the rank-48 J candidate circuit."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-cycle", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--circuit-seed", type=int, default=20260731)
    parser.add_argument("--fixed-mlp-neurons", type=int, nargs="+", required=True)
    parser.add_argument("--random-draws", type=int, default=10)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.04)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_experiment(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
