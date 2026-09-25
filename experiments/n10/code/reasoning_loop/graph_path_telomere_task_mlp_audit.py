from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    Permutation,
    _expand_all_starts,
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import (
    _controlled_loop,
    load_task_mlp_modules,
)
from reasoning_loop.graph_path_telomere_task_lora_j import (
    load_task_lora_modules,
)
from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map


Operator = Callable[[torch.Tensor], torch.Tensor]


def _segments(values: list[float], start: int, stop: int) -> float | None:
    segment = values[start:stop]
    return sum(segment) / len(segment) if segment else None


def _canonical_training_streams(
    payload: dict[str, Any],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    artifact = payload.get("initial_affine_artifact")
    initialization_mode = payload.get("initialization_mode", "affine_svd")
    if initialization_mode == "identity":
        streams: list[tuple[str, int, int, int, int, int]] = []
    elif not artifact:
        raise ValueError(
            "canonical graph audit requires initial_affine_artifact metadata"
        )
    else:
        initializer_summary_path = Path(artifact).parent / "summary.json"
        if not initializer_summary_path.exists():
            raise ValueError(
                "canonical graph audit cannot find initializer summary: "
                f"{initializer_summary_path}"
            )
        initializer = json.loads(
            initializer_summary_path.read_text(encoding="utf-8")
        )
        graphs = int(initializer["graphs"])
        batch_size = int(initializer.get("batch_size", 32))
        if graphs % batch_size:
            raise ValueError("initializer graphs are not divisible by batch size")
        streams = [
            (
                "boundary_initializer",
                int(initializer["data_seed"]),
                0,
                1,
                batch_size,
                graphs // batch_size,
            )
        ]
    for stage in payload["stages"]:
        streams.append(
            (
                str(stage["name"]),
                int(stage["data_seed"]),
                1000,
                int(stage["rounds"]),
                int(stage["batch_size"]),
                int(stage["batches_per_round"]),
            )
        )
    return tuple(streams)


@torch.no_grad()
def evaluate_partition(
    *,
    permutations: list[Permutation],
    operators: dict[str, Operator],
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    continuation_loops: int,
    placement: str,
) -> dict[str, Any]:
    """Evaluate all controllers on identical graphs and starts.

    The controller states are concatenated so the frozen Transformer executes
    one large forward per cycle.  Each controller still receives and updates
    only its own slice of the batch.
    """

    successors_all, starts_all = _expand_all_starts(permutations, device=device)
    labels = list(operators)
    jump = phase_positions[3] - phase_positions[2]
    correct = {label: [0] * continuation_loops for label in labels}
    total = 0
    for offset in range(0, successors_all.shape[0], batch_size):
        successors = successors_all[offset : offset + batch_size]
        starts = starts_all[offset : offset + batch_size]
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        state = torch.cat([initial for _ in labels], dim=0)

        def combined_operator(value: torch.Tensor) -> torch.Tensor:
            chunks = value.chunk(len(labels), dim=0)
            return torch.cat(
                [operators[label](chunk) for label, chunk in zip(labels, chunks)],
                dim=0,
            )

        for cycle in range(1, continuation_loops + 1):
            step = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=state,
                loop_index=cfg.max_loops + cycle - 1,
                positions=positions,
                operator=combined_operator,
                placement=placement,
            )
            target = advance_nodes(successors, endpoint, steps=jump * cycle)
            for label, logits in zip(labels, step.logits.chunk(len(labels), dim=0)):
                correct[label][cycle - 1] += int(
                    logits.argmax(dim=-1).eq(target).sum().item()
                )
            state = step.state
        total += successors.shape[0]

    curves: dict[str, Any] = {}
    for label, counts in correct.items():
        accuracy = [value / total for value in counts]
        curves[label] = {
            "accuracy_by_cycle": accuracy,
            "auc_1_24": _segments(accuracy, 0, 24),
            "auc_25_48": _segments(accuracy, 24, 48),
            "auc_49_64": _segments(accuracy, 48, 64),
            "auc_65_96": _segments(accuracy, 64, 96),
            "auc_97_128": _segments(accuracy, 96, 128),
            "final_accuracy": accuracy[-1],
            "accuracy_at_cycles": {
                str(cycle): accuracy[cycle - 1]
                for cycle in (1, 2, 8, 16, 24, 40, 48, 64, 80, 96, 112, 128)
                if cycle <= len(accuracy)
            },
        }
    return {
        "permutations": len(permutations),
        "examples_all_starts": total,
        "curves": curves,
    }


def aggregate_widths(
    curves: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, int]],
) -> dict[str, Any]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for label, item in metadata.items():
        grouped[item["hidden_width"]].append(label)
    result: dict[str, Any] = {}
    for width, labels in sorted(grouped.items()):
        metrics: dict[str, Any] = {"labels": sorted(labels)}
        for key in (
            "auc_1_24",
            "auc_25_48",
            "auc_49_64",
            "auc_65_96",
            "auc_97_128",
            "final_accuracy",
        ):
            raw_values = [curves[label][key] for label in labels]
            if any(value is None for value in raw_values):
                metrics[key] = None
                continue
            values = torch.tensor(raw_values, dtype=torch.float64)
            metrics[key] = {
                "mean": float(values.mean().item()),
                "std_population": float(values.std(unbiased=False).item()),
                "values": [float(value) for value in values.tolist()],
            }
        result[str(width)] = metrics
    return result


def aggregate_ranks(
    curves: dict[str, dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for label, item in metadata.items():
        grouped[int(item["rank"])].append(label)
    result: dict[str, Any] = {}
    for rank, labels in sorted(grouped.items()):
        metrics: dict[str, Any] = {"labels": sorted(labels)}
        for key in (
            "auc_1_24",
            "auc_25_48",
            "auc_49_64",
            "auc_65_96",
            "auc_97_128",
            "final_accuracy",
        ):
            values = torch.tensor(
                [curves[label][key] for label in labels],
                dtype=torch.float64,
            )
            metrics[key] = {
                "mean": float(values.mean().item()),
                "std_population": float(values.std(unbiased=False).item()),
                "values": [float(value) for value in values.tolist()],
            }
        loop64 = torch.tensor(
            [curves[label]["accuracy_by_cycle"][63] for label in labels],
            dtype=torch.float64,
        )
        metrics["accuracy_loop64"] = {
            "mean": float(loop64.mean().item()),
            "std_population": float(loop64.std(unbiased=False).item()),
            "values": [float(value) for value in loop64.tolist()],
        }
        result[str(rank)] = metrics
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict unseen-graph audit for task-aware residual MLP-J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--affine-artifact", type=Path, required=True)
    parser.add_argument("--affine-label", default="task")
    parser.add_argument("--mlp-artifacts", type=Path, nargs="+", default=())
    parser.add_argument("--lora-artifacts", type=Path, nargs="+", default=())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.12)
    parser.add_argument(
        "--training-graph-protocol",
        choices=("canonical_artifact", "legacy_primary"),
        default="canonical_artifact",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.mlp_artifacts and not args.lora_artifacts:
        raise ValueError("at least one MLP or LoRA artifact is required")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction, device=torch.cuda.current_device()
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    if cfg.node_count != 8 or cfg.max_depth != 8 or cfg.max_loops != 8:
        raise ValueError("audit requires the D8L8 N8 checkpoint")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]

    affine, affine_checkpoint = load_unit_j_map(
        args.affine_artifact, label=args.affine_label, device=device
    )
    if affine_checkpoint != str(args.checkpoint):
        raise ValueError("affine J and frozen model checkpoints differ")
    operators: dict[str, Operator] = {
        "identity_no_J": lambda value: value,
        "reference_affine": affine,
    }
    metadata: dict[str, dict[str, Any]] = {}
    placements: set[str] = set()
    lora_payloads: list[dict[str, Any]] = []
    for artifact in args.mlp_artifacts:
        checkpoint, artifact_positions, modules, payload = load_task_mlp_modules(
            artifact, device=device
        )
        if checkpoint != str(args.checkpoint):
            raise ValueError(f"{artifact} belongs to another checkpoint")
        if artifact_positions != positions:
            raise ValueError(f"{artifact} uses different intervention positions")
        placements.add(str(payload.get("placement", "pre_block2")))
        overlap = set(operators) & set(modules)
        if overlap:
            raise ValueError(f"duplicate controller labels: {sorted(overlap)}")
        operators.update(modules)
        for label, item in payload["modules"].items():
            metadata[label] = {
                "controller_family": "residual_mlp",
                "hidden_width": int(item["hidden_width"]),
                "initialization_seed": int(item["initialization_seed"]),
                "parameter_count": int(item["parameter_count"]),
            }
    for artifact in args.lora_artifacts:
        checkpoint, artifact_positions, modules, payload = (
            load_task_lora_modules(artifact, device=device)
        )
        if checkpoint != str(args.checkpoint):
            raise ValueError(f"{artifact} belongs to another checkpoint")
        if artifact_positions != positions:
            raise ValueError(f"{artifact} uses different intervention positions")
        placements.add(str(payload.get("placement", "loop_boundary")))
        lora_payloads.append(payload)
        overlap = set(operators) & set(modules)
        if overlap:
            raise ValueError(f"duplicate controller labels: {sorted(overlap)}")
        operators.update(modules)
        for label, item in payload["modules"].items():
            metadata[label] = {
                "controller_family": "residual_lora",
                "rank": int(item["rank"]),
                "initialization_seed": int(item["initialization_seed"]),
                "parameter_count": int(item["parameter_count"]),
            }
    if len(placements) != 1:
        raise ValueError(
            "all audited MLP artifacts must use one common placement; got "
            f"{sorted(placements)}"
        )
    placement = next(iter(placements))

    if args.training_graph_protocol == "canonical_artifact":
        if not lora_payloads:
            raise ValueError("canonical graph audit requires a LoRA artifact")
        streams = _canonical_training_streams(lora_payloads[0])
        for payload in lora_payloads[1:]:
            if _canonical_training_streams(payload) != streams:
                raise ValueError("audited controllers used different graph streams")
        seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
            device=device,
            node_count=cfg.node_count,
            streams=streams,
        )
    else:
        streams = None
        seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
            device=device,
            node_count=cfg.node_count,
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
        permutations=sampled_unseen,
        operators=operators,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        placement=placement,
    )
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            "frozen backbone: "
            + (
                str(lora_payloads[0].get("backbone_loss_description", "not recorded"))
                if lora_payloads
                else "not recorded"
            )
            + "; audited controllers use the loss recorded in their artifacts"
        ),
        "controller_placement": placement,
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "graph_universe": len(universe),
        "training_graph_draws": total_draws,
        "training_graph_protocol": args.training_graph_protocol,
        "training_graph_streams": streams,
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs": len(unseen),
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "strictly_unseen_permutations": args.sample_per_partition,
            "all_starts_per_permutation": cfg.node_count,
            "matched_cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "continuation_loops": args.continuation_loops,
        "operators": metadata,
        "strictly_unseen": result,
        "width_aggregates": aggregate_widths(
            result["curves"],
            {
                label: item
                for label, item in metadata.items()
                if item["controller_family"] == "residual_mlp"
            },
        ),
        "rank_aggregates": aggregate_ranks(
            result["curves"],
            {
                label: item
                for label, item in metadata.items()
                if item["controller_family"] == "residual_lora"
            },
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
