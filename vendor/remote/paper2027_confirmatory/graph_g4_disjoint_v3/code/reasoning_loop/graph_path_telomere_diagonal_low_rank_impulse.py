from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine, run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import intervention_groups
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


Operator = Callable[[torch.Tensor], torch.Tensor]


def _affine(weight: torch.Tensor, bias: torch.Tensor) -> VectorAffine:
    dimension = int(weight.shape[0])
    return VectorAffine(
        weight=weight.float(),
        bias=bias.float(),
        update_rank=dimension,
        fit_dimension=dimension,
        retained_fit_energy=1.0,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    full_operator: Operator,
    damage_operators: dict[str, Operator],
    damage_cycles: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    conditions: dict[str, tuple[int | None, Operator]] = {
        "always_full": (None, full_operator)
    }
    for damage_label, operator in damage_operators.items():
        for cycle in damage_cycles:
            conditions[f"{damage_label}_at_{cycle}"] = (cycle, operator)
    labels = tuple(conditions)
    correct = {label: [0] * continuation_loops for label in labels}
    relative_mse = {label: [0.0] * continuation_loops for label in labels}
    total = 0
    jump = phase_positions[3] - phase_positions[2]

    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        state = torch.cat([initial for _ in labels], dim=0)
        for cycle in range(1, continuation_loops + 1):
            chunks = state.chunk(len(labels), dim=0)

            def combined_operator(value: torch.Tensor) -> torch.Tensor:
                value_chunks = value.chunk(len(labels), dim=0)
                outputs = []
                for label, value_chunk in zip(labels, value_chunks):
                    damage_cycle, damage_operator = conditions[label]
                    operator = (
                        damage_operator
                        if damage_cycle == cycle
                        else full_operator
                    )
                    outputs.append(operator(value_chunk))
                return torch.cat(outputs, dim=0)

            step = _controlled_loop(
                loop_runner=run_one_loop,
                model=model,
                state=torch.cat(chunks, dim=0),
                loop_index=cfg.max_loops + cycle - 1,
                positions=positions,
                operator=combined_operator,
                placement="loop_boundary",
            )
            target = advance_nodes(successors, endpoint, steps=jump * cycle)
            next_chunks = step.state.chunk(len(labels), dim=0)
            logits_chunks = step.logits.chunk(len(labels), dim=0)
            reference = next_chunks[0].float()
            reference_scale = reference.var(unbiased=False).clamp_min(1e-8)
            for label, next_state, logits in zip(
                labels, next_chunks, logits_chunks
            ):
                correct[label][cycle - 1] += int(
                    logits.argmax(dim=-1).eq(target).sum().item()
                )
                relative_mse[label][cycle - 1] += float(
                    (next_state.float() - reference).square().mean()
                    / reference_scale
                ) * batch_size
            state = step.state
        total += batch_size

    curves: dict[str, Any] = {}
    for label in labels:
        accuracy = [count / total for count in correct[label]]
        mse = [value / total for value in relative_mse[label]]
        damage_cycle = conditions[label][0]
        report_cycles = {1, 2, 8, 16, 24, 32, 40, 48, 56, 64}
        if damage_cycle is not None:
            report_cycles.update(
                cycle
                for cycle in (
                    damage_cycle,
                    damage_cycle + 1,
                    damage_cycle + 2,
                    damage_cycle + 4,
                    damage_cycle + 8,
                )
                if cycle <= continuation_loops
            )
        curves[label] = {
            "damage_cycle": damage_cycle,
            "accuracy_by_cycle": accuracy,
            "relative_mse_to_full_by_cycle": mse,
            "selected_cycles": {
                str(cycle): {
                    "accuracy": accuracy[cycle - 1],
                    "relative_mse_to_full": mse[cycle - 1],
                }
                for cycle in sorted(report_cycles)
                if cycle <= continuation_loops
            },
            "final_accuracy": accuracy[-1],
            "final_relative_mse_to_full": mse[-1],
        }
    return {
        "examples": total,
        "continuation_loops": continuation_loops,
        "seed": seed,
        "curves": curves,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
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
    if payload["checkpoint"] != str(args.checkpoint):
        raise ValueError("controller and backbone checkpoints differ")
    item = payload["modules"][args.label]
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
    full = _affine(diagonal_weight + correction, bias)
    damages = {
        "remove_AB": _affine(diagonal_weight, bias),
        "replace_D_by_mean": _affine(
            diagonal.mean() * identity + correction, bias
        ),
        "replace_D_by_I": _affine(identity + correction, bias),
        "remove_bias": _affine(diagonal_weight + correction, zero_bias),
    }
    evaluation = evaluate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        full_operator=full,
        damage_operators=damages,
        damage_cycles=tuple(args.damage_cycles),
        device=device,
        batch_size=args.batch_size,
        batches=args.batches,
        continuation_loops=args.continuation_loops,
        seed=args.seed,
    )
    rows = [
        {
            "condition": label,
            "cycle": cycle,
            "accuracy": curve["accuracy_by_cycle"][cycle - 1],
            "relative_mse_to_full": curve[
                "relative_mse_to_full_by_cycle"
            ][cycle - 1],
        }
        for label, curve in evaluation["curves"].items()
        for cycle in range(1, args.continuation_loops + 1)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "impulse_curves.csv", rows)
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
        "damage_cycles": list(args.damage_cycles),
        "evaluation": evaluation,
        "files": {"curves": "impulse_curves.csv"},
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-cycle component-damage audit for diagonal low-rank J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--damage-cycles", type=int, nargs="+", default=(1, 32))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_experiment(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
