from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import load_task_lora_modules
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop


Schedule = Callable[[int], bool]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _segment(values: list[float], start: int, stop: int) -> float | None:
    selected = values[start:stop]
    return sum(selected) / len(selected) if selected else None


def schedules() -> dict[str, Schedule]:
    return {
        "no_J": lambda _cycle: False,
        "J_every_loop": lambda _cycle: True,
        "J_first_only": lambda cycle: cycle == 1,
        "J_first_8": lambda cycle: cycle <= 8,
        "J_first_16": lambda cycle: cycle <= 16,
        "J_first_32": lambda cycle: cycle <= 32,
        "J_every_2": lambda cycle: (cycle - 1) % 2 == 0,
        "J_every_4": lambda cycle: (cycle - 1) % 4 == 0,
        "J_skip_cycle_32": lambda cycle: cycle != 32,
        "J_skip_cycle_64": lambda cycle: cycle != 64,
    }


@torch.no_grad()
def evaluate(
    *,
    model,
    cfg,
    operator,
    positions: tuple[int, ...],
    phase_positions: list[int],
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    policies = schedules()
    labels = tuple(policies) + ("exact_H7_every_loop",)
    correct = {label: [0] * continuation_loops for label in labels}
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
        states = {label: initial.clone() for label in policies}
        for cycle in range(1, continuation_loops + 1):
            loop_index = cfg.max_loops + cycle - 1
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            target = advance_nodes(
                successors,
                endpoint,
                steps=jump * cycle,
            )
            for label, policy in policies.items():
                if policy(cycle):
                    step = _controlled_loop(
                        loop_runner=run_one_loop,
                        model=model,
                        state=states[label],
                        loop_index=loop_index,
                        positions=positions,
                        operator=operator,
                        placement="loop_boundary",
                    )
                else:
                    step = run_one_loop(
                        model,
                        states[label],
                        loop_index=loop_index,
                    )
                correct[label][cycle - 1] += int(
                    step.logits.argmax(dim=-1).eq(target).sum().item()
                )
                states[label] = step.state
            exact = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=7,
                phase_position=phase_positions[7],
            )
            exact_step = run_one_loop(model, exact, loop_index=loop_index)
            correct["exact_H7_every_loop"][cycle - 1] += int(
                exact_step.logits.argmax(dim=-1).eq(target).sum().item()
            )
        total += batch_size
    rows = [
        {
            "condition": label,
            "cycle": cycle,
            "accuracy": correct[label][cycle - 1] / total,
            "examples": total,
        }
        for label in labels
        for cycle in range(1, continuation_loops + 1)
    ]
    curves = {}
    for label in labels:
        values = [count / total for count in correct[label]]
        curves[label] = {
            "accuracy_by_cycle": values,
            "auc_1_24": _segment(values, 0, 24),
            "auc_25_48": _segment(values, 24, 48),
            "auc_49_64": _segment(values, 48, 64),
            "auc_65_96": _segment(values, 64, 96),
            "auc_97_128": _segment(values, 96, 128),
            "final_accuracy": values[-1],
        }
    return rows, {
        "examples": total,
        "continuation_loops": continuation_loops,
        "seed": seed,
        "curves": curves,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dose-schedule audit for a loop-boundary affine J."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    return parser.parse_args(argv)


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, operators, payload = load_task_lora_modules(
        args.operator_artifact,
        device=device,
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if payload.get("placement") != "loop_boundary":
        raise ValueError("schedule audit requires a loop-boundary operator")
    if args.operator_label not in operators:
        raise ValueError(f"operator label not found: {args.operator_label}")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    rows, evaluation = evaluate(
        model=model,
        cfg=cfg,
        operator=operators[args.operator_label],
        positions=positions,
        phase_positions=phase_positions,
        device=device,
        batch_size=args.batch_size,
        batches=args.batches,
        continuation_loops=args.continuation_loops,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "schedule_curves.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "frozen_model_loss_placement": payload.get(
            "backbone_loss_description",
            "not recorded",
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller_placement": "loop_boundary",
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "evaluation": evaluation,
        "files": {"curves": "schedule_curves.csv"},
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
