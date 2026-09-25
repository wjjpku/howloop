from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_mlp_j import _controlled_loop
from reasoning_loop.graph_path_telomere_unit_j import exact_interfaces


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


@torch.no_grad()
def evaluate_exact_placement_oracles(
    *,
    model,
    cfg,
    phase_positions: list[int],
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare exact young-state replacement at two recurrent sites.

    Every oracle is constructed from the same graph and the same current node
    as the live trajectory.  The only changed variable is represented age.
    Shuffled-graph controls preserve the current node and age while replacing
    graph identity, so an oracle cannot succeed merely by supplying a generic
    phase/age vector.
    """

    set_seed(seed)
    positions = intervention_groups(cfg.node_count)["all"]
    jump = phase_positions[3] - phase_positions[2]
    labels = (
        "no_control",
        "exact_H7_loop_boundary",
        "exact_H7_pre_block2",
        "shuffled_H7_loop_boundary",
        "shuffled_H7_pre_block2",
    )
    correct = {label: [0] * continuation_loops for label in labels}
    total = 0

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
        states = {label: initial.clone() for label in labels}
        shuffled_successors = successors.roll(shifts=1, dims=0)

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

            steps = {
                "no_control": run_one_loop(
                    model,
                    states["no_control"],
                    loop_index=loop_index,
                )
            }

            exact_boundary = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=7,
                phase_position=phase_positions[7],
            )
            shuffled_boundary = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=shuffled_successors,
                current=current,
                age=7,
                phase_position=phase_positions[7],
            )
            steps["exact_H7_loop_boundary"] = run_one_loop(
                model,
                exact_boundary,
                loop_index=loop_index,
            )
            steps["shuffled_H7_loop_boundary"] = run_one_loop(
                model,
                shuffled_boundary,
                loop_index=loop_index,
            )

            exact_pre_block2 = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=successors,
                current=current,
                ages=(7,),
                loop_index=loop_index,
            )[7]
            shuffled_pre_block2 = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=shuffled_successors,
                current=current,
                ages=(7,),
                loop_index=loop_index,
            )[7]

            for label, oracle in (
                ("exact_H7_pre_block2", exact_pre_block2),
                ("shuffled_H7_pre_block2", shuffled_pre_block2),
            ):
                steps[label] = _controlled_loop(
                    loop_runner=run_one_loop,
                    model=model,
                    state=states[label],
                    loop_index=loop_index,
                    positions=positions,
                    operator=lambda _value, oracle=oracle: oracle,
                    placement="pre_block2",
                )

            for label, step in steps.items():
                correct[label][cycle - 1] += int(
                    step.logits.argmax(dim=-1).eq(target).sum().item()
                )
                states[label] = step.state
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
    curves: dict[str, Any] = {}
    for label in labels:
        values = [value / total for value in correct[label]]
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
        description="Exact-young causal comparison of two J insertion sites."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.04)
    return parser.parse_args(argv)


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
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
        or cfg.d_model != 256
    ):
        raise ValueError("experiment requires frozen D8L8 N8 d256 B2")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    rows, evaluation = evaluate_exact_placement_oracles(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        device=device,
        batch_size=args.batch_size,
        batches=args.batches,
        continuation_loops=args.continuation_loops,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "oracle_curves.csv", rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "frozen_model_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "intervention": (
            "same-graph same-current exact H7 age replacement at either the "
            "loop boundary or the Block1-FFN/Block2-attention interface"
        ),
        "evaluation": evaluation,
        "files": {"curves": "oracle_curves.csv"},
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
