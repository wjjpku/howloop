"""Evaluate whether a one-shot D8L6 jump controller closes under reuse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_jump_controller import (
    JumpMode,
    _matched_target_state,
    apply_vector_map,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine, run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes, cache_states_with_initial
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> tuple[int, int]:
    keep = target.ne(current)
    prediction = logits.argmax(dim=-1)
    return int(prediction[keep].eq(target[keep]).sum()), int(keep.sum())


def load_controller(path: Path, mode: str, rank: int, device: torch.device) -> VectorAffine:
    payload = torch.load(path, map_location=device, weights_only=False)
    item = payload["shared"][mode][str(rank)]
    return VectorAffine(
        weight=item["weight"].to(device),
        bias=item["bias"].to(device),
        update_rank=int(item["update_rank"]),
        fit_dimension=int(item["fit_dimension"]),
        retained_fit_energy=float(item["retained_fit_energy"]),
    )


def add(bucket: dict[str, int], prefix: str, counts: tuple[int, int]) -> None:
    bucket[f"{prefix}_correct"] += counts[0]
    bucket[f"{prefix}_count"] += counts[1]


def finalize(accumulators: dict[int, dict[str, int]]) -> list[dict[str, Any]]:
    rows = []
    for cycle, bucket in sorted(accumulators.items()):
        row: dict[str, Any] = {"cycle": cycle}
        row.update(bucket)
        for prefix in (
            "controlled_pre_target",
            "controlled_pre_current",
            "controlled_post_target",
            "no_control_post_target",
            "shuffled_post_target",
            "oracle_post_target",
        ):
            row[f"{prefix}_strict_accuracy"] = (
                bucket[f"{prefix}_correct"] / bucket[f"{prefix}_count"]
                if bucket[f"{prefix}_count"]
                else float("nan")
            )
        rows.append(row)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("one", "two"), default="two")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--reference-age", type=int, default=1)
    parser.add_argument("--reference-path-before", type=int, default=2)
    parser.add_argument("--programmed-jump", type=int, default=2)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=202608095)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.06)
    return parser.parse_args(argv)


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch size")
    if args.cycles < 1:
        raise ValueError("cycles must be positive")
    device = pick_device(args.device)
    if device.type == "cuda": 
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    if (cfg.max_depth, cfg.max_loops, cfg.n_layers) != (8, 6, 2):
        raise ValueError("closure evaluator requires D8L6 with two shared blocks")
    controller = load_controller(args.controllers, args.mode, args.rank, device)
    positions = tuple(range(cfg.seq_len))
    oracle_mode = JumpMode(
        name=args.mode,
        reference_age=args.reference_age,
        reference_path_before=args.reference_path_before,
        programmed_jump=args.programmed_jump,
    )
    set_seed(args.seed)
    accumulators: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _ in range(args.examples // args.batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            args.batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = advance_nodes(successors, start, steps=cfg.max_depth)
        terminal = cache_states_with_initial(model, tokens, loops=cfg.max_loops)[-1]
        controlled_state = terminal
        no_control_state = terminal
        shuffled_state = terminal

        for cycle in range(1, args.cycles + 1):
            current = advance_nodes(
                successors,
                endpoint,
                steps=args.programmed_jump * (cycle - 1),
            )
            target = advance_nodes(
                successors,
                endpoint,
                steps=args.programmed_jump * cycle,
            )
            bucket = accumulators[cycle]

            controlled_boundary = apply_vector_map(
                controlled_state,
                positions=positions,
                controller=controller,
            )
            pre_logits = logits_from_raw_state(model, controlled_boundary)
            add(
                bucket,
                "controlled_pre_target",
                strict_counts(pre_logits, target, current),
            )
            add(
                bucket,
                "controlled_pre_current",
                strict_counts(pre_logits, current, target),
            )
            controlled_step = run_one_loop(
                model,
                controlled_boundary,
                loop_index=cfg.max_loops + cycle - 1,
            )
            add(
                bucket,
                "controlled_post_target",
                strict_counts(controlled_step.logits, target, current),
            )
            controlled_state = controlled_step.state

            no_control_step = run_one_loop(
                model,
                no_control_state,
                loop_index=cfg.max_loops + cycle - 1,
            )
            add(
                bucket,
                "no_control_post_target",
                strict_counts(no_control_step.logits, target, current),
            )
            no_control_state = no_control_step.state

            shuffled_boundary = apply_vector_map(
                shuffled_state,
                positions=positions,
                controller=controller,
            ).roll(1, dims=0)
            shuffled_step = run_one_loop(
                model,
                shuffled_boundary,
                loop_index=cfg.max_loops + cycle - 1,
            )
            add(
                bucket,
                "shuffled_post_target",
                strict_counts(shuffled_step.logits, target, current),
            )
            shuffled_state = shuffled_step.state

            shifted_start = advance_nodes(
                successors,
                start,
                steps=args.programmed_jump * (cycle - 1),
            )
            oracle_boundary = _matched_target_state(
                model=model,
                cfg=cfg,
                successors=successors,
                start=shifted_start,
                mode=oracle_mode,
            )
            oracle_step = run_one_loop(
                model,
                oracle_boundary,
                loop_index=cfg.max_loops + cycle - 1,
            )
            add(
                bucket,
                "oracle_post_target",
                strict_counts(oracle_step.logits, target, current),
            )

    rows = finalize(accumulators)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": checkpoint_payload.get("loss_mode", "final_only"),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller_artifact": str(args.controllers),
        "controller_sha256": file_sha256(args.controllers),
        "controller_parameterization": "row-vector I plus rank-r delta plus bias",
        "controller_site": "loop boundary before the next shared loop",
        "mode": args.mode,
        "rank": args.rank,
        "reference_age": args.reference_age,
        "reference_path_before": args.reference_path_before,
        "programmed_jump": args.programmed_jump,
        "examples": args.examples,
        "seed": args.seed,
        "cycles": args.cycles,
        "rows": rows,
        "interpretation_boundary": (
            "Cycle 1 tests one-shot reachability. Later cycles test closure under "
            "the controller's own states; they are not implied by the one-shot fit."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "closure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_experiment(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
