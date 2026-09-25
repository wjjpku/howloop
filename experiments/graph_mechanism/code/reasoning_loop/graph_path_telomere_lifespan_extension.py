from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


Controller = Literal[
    "baseline",
    "additive",
    "reverse_additive",
    "exact_replace",
    "anchor_replace",
    "shuffled_replace",
    "wrong_node_replace",
]

CONTROLLERS: tuple[Controller, ...] = (
    "baseline",
    "additive",
    "reverse_additive",
    "exact_replace",
    "anchor_replace",
    "shuffled_replace",
    "wrong_node_replace",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extension_position_groups(
    cfg: GraphPathConfig,
) -> dict[str, tuple[int, ...]]:
    base = explicit_depth_position_groups(cfg.node_count)
    graph = base["graph"]
    query_work = base["query_metadata"] + base["answer"]
    registers = base["start"] + base["depth"] + base["answer"]
    return {
        "answer": base["answer"],
        "registers": registers,
        "query_work": query_work,
        "graph": graph,
        "graph_answer": graph + base["answer"],
        "all": tuple(range(cfg.seq_len)),
    }


def apply_rejuvenation(
    state: torch.Tensor,
    *,
    reference: torch.Tensor,
    anchor: torch.Tensor,
    wrong_node_reference: torch.Tensor,
    positions: tuple[int, ...],
    controller: Controller,
) -> torch.Tensor:
    if not (
        state.shape
        == reference.shape
        == anchor.shape
        == wrong_node_reference.shape
    ):
        raise ValueError("all recurrent states must have identical shapes")
    index = list(positions)
    result = state.clone()
    if controller == "baseline":
        return result
    if controller == "additive":
        result[:, index] += reference[:, index] - anchor[:, index]
    elif controller == "reverse_additive":
        result[:, index] -= reference[:, index] - anchor[:, index]
    elif controller == "exact_replace":
        result[:, index] = reference[:, index]
    elif controller == "anchor_replace":
        result[:, index] = anchor[:, index]
    elif controller == "shuffled_replace":
        result[:, index] = reference[:, index].roll(1, dims=0)
    elif controller == "wrong_node_replace":
        result[:, index] = wrong_node_reference[:, index]
    else:
        raise ValueError(f"unknown controller: {controller}")
    return result


def select_executable_phase(
    rows: list[dict[str, str]],
    *,
    max_depth: int,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["group"] == "all"
        and row["mode"] == "matched"
        and int(float(row["target_offset"])) in {1, 2}
        and 0 <= int(float(row["reference_path_before"])) <= max_depth
    ]
    if not eligible:
        raise ValueError("phase grid contains no executable all-state phase")
    best = max(
        eligible,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["margin"]),
        ),
    )
    jump = int(float(best["target_offset"]))
    return {
        "reference_age": int(float(best["reference_age"])),
        "reference_path_before": int(
            float(best["reference_path_before"])
        ),
        "programmed_jump": jump,
        "selection_accuracy": float(best["accuracy"]),
        "selection_margin": float(best["margin"]),
        "trajectory_resolved": best["reference_resolved"] == "True",
    }


def load_phase_candidate(summary_path: Path) -> dict[str, Any]:
    grid_path = summary_path.with_name("matched_phase_grid_rows.csv")
    if grid_path.exists():
        with grid_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        max_depth = int(
            summary.get("config", {}).get(
                "max_depth",
                summary.get("query_depth", 8),
            )
        )
        return select_executable_phase(rows, max_depth=max_depth)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best_closed_loop_control")
    if best is None:
        best = summary["selected_matched_phase_reset"]
    return {
        "reference_age": int(best["reference_age"]),
        "reference_path_before": int(best["reference_path_before"]),
        "programmed_jump": int(best["programmed_jump"]),
        "selection_accuracy": float(
            best.get(
                "selection_accuracy",
                best.get("accuracy_by_extra_loop", [float("nan")])[0],
            )
        ),
        "selection_margin": float("nan"),
        "trajectory_resolved": True,
    }


def _accumulate(
    bucket: dict[str, float],
    metrics: dict[str, float | int],
) -> None:
    count = int(metrics["valid_count"])
    if not count:
        return
    bucket["correct"] += float(metrics["accuracy"]) * count
    bucket["probability"] += float(metrics["probability"]) * count
    bucket["margin"] += float(metrics["margin"]) * count
    bucket["count"] += count


def _blank_bucket() -> dict[str, float]:
    return {
        "correct": 0.0,
        "probability": 0.0,
        "margin": 0.0,
        "count": 0.0,
    }


@torch.no_grad()
def evaluate_lifespan_extension(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    candidate: dict[str, Any],
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    reference_age = int(candidate["reference_age"])
    reference_position = int(candidate["reference_path_before"])
    jump = int(candidate["programmed_jump"])
    if jump not in {1, 2}:
        raise ValueError("programmed jump must be one or two")
    groups = extension_position_groups(cfg)
    accumulators = {
        (group, controller, extra_loop): _blank_bucket()
        for group in groups
        for controller in CONTROLLERS
        for extra_loop in range(1, extra_loops + 1)
    }
    clean_accumulators = {
        extra_loop: _blank_bucket()
        for extra_loop in range(1, extra_loops + 1)
    }
    wrong_target_accumulators = {
        (group, extra_loop): _blank_bucket()
        for group in groups
        for extra_loop in range(1, extra_loops + 1)
    }
    set_seed(seed)
    path_positions = cfg.max_depth + 2 * extra_loops + 2
    for _ in range(batches):
        tokens, targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        terminal = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )[-1]
        all_targets = _all_targets(start, targets)
        endpoint = all_targets[:, cfg.max_depth]
        donors = []
        for extra_loop in range(1, extra_loops + 1):
            current_position = (
                cfg.max_depth + jump * (extra_loop - 1)
            )
            reference_start = advance_nodes(
                successors,
                start,
                steps=current_position - reference_position,
            )
            anchor_start = advance_nodes(
                successors,
                start,
                steps=current_position - cfg.max_depth,
            )
            wrong_start = advance_nodes(
                successors,
                reference_start,
                steps=1,
            )
            reference_tokens, _, _, _ = fixed_depth_batch(
                cfg,
                batch_size,
                device,
                path_positions=cfg.max_depth,
                successors=successors,
                start=reference_start,
            )
            anchor_tokens, _, _, _ = fixed_depth_batch(
                cfg,
                batch_size,
                device,
                path_positions=cfg.max_depth,
                successors=successors,
                start=anchor_start,
            )
            wrong_tokens, _, _, _ = fixed_depth_batch(
                cfg,
                batch_size,
                device,
                path_positions=cfg.max_depth,
                successors=successors,
                start=wrong_start,
            )
            reference = cache_states_with_initial(
                model,
                reference_tokens,
                loops=max(1, reference_age),
            )[reference_age]
            anchor = cache_states_with_initial(
                model,
                anchor_tokens,
                loops=cfg.max_loops,
            )[-1]
            wrong_reference = cache_states_with_initial(
                model,
                wrong_tokens,
                loops=max(1, reference_age),
            )[reference_age]
            donors.append((reference, anchor, wrong_reference))

            clean_state = apply_shared_stack(
                model,
                reference,
                loop_index=cfg.max_loops + extra_loop - 1,
            )
            clean_logits = logits_from_raw_state(model, clean_state)
            clean_metrics = _masked_metrics(
                clean_logits,
                all_targets[
                    :, cfg.max_depth + jump * extra_loop
                ],
                endpoint=endpoint,
            )
            _accumulate(
                clean_accumulators[extra_loop],
                clean_metrics,
            )

        for group, positions in groups.items():
            states = {
                controller: terminal.clone()
                for controller in CONTROLLERS
            }
            for extra_loop, (
                reference,
                anchor,
                wrong_reference,
            ) in enumerate(donors, start=1):
                intended_target = all_targets[
                    :, cfg.max_depth + jump * extra_loop
                ]
                wrong_target = all_targets[
                    :, cfg.max_depth + jump * extra_loop + 1
                ]
                for controller in CONTROLLERS:
                    state = apply_rejuvenation(
                        states[controller],
                        reference=reference,
                        anchor=anchor,
                        wrong_node_reference=wrong_reference,
                        positions=positions,
                        controller=controller,
                    )
                    state = apply_shared_stack(
                        model,
                        state,
                        loop_index=cfg.max_loops + extra_loop - 1,
                    )
                    states[controller] = state
                    logits = logits_from_raw_state(model, state)
                    metrics = _masked_metrics(
                        logits,
                        intended_target,
                        endpoint=endpoint,
                    )
                    _accumulate(
                        accumulators[
                            (group, controller, extra_loop)
                        ],
                        metrics,
                    )
                    if controller == "wrong_node_replace":
                        wrong_metrics = _masked_metrics(
                            logits,
                            wrong_target,
                            endpoint=endpoint,
                        )
                        _accumulate(
                            wrong_target_accumulators[
                                (group, extra_loop)
                            ],
                            wrong_metrics,
                        )

    rows: list[dict[str, Any]] = []
    for group, positions in groups.items():
        replaced_fraction = len(positions) / cfg.seq_len
        for controller in CONTROLLERS:
            for extra_loop in range(1, extra_loops + 1):
                bucket = accumulators[
                    (group, controller, extra_loop)
                ]
                count = int(bucket["count"])
                wrong_bucket = wrong_target_accumulators[
                    (group, extra_loop)
                ]
                wrong_count = int(wrong_bucket["count"])
                rows.append(
                    {
                        "group": group,
                        "position_count": len(positions),
                        "replaced_fraction": replaced_fraction,
                        "controller": controller,
                        "extra_loop": extra_loop,
                        "target_offset": jump * extra_loop,
                        "accuracy": (
                            bucket["correct"] / count
                            if count
                            else float("nan")
                        ),
                        "probability": (
                            bucket["probability"] / count
                            if count
                            else float("nan")
                        ),
                        "margin": (
                            bucket["margin"] / count
                            if count
                            else float("nan")
                        ),
                        "valid_count": count,
                        "wrong_node_target_accuracy": (
                            wrong_bucket["correct"] / wrong_count
                            if controller == "wrong_node_replace"
                            and wrong_count
                            else float("nan")
                        ),
                    }
                )
    for extra_loop, bucket in clean_accumulators.items():
        count = int(bucket["count"])
        rows.append(
            {
                "group": "all",
                "position_count": cfg.seq_len,
                "replaced_fraction": 1.0,
                "controller": "clean_reference",
                "extra_loop": extra_loop,
                "target_offset": jump * extra_loop,
                "accuracy": (
                    bucket["correct"] / count
                    if count
                    else float("nan")
                ),
                "probability": (
                    bucket["probability"] / count
                    if count
                    else float("nan")
                ),
                "margin": (
                    bucket["margin"] / count
                    if count
                    else float("nan")
                ),
                "valid_count": count,
                "wrong_node_target_accuracy": float("nan"),
            }
        )
    return rows


def summarize_extension(
    rows: list[dict[str, Any]],
    *,
    minimum_accuracy: float = 0.5,
) -> dict[str, Any]:
    exact = [
        row for row in rows if row["controller"] == "exact_replace"
    ]
    group_summaries = []
    for group in sorted({str(row["group"]) for row in exact}):
        curve = sorted(
            [row for row in exact if row["group"] == group],
            key=lambda row: int(row["extra_loop"]),
        )
        accuracies = [float(row["accuracy"]) for row in curve]
        sustained = 0
        for accuracy in accuracies:
            if accuracy < minimum_accuracy:
                break
            sustained += 1
        group_summaries.append(
            {
                "group": group,
                "position_count": int(curve[0]["position_count"]),
                "replaced_fraction": float(
                    curve[0]["replaced_fraction"]
                ),
                "accuracy_by_extra_loop": accuracies,
                "mean_accuracy": float(np.mean(accuracies)),
                "sustained_loops_at_threshold": sustained,
            }
        )
    best_accuracy = max(
        group_summaries,
        key=lambda row: row["mean_accuracy"],
    )
    viable = [
        row
        for row in group_summaries
        if row["sustained_loops_at_threshold"] >= 4
    ]
    minimal_viable = (
        min(
            viable,
            key=lambda row: (
                row["position_count"],
                -row["mean_accuracy"],
            ),
        )
        if viable
        else None
    )
    return {
        "threshold": minimum_accuracy,
        "best_exact_replacement": best_accuracy,
        "minimal_four_loop_replacement": minimal_viable,
        "groups": group_summaries,
    }


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    phase_summary: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    candidate = load_phase_candidate(phase_summary)
    if candidate["reference_path_before"] > cfg.max_depth:
        raise ValueError("reference phase lies after the trained endpoint")
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = evaluate_lifespan_extension(
        model=model,
        cfg=cfg,
        candidate=candidate,
        device=device,
        batch_size=batch_size,
        batches=batches,
        extra_loops=extra_loops,
        seed=seed,
    )
    _write_csv(run_dir / "lifespan_extension_rows.csv", rows)
    extension_summary = summarize_extension(rows)
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "config": asdict(cfg),
        "candidate": candidate,
        "sample_size": batch_size * batches,
        "extra_loops": extra_loops,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
        **extension_summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path, Path]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,PHASE_SUMMARY"
        )
    paths = parts[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,PHASE_SUMMARY"
        )
    return parts[0], Path(paths[0]), Path(paths[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether high-dimensional recurrent phase states can "
            "rejuvenate a graph-path model beyond its trained loop horizon."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = {}
    for run_index, (name, checkpoint, phase_summary) in enumerate(
        args.run
    ):
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            phase_summary=phase_summary,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            extra_loops=args.extra_loops,
            seed=args.seed + 1000 * run_index,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
