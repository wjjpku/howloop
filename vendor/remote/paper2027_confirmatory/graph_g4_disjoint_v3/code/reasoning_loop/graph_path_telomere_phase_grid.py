from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    FunctionalIntervention,
    _roll_trace,
    run_instrumented_state,
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
    _matched_phase_selection_rows,
    _trajectory,
    advance_nodes,
    cache_states_with_initial,
    telomere_position_groups,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_metric_rows(
    batches: list[list[dict[str, Any]]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    accumulated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rows in batches:
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            bucket = accumulated.setdefault(
                key,
                {
                    "template": {
                        field: value
                        for field, value in row.items()
                        if field
                        not in {
                            "accuracy",
                            "probability",
                            "margin",
                            "valid_count",
                        }
                    },
                    "correct": 0.0,
                    "probability": 0.0,
                    "margin": 0.0,
                    "count": 0,
                },
            )
            count = int(row["valid_count"])
            if count and math.isfinite(float(row["accuracy"])):
                bucket["correct"] += float(row["accuracy"]) * count
                bucket["probability"] += float(row["probability"]) * count
                bucket["margin"] += float(row["margin"]) * count
                bucket["count"] += count
    combined: list[dict[str, Any]] = []
    for bucket in accumulated.values():
        count = int(bucket["count"])
        combined.append(
            {
                **bucket["template"],
                "accuracy": (
                    bucket["correct"] / count if count else float("nan")
                ),
                "probability": (
                    bucket["probability"] / count
                    if count
                    else float("nan")
                ),
                "margin": (
                    bucket["margin"] / count if count else float("nan")
                ),
                "valid_count": count,
            }
        )
    return combined


@torch.no_grad()
def evaluate_phase_grid(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    trajectory_positions: list[int],
    trajectory_accuracies: list[float],
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups = telomere_position_groups(cfg)
    dummy_profiles = {
        group: type(
            "PositionsOnly",
            (),
            {"positions": positions},
        )()
        for group, positions in groups.items()
    }
    batch_rows = []
    for batch_index in range(batches):
        batch_rows.append(
            _matched_phase_selection_rows(
                model=model,
                cfg=cfg,
                profiles=dummy_profiles,
                trajectory_positions=trajectory_positions,
                trajectory_accuracies=trajectory_accuracies,
                device=device,
                batch_size=batch_size,
                path_positions=cfg.max_depth + 2 * extra_loops,
                seed=seed + batch_index,
            )
        )
    return _aggregate_metric_rows(
        batch_rows,
        key_fields=(
            "group",
            "reference_age",
            "mode",
            "target_offset",
        ),
    )


@torch.no_grad()
def evaluate_closed_loop_grid(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    candidates: list[dict[str, Any]],
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    normalized_candidates = {
        (
            int(item["reference_age"]),
            int(item["reference_path_before"]),
            int(item["programmed_jump"]),
        )
        for item in candidates
        if int(item["programmed_jump"]) in {1, 2}
        and 0 <= int(item["reference_path_before"]) <= cfg.max_depth
    }
    if not normalized_candidates:
        raise ValueError("closed-loop grid has no active candidates")
    groups = telomere_position_groups(cfg)
    modes = ("matched", "batch_shuffled", "reverse")
    accumulators: dict[
        tuple[int, int, int, str, str, int], dict[str, float]
    ] = {}
    for reference_age, reference_position, jump in normalized_candidates:
        for group in groups:
            for mode in modes:
                for extra_loop in range(1, extra_loops + 1):
                    accumulators[
                        (
                            reference_age,
                            reference_position,
                            jump,
                            group,
                            mode,
                            extra_loop,
                        )
                    ] = {
                        "correct": 0.0,
                        "probability": 0.0,
                        "margin": 0.0,
                        "count": 0.0,
                    }
    set_seed(seed)
    path_positions = cfg.max_depth + 2 * extra_loops
    for _ in range(batches):
        tokens, targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        terminal = cache_states_with_initial(
            model, tokens, loops=cfg.max_loops
        )[-1]
        all_targets = _all_targets(start, targets)
        endpoint = all_targets[:, cfg.max_depth]
        for reference_age, reference_position, jump in normalized_candidates:
            deltas: list[torch.Tensor] = []
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
                reference_state = cache_states_with_initial(
                    model,
                    reference_tokens,
                    loops=max(1, reference_age),
                )[reference_age]
                anchor_state = cache_states_with_initial(
                    model,
                    anchor_tokens,
                    loops=cfg.max_loops,
                )[-1]
                deltas.append(reference_state - anchor_state)
            for group, positions_tuple in groups.items():
                positions = list(positions_tuple)
                for mode in modes:
                    state = terminal.clone()
                    for extra_loop, full_delta in enumerate(
                        deltas, start=1
                    ):
                        delta = full_delta[:, positions]
                        delta = (
                            delta.roll(1, dims=0)
                            if mode == "batch_shuffled"
                            else -delta
                            if mode == "reverse"
                            else delta
                        )
                        state[:, positions] += delta
                        state = model.apply_loop(
                            state,
                            loop_index=cfg.max_loops + extra_loop - 1,
                        )
                        logits = model.unembed(
                            model.ln_final(state[:, -1])
                        )[:, : cfg.node_count]
                        metrics = _masked_metrics(
                            logits,
                            all_targets[
                                :, cfg.max_depth + jump * extra_loop
                            ],
                            endpoint=endpoint,
                        )
                        key = (
                            reference_age,
                            reference_position,
                            jump,
                            group,
                            mode,
                            extra_loop,
                        )
                        bucket = accumulators[key]
                        count = int(metrics["valid_count"])
                        if count:
                            bucket["correct"] += (
                                float(metrics["accuracy"]) * count
                            )
                            bucket["probability"] += (
                                float(metrics["probability"]) * count
                            )
                            bucket["margin"] += (
                                float(metrics["margin"]) * count
                            )
                            bucket["count"] += count
    rows: list[dict[str, Any]] = []
    for key, bucket in accumulators.items():
        (
            reference_age,
            reference_position,
            jump,
            group,
            mode,
            extra_loop,
        ) = key
        count = bucket["count"]
        rows.append(
            {
                "reference_age": reference_age,
                "reference_path_before": reference_position,
                "programmed_jump": jump,
                "group": group,
                "mode": mode,
                "extra_loop": extra_loop,
                "target_offset": jump * extra_loop,
                "accuracy": (
                    bucket["correct"] / count if count else float("nan")
                ),
                "probability": (
                    bucket["probability"] / count
                    if count
                    else float("nan")
                ),
                "margin": (
                    bucket["margin"] / count if count else float("nan")
                ),
                "valid_count": int(count),
            }
        )
    return rows


def _metric_pair(
    logits: torch.Tensor,
    *,
    target: torch.Tensor,
    endpoint: torch.Tensor,
) -> tuple[float, float]:
    metrics = _masked_metrics(logits, target, endpoint=endpoint)
    return float(metrics["accuracy"]), float(metrics["margin"])


@torch.no_grad()
def evaluate_executor(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    selected: dict[str, Any],
    batch_size: int,
    batches: int,
    seed: int,
) -> list[dict[str, Any]]:
    group = str(selected["group"])
    positions = telomere_position_groups(cfg)[group]
    reference_age = int(selected["reference_age"])
    reference_position = int(selected["reference_path_before"])
    jump = int(selected["programmed_jump"])
    if jump not in {1, 2}:
        raise ValueError("executor evaluation requires an active phase")
    site_specs = [
        ("head", block, head)
        for block in range(cfg.n_layers)
        for head in range(cfg.n_heads)
    ] + [
        ("mlp", block, None)
        for block in range(cfg.n_layers)
    ]
    accumulators: dict[tuple[str, int, int | None], dict[str, float]] = {
        spec: {
            "baseline_accuracy": 0.0,
            "baseline_margin": 0.0,
            "zero_accuracy": 0.0,
            "zero_margin": 0.0,
            "shuffle_accuracy": 0.0,
            "shuffle_margin": 0.0,
        }
        for spec in site_specs
    }
    set_seed(seed)
    for _ in range(batches):
        tokens, targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump,
        )
        terminal = cache_states_with_initial(
            model, tokens, loops=cfg.max_loops
        )[-1]
        reference_start = advance_nodes(
            successors,
            start,
            steps=cfg.max_depth - reference_position,
        )
        reference_tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
            successors=successors,
            start=reference_start,
        )
        reference_state = cache_states_with_initial(
            model,
            reference_tokens,
            loops=max(1, reference_age),
        )[reference_age]
        patched = terminal.clone()
        patched[:, list(positions)] += (
            reference_state[:, list(positions)]
            - terminal[:, list(positions)]
        )
        baseline_logits, baseline_trace = run_instrumented_state(
            model,
            patched,
            loop_indices=(cfg.max_loops,),
        )
        shuffled_trace = _roll_trace(baseline_trace)
        all_targets = _all_targets(start, targets)
        endpoint = all_targets[:, cfg.max_depth]
        target = all_targets[:, cfg.max_depth + jump]
        baseline_accuracy, baseline_margin = _metric_pair(
            baseline_logits, target=target, endpoint=endpoint
        )
        for kind, block, head in site_specs:
            intervention = FunctionalIntervention(
                site=block,
                component=(
                    "head_context" if kind == "head" else "mlp_out"
                ),
                mode="zero",
                heads=(head,) if head is not None else None,
            )
            zero_logits, _ = run_instrumented_state(
                model,
                patched,
                loop_indices=(cfg.max_loops,),
                interventions=(intervention,),
            )
            patch_intervention = FunctionalIntervention(
                site=block,
                component=(
                    "head_context" if kind == "head" else "mlp_out"
                ),
                mode="patch",
                heads=(head,) if head is not None else None,
            )
            shuffle_logits, _ = run_instrumented_state(
                model,
                patched,
                loop_indices=(cfg.max_loops,),
                interventions=(patch_intervention,),
                donor_trace=shuffled_trace,
            )
            zero_accuracy, zero_margin = _metric_pair(
                zero_logits, target=target, endpoint=endpoint
            )
            shuffle_accuracy, shuffle_margin = _metric_pair(
                shuffle_logits, target=target, endpoint=endpoint
            )
            bucket = accumulators[(kind, block, head)]
            bucket["baseline_accuracy"] += baseline_accuracy
            bucket["baseline_margin"] += baseline_margin
            bucket["zero_accuracy"] += zero_accuracy
            bucket["zero_margin"] += zero_margin
            bucket["shuffle_accuracy"] += shuffle_accuracy
            bucket["shuffle_margin"] += shuffle_margin
    rows: list[dict[str, Any]] = []
    for (kind, block, head), bucket in accumulators.items():
        averaged = {key: value / batches for key, value in bucket.items()}
        zero_accuracy_drop = (
            averaged["baseline_accuracy"] - averaged["zero_accuracy"]
        )
        zero_margin_drop = (
            averaged["baseline_margin"] - averaged["zero_margin"]
        )
        shuffle_accuracy_drop = (
            averaged["baseline_accuracy"] - averaged["shuffle_accuracy"]
        )
        shuffle_margin_drop = (
            averaged["baseline_margin"] - averaged["shuffle_margin"]
        )
        rows.append(
            {
                "group": group,
                "reference_age": reference_age,
                "reference_path_before": reference_position,
                "programmed_jump": jump,
                "component": (
                    f"B{block + 1}.H{head}"
                    if head is not None
                    else f"B{block + 1}.MLP"
                ),
                **averaged,
                "zero_accuracy_drop": zero_accuracy_drop,
                "zero_margin_drop": zero_margin_drop,
                "shuffle_accuracy_drop": shuffle_accuracy_drop,
                "shuffle_margin_drop": shuffle_margin_drop,
                "strongly_used": (
                    (
                        zero_accuracy_drop >= 0.03
                        or zero_margin_drop >= 0.5
                    )
                    and (
                        shuffle_accuracy_drop >= 0.03
                        or shuffle_margin_drop >= 0.5
                    )
                ),
            }
        )
    return rows


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    prior_summary: Path,
    out_dir: Path,
    device: torch.device,
    trajectory_batch_size: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    seed: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint, device)
    prior = json.loads(prior_summary.read_text())
    trajectory_positions, trajectory_accuracies = _trajectory(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=trajectory_batch_size,
        path_positions=cfg.max_depth + 2 * extra_loops,
        seed=seed,
    )
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    grid_rows = evaluate_phase_grid(
        model=model,
        cfg=cfg,
        device=device,
        trajectory_positions=trajectory_positions,
        trajectory_accuracies=trajectory_accuracies,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=seed + 100,
    )
    _write_csv(run_dir / "matched_phase_grid_rows.csv", grid_rows)
    selected = prior["selected_matched_phase_reset"]
    resolved_candidates = [
        {
            "reference_age": row["reference_age"],
            "reference_path_before": row["reference_path_before"],
            "programmed_jump": row["reference_jump"],
        }
        for row in grid_rows
        if row["reference_resolved"]
        and row["reference_jump"] in {1, 2}
    ]
    resolved_candidates.append(selected)
    closed_loop_rows = evaluate_closed_loop_grid(
        model=model,
        cfg=cfg,
        device=device,
        candidates=resolved_candidates,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        extra_loops=extra_loops,
        seed=seed + 150,
    )
    _write_csv(run_dir / "closed_loop_grid_rows.csv", closed_loop_rows)
    executor_rows = evaluate_executor(
        model=model,
        cfg=cfg,
        device=device,
        selected=selected,
        batch_size=evaluation_batch_size,
        batches=evaluation_batches,
        seed=seed + 200,
    )
    _write_csv(run_dir / "renewed_executor_rows.csv", executor_rows)

    active_grid = [
        row
        for row in grid_rows
        if row["reference_resolved"]
        and row["reference_jump"] in {1, 2}
        and row["target_offset"] == row["reference_jump"]
    ]
    matched = [row for row in active_grid if row["mode"] == "matched"]
    controls = {
        (
            row["group"],
            row["reference_age"],
            row["target_offset"],
        ): row
        for row in active_grid
        if row["mode"] == "batch_shuffled"
    }
    phase_effects = []
    for row in matched:
        control = controls[
            (row["group"], row["reference_age"], row["target_offset"])
        ]
        phase_effects.append(
            {
                "reference_age": int(row["reference_age"]),
                "reference_jump": int(row["reference_jump"]),
                "group": str(row["group"]),
                "matched_accuracy": float(row["accuracy"]),
                "shuffled_accuracy": float(control["accuracy"]),
                "specificity": float(row["accuracy"])
                - float(control["accuracy"]),
            }
        )
    best_by_age = []
    for age in sorted({row["reference_age"] for row in phase_effects}):
        best_by_age.append(
            max(
                (
                    row
                    for row in phase_effects
                    if row["reference_age"] == age
                ),
                key=lambda row: row["specificity"],
            )
        )
    strong_executor = [
        row["component"] for row in executor_rows if row["strongly_used"]
    ]
    shuffled_lookup = {
        (
            row["reference_age"],
            row["reference_path_before"],
            row["programmed_jump"],
            row["group"],
            row["extra_loop"],
        ): row
        for row in closed_loop_rows
        if row["mode"] == "batch_shuffled"
    }
    closed_loop_scores = []
    candidate_keys = {
        (
            row["reference_age"],
            row["reference_path_before"],
            row["programmed_jump"],
            row["group"],
        )
        for row in closed_loop_rows
        if row["mode"] == "matched"
    }
    for candidate_key in candidate_keys:
        matched_curve = sorted(
            (
                row
                for row in closed_loop_rows
                if row["mode"] == "matched"
                and (
                    row["reference_age"],
                    row["reference_path_before"],
                    row["programmed_jump"],
                    row["group"],
                )
                == candidate_key
            ),
            key=lambda row: row["extra_loop"],
        )
        control_curve = [
            shuffled_lookup[
                (*candidate_key, row["extra_loop"])
            ]
            for row in matched_curve
        ]
        closed_loop_scores.append(
            {
                "reference_age": int(candidate_key[0]),
                "reference_path_before": int(candidate_key[1]),
                "programmed_jump": int(candidate_key[2]),
                "group": str(candidate_key[3]),
                "accuracy_by_extra_loop": [
                    float(row["accuracy"]) for row in matched_curve
                ],
                "shuffled_accuracy_by_extra_loop": [
                    float(row["accuracy"]) for row in control_curve
                ],
                "mean_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in matched_curve])
                ),
                "mean_specificity": float(
                    np.mean(
                        [
                            float(row["accuracy"])
                            - float(control["accuracy"])
                            for row, control in zip(
                                matched_curve,
                                control_curve,
                                strict=True,
                            )
                        ]
                    )
                ),
            }
        )
    best_closed_loop = max(
        closed_loop_scores,
        key=lambda row: (row["mean_accuracy"], row["mean_specificity"]),
    )
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "trajectory_positions_including_initial": trajectory_positions,
        "trajectory_accuracies_including_initial": trajectory_accuracies,
        "best_matched_phase_effect_by_age": best_by_age,
        "renewed_executor_strong_components": strong_executor,
        "best_closed_loop_control": best_closed_loop,
        "selected_matched_phase_reset": selected,
        "sample_sizes": {
            "trajectory": trajectory_batch_size,
            "evaluation": evaluation_batch_size * evaluation_batches,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path, Path]:
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,SUMMARY"
        )
    paths = parts[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,SUMMARY"
        )
    return parts[0], Path(paths[0]), Path(paths[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate phase-specific jump control and locate the executor "
            "that consumes a renewed graph-path phase state."
        )
    )
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trajectory-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = {}
    for name, checkpoint, prior_summary in args.run:
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            prior_summary=prior_summary,
            out_dir=args.out_dir,
            device=device,
            trajectory_batch_size=args.trajectory_batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            evaluation_batches=args.evaluation_batches,
            extra_loops=args.extra_loops,
            seed=args.seed,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
