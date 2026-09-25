from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_telomere_lifespan_extension import (
    _accumulate,
    _blank_bucket,
    extension_position_groups,
)
from reasoning_loop.graph_path_telomere_overloop import (
    _all_targets,
    _masked_metrics,
    advance_nodes,
    cache_states_with_initial,
)
from reasoning_loop.graph_path_telomere_rejuvenator import FlattenedAffine
from reasoning_loop.graph_path_telomere_single_age import (
    fit_single_age_map,
    homogeneous_age_matrix,
    relative_mse,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_rows(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, parts in grouped.items():
        result = dict(zip(keys, group, strict=True))
        numeric_keys = [
            key
            for key, value in parts[0].items()
            if key not in keys and isinstance(value, (int, float))
        ]
        for key in numeric_keys:
            values = [float(part[key]) for part in parts]
            result[key] = sum(values) / len(values)
            result[f"{key}_min"] = min(values)
            result[f"{key}_max"] = max(values)
        output.append(result)
    return sorted(output, key=lambda row: tuple(row[key] for key in keys))


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


def exact_predecessor(
    successors: torch.Tensor,
    current: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    if successors.ndim != 2:
        raise ValueError("successors must have shape [batch, node]")
    if current.shape != (successors.shape[0],):
        raise ValueError("current must have shape [batch]")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    sources = torch.arange(
        successors.shape[1],
        device=successors.device,
    )[None, :].expand_as(successors)
    inverse = torch.empty_like(successors)
    inverse.scatter_(1, successors, sources)
    return advance_nodes(inverse, current, steps=steps)


@torch.no_grad()
def _aligned_state_at_age(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    successors: torch.Tensor,
    current: torch.Tensor,
    age: int,
    phase_position: int,
) -> torch.Tensor:
    start = exact_predecessor(
        successors,
        current,
        steps=phase_position,
    )
    tokens, _, _, _ = fixed_depth_batch(
        cfg,
        current.shape[0],
        current.device,
        path_positions=cfg.max_depth,
        successors=successors,
        start=start,
    )
    return cache_states_with_initial(
        model,
        tokens,
        loops=max(1, age),
    )[age]


def _apply_answer_map(
    state: torch.Tensor,
    *,
    answer_positions: tuple[int, ...],
    age_map: FlattenedAffine,
    mode: str,
) -> torch.Tensor:
    if mode not in {"matched", "shuffled", "reverse"}:
        raise ValueError(f"unknown age-map mode: {mode}")
    index = list(answer_positions)
    result = state.clone()
    rejuvenated = age_map(result[:, index])
    if mode == "shuffled":
        rejuvenated = rejuvenated.roll(1, dims=0)
    elif mode == "reverse":
        rejuvenated = 2 * result[:, index] - rejuvenated
    result[:, index] = rejuvenated
    return result


def _repeat_answer_map(
    state: torch.Tensor,
    *,
    answer_positions: tuple[int, ...],
    age_map: FlattenedAffine,
    applications: int,
    mode: str,
) -> torch.Tensor:
    result = state
    for _ in range(applications):
        result = _apply_answer_map(
            result,
            answer_positions=answer_positions,
            age_map=age_map,
            mode=mode,
        )
    return result


@torch.no_grad()
def collect_h3_h2_pairs(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    answer_positions: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources = []
    targets = []
    set_seed(seed)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        source = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=3,
            phase_position=phase_positions[3],
        )
        target = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=current,
            age=2,
            phase_position=phase_positions[2],
        )
        sources.append(source[:, list(answer_positions)])
        targets.append(target[:, list(answer_positions)])
    return torch.cat(sources), torch.cat(targets)


@torch.no_grad()
def evaluate_age_ladder(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    answer_positions: tuple[int, ...],
    age_map: FlattenedAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
) -> list[dict[str, Any]]:
    parts = []
    set_seed(seed)
    for batch_index in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        states = {
            age: _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=age,
                phase_position=phase_positions[age],
            )
            for age in range(2, cfg.max_loops + 1)
        }
        for source_age in range(3, cfg.max_loops + 1):
            prediction = states[source_age]
            for applications in range(1, source_age - 1):
                prediction = _apply_answer_map(
                    prediction,
                    answer_positions=answer_positions,
                    age_map=age_map,
                    mode="matched",
                )
                target_age = source_age - applications
                oracle = states[target_age]
                answer_prediction = prediction[
                    :, list(answer_positions)
                ]
                answer_oracle = oracle[:, list(answer_positions)]
                direct_accuracy = _accuracy(
                    logits_from_raw_state(model, prediction),
                    current,
                )
                jump = (
                    phase_positions[target_age + 1]
                    - phase_positions[target_age]
                )
                next_target = advance_nodes(
                    successors,
                    current,
                    steps=jump,
                )
                executed = apply_shared_stack(
                    model,
                    prediction,
                    loop_index=target_age,
                )
                oracle_executed = apply_shared_stack(
                    model,
                    oracle,
                    loop_index=target_age,
                )
                parts.append(
                    {
                        "batch": batch_index,
                        "source_age": source_age,
                        "applications": applications,
                        "target_age": target_age,
                        "target_next_jump": jump,
                        "answer_relative_mse": relative_mse(
                            answer_prediction,
                            answer_oracle,
                        ),
                        "direct_current_accuracy": direct_accuracy,
                        "next_execution_accuracy": _accuracy(
                            logits_from_raw_state(model, executed),
                            next_target,
                        ),
                        "oracle_next_execution_accuracy": _accuracy(
                            logits_from_raw_state(
                                model,
                                oracle_executed,
                            ),
                            next_target,
                        ),
                    }
                )
    return _mean_rows(
        parts,
        keys=("source_age", "applications", "target_age"),
    )


def _finalize_curve_rows(
    accumulators: dict[tuple[str, int], dict[str, float]],
    *,
    start_mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for condition, extra_loop in sorted(accumulators):
        bucket = accumulators[(condition, extra_loop)]
        count = int(bucket["count"])
        rows.append(
            {
                "start_mode": start_mode,
                "condition": condition,
                "extra_loop": extra_loop,
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
            }
        )
    return rows


@torch.no_grad()
def evaluate_closed_loop(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    phase_positions: list[int],
    answer_positions: tuple[int, ...],
    age_map: FlattenedAffine,
    device: torch.device,
    batch_size: int,
    batches: int,
    extra_loops: int,
    seed: int,
    start_mode: str,
) -> list[dict[str, Any]]:
    if start_mode not in {"exact_h2", "terminal_h8"}:
        raise ValueError("start_mode must be exact_h2 or terminal_h8")
    jump = phase_positions[3] - phase_positions[2]
    if jump < 1:
        raise ValueError("age-2 phase must execute a positive graph jump")
    conditions = (
        ("learned", "matched"),
        ("no_rejuvenation", "matched"),
        ("batch_shuffled", "shuffled"),
        ("reverse", "reverse"),
        ("oracle_young", "matched"),
    )
    if start_mode == "terminal_h8":
        conditions = (
            *conditions,
            ("under_rewind", "matched"),
            ("over_rewind", "matched"),
        )
    accumulators = {
        (condition, extra_loop): _blank_bucket()
        for condition, _ in conditions
        for extra_loop in range(1, extra_loops + 1)
    }

    set_seed(seed)
    for _ in range(batches):
        tokens, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth + jump * extra_loops,
        )
        all_targets = _all_targets(start, path_targets)
        endpoint = all_targets[:, cfg.max_depth]
        terminal = cache_states_with_initial(
            model,
            tokens,
            loops=cfg.max_loops,
        )[-1]
        exact_h2 = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=2,
            phase_position=phase_positions[2],
        )
        if start_mode == "exact_h2":
            states = {
                condition: exact_h2.clone()
                for condition, _ in conditions
            }
        else:
            rewind = cfg.max_loops - 2
            states = {}
            for condition, mode in conditions:
                applications = rewind
                if condition == "no_rejuvenation":
                    applications = 0
                elif condition == "oracle_young":
                    states[condition] = exact_h2.clone()
                    continue
                elif condition == "under_rewind":
                    applications -= 1
                elif condition == "over_rewind":
                    applications += 1
                states[condition] = _repeat_answer_map(
                    terminal,
                    answer_positions=answer_positions,
                    age_map=age_map,
                    applications=applications,
                    mode=mode,
                )

        for extra_loop in range(1, extra_loops + 1):
            target = all_targets[
                :, cfg.max_depth + jump * extra_loop
            ]
            for condition, _ in conditions:
                states[condition] = apply_shared_stack(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + extra_loop - 1,
                )
                metrics = _masked_metrics(
                    logits_from_raw_state(
                        model,
                        states[condition],
                    ),
                    target,
                    endpoint=endpoint,
                )
                _accumulate(
                    accumulators[(condition, extra_loop)],
                    metrics,
                )
            if extra_loop == extra_loops:
                continue
            for condition, mode in conditions:
                if condition == "no_rejuvenation":
                    continue
                if condition == "oracle_young":
                    states[condition] = _aligned_state_at_age(
                        model=model,
                        cfg=cfg,
                        successors=successors,
                        current=target,
                        age=2,
                        phase_position=phase_positions[2],
                    )
                    continue
                states[condition] = _apply_answer_map(
                    states[condition],
                    answer_positions=answer_positions,
                    age_map=age_map,
                    mode=mode,
                )
    return _finalize_curve_rows(
        accumulators,
        start_mode=start_mode,
    )


def _curve(
    rows: list[dict[str, Any]],
    *,
    start_mode: str,
    condition: str,
) -> list[float]:
    return [
        float(row["accuracy"])
        for row in sorted(
            (
                row
                for row in rows
                if row["start_mode"] == start_mode
                and row["condition"] == condition
            ),
            key=lambda row: int(row["extra_loop"]),
        )
    ]


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    out_dir: Path,
    device_name: str,
    calibration_batch_size: int,
    calibration_batches: int,
    heldout_batch_size: int,
    heldout_batches: int,
    evaluation_batch_size: int,
    evaluation_batches: int,
    extra_loops: int,
    ridge: float,
    seed: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.08")
        )
        if not 0 < fraction <= 1:
            raise ValueError("CUDA memory fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if not (
        cfg.node_count == 8
        and cfg.max_depth == 8
        and cfg.max_loops == 8
        and cfg.n_layers == 2
    ):
        raise ValueError("experiment requires the Graph D8L8 two-block model")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary[
            "trajectory_positions_including_initial"
        ]
    ]
    if len(phase_positions) != cfg.max_loops + 1:
        raise ValueError("phase trajectory must include ages 0 through 8")
    answer_positions = extension_position_groups(cfg)["answer"]

    calibration_source, calibration_target = collect_h3_h2_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        answer_positions=answer_positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=seed,
    )
    age_map = fit_single_age_map(
        calibration_source,
        calibration_target,
        ridge=ridge,
    )
    if not isinstance(age_map, FlattenedAffine):
        raise TypeError("answer-only map must be a flattened affine map")
    heldout_source, heldout_target = collect_h3_h2_pairs(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        answer_positions=answer_positions,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=seed + 1,
    )
    heldout_h3_h2_relative_mse = relative_mse(
        age_map(heldout_source),
        heldout_target,
    )
    age_rows = evaluate_age_ladder(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        answer_positions=answer_positions,
        age_map=age_map,
        device=device,
        batch_size=heldout_batch_size,
        batches=heldout_batches,
        seed=seed + 2,
    )
    closed_loop_rows = []
    for offset, start_mode in enumerate(("exact_h2", "terminal_h8")):
        closed_loop_rows.extend(
            evaluate_closed_loop(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                answer_positions=answer_positions,
                age_map=age_map,
                device=device,
                batch_size=evaluation_batch_size,
                batches=evaluation_batches,
                extra_loops=extra_loops,
                seed=seed + 10 + offset,
                start_mode=start_mode,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    homogeneous = homogeneous_age_matrix(age_map)
    torch.save(
        {
            "format_version": 1,
            "kind": "graph_path_simple_one_step_telomere",
            "checkpoint": str(checkpoint),
            "checkpoint_seed": payload.get("initialization_seed"),
            "training_relation": "H3(current) -> H2(current)",
            "positions": list(answer_positions),
            "ridge": ridge,
            "homogeneous_matrix": homogeneous.cpu(),
            "weight": age_map.affine.weight[0].cpu(),
            "bias": age_map.affine.bias[0].cpu(),
        },
        out_dir / "simple_one_step_R.pt",
    )
    _write_csv(out_dir / "age_ladder_rows.csv", age_rows)
    _write_csv(out_dir / "closed_loop_rows.csv", closed_loop_rows)

    exact_curve = _curve(
        closed_loop_rows,
        start_mode="exact_h2",
        condition="learned",
    )
    terminal_curve = _curve(
        closed_loop_rows,
        start_mode="terminal_h8",
        condition="learned",
    )
    terminal_rewind = next(
        row
        for row in age_rows
        if int(row["source_age"]) == 8
        and int(row["target_age"]) == 2
    )
    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "checkpoint_seed": payload.get("initialization_seed"),
        "config": {
            "node_count": cfg.node_count,
            "max_depth": cfg.max_depth,
            "max_loops": cfg.max_loops,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_mlp": cfg.d_mlp,
        },
        "device": str(device),
        "phase_positions": phase_positions,
        "training": {
            "relation": "H3(current) -> H2(current)",
            "loss": "closed-form ridge hidden-state MSE",
            "ridge": ridge,
            "calibration_examples": int(calibration_source.shape[0]),
            "excluded": [
                "task CE",
                "multi-age loss",
                "R power loss",
                "closed-loop loss",
                "DAgger or rollout-state training",
            ],
        },
        "operator": {
            "form": "R(h) = hW + b on the answer position",
            "parameter_count": int(
                age_map.affine.weight[0].numel()
                + age_map.affine.bias[0].numel()
            ),
            "homogeneous_shape": list(homogeneous.shape),
        },
        "heldout_examples": int(heldout_source.shape[0]),
        "heldout_h3_h2_relative_mse": heldout_h3_h2_relative_mse,
        "terminal_h8_to_h2": terminal_rewind,
        "exact_h2_closed_loop_accuracy": exact_curve,
        "exact_h2_closed_loop_auc": float(np.mean(exact_curve)),
        "terminal_h8_closed_loop_accuracy": terminal_curve,
        "terminal_h8_closed_loop_auc": float(
            np.mean(terminal_curve)
        ),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
        "files": {
            "operator": "simple_one_step_R.pt",
            "age_ladder": "age_ladder_rows.csv",
            "closed_loop": "closed_loop_rows.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit R only on D8L8 H3->H2 and test zero-shot powers and "
            "closed-loop reuse."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-batch-size", type=int, default=512)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--heldout-batch-size", type=int, default=256)
    parser.add_argument("--heldout-batches", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--extra-loops", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        out_dir=args.out_dir,
        device_name=args.device,
        calibration_batch_size=args.calibration_batch_size,
        calibration_batches=args.calibration_batches,
        heldout_batch_size=args.heldout_batch_size,
        heldout_batches=args.heldout_batches,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_batches=args.evaluation_batches,
        extra_loops=args.extra_loops,
        ridge=args.ridge,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
