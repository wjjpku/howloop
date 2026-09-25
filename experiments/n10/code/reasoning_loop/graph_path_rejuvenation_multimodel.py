from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_rejuvenation_circuit import (
    MatchedAgeBatch,
    attention_function_rows,
    behavior_metrics,
    collect_matched_age_batch,
    component_localization_rows,
    discover_component_circuit,
    evaluate_state,
    qkv_mediation_rows,
    stage_readout_rows,
    validate_component_circuit,
)
from reasoning_loop.graph_path_telomere_lifespan_extension import (
    extension_position_groups,
)
from reasoning_loop.graph_path_telomere_rejuvenator import (
    FlattenedAffine,
    PositionwiseAffine,
    _relative_mse,
    collect_rejuvenator_pairs,
    evaluate_rejuvenator,
    fit_positionwise_affine,
    select_rejuvenator_group,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)
from reasoning_loop.graph_path_functional_circuit import (
    run_instrumented_state,
)


AffineMap = PositionwiseAffine | FlattenedAffine


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_single_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> AffineMap:
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError(
            "source and target must share [sample, position, feature] shape"
        )
    position_count = source.shape[1]
    feature_count = source.shape[2]
    if position_count > 4:
        return fit_positionwise_affine(source, target, ridge=ridge)
    affine = fit_positionwise_affine(
        source.reshape(source.shape[0], 1, -1),
        target.reshape(target.shape[0], 1, -1),
        ridge=ridge,
    )
    return FlattenedAffine(
        affine=affine,
        position_count=position_count,
        feature_count=feature_count,
    )


def apply_affine_to_state(
    state: torch.Tensor,
    *,
    positions: Sequence[int],
    affine: AffineMap,
    mode: str = "matched",
) -> torch.Tensor:
    if mode not in {"matched", "shuffled", "reverse"}:
        raise ValueError(f"unknown affine intervention mode: {mode}")
    index = list(positions)
    old = state[:, index]
    prediction = affine(old)
    if mode == "shuffled":
        prediction = prediction.roll(1, dims=0)
    elif mode == "reverse":
        prediction = 2 * old - prediction
    result = state.clone()
    result[:, index] = prediction
    return result


def affine_payload(affine: AffineMap) -> dict[str, Any]:
    if isinstance(affine, FlattenedAffine):
        return {
            "structure": "flattened_affine",
            "position_count": affine.position_count,
            "feature_count": affine.feature_count,
            "weight": affine.affine.weight.detach().cpu(),
            "bias": affine.affine.bias.detach().cpu(),
        }
    return {
        "structure": "positionwise_affine",
        "position_count": int(affine.weight.shape[0]),
        "feature_count": int(affine.weight.shape[1]),
        "weight": affine.weight.detach().cpu(),
        "bias": affine.bias.detach().cpu(),
    }


def select_affine_by_state_mse(
    *,
    calibration_source: torch.Tensor,
    calibration_target: torch.Tensor,
    validation_source: torch.Tensor,
    validation_target: torch.Tensor,
    ridge_values: Sequence[float],
) -> tuple[AffineMap, float, list[dict[str, Any]]]:
    rows = []
    maps = {}
    for ridge in ridge_values:
        affine = fit_single_affine(
            calibration_source,
            calibration_target,
            ridge=float(ridge),
        )
        maps[float(ridge)] = affine
        prediction = affine(validation_source)
        rows.append(
            {
                "ridge": float(ridge),
                "validation_state_relative_mse": _relative_mse(
                    prediction,
                    validation_target,
                ),
                "validation_state_mse": float(
                    (
                        prediction.float()
                        - validation_target.float()
                    )
                    .square()
                    .mean()
                ),
            }
        )
    best = min(
        rows,
        key=lambda row: (
            float(row["validation_state_relative_mse"]),
            float(row["validation_state_mse"]),
            float(row["ridge"]),
        ),
    )
    selected_ridge = float(best["ridge"])
    return maps[selected_ridge], selected_ridge, rows


def fit_age_basis(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    max_rank: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    delta = (target.float() - source.float()).flatten(1)
    mean_delta = delta.mean(dim=0)
    centered = delta - mean_delta
    total_variance = float(centered.square().sum().clamp_min(1e-12))
    available_rank = min(
        int(max_rank),
        centered.shape[0] - 1,
        centered.shape[1],
    )
    if available_rank <= 0:
        return (
            mean_delta,
            centered.new_zeros((0, centered.shape[1])),
            centered.new_zeros((0,)),
            total_variance,
        )
    set_seed(seed)
    _, singular_values, directions = torch.pca_lowrank(
        centered,
        q=available_rank,
        center=False,
        niter=4,
    )
    return (
        mean_delta,
        directions.T.contiguous(),
        singular_values,
        total_variance,
    )


def project_affine_update(
    source: torch.Tensor,
    prediction: torch.Tensor,
    *,
    mean_delta: torch.Tensor,
    directions: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if not 0 <= rank <= directions.shape[0]:
        raise ValueError("rank is outside the available age basis")
    flat_source = source.float().flatten(1)
    flat_update = prediction.float().flatten(1) - flat_source
    centered_update = flat_update - mean_delta
    if rank:
        basis = directions[:rank]
        centered_update = (centered_update @ basis.T) @ basis
    else:
        centered_update = torch.zeros_like(centered_update)
    projected = mean_delta + centered_update
    return (flat_source + projected).reshape_as(source).to(source.dtype)


def random_directions(
    *,
    feature_count: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    if rank == 0:
        return torch.zeros(0, feature_count, device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(
        feature_count,
        rank,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return torch.linalg.qr(matrix, mode="reduced").Q.T.contiguous()


@torch.no_grad()
def age_rank_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    positions: Sequence[int],
    affine: AffineMap,
    mean_delta: torch.Tensor,
    directions: torch.Tensor,
    singular_values: torch.Tensor,
    total_variance: float,
    ranks: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    index = list(positions)
    source = batch.terminal[:, index]
    prediction = affine(source)
    rows = []
    allowed_ranks = sorted(
        {
            max(0, min(int(rank), directions.shape[0]))
            for rank in ranks
        }
    )
    for rank in allowed_ranks:
        projected = project_affine_update(
            source,
            prediction,
            mean_delta=mean_delta,
            directions=directions,
            rank=rank,
        )
        random_basis = random_directions(
            feature_count=mean_delta.numel(),
            rank=rank,
            device=source.device,
            dtype=source.float().dtype,
            seed=seed + rank,
        )
        random_projected = project_affine_update(
            source,
            prediction,
            mean_delta=mean_delta,
            directions=random_basis,
            rank=rank,
        )
        full_update = prediction.float() - source.float()
        projected_update = projected.float() - source.float()
        complement = source.float() + full_update - projected_update
        conditions = (
            ("age_subspace", projected),
            ("age_subspace_complement", complement.to(source.dtype)),
            ("random_output_subspace", random_projected),
        )
        for condition, transformed in conditions:
            state = batch.terminal.clone()
            state[:, index] = transformed
            _, metrics = evaluate_state(
                model=model,
                cfg=cfg,
                state=state,
                batch=batch,
            )
            retained = (
                float(
                    singular_values[:rank].square().sum()
                    / total_variance
                )
                if rank
                else 0.0
            )
            rows.append(
                {
                    "rank": rank,
                    "condition": condition,
                    "retained_age_variance": retained,
                    "state_relative_mse": _relative_mse(
                        transformed,
                        batch.young[:, index],
                    ),
                    **metrics,
                }
            )
    return rows


def _metric_row(
    *,
    condition: str,
    logits: torch.Tensor,
    batch: MatchedAgeBatch,
) -> dict[str, Any]:
    return {
        "condition": condition,
        **behavior_metrics(
            logits,
            target=batch.target,
            endpoint=batch.endpoint,
        ),
    }


@torch.no_grad()
def one_step_behavior(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: MatchedAgeBatch,
    positions: Sequence[int],
    affine: AffineMap,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    index = list(positions)
    states = {
        "terminal": batch.terminal,
        "single_J": apply_affine_to_state(
            batch.terminal,
            positions=positions,
            affine=affine,
        ),
        "single_J_shuffled": apply_affine_to_state(
            batch.terminal,
            positions=positions,
            affine=affine,
            mode="shuffled",
        ),
        "single_J_reverse": apply_affine_to_state(
            batch.terminal,
            positions=positions,
            affine=affine,
            mode="reverse",
        ),
    }
    oracle = batch.terminal.clone()
    oracle[:, index] = batch.young[:, index]
    states["oracle_young"] = oracle
    rows = []
    for condition, state in states.items():
        rows.append(
            _metric_row(
                condition=f"{condition}_before_shared_stack",
                logits=logits_from_raw_state(model, state),
                batch=batch,
            )
        )
        logits, metrics = evaluate_state(
            model=model,
            cfg=cfg,
            state=state,
            batch=batch,
        )
        del logits
        rows.append(
            {
                "condition": f"{condition}_after_shared_stack",
                **metrics,
            }
        )
    return rows, states


def _find_metric(
    rows: Sequence[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    return next(row for row in rows if row["condition"] == condition)


@torch.no_grad()
def analyze_model(
    *,
    name: str,
    checkpoint: Path,
    lifespan_summary_path: Path,
    out_dir: Path,
    device: torch.device,
    calibration_batch_size: int,
    calibration_batches: int,
    validation_size: int,
    discovery_size: int,
    evaluation_size: int,
    lifespan_batch_size: int,
    lifespan_batches: int,
    extra_loops: int,
    ridge_values: Sequence[float],
    ranks: Sequence[int],
    max_age_rank: int,
    random_circuit_subsets: int,
    seed: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    lifespan_summary = json.loads(
        lifespan_summary_path.read_text(encoding="utf-8")
    )
    candidate = lifespan_summary["candidate"]
    group = select_rejuvenator_group(lifespan_summary)
    positions = extension_position_groups(cfg)[group]
    calibration = collect_rejuvenator_pairs(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        device=device,
        batch_size=calibration_batch_size,
        batches=calibration_batches,
        seed=seed,
    )
    validation = collect_rejuvenator_pairs(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        device=device,
        batch_size=validation_size,
        batches=1,
        seed=seed + 1,
    )
    affine, selected_ridge, ridge_rows = select_affine_by_state_mse(
        calibration_source=calibration["initial_source"],
        calibration_target=calibration["initial_target"],
        validation_source=validation["initial_source"],
        validation_target=validation["initial_target"],
        ridge_values=ridge_values,
    )
    mean_delta, directions, age_singular_values, total_age_variance = (
        fit_age_basis(
            calibration["initial_source"],
            calibration["initial_target"],
            max_rank=max_age_rank,
            seed=seed + 2,
        )
    )
    evaluation = collect_matched_age_batch(
        model=model,
        cfg=cfg,
        candidate=candidate,
        batch_size=evaluation_size,
        device=device,
        seed=seed + 3,
    )
    behavior_rows, states = one_step_behavior(
        model=model,
        cfg=cfg,
        batch=evaluation,
        positions=positions,
        affine=affine,
    )
    rank_rows = age_rank_rows(
        model=model,
        cfg=cfg,
        batch=evaluation,
        positions=positions,
        affine=affine,
        mean_delta=mean_delta,
        directions=directions,
        singular_values=age_singular_values,
        total_variance=total_age_variance,
        ranks=ranks,
        seed=seed + 4,
    )
    lifespan_rows = evaluate_rejuvenator(
        model=model,
        cfg=cfg,
        candidate=candidate,
        positions=positions,
        initial_map=affine,
        cycle_map=affine,
        device=device,
        batch_size=lifespan_batch_size,
        batches=lifespan_batches,
        extra_loops=extra_loops,
        seed=seed + 5,
    )
    discovery = collect_matched_age_batch(
        model=model,
        cfg=cfg,
        candidate=candidate,
        batch_size=discovery_size,
        device=device,
        seed=seed + 6,
    )
    discovery_state = apply_affine_to_state(
        discovery.terminal,
        positions=positions,
        affine=affine,
    )
    full_accuracy = float(
        _find_metric(
            behavior_rows,
            "single_J_after_shared_stack",
        )["accuracy"]
    )
    selected_nodes, circuit_search_rows = discover_component_circuit(
        model=model,
        cfg=cfg,
        batch=discovery,
        rejuvenated_state=discovery_state,
        target_recovery=0.90,
        min_accuracy=max(0.0, full_accuracy - 0.02),
    )
    circuit_rows = validate_component_circuit(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=states["single_J"],
        selected=selected_nodes,
        random_subsets=random_circuit_subsets,
        exhaustive_subsets=False,
        seed=seed + 7,
    )
    oracle_circuit_rows = validate_component_circuit(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=states["oracle_young"],
        selected=selected_nodes,
        random_subsets=0,
        exhaustive_subsets=False,
        seed=seed + 8,
    )
    localization_rows, _, old_trace, _, clean_trace = (
        component_localization_rows(
            model=model,
            cfg=cfg,
            batch=evaluation,
            rejuvenated_state=states["single_J"],
        )
    )
    qkv_rows = qkv_mediation_rows(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=states["single_J"],
    )
    oracle_qkv_rows = qkv_mediation_rows(
        model=model,
        cfg=cfg,
        batch=evaluation,
        rejuvenated_state=states["oracle_young"],
    )
    _, oracle_trace = run_instrumented_state(
        model,
        states["oracle_young"],
        loop_indices=(cfg.max_loops,),
    )
    attention_rows = attention_function_rows(
        cfg=cfg,
        batch=evaluation,
        traces={
            "terminal": old_trace,
            "single_J": clean_trace,
            "oracle_young": oracle_trace,
        },
    )
    stage_rows = stage_readout_rows(
        model=model,
        batch=evaluation,
        initial_states={
            "terminal": evaluation.terminal,
            "single_J": states["single_J"],
            "oracle_young": states["oracle_young"],
        },
        traces={
            "terminal": old_trace,
            "single_J": clean_trace,
            "oracle_young": oracle_trace,
        },
    )
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "ridge_selection_rows.csv", ridge_rows)
    _write_csv(run_dir / "one_step_behavior_rows.csv", behavior_rows)
    _write_csv(run_dir / "age_rank_rows.csv", rank_rows)
    _write_csv(run_dir / "single_J_lifespan_rows.csv", lifespan_rows)
    _write_csv(run_dir / "component_localization_rows.csv", localization_rows)
    _write_csv(run_dir / "circuit_search_rows.csv", circuit_search_rows)
    _write_csv(run_dir / "circuit_validation_rows.csv", circuit_rows)
    _write_csv(
        run_dir / "oracle_circuit_transfer_rows.csv",
        oracle_circuit_rows,
    )
    _write_csv(run_dir / "qkv_mediation_rows.csv", qkv_rows)
    _write_csv(
        run_dir / "oracle_qkv_mediation_rows.csv",
        oracle_qkv_rows,
    )
    _write_csv(run_dir / "attention_function_rows.csv", attention_rows)
    _write_csv(run_dir / "stage_readout_rows.csv", stage_rows)
    torch.save(
        {
            "format_version": 1,
            "name": name,
            "checkpoint": str(checkpoint),
            "candidate": candidate,
            "position_group": group,
            "positions": tuple(int(position) for position in positions),
            "ridge_selection_metric": "heldout_state_relative_mse",
            "selected_ridge": selected_ridge,
            "affine": affine_payload(affine),
            "mean_age_delta": mean_delta.detach().cpu(),
            "age_directions": directions.detach().cpu(),
            "age_singular_values": age_singular_values.detach().cpu(),
        },
        run_dir / "single_rejuvenation_matrix.pt",
    )
    target_rank_accuracy = min(0.90, full_accuracy - 0.02)
    sufficient_ranks = [
        int(row["rank"])
        for row in rank_rows
        if row["condition"] == "age_subspace"
        and float(row["accuracy"]) >= target_rank_accuracy
    ]
    minimal_rank = min(sufficient_ranks) if sufficient_ranks else None
    circuit_only = _find_metric(circuit_rows, "circuit_only")
    complement_only = _find_metric(circuit_rows, "complement_only")
    oracle_circuit_only = _find_metric(
        oracle_circuit_rows,
        "circuit_only",
    )
    learned_curve = [
        float(row["accuracy"])
        for row in lifespan_rows
        if row["condition"] == "learned"
    ]
    shuffled_curve = [
        float(row["accuracy"])
        for row in lifespan_rows
        if row["condition"] == "learned_shuffled"
    ]
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "config": asdict(cfg),
        "candidate": candidate,
        "position_group": group,
        "position_count": len(positions),
        "map_structure": affine_payload(affine)["structure"],
        "matrix_is_checkpoint_specific": True,
        "single_matrix_used_at_every_extra_loop": True,
        "ridge_selection_metric": "heldout_state_relative_mse",
        "selected_ridge": selected_ridge,
        "heldout_state_relative_mse": _relative_mse(
            affine(validation["initial_source"]),
            validation["initial_target"],
        ),
        "terminal_behavior": _find_metric(
            behavior_rows,
            "terminal_after_shared_stack",
        ),
        "pre_stack_single_J_behavior": _find_metric(
            behavior_rows,
            "single_J_before_shared_stack",
        ),
        "single_J_behavior": _find_metric(
            behavior_rows,
            "single_J_after_shared_stack",
        ),
        "oracle_young_behavior": _find_metric(
            behavior_rows,
            "oracle_young_after_shared_stack",
        ),
        "shuffled_J_behavior": _find_metric(
            behavior_rows,
            "single_J_shuffled_after_shared_stack",
        ),
        "reverse_J_behavior": _find_metric(
            behavior_rows,
            "single_J_reverse_after_shared_stack",
        ),
        "minimal_sufficient_age_rank": minimal_rank,
        "age_variance_at_minimal_rank": (
            next(
                float(row["retained_age_variance"])
                for row in rank_rows
                if row["condition"] == "age_subspace"
                and int(row["rank"]) == minimal_rank
            )
            if minimal_rank is not None
            else None
        ),
        "single_J_lifespan_accuracy": learned_curve,
        "single_J_lifespan_mean_accuracy": float(
            np.mean(learned_curve)
        ),
        "single_J_shuffled_lifespan_mean_accuracy": float(
            np.mean(shuffled_curve)
        ),
        "selected_circuit_nodes": [
            node.label for node in selected_nodes
        ],
        "circuit_only": circuit_only,
        "complement_only": complement_only,
        "oracle_young_circuit_only": oracle_circuit_only,
        "sample_sizes": {
            "calibration": (
                calibration_batch_size * calibration_batches
            ),
            "ridge_validation": validation_size,
            "circuit_discovery": discovery_size,
            "final_evaluation": evaluation_size,
            "lifespan_evaluation": (
                lifespan_batch_size * lifespan_batches
            ),
        },
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device)) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path, Path]:
    name_and_paths = text.split("=", 1)
    if len(name_and_paths) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,LIFESPAN_SUMMARY"
        )
    paths = name_and_paths[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError(
            "run must be NAME=CHECKPOINT,LIFESPAN_SUMMARY"
        )
    return name_and_paths[0], Path(paths[0]), Path(paths[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one checkpoint-specific rejuvenation matrix per model, "
            "select it without answer labels, and test one-step circuit "
            "reactivation plus repeated single-matrix lifespan."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration-batch-size", type=int, default=256)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--discovery-size", type=int, default=256)
    parser.add_argument("--evaluation-size", type=int, default=512)
    parser.add_argument("--lifespan-batch-size", type=int, default=128)
    parser.add_argument("--lifespan-batches", type=int, default=4)
    parser.add_argument("--extra-loops", type=int, default=8)
    parser.add_argument(
        "--ridge-values",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument("--max-age-rank", type=int, default=128)
    parser.add_argument("--random-circuit-subsets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026073001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("cuda memory fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    summaries = {}
    for run_index, (name, checkpoint, lifespan_summary) in enumerate(
        args.run
    ):
        summaries[name] = analyze_model(
            name=name,
            checkpoint=checkpoint,
            lifespan_summary_path=lifespan_summary,
            out_dir=args.out_dir,
            device=device,
            calibration_batch_size=args.calibration_batch_size,
            calibration_batches=args.calibration_batches,
            validation_size=args.validation_size,
            discovery_size=args.discovery_size,
            evaluation_size=args.evaluation_size,
            lifespan_batch_size=args.lifespan_batch_size,
            lifespan_batches=args.lifespan_batches,
            extra_loops=args.extra_loops,
            ridge_values=args.ridge_values,
            ranks=args.ranks,
            max_age_rank=args.max_age_rank,
            random_circuit_subsets=args.random_circuit_subsets,
            seed=args.seed + 1000 * run_index,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
