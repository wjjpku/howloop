from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_robust_j import WeightedAffineStats
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_task_mlp_j import (
    evaluate_random_unit_every,
)
from reasoning_loop.graph_path_telomere_unit_j import _add_selected_pair
from reasoning_loop.graph_path_telomere_unit_j import exact_interfaces


class IdentityJ(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _curve_summary(values: list[float]) -> dict[str, Any]:
    def segment(start: int, stop: int) -> float | None:
        selected = values[start:stop]
        return sum(selected) / len(selected) if selected else None

    return {
        "accuracy_by_cycle": values,
        "auc_1_24": segment(0, 24),
        "auc_25_48": segment(24, 48),
        "auc_49_64": segment(48, 64),
    }


def last_active_age(phase_positions: Sequence[int]) -> int:
    """Return the final boundary age whose next loop still advances the task.

    A one-step trajectory ``0,1,...,8`` returns 7.  A two-step trajectory
    ``0,2,4,6,8,8,...`` returns 3.  This prevents a held H7 state from being
    mislabeled as a young execution interface on compressed schedules.
    """

    if len(phase_positions) < 4:
        raise ValueError("phase trajectory must contain at least four ages")
    jump = int(phase_positions[3]) - int(phase_positions[2])
    if jump <= 0:
        raise ValueError("phase trajectory must define a positive task step")
    active = [
        age
        for age in range(2, len(phase_positions) - 1)
        if int(phase_positions[age + 1]) - int(phase_positions[age]) == jump
    ]
    if not active:
        raise ValueError("phase trajectory contains no active transition")
    return max(active)


@torch.no_grad()
def collect_boundary_adjacent_pairs(
    *,
    stats: WeightedAffineStats,
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    answer_weight: int,
    identity_weight: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    placement: str,
    active_age: int,
) -> None:
    set_seed(seed)
    answer_relative_position = positions.index(cfg.seq_len - 1)
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        current = path_targets[:, cfg.max_depth - 1]
        if placement == "loop_boundary":
            interfaces = {
                age: _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=age,
                    phase_position=phase_positions[age],
                )[:, list(positions)]
                for age in range(2, 9)
            }
        elif placement == "pre_block2":
            interfaces = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=successors,
                current=current,
                ages=tuple(range(2, 9)),
                loop_index=cfg.max_loops,
            )
        else:
            raise ValueError(f"unsupported placement: {placement}")
        if identity_weight > 0:
            _add_selected_pair(
                stats,
                source=interfaces[2],
                target=interfaces[2],
                answer_relative_position=answer_relative_position,
                answer_weight=answer_weight * identity_weight,
                group=f"{placement}_H2_identity",
            )
        for age in range(3, 9):
            target_age = min(age - 1, active_age)
            _add_selected_pair(
                stats,
                source=interfaces[age],
                target=interfaces[target_age],
                answer_relative_position=answer_relative_position,
                answer_weight=answer_weight,
                group=f"{placement}_H{age}_to_H{target_age}",
            )


def _centered_moments(
    stats: WeightedAffineStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = stats.sum_weight
    mean_source = stats.sum_source / weight
    mean_target = stats.sum_target / weight
    gram = stats.sum_source_source - torch.outer(
        stats.sum_source, stats.sum_source
    ) / weight
    source_target = stats.sum_source_source + stats.sum_source_update
    cross = source_target - torch.outer(
        stats.sum_source, stats.sum_target
    ) / weight
    target_gram = stats.sum_target_target - torch.outer(
        stats.sum_target, stats.sum_target
    ) / weight
    return mean_source, mean_target, gram, cross, target_gram


def explicit_diagonal_reduced_rank_fit(
    stats: WeightedAffineStats,
    *,
    rank: int,
    ridge: float,
    iterations: int,
    diagonal_init: str,
    device: torch.device,
) -> tuple[VectorAffine, dict[str, Any]]:
    """Alternating exact least-squares updates for W=diag(d)+L, rank(L)<=r."""

    mean_x, mean_y, gram, cross, target_gram = _centered_moments(stats)
    dimension = stats.dimension
    identity = torch.eye(dimension, dtype=torch.float64)
    scale = gram.diagonal().mean().clamp_min(1e-12)
    penalty = ridge * scale
    system = gram + penalty * identity
    dense = stats.fit(ridge=ridge, device=torch.device("cpu"))
    if diagonal_init == "dense_diagonal":
        diagonal = torch.diagonal(dense.weight.double()).clone()
    elif diagonal_init == "one":
        diagonal = torch.ones(dimension, dtype=torch.float64)
    else:
        raise ValueError(f"unknown diagonal_init: {diagonal_init}")
    low_rank = torch.zeros_like(gram)
    history: list[dict[str, float]] = []
    target_variance = torch.trace(target_gram).clamp_min(1e-12)

    for iteration in range(1, iterations + 1):
        previous_weight = torch.diag(diagonal) + low_rank
        residual_cross = cross - gram * diagonal.unsqueeze(0)
        unconstrained = torch.linalg.solve(system, residual_cross)
        fitted_covariance = residual_cross.T @ unconstrained
        fitted_covariance = 0.5 * (
            fitted_covariance + fitted_covariance.T
        )
        _, eigenvectors = torch.linalg.eigh(fitted_covariance)
        output_basis = eigenvectors[:, -rank:]
        low_rank = (
            unconstrained
            @ output_basis
            @ output_basis.transpose(0, 1)
        )
        diagonal = (
            torch.diagonal(cross)
            - torch.diagonal(gram @ low_rank)
            + penalty
        ) / (torch.diagonal(gram) + penalty)
        weight = torch.diag(diagonal) + low_rank
        delta = weight - previous_weight
        prediction_sse = (
            torch.trace(target_gram)
            - 2.0 * torch.trace(weight.T @ cross)
            + torch.trace(weight.T @ gram @ weight)
        )
        penalized_objective = prediction_sse + penalty * (
            low_rank.square().sum() + (diagonal - 1.0).square().sum()
        )
        history.append(
            {
                "iteration": float(iteration),
                "relative_weight_change": float(
                    delta.norm() / weight.norm().clamp_min(1e-12)
                ),
                "relative_state_mse": float(
                    prediction_sse / target_variance
                ),
                "penalized_objective_per_target_variance": float(
                    penalized_objective / target_variance
                ),
            }
        )

    weight = torch.diag(diagonal) + low_rank
    bias = mean_y - mean_x @ weight
    correction_singular = torch.linalg.svdvals(low_rank)
    threshold = max(float(correction_singular[0]) * 1e-8, 1e-10)
    fitted = VectorAffine(
        weight=weight.to(device=device, dtype=torch.float32),
        bias=bias.to(device=device, dtype=torch.float32),
        update_rank=rank,
        fit_dimension=dimension,
        retained_fit_energy=1.0,
    )
    diagnostics = {
        "rank": rank,
        "iterations": iterations,
        "ridge": ridge,
        "diagonal_init": diagonal_init,
        "parameter_count": 2 * dimension * rank + 2 * dimension,
        "numerical_correction_rank": int(
            correction_singular.gt(threshold).sum()
        ),
        "correction_spectral_norm": float(correction_singular[0]),
        "diagonal_mean": float(diagonal.mean()),
        "diagonal_std": float(diagonal.std(unbiased=False)),
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "history": history,
    }
    return fitted, diagnostics


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.graphs <= 0 or args.graphs % args.batch_size:
        raise ValueError("graphs must be positive and divisible by batch_size")
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
        or cfg.d_model != 256
    ):
        raise ValueError("experiment requires the frozen D8L8 N8 d256 model")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    task_step_size = phase_positions[3] - phase_positions[2]
    active_age = last_active_age(phase_positions)
    positions = intervention_groups(cfg.node_count)["all"]
    stats = WeightedAffineStats(cfg.d_model)
    collect_boundary_adjacent_pairs(
        stats=stats,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        answer_weight=args.answer_weight,
        identity_weight=args.identity_weight,
        device=device,
        batch_size=args.batch_size,
        batches=args.graphs // args.batch_size,
        seed=args.data_seed,
        placement=args.placement,
        active_age=active_age,
    )
    dense = stats.fit(ridge=args.ridge, device=device)
    maps: dict[str, Any] = {
        "identity_no_J": IdentityJ().to(device).eval(),
        "dense_boundary_ridge": dense,
    }
    diagnostics: dict[str, Any] = {}
    for rank in args.ranks:
        label = f"explicit_diag_rrr_r{rank}"
        fitted, item = explicit_diagonal_reduced_rank_fit(
            stats,
            rank=rank,
            ridge=args.ridge,
            iterations=args.iterations,
            diagonal_init=args.diagonal_init,
            device=device,
        )
        maps[label] = fitted
        diagnostics[label] = item

    rows = evaluate_random_unit_every(
        maps=maps,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.evaluation_batch_size,
        batches=args.evaluation_batches,
        continuation_loops=args.evaluation_loops,
        seed=args.evaluation_seed,
        placement=args.placement,
    )
    curves = {
        label: _curve_summary(
            [float(row["accuracy"]) for row in rows if row["variant"] == label]
        )
        for label in maps
    }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "random_graph_closed_loop.csv", rows)
    map_payload = {
        label: {
            "weight": operator.weight.detach().cpu(),
            "bias": operator.bias.detach().cpu(),
            "rank": operator.update_rank,
            "parameterization": (
                "full" if label == "dense_boundary_ridge" else "diagonal_low_rank"
            ),
        }
        for label, operator in maps.items()
        if isinstance(operator, VectorAffine)
    }
    artifact_tmp = out_dir / "unit_j_maps.pt.tmp"
    artifact = out_dir / "unit_j_maps.pt"
    torch.save(
        {
            "kind": "graph_path_telomere_unit_j",
            "checkpoint": str(args.checkpoint),
            "positions": positions,
            "placement": args.placement,
            "pairing": "phase_aware_rollback_to_last_active_age",
            "last_active_age": active_age,
            "maps": map_payload,
        },
        artifact_tmp,
    )
    os.replace(artifact_tmp, artifact)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": (
            f"frozen backbone: {args.backbone_loss_description}; explicit J uses only "
            "weighted hidden-state least squares, no CE and no gradient descent"
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller_placement": args.placement,
        "task_step_size": task_step_size,
        "last_active_age": active_age,
        "pairing": "H_age -> H_min(age-1,last_active_age), plus H2 identity",
        "regression": "alternating closed-form diagonal + reduced-rank ridge",
        "raw_vector_rows": stats.raw_rows,
        "effective_weighted_rows": stats.effective_rows,
        "group_effective_rows": dict(stats.group_effective_rows),
        "graphs": args.graphs,
        "batch_size": args.batch_size,
        "data_seed": args.data_seed,
        "answer_weight": args.answer_weight,
        "identity_weight": args.identity_weight,
        "ridge": args.ridge,
        "ranks": list(args.ranks),
        "diagnostics": diagnostics,
        "random_graph_evaluation": {
            "examples": args.evaluation_batch_size * args.evaluation_batches,
            "loops": args.evaluation_loops,
            "seed": args.evaluation_seed,
            "curves": curves,
        },
        "files": {
            "artifact": "unit_j_maps.pt",
            "evaluation": "random_graph_closed_loop.csv",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicit diagonal plus reduced-rank boundary-J regression."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone-loss-description", default="final-only CE at loop 8"
    )
    parser.add_argument(
        "--placement",
        choices=("loop_boundary", "pre_block2"),
        default="loop_boundary",
    )
    parser.add_argument("--graphs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=20260731)
    parser.add_argument("--answer-weight", type=int, default=28)
    parser.add_argument("--identity-weight", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--diagonal-init",
        choices=("dense_diagonal", "one"),
        default="dense_diagonal",
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=(8, 16, 32, 48, 64, 96, 128)
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--evaluation-loops", type=int, default=64)
    parser.add_argument("--evaluation-seed", type=int, default=212004)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_experiment(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
