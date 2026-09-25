from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import run_instrumented_state
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    cycle_type,
    reconstruct_primary_training_graphs,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import load_task_lora_modules
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
)
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


STATE_SITES = (
    "boundary_pre_J",
    "boundary_post_J",
    "post_Block1",
    "post_Block2",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _apply_operator(
    state: torch.Tensor,
    *,
    positions: tuple[int, ...],
    operator,
) -> torch.Tensor:
    result = state.clone()
    index = list(positions)
    result[:, index] = operator(result[:, index]).to(dtype=result.dtype)
    return result


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(target).float().mean())


def _relative_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = (left.float() - right.float()).square().mean().sqrt()
    scale = 0.5 * (
        left.float().square().mean().sqrt()
        + right.float().square().mean().sqrt()
    )
    return float(difference / scale.clamp_min(1e-12))


def _relative_mse(left: torch.Tensor, right: torch.Tensor) -> float:
    variance = (
        right.float() - right.float().mean(dim=0, keepdim=True)
    ).square().mean().clamp_min(1e-12)
    return float((left.float() - right.float()).square().mean() / variance)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float(), right.float(), dim=-1).mean()
    )


def _spectral_stats(values: np.ndarray) -> dict[str, float]:
    centered = values.astype(np.float64) - values.astype(np.float64).mean(
        axis=0,
        keepdims=True,
    )
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = np.square(singular) / max(1, len(values) - 1)
    trace = float(eigenvalues.sum())
    if trace <= 1e-30:
        return {
            "spectral_entropy": float("nan"),
            "effective_rank": 0.0,
            "pc1_fraction": float("nan"),
            "covariance_trace": 0.0,
        }
    probability = eigenvalues / trace
    positive = probability > 0
    entropy = float(
        -(probability[positive] * np.log(probability[positive])).sum()
    )
    return {
        "spectral_entropy": entropy / np.log(values.shape[-1]),
        "effective_rank": float(np.exp(entropy)),
        "pc1_fraction": float(probability[0]),
        "covariance_trace": trace,
    }


def _sample_unseen_eight_cycles(
    *,
    seen: set[tuple[int, ...]],
    count: int,
    seed: int,
) -> list[tuple[int, ...]]:
    candidates = [
        item
        for item in itertools.permutations(range(8))
        if item not in seen and cycle_type(item) == (8,)
    ]
    if count > len(candidates):
        raise ValueError("not enough strictly unseen 8-cycles")
    random.Random(seed).shuffle(candidates)
    return candidates[:count]


def _plot_dynamics(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, str]:
    pooled = [row for row in rows if row["replica"] == "pooled"]
    cycles = sorted({int(row["cycle"]) for row in pooled})
    figures: dict[str, str] = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (
        ("accuracy", "successor accuracy"),
        ("l2_norm_mean", "answer-state L2 norm"),
        ("cosine_to_exact_H7", "cosine to exact H7"),
        ("relative_mse_to_exact_H7", "relative MSE to exact H7"),
    )
    for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
        for site in STATE_SITES:
            selected = sorted(
                (row for row in pooled if row["site"] == site),
                key=lambda row: int(row["cycle"]),
            )
            axis.plot(
                [int(row["cycle"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                label=site,
            )
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for suffix in ("pdf", "png"):
        path = out_dir / f"boundary_dynamics.{suffix}"
        fig.savefig(path, dpi=190 if suffix == "png" else None)
        figures[f"dynamics_{suffix}"] = path.name
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    spectral_metrics = (
        ("spectral_entropy", "normalized spectral entropy"),
        ("effective_rank", "effective rank"),
        ("pc1_fraction", "PC1 variance fraction"),
        ("covariance_trace", "covariance trace"),
    )
    for axis, (metric, title) in zip(axes.flat, spectral_metrics, strict=True):
        for site in STATE_SITES:
            selected = sorted(
                (row for row in pooled if row["site"] == site),
                key=lambda row: int(row["cycle"]),
            )
            axis.plot(cycles, [float(row[metric]) for row in selected], marker="o", label=site)
        axis.set_title(title)
        axis.grid(alpha=0.2)
        if metric == "covariance_trace":
            axis.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    for suffix in ("pdf", "png"):
        path = out_dir / f"boundary_spectral_dynamics.{suffix}"
        fig.savefig(path, dpi=190 if suffix == "png" else None)
        figures[f"spectral_{suffix}"] = path.name
    plt.close(fig)
    return figures


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    checkpoint, positions, operators, artifact_payload = load_task_lora_modules(
        args.operator_artifact,
        device=device,
    )
    if checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if artifact_payload.get("placement") != "loop_boundary":
        raise ValueError("boundary dynamics requires loop-boundary J")
    if args.operator_label not in operators:
        raise ValueError(f"operator not found: {args.operator_label}")
    operator = operators[args.operator_label]
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    streams = _canonical_training_streams(artifact_payload)
    seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
        device=device,
        node_count=cfg.node_count,
        streams=streams,
    )
    sampled = _sample_unseen_eight_cycles(
        seen=seen,
        count=args.permutations_per_replica * args.replicas,
        seed=args.sample_seed,
    )
    matched_cycles = tuple(
        range(args.matched_period, args.continuation_loops + 1, args.matched_period)
    )
    dynamics_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    all_values: dict[str, list[np.ndarray]] = {site: [] for site in STATE_SITES}

    for replica in range(args.replicas):
        part = sampled[
            replica * args.permutations_per_replica :
            (replica + 1) * args.permutations_per_replica
        ]
        successors, starts = _expand_all_starts(part, device=device)
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        state = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        replica_values = {
            site: np.empty(
                (len(successors), len(matched_cycles), cfg.d_model),
                dtype=np.float32,
            )
            for site in STATE_SITES
        }
        snapshot_index = 0
        for cycle in range(1, args.continuation_loops + 1):
            loop_index = cfg.max_loops + cycle - 1
            current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
            target = advance_nodes(successors, endpoint, steps=jump * cycle)
            post_j = _apply_operator(state, positions=positions, operator=operator)
            logits, trace = run_instrumented_state(
                model,
                post_j,
                loop_indices=(loop_index,),
            )
            next_state = trace.sites[-1].hidden_out

            if cycle in args.commutator_cycles:
                raw_step = run_one_loop(model, state, loop_index=loop_index)
                j_after_f = _apply_operator(
                    raw_step.state,
                    positions=positions,
                    operator=operator,
                )
                commutator_rows.append(
                    {
                        "replica": replica,
                        "cycle": cycle,
                        "examples": len(successors),
                        "relative_rms_F_after_J_vs_J_after_F_all": _relative_rms(
                            next_state,
                            j_after_f,
                        ),
                        "relative_rms_F_after_J_vs_J_after_F_answer": _relative_rms(
                            next_state[:, -1],
                            j_after_f[:, -1],
                        ),
                        "answer_cosine_F_after_J_vs_J_after_F": _cosine(
                            next_state[:, -1],
                            j_after_f[:, -1],
                        ),
                        "accuracy_F_after_J": _accuracy(logits, target),
                        "accuracy_J_after_F_readout": _accuracy(
                            logits_from_raw_state(model, j_after_f),
                            target,
                        ),
                    }
                )

            if cycle in matched_cycles:
                exact = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=current,
                    age=7,
                    phase_position=phase_positions[7],
                )
                values = {
                    "boundary_pre_J": state[:, -1],
                    "boundary_post_J": post_j[:, -1],
                    "post_Block1": trace.sites[0].hidden_out[:, -1],
                    "post_Block2": next_state[:, -1],
                }
                exact_logits, exact_trace = run_instrumented_state(
                    model,
                    exact,
                    loop_indices=(loop_index,),
                )
                exact_values = {
                    "boundary_pre_J": exact[:, -1],
                    "boundary_post_J": exact[:, -1],
                    "post_Block1": exact_trace.sites[0].hidden_out[:, -1],
                    "post_Block2": exact_trace.sites[-1].hidden_out[:, -1],
                }
                accuracy = _accuracy(logits, target)
                exact_accuracy = _accuracy(exact_logits, target)
                for site in STATE_SITES:
                    live = values[site]
                    young = exact_values[site]
                    replica_values[site][:, snapshot_index] = live.float().cpu().numpy()
                    dynamics_rows.append(
                        {
                            "replica": replica,
                            "cycle": cycle,
                            "site": site,
                            "examples": len(successors),
                            "accuracy": accuracy,
                            "exact_H7_accuracy": exact_accuracy,
                            "l2_norm_mean": float(live.float().norm(dim=-1).mean()),
                            "l2_norm_std": float(live.float().norm(dim=-1).std(unbiased=False)),
                            "cosine_to_exact_H7": _cosine(live, young),
                            "relative_mse_to_exact_H7": _relative_mse(live, young),
                        }
                    )
                snapshot_index += 1
            state = next_state

        replica_dir = args.out_dir / f"replica_{replica}"
        replica_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            replica_dir / "boundary_answer_states.npz",
            matched_cycles=np.asarray(matched_cycles, dtype=np.int64),
            starts=starts.cpu().numpy(),
            successors=successors.cpu().numpy(),
            **replica_values,
        )
        for site in STATE_SITES:
            all_values[site].append(replica_values[site])

    for site in STATE_SITES:
        pooled = np.concatenate(all_values[site], axis=0)
        for index, cycle in enumerate(matched_cycles):
            stats = _spectral_stats(pooled[:, index])
            source_rows = [
                row
                for row in dynamics_rows
                if row["site"] == site and int(row["cycle"]) == cycle
            ]
            weights = np.asarray([int(row["examples"]) for row in source_rows])
            row = {
                "replica": "pooled",
                "cycle": cycle,
                "site": site,
                "examples": int(weights.sum()),
                **stats,
            }
            for metric in (
                "accuracy",
                "exact_H7_accuracy",
                "l2_norm_mean",
                "l2_norm_std",
                "cosine_to_exact_H7",
                "relative_mse_to_exact_H7",
            ):
                row[metric] = float(
                    np.average(
                        [float(item[metric]) for item in source_rows],
                        weights=weights,
                    )
                )
            dynamics_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "boundary_dynamics.csv", dynamics_rows)
    _write_csv(args.out_dir / "function_commutator.csv", commutator_rows)
    figures = _plot_dynamics(rows=dynamics_rows, out_dir=args.out_dir)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_description": artifact_payload.get(
            "backbone_loss_description",
            "not recorded",
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "controller_placement": "loop_boundary",
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "strict_unseen_protocol": {
            "training_graph_streams": streams,
            "training_graph_draws": total_draws,
            "unique_training_graphs": len(seen),
            "unique_after_stage": unique_after_stage,
            "cycle_type": [8],
            "permutations_per_replica": args.permutations_per_replica,
            "all_starts": True,
            "replicas": args.replicas,
        },
        "matched_cycles": matched_cycles,
        "commutator_definition": (
            "compare F(J(h)) with J(F(h)) in the same loop-boundary state "
            "space; a small value would support approximate commutation"
        ),
        "commutator": commutator_rows,
        "figures": figures,
        "files": {
            "dynamics": "boundary_dynamics.csv",
            "commutator": "function_commutator.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boundary-native dynamics and F/J commutator audit."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--permutations-per-replica", type=int, default=64)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--matched-period", type=int, default=8)
    parser.add_argument(
        "--commutator-cycles",
        type=int,
        nargs="+",
        default=(1, 32, 64, 96, 128),
    )
    parser.add_argument("--sample-seed", type=int, default=20260731)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    summary = run_experiment(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
