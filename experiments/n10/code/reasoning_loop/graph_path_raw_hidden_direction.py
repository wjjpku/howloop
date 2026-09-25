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

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import explicit_depth_position_groups
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    cycle_type,
    reconstruct_primary_training_graphs,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_task_lora_j import load_task_lora_modules
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sample_eight_cycles(
    *, seen: set[tuple[int, ...]], count: int, seed: int
) -> list[tuple[int, ...]]:
    candidates = [
        item
        for item in itertools.permutations(range(8))
        if item not in seen and cycle_type(item) == (8,)
    ]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(
            f"requested {count} unseen 8-cycles but only {len(candidates)} remain"
        )
    return candidates[:count]


def _pca(values: np.ndarray) -> dict[str, np.ndarray | float]:
    # Remove each graph/start trajectory's first sampled state before fitting.
    delta = values.astype(np.float64) - values[:, :1].astype(np.float64)
    flat = delta.reshape(-1, delta.shape[-1])
    center = flat.mean(axis=0, keepdims=True)
    centered = flat - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular)
    explained = variance / max(float(variance.sum()), 1e-30)
    scores = ((flat - center) @ vt[:2].T).reshape(values.shape[0], values.shape[1], 2)
    return {
        "scores": scores,
        "components": vt[:16],
        "explained": explained[:16],
        "center": center[0],
    }


def _pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    unit = vectors / np.maximum(norm, 1e-12)
    return unit @ unit.T


def _step_metrics(values: np.ndarray, *, first_loop: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    steps = np.diff(values.astype(np.float64), axis=1)
    mean_steps = steps.mean(axis=0)
    pairwise = _pairwise_cosine(mean_steps)
    mean_norm = np.linalg.norm(mean_steps, axis=-1)
    rms_norm = np.sqrt(np.square(steps).sum(axis=-1).mean(axis=0))
    coherence = np.square(mean_norm) / np.maximum(np.square(rms_norm), 1e-24)
    unit = steps / np.maximum(np.linalg.norm(steps, axis=-1, keepdims=True), 1e-12)
    adjacent = (unit[:, 1:] * unit[:, :-1]).sum(axis=-1).mean(axis=0)
    rows: list[dict[str, Any]] = []
    for index in range(steps.shape[1]):
        rows.append(
            {
                "from_loop": first_loop + index,
                "to_loop": first_loop + index + 1,
                "mean_step_norm": float(mean_norm[index]),
                "rms_per_example_step_norm": float(rms_norm[index]),
                "shared_direction_energy_fraction": float(coherence[index]),
                "mean_sample_adjacent_step_cosine": (
                    float(adjacent[index]) if index < len(adjacent) else None
                ),
            }
        )
    return rows, pairwise


def _linear_direction_diagnostic(values: np.ndarray) -> dict[str, float]:
    mean = values.astype(np.float64).mean(axis=0)
    displacement = mean - mean[:1]
    final = displacement[-1]
    final_norm = np.linalg.norm(final)
    if final_norm <= 1e-12:
        return {
            "linear_coordinate_slope": 0.0,
            "linear_coordinate_r2": 0.0,
            "mean_orthogonal_fraction": 0.0,
        }
    direction = final / final_norm
    coordinate = displacement @ direction
    index = np.arange(len(coordinate), dtype=np.float64)
    design = np.stack([index, np.ones_like(index)], axis=1)
    coefficient, *_ = np.linalg.lstsq(design, coordinate, rcond=None)
    fitted = design @ coefficient
    residual = np.square(coordinate - fitted).sum()
    total = np.square(coordinate - coordinate.mean()).sum()
    orthogonal = displacement - coordinate[:, None] * direction[None]
    orthogonal_fraction = np.linalg.norm(orthogonal, axis=-1) / np.maximum(
        np.linalg.norm(displacement, axis=-1), 1e-12
    )
    return {
        "linear_coordinate_slope": float(coefficient[0]),
        "linear_coordinate_r2": float(1.0 - residual / max(float(total), 1e-24)),
        "mean_orthogonal_fraction": float(orthogonal_fraction[1:].mean()),
    }


def _plot_trajectories(
    *, out_dir: Path, loops: np.ndarray, states: dict[str, np.ndarray]
) -> dict[str, str]:
    detail_mask = loops <= min(16, int(loops.max()))
    periodic_mask = loops % 8 == 0
    display = {
        "answer_position": "Answer token (p28)",
        "start_register_position": "Start register (p26)",
        "graph_destination_node0_position": "Graph destination slot (p3)",
    }
    fig, axes = plt.subplots(len(states), 2, figsize=(13, 11), constrained_layout=True)
    for row, (role, values) in enumerate(states.items()):
        for column, (mask, title) in enumerate(
            ((detail_mask, "successive raw loops"), (periodic_mask, "same-content 8-cycle snapshots"))
        ):
            selected_loops = loops[mask]
            selected = values[:, mask]
            payload = _pca(selected)
            scores = payload["scores"]
            mean = scores.mean(axis=0)
            axis = axes[row, column]
            axis.plot(mean[:, 0], mean[:, 1], color="#165DFF", linewidth=2.2)
            axis.scatter(mean[:, 0], mean[:, 1], c=selected_loops, cmap="viridis", s=42, edgecolor="black", linewidth=0.25)
            annotated = (
                set(int(value) for value in selected_loops)
                if column == 1
                else {1, 2, 4, 6, 8, 9, 12, 16}
            )
            for index, loop in enumerate(selected_loops):
                if int(loop) in annotated:
                    axis.annotate(f"H{int(loop)}", mean[index], xytext=(4, 4), textcoords="offset points", fontsize=8)
                if index + 1 < len(mean):
                    axis.annotate("", xy=mean[index + 1], xytext=mean[index], arrowprops={"arrowstyle": "->", "color": "#165DFF", "lw": 1.1})
            explained = payload["explained"]
            axis.set_title(f"{display[role]}: {title}\npopulation mean; PC1+PC2={float(explained[:2].sum()):.3f}")
            axis.set_xlabel("content-centered PC1")
            axis.set_ylabel("content-centered PC2")
            axis.grid(alpha=0.18)
            axis.margins(0.15)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"raw_same_position_direction_trajectory.{suffix}", dpi=200 if suffix == "png" else None)
    plt.close(fig)
    return {
        "trajectory_png": "raw_same_position_direction_trajectory.png",
        "trajectory_pdf": "raw_same_position_direction_trajectory.pdf",
    }


def _plot_direction_metrics(
    *, out_dir: Path, loops: np.ndarray, states: dict[str, np.ndarray]
) -> dict[str, str]:
    detail_mask = loops <= min(16, int(loops.max()))
    periodic_mask = loops % 8 == 0
    display = {
        "answer_position": "Answer p28",
        "start_register_position": "Start register p26",
        "graph_destination_node0_position": "Graph destination p3",
    }
    fig, axes = plt.subplots(len(states), 2, figsize=(15, 11), constrained_layout=True)
    for row, (role, values) in enumerate(states.items()):
        _, detail_cos = _step_metrics(values[:, detail_mask], first_loop=int(loops[detail_mask][0]))
        periodic_values = values[:, periodic_mask]
        periodic_delta = np.diff(periodic_values, axis=1)
        periodic_cos = _pairwise_cosine(periodic_delta.mean(axis=0))
        for column, (matrix, labels, title) in enumerate(
            (
                (detail_cos, [f"{a}->{a+1}" for a in loops[detail_mask][:-1]], "successive mean-step directions"),
                (periodic_cos, [f"{a}->{a+8}" for a in loops[periodic_mask][:-1]], "same-content 8-loop drift directions"),
            )
        ):
            axis = axes[row, column]
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            axis.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=7)
            axis.set_yticks(range(len(labels)), labels=labels, fontsize=7)
            axis.set_title(f"{display[role]}: {title}")
    fig.colorbar(image, ax=axes, shrink=0.72, label="cosine in original 256-D space")
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"raw_highdim_direction_cosines.{suffix}", dpi=200 if suffix == "png" else None)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for role, values in states.items():
        rows, _ = _step_metrics(values, first_loop=int(loops[0]))
        axes[0].plot([row["to_loop"] for row in rows], [row["rms_per_example_step_norm"] for row in rows], label=role)
        axes[1].plot([row["to_loop"] for row in rows], [row["shared_direction_energy_fraction"] for row in rows], label=role)
    axes[0].set_ylabel("RMS movement norm")
    axes[1].set_ylabel("shared-direction energy fraction")
    axes[1].set_xlabel("destination loop")
    for axis in axes:
        axis.axvline(8, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.18)
        axis.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"raw_movement_norm_and_coherence.{suffix}", dpi=200 if suffix == "png" else None)
    plt.close(fig)
    return {
        "cosine_png": "raw_highdim_direction_cosines.png",
        "cosine_pdf": "raw_highdim_direction_cosines.pdf",
        "movement_png": "raw_movement_norm_and_coherence.png",
        "movement_pdf": "raw_movement_norm_and_coherence.pdf",
    }


@torch.no_grad()
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    if device.type == "cuda": torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=torch.cuda.current_device())
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    if (cfg.node_count, cfg.max_depth, cfg.max_loops, cfg.n_layers, cfg.d_model) != (8, 8, 8, 2, 256):
        raise ValueError("experiment requires the D8L8 N8 d256 backbone")
    model.eval()

    seen: set[tuple[int, ...]] = set()
    streams = None
    unique_after_stage: dict[str, int] = {}
    training_draws = 0
    if args.training_artifact is not None:
        artifact_checkpoint, _, _, payload = load_task_lora_modules(args.training_artifact, device=device)
        if artifact_checkpoint != str(args.checkpoint):
            raise ValueError("training artifact and backbone checkpoints differ")
        streams = _canonical_training_streams(payload)
        seen, unique_after_stage, training_draws = reconstruct_primary_training_graphs(device=device, node_count=8, streams=streams)
    sampled = _sample_eight_cycles(seen=seen, count=args.permutations, seed=args.sample_seed)
    successors, starts = _expand_all_starts(sampled, device=device)
    tokens, _, _, _ = fixed_depth_batch(cfg, len(successors), device, path_positions=cfg.max_depth, successors=successors, start=starts)
    groups = explicit_depth_position_groups(cfg.node_count)
    positions = {
        "answer_position": groups["answer"][0],
        "start_register_position": groups["start"][0],
        "graph_destination_node0_position": groups["destination"][0],
    }
    state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    captured = {label: np.empty((len(successors), args.loops, cfg.d_model), dtype=np.float32) for label in positions}
    accuracy: list[float] = []
    for loop_index in range(args.loops):
        state = apply_shared_stack(model, state, loop_index=loop_index)
        prediction = logits_from_raw_state(model, state).argmax(dim=-1)
        target = advance_nodes(successors, starts, steps=loop_index + 1)
        accuracy.append(float(prediction.eq(target).float().mean()))
        for label, position in positions.items():
            captured[label][:, loop_index] = state[:, position].float().cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loops = np.arange(1, args.loops + 1, dtype=np.int64)
    np.savez_compressed(args.out_dir / "raw_same_position_states.npz", loops=loops, successors=successors.cpu().numpy(), starts=starts.cpu().numpy(), accuracy=np.asarray(accuracy), **captured)
    rows: list[dict[str, Any]] = []
    periodic = loops % 8 == 0
    role_summary: dict[str, Any] = {}
    for role, values in captured.items():
        step_rows, _ = _step_metrics(values, first_loop=1)
        for row in step_rows:
            row["role"] = role
        rows.extend(step_rows)
        detail_pca = _pca(values[:, loops <= min(16, args.loops)])
        periodic_pca = _pca(values[:, periodic])
        periodic_values = values[:, periodic]
        periodic_steps = np.diff(periodic_values.astype(np.float64), axis=1)
        periodic_mean = periodic_steps.mean(axis=0)
        periodic_pairwise = _pairwise_cosine(periodic_mean)
        off_diagonal = periodic_pairwise[np.triu_indices_from(periodic_pairwise, k=1)]
        off_diagonal_mean = float(off_diagonal.mean()) if len(off_diagonal) else 1.0
        off_diagonal_min = float(off_diagonal.min()) if len(off_diagonal) else 1.0
        role_summary[role] = {
            "position": positions[role],
            "detail_pc1_fraction": float(detail_pca["explained"][0]),
            "detail_pc1_pc2_fraction": float(np.asarray(detail_pca["explained"])[:2].sum()),
            "matched_pc1_fraction": float(periodic_pca["explained"][0]),
            "matched_pc1_pc2_fraction": float(np.asarray(periodic_pca["explained"])[:2].sum()),
            "matched_drift_pairwise_cosine_mean": off_diagonal_mean,
            "matched_drift_pairwise_cosine_min": off_diagonal_min,
            "H8_to_H64_distance_mean": float(np.linalg.norm(periodic_values[:, -1] - periodic_values[:, 0], axis=-1).mean()),
            **_linear_direction_diagnostic(periodic_values),
        }
    _write_csv(args.out_dir / "raw_step_direction_metrics.csv", rows)
    figures = _plot_trajectories(out_dir=args.out_dir, loops=loops, states=captured)
    figures.update(_plot_direction_metrics(out_dir=args.out_dir, loops=loops, states=captured))
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "condition": "raw frozen backbone; no J is applied at any loop",
        "model": {"node_count": 8, "max_depth": 8, "trained_loops": 8, "shared_blocks": 2, "d_model": 256},
        "sampling": {"permutations": args.permutations, "cycle_type": [8], "all_8_starts": True, "examples": len(successors), "seed": args.sample_seed, "excluded_J_training_graphs": args.training_artifact is not None, "training_graph_draws": training_draws, "unique_training_graphs": len(seen), "unique_after_stage": unique_after_stage, "training_streams": streams},
        "loops": args.loops,
        "accuracy_by_loop": accuracy,
        "accuracy_at": {str(loop): accuracy[loop - 1] for loop in (1, 2, 4, 8, 16, 24, 32, 48, 64) if loop <= args.loops},
        "roles": role_summary,
        "figures": figures,
        "files": {"states": "raw_same_position_states.npz", "step_metrics": "raw_step_direction_metrics.csv"},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw D8L8 same-position hidden-state direction trajectories.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--training-artifact", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loops", type=int, default=64)
    parser.add_argument("--permutations", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=20260806)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run_experiment(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
