"""Measure accuracy distributions for zero through five J calls/compositions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_composition_laws import (
    forward_steps,
    load_bank,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeTrajectory,
    _run_mixed_trajectory,
    _trajectory_text,
    sample_bounded_bridge,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trajectories", type=int, default=112)
    parser.add_argument("--roundtrip-batches", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def no_j_trajectory(rng: np.random.Generator) -> AgeTrajectory:
    candidates = [(start, end) for start in range(1, 9) for end in range(start, 9)]
    start, end = candidates[int(rng.integers(0, len(candidates)))]
    actions = (1,) * (end - start)
    ages = tuple(range(start, end))
    return AgeTrajectory(
        start_age=start,
        end_age=end,
        extra_backs=0,
        actions=actions,
        ages=ages,
        mandatory_rollback_source=None,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_mixed_exact_counts(
    *,
    model,
    cfg,
    bank,
    phase_positions: list[int],
    device: torch.device,
    trajectories: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = tuple(range(cfg.seq_len))
    for j_count in range(6):
        # Reset both model-data RNG and path RNG so graph batches are matched
        # across J counts; path constraints alone vary.
        set_seed(seed)
        rng = np.random.default_rng(seed)
        for trajectory_index in range(trajectories):
            trajectory = (
                no_j_trajectory(rng)
                if j_count == 0
                else sample_bounded_bridge(
                    rng=rng,
                    max_total_backs=j_count,
                    minimum_total_backs=j_count,
                    max_consecutive_backs=j_count,
                    required_consecutive_backs=0,
                    mandatory_rollback_source=None,
                )
            )
            _, path_targets, successors, _ = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            endpoint = path_targets[:, cfg.max_depth - 1]
            initial_state = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=trajectory.start_age,
                phase_position=phase_positions[trajectory.start_age],
            )
            _, logits, target = _run_mixed_trajectory(
                model=model,
                cfg=cfg,
                bank=bank,
                positions=positions,
                successors=successors,
                endpoint=endpoint,
                initial_state=initial_state,
                trajectory=trajectory,
                condition="learned",
                phase_positions=phase_positions,
                rollback_composition="product",
            )
            rows.append(
                {
                    "evaluation": "random_mixed_exact_count",
                    "j_count": j_count,
                    "trajectory": trajectory_index,
                    "accuracy": float(
                        logits.argmax(dim=-1).eq(target).float().mean()
                    ),
                    "start_age": trajectory.start_age,
                    "end_age": trajectory.end_age,
                    "forward_count": trajectory.forward_count,
                    "back_count": trajectory.back_count,
                    "max_consecutive_j": trajectory.max_rollback_run,
                    "rollback_sources": ",".join(
                        map(str, trajectory.rollback_sources)
                    ),
                    "age_path": _trajectory_text(trajectory),
                    "examples": batch_size,
                }
            )
    return rows


@torch.no_grad()
def evaluate_controlled_roundtrips(
    *,
    model,
    cfg,
    bank,
    phase_positions: list[int],
    device: torch.device,
    batches: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positions = tuple(range(cfg.seq_len))
    jump = phase_positions[2] - phase_positions[1]
    for j_count in range(6):
        set_seed(seed)
        for batch_index in range(batches):
            _, path_targets, successors, _ = fixed_depth_batch(
                cfg, batch_size, device, path_positions=cfg.max_depth
            )
            current = path_targets[:, cfg.max_depth - 1]
            state = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=8,
                phase_position=phase_positions[8],
            )
            source_ages = list(range(8, 8 - j_count, -1))
            if source_ages:
                state = bank.rollback_composed(
                    state,
                    source_ages=source_ages,
                    positions=positions,
                    composition="product",
                )
                state = forward_steps(
                    model,
                    state,
                    start_age=8 - j_count,
                    steps=j_count,
                )
            target = advance_nodes(
                successors, current, steps=j_count * jump
            )
            logits = logits_from_raw_state(model, state)
            rows.append(
                {
                    "evaluation": "controlled_H8_roundtrip",
                    "j_count": j_count,
                    "trajectory": batch_index,
                    "accuracy": float(
                        logits.argmax(dim=-1).eq(target).float().mean()
                    ),
                    "start_age": 8,
                    "end_age": 8,
                    "forward_count": j_count,
                    "back_count": j_count,
                    "max_consecutive_j": j_count,
                    "rollback_sources": ",".join(map(str, source_ages)),
                    "age_path": ",".join(
                        map(
                            str,
                            [8]
                            + list(range(7, 7 - j_count, -1))
                            + list(range(9 - j_count, 9)),
                        )
                    ),
                    "examples": batch_size,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for evaluation in sorted({row["evaluation"] for row in rows}):
        for j_count in range(6):
            values = np.array(
                [
                    float(row["accuracy"])
                    for row in rows
                    if row["evaluation"] == evaluation
                    and int(row["j_count"]) == j_count
                ]
            )
            result.append(
                {
                    "evaluation": evaluation,
                    "j_count": j_count,
                    "trajectories_or_batches": len(values),
                    "examples_per_point": int(
                        next(
                            row["examples"]
                            for row in rows
                            if row["evaluation"] == evaluation
                            and int(row["j_count"]) == j_count
                        )
                    ),
                    "accuracy_mean": float(values.mean()),
                    "accuracy_std": float(values.std(ddof=1)),
                    "accuracy_min": float(values.min()),
                    "accuracy_p10": float(np.quantile(values, 0.10)),
                    "accuracy_p25": float(np.quantile(values, 0.25)),
                    "accuracy_median": float(np.median(values)),
                    "accuracy_p75": float(np.quantile(values, 0.75)),
                    "accuracy_p90": float(np.quantile(values, 0.90)),
                    "accuracy_max": float(values.max()),
                }
            )
    return result


def make_plot(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    configurations = (
        (
            "random_mixed_exact_count",
            "Random mixed F/J paths\n(exact total number of J calls)",
        ),
        (
            "controlled_H8_roundtrip",
            "Controlled H8: consecutive J product, then same number of F loops",
        ),
    )
    rng = np.random.default_rng(831777)
    for axis, (evaluation, title) in zip(axes, configurations, strict=True):
        groups = [
            [
                float(row["accuracy"])
                for row in rows
                if row["evaluation"] == evaluation and int(row["j_count"]) == count
            ]
            for count in range(6)
        ]
        axis.boxplot(
            groups,
            positions=range(6),
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#b8d8eb", "alpha": 0.75},
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for count, values in enumerate(groups):
            jitter = rng.normal(0, 0.055, size=len(values))
            axis.scatter(
                count + jitter,
                values,
                s=10,
                alpha=0.28,
                color="#1f77b4",
                linewidths=0,
            )
            mean = float(np.mean(values))
            axis.scatter(count, mean, marker="D", s=50, color="#d62728", zorder=5)
            axis.text(
                count,
                min(1.015, mean + 0.025),
                f"{100 * mean:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#8b0000",
            )
        axis.set_title(title)
        axis.set_xlabel("number of J calls")
        axis.set_ylabel("accuracy per 64-graph point")
        axis.set_xticks(range(6), labels=range(6))
        axis.set_ylim(0.55, 1.035)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "D8L8 seed0, product-J r64/s24: accuracy distribution by composition count",
        fontsize=15,
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    bank, bank_payload = load_bank(
        args.bank_artifact, dimension=cfg.d_model, device=device
    )
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase["trajectory_positions_including_initial"]
    ]
    mixed_rows = evaluate_mixed_exact_counts(
        model=model,
        cfg=cfg,
        bank=bank,
        phase_positions=phase_positions,
        device=device,
        trajectories=args.trajectories,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    roundtrip_rows = evaluate_controlled_roundtrips(
        model=model,
        cfg=cfg,
        bank=bank,
        phase_positions=phase_positions,
        device=device,
        batches=args.roundtrip_batches,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    rows = mixed_rows + roundtrip_rows
    summary_rows = summarize(rows)
    write_csv(args.out_dir / "composition_count_points.csv", rows)
    write_csv(args.out_dir / "composition_count_summary.csv", summary_rows)
    make_plot(rows, args.out_dir / "composition_count_distributions.png")
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_rank": bank_payload.get("rank"),
        "bank_stage_rank": bank_payload.get("stage_rank"),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "rollback_composition": bank_payload.get("rollback_composition") or "product",
        "random_trajectory_points_per_count": args.trajectories,
        "controlled_roundtrip_points_per_count": args.roundtrip_batches,
        "examples_per_point": args.batch_size,
        "summary": summary_rows,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
