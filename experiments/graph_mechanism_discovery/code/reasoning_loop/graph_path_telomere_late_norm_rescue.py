from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    reconstruct_primary_training_graphs,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_norm_lifespan import (
    _select_disjoint_unseen_eight_cycles,
    _token_normmatched,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import (
    exact_interfaces,
    load_unit_j_map,
)
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


CONDITIONS = (
    "learned_J",
    "postJ_answer_norm_after_onset",
    "postJ_all_token_norm_after_onset",
    "output_answer_norm_after_onset",
    "output_all_token_norm_after_onset",
    "exact_H7",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _answer_normmatched(
    value: torch.Tensor,
    oracle: torch.Tensor,
    *,
    answer_index: int,
) -> torch.Tensor:
    result = value.clone()
    live = result[:, answer_index]
    target = oracle[:, answer_index]
    scale = (
        target.float().norm(dim=-1, keepdim=True)
        / live.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    result[:, answer_index] = live * scale.to(dtype=live.dtype)
    return result


def _run_map_step(
    *,
    condition: str,
    cycle: int,
    onset_cycle: int,
    model,
    state: torch.Tensor,
    loop_index: int,
    positions: tuple[int, ...],
    answer_index: int,
    oracle_interface: torch.Tensor,
    age_map,
):
    if condition == "exact_H7":
        return run_one_loop(
            model,
            state,
            loop_index=loop_index,
            block2_position_override=(positions, oracle_interface),
        )

    def transform(value: torch.Tensor) -> torch.Tensor:
        mapped = age_map(value)
        if cycle < onset_cycle:
            return mapped
        if condition == "postJ_answer_norm_after_onset":
            return _answer_normmatched(
                mapped,
                oracle_interface,
                answer_index=answer_index,
            )
        if condition == "postJ_all_token_norm_after_onset":
            return _token_normmatched(mapped, oracle_interface)
        return mapped

    return run_one_loop(
        model,
        state,
        loop_index=loop_index,
        block2_position_transform=(positions, transform),
    )


def _output_state_and_logits(
    *,
    condition: str,
    cycle: int,
    onset_cycle: int,
    model,
    step,
    exact_output: torch.Tensor,
    answer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = step.state
    if cycle >= onset_cycle:
        if condition == "output_answer_norm_after_onset":
            state = _answer_normmatched(
                state,
                exact_output,
                answer_index=answer_index,
            )
        elif condition == "output_all_token_norm_after_onset":
            state = _token_normmatched(state, exact_output)
    return state, logits_from_raw_state(model, state)


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    j_artifact: Path,
    j_label: str,
    out_dir: Path,
    device_name: str,
    sample_count: int,
    sample_seeds: Sequence[int],
    batch_size: int,
    continuation_loops: int,
    operating_age: int,
    onset_cycle: int,
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.05")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if (
        cfg.node_count != 8
        or cfg.max_depth != 8
        or cfg.max_loops != 8
        or cfg.n_layers != 2
    ):
        raise ValueError("experiment is fixed to the N8 D8L8 two-block model")
    phase_summary = json.loads(
        phase_summary_path.read_text(encoding="utf-8")
    )
    phase_positions = [
        int(value)
        for value in phase_summary["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    answer_index = positions.index(cfg.seq_len - 1)
    age_map, map_checkpoint = load_unit_j_map(
        j_artifact,
        label=j_label,
        device=device,
    )
    if map_checkpoint != str(checkpoint):
        raise ValueError("J and frozen model checkpoints differ")
    seen, _, _ = reconstruct_primary_training_graphs(
        device=device,
        node_count=cfg.node_count,
    )
    samples = _select_disjoint_unseen_eight_cycles(
        seen=seen,
        sample_count=sample_count,
        sample_seeds=sample_seeds,
    )
    jump = phase_positions[3] - phase_positions[2]
    correct: dict[tuple[int, str, int], int] = defaultdict(int)
    counts: dict[tuple[int, str, int], int] = defaultdict(int)
    output_norm_sum: dict[tuple[int, str, int], float] = defaultdict(float)
    postj_norm_sum: dict[tuple[int, str, int], float] = defaultdict(float)

    for sample_seed in sample_seeds:
        successors_all, starts_all = _expand_all_starts(
            samples[int(sample_seed)],
            device=device,
        )
        for offset in range(0, successors_all.shape[0], batch_size):
            successors = successors_all[offset : offset + batch_size]
            starts = starts_all[offset : offset + batch_size]
            endpoint = advance_nodes(
                successors,
                starts,
                steps=cfg.max_depth,
            )
            initial = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=8,
                phase_position=phase_positions[8],
            )
            states = {condition: initial.clone() for condition in CONDITIONS}
            for cycle_index in range(continuation_loops):
                cycle = cycle_index + 1
                current = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * cycle_index,
                )
                target = advance_nodes(
                    successors,
                    endpoint,
                    steps=jump * cycle,
                )
                oracle_interface = exact_interfaces(
                    model=model,
                    cfg=cfg,
                    phase_positions=phase_positions,
                    positions=positions,
                    successors=successors,
                    current=current,
                    ages=(operating_age,),
                    loop_index=cfg.max_loops + cycle_index,
                )[operating_age]
                exact_step = _run_map_step(
                    condition="exact_H7",
                    cycle=cycle,
                    onset_cycle=onset_cycle,
                    model=model,
                    state=states["exact_H7"],
                    loop_index=cfg.max_loops + cycle_index,
                    positions=positions,
                    answer_index=answer_index,
                    oracle_interface=oracle_interface,
                    age_map=age_map,
                )
                steps = {"exact_H7": exact_step}
                for condition in CONDITIONS:
                    if condition == "exact_H7":
                        continue
                    steps[condition] = _run_map_step(
                        condition=condition,
                        cycle=cycle,
                        onset_cycle=onset_cycle,
                        model=model,
                        state=states[condition],
                        loop_index=cfg.max_loops + cycle_index,
                        positions=positions,
                        answer_index=answer_index,
                        oracle_interface=oracle_interface,
                        age_map=age_map,
                    )
                next_states = {}
                for condition in CONDITIONS:
                    state, logits = _output_state_and_logits(
                        condition=condition,
                        cycle=cycle,
                        onset_cycle=onset_cycle,
                        model=model,
                        step=steps[condition],
                        exact_output=exact_step.state,
                        answer_index=cfg.seq_len - 1,
                    )
                    key = (int(sample_seed), condition, cycle)
                    correct[key] += int(
                        logits.argmax(dim=-1).eq(target).sum().item()
                    )
                    counts[key] += int(successors.shape[0])
                    output_norm_sum[key] += float(
                        state[:, -1].float().norm(dim=-1).sum().item()
                    )
                    postj_norm_sum[key] += float(
                        steps[condition]
                        .block2_hidden_in[:, -1]
                        .float()
                        .norm(dim=-1)
                        .sum()
                        .item()
                    )
                    next_states[condition] = state
                states = next_states

    rows = []
    for key in sorted(counts):
        sample_seed, condition, cycle = key
        count = counts[key]
        rows.append(
            {
                "sample_seed": sample_seed,
                "condition": condition,
                "cycle": cycle,
                "correct": correct[key],
                "count": count,
                "accuracy": correct[key] / count,
                "answer_postJ_norm": postj_norm_sum[key] / count,
                "answer_output_norm": output_norm_sum[key] / count,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "late_norm_rescue.csv", rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "learned_J": "#1f77b4",
        "postJ_answer_norm_after_onset": "#ff7f0e",
        "postJ_all_token_norm_after_onset": "#d62728",
        "output_answer_norm_after_onset": "#2ca02c",
        "output_all_token_norm_after_onset": "#9467bd",
        "exact_H7": "#111111",
    }
    labels = {
        "learned_J": "learned J",
        "postJ_answer_norm_after_onset": "match answer norm after J",
        "postJ_all_token_norm_after_onset": "match every norm after J",
        "output_answer_norm_after_onset": "match answer norm at output",
        "output_all_token_norm_after_onset": "match every norm at output",
        "exact_H7": "exact H7",
    }
    means: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("accuracy", "answer_postJ_norm", "answer_output_norm"):
            means[(str(row["condition"]), int(row["cycle"]), metric)].append(
                float(row[metric])
            )
    cycles = range(1, continuation_loops + 1)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 11),
        sharex=True,
        constrained_layout=True,
    )
    for condition in CONDITIONS:
        for axis, metric in zip(
            axes,
            ("accuracy", "answer_postJ_norm", "answer_output_norm"),
            strict=True,
        ):
            values = [
                float(np.mean(means[(condition, cycle, metric)]))
                for cycle in cycles
            ]
            axis.plot(
                cycles,
                values,
                color=colors[condition],
                label=labels[condition],
                linewidth=1.8,
            )
    for axis in axes:
        axis.axvline(
            onset_cycle,
            color="#777777",
            linestyle="--",
            linewidth=1,
        )
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("Current-step accuracy")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_ylabel("Answer norm after J")
    axes[2].set_ylabel("Answer norm at loop output")
    axes[2].set_xlabel("Continuation loop after H8")
    figure.suptitle(
        f"Late norm-only rescue beginning at continuation loop {onset_cycle}"
    )
    figure_path = out_dir / "late_norm_rescue.png"
    figure.savefig(figure_path, dpi=190)
    plt.close(figure)

    mean_accuracy = {
        condition: {
            cycle: float(
                np.mean(means[(condition, cycle, "accuracy")])
            )
            for cycle in cycles
        }
        for condition in CONDITIONS
    }
    windows = ((1, onset_cycle - 1), (onset_cycle, 64), (65, 96))
    auc = {
        condition: {
            f"auc_{start}_{min(end, continuation_loops)}": float(
                np.mean(
                    [
                        mean_accuracy[condition][cycle]
                        for cycle in range(
                            start,
                            min(end, continuation_loops) + 1,
                        )
                    ]
                )
            )
            for start, end in windows
            if start <= continuation_loops
        }
        for condition in CONDITIONS
    }
    payload = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "J_artifact": str(j_artifact),
        "J_label": j_label,
        "loss_placement": (
            "no training; late norm-only causal intervention on frozen "
            "D8L8 seed0 and its task-aware J"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "dataset": {
            "distribution": (
                "strictly unseen single 8-cycle permutations; all 8 starts"
            ),
            "sample_count_per_replica": sample_count,
            "sample_seeds": [int(value) for value in sample_seeds],
            "replicas_are_disjoint": True,
        },
        "onset_cycle": onset_cycle,
        "conditions": list(CONDITIONS),
        "accuracy_windows": auc,
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "peak_cuda_reserved_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "trajectory": "late_norm_rescue.csv",
            "figure": figure_path.name,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--j-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument(
        "--sample-seeds",
        type=int,
        nargs="+",
        default=(20260801, 20260802),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--continuation-loops", type=int, default=96)
    parser.add_argument("--operating-age", type=int, default=7)
    parser.add_argument("--onset-cycle", type=int, default=48)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=4.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        j_artifact=args.j_artifact,
        j_label=args.j_label,
        out_dir=args.out_dir,
        device_name=args.device,
        sample_count=args.sample_count,
        sample_seeds=args.sample_seeds,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        operating_age=args.operating_age,
        onset_cycle=args.onset_cycle,
        physical_gpu=args.physical_gpu,
        prelaunch_used_mib=args.prelaunch_used_mib,
        prelaunch_free_mib=args.prelaunch_free_mib,
        declared_peak_gib=args.declared_peak_gib,
        reserve_gib=args.reserve_gib,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
