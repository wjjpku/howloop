from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    reconstruct_primary_training_graphs,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_norm_lifespan import (
    _select_disjoint_unseen_eight_cycles,
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


@dataclass(frozen=True)
class Condition:
    name: str
    mode: str
    scope: str = "all"
    alpha: float = 0.0
    timing: str = "repeat"
    shuffled: bool = False


CONDITIONS = (
    Condition("learned_J", mode="none"),
    Condition("radial_answer_repeat", mode="radial", scope="answer"),
    Condition(
        "ln_direction_answer_a100_repeat",
        mode="ln_direction",
        scope="answer",
        alpha=1.0,
    ),
    Condition(
        "ln_direction_all_a025_repeat",
        mode="ln_direction",
        alpha=0.25,
    ),
    Condition(
        "ln_direction_all_a050_repeat",
        mode="ln_direction",
        alpha=0.50,
    ),
    Condition(
        "ln_direction_all_a075_repeat",
        mode="ln_direction",
        alpha=0.75,
    ),
    Condition(
        "ln_direction_all_a100_repeat",
        mode="ln_direction",
        alpha=1.0,
    ),
    Condition(
        "ln_direction_all_a100_once",
        mode="ln_direction",
        alpha=1.0,
        timing="once",
    ),
    Condition(
        "ln_direction_all_a100_shuffled",
        mode="ln_direction",
        alpha=1.0,
        shuffled=True,
    ),
    Condition("exact_H7", mode="exact"),
)
CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}
RAW_METRIC_NAMES = (
    "answer_postJ_norm",
    "answer_postJ_ln_cosine",
    "all_postJ_ln_cosine",
)


@dataclass
class SumCount:
    total: float = 0.0
    count: int = 0

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().float().flatten()
        self.total += float(flat.sum().item())
        self.count += int(flat.numel())

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _centered_direction(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = value.float().mean(dim=-1, keepdim=True)
    centered = value.float() - mean
    centered_norm = centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return mean, centered / centered_norm, centered_norm


def slerp_layernorm_direction(
    live: torch.Tensor,
    oracle: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Replace only centered direction, preserving live mean and raw L2 norm."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    live_mean, live_direction, live_centered_norm = _centered_direction(live)
    _, oracle_direction, _ = _centered_direction(oracle)
    cosine = (live_direction * oracle_direction).sum(
        dim=-1,
        keepdim=True,
    ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cosine)
    sin_theta = torch.sin(theta)
    left = torch.sin((1.0 - alpha) * theta) / sin_theta.clamp_min(1e-7)
    right = torch.sin(alpha * theta) / sin_theta.clamp_min(1e-7)
    direction = left * live_direction + right * oracle_direction
    near_parallel = sin_theta.abs() < 1e-5
    linear = (1.0 - alpha) * live_direction + alpha * oracle_direction
    direction = torch.where(near_parallel, linear, direction)
    direction = F.normalize(direction, dim=-1, eps=1e-8)
    result = live_mean + direction * live_centered_norm
    return result.to(dtype=live.dtype)


def _radial_answer_patch(
    live: torch.Tensor,
    oracle: torch.Tensor,
    *,
    answer_index: int,
) -> torch.Tensor:
    result = live.clone()
    answer = result[:, answer_index]
    target = oracle[:, answer_index]
    scale = (
        target.float().norm(dim=-1, keepdim=True)
        / answer.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    result[:, answer_index] = answer * scale.to(dtype=answer.dtype)
    return result


def _should_patch(
    condition: Condition,
    *,
    cycle: int,
    onset_cycle: int,
) -> bool:
    if condition.mode in ("none", "exact"):
        return False
    if condition.timing == "once":
        return cycle == onset_cycle
    if condition.timing == "repeat":
        return cycle >= onset_cycle
    raise ValueError(f"unknown timing: {condition.timing}")


def _angular_patch(
    live: torch.Tensor,
    oracle: torch.Tensor,
    *,
    scope: str,
    alpha: float,
    answer_index: int,
) -> torch.Tensor:
    if scope == "all":
        return slerp_layernorm_direction(live, oracle, alpha=alpha)
    if scope == "answer":
        result = live.clone()
        result[:, answer_index] = slerp_layernorm_direction(
            live[:, answer_index],
            oracle[:, answer_index],
            alpha=alpha,
        )
        return result
    raise ValueError(f"unknown scope: {scope}")


def _run_condition_step(
    *,
    condition: Condition,
    cycle: int,
    onset_cycle: int,
    model,
    state: torch.Tensor,
    loop_index: int,
    positions: tuple[int, ...],
    answer_index: int,
    oracle_interface: torch.Tensor,
    shuffled_oracle: torch.Tensor,
    age_map,
    attention_answer_override: torch.Tensor | None = None,
    mlp_answer_override: torch.Tensor | None = None,
):
    if condition.mode == "exact":
        step = run_one_loop(
            model,
            state,
            loop_index=loop_index,
            block2_position_override=(positions, oracle_interface),
            attention_answer_override=attention_answer_override,
            mlp_answer_override=mlp_answer_override,
        )
        return step, None

    capture: dict[str, torch.Tensor | bool] = {}

    def transform(value: torch.Tensor) -> torch.Tensor:
        mapped = age_map(value)
        patched = mapped
        applied = _should_patch(
            condition,
            cycle=cycle,
            onset_cycle=onset_cycle,
        )
        if applied:
            target = shuffled_oracle if condition.shuffled else oracle_interface
            if condition.mode == "ln_direction":
                patched = _angular_patch(
                    mapped,
                    target,
                    scope=condition.scope,
                    alpha=condition.alpha,
                    answer_index=answer_index,
                )
            elif condition.mode == "radial":
                patched = _radial_answer_patch(
                    mapped,
                    target,
                    answer_index=answer_index,
                )
            else:
                raise ValueError(f"unknown intervention mode: {condition.mode}")
        capture["mapped"] = mapped.detach()
        capture["patched"] = patched.detach()
        capture["applied"] = applied
        return patched

    step = run_one_loop(
        model,
        state,
        loop_index=loop_index,
        block2_position_transform=(positions, transform),
        attention_answer_override=attention_answer_override,
        mlp_answer_override=mlp_answer_override,
    )
    return step, capture


def _group_mean_norm(
    value: torch.Tensor,
    positions: tuple[int, ...],
) -> torch.Tensor:
    return value[:, list(positions)].float().norm(dim=-1).mean(dim=1)


def _group_mean_cosine(
    value: torch.Tensor,
    target: torch.Tensor,
    positions: tuple[int, ...],
) -> torch.Tensor:
    return F.cosine_similarity(
        value[:, list(positions)].float(),
        target[:, list(positions)].float(),
        dim=-1,
        eps=1e-8,
    ).mean(dim=1)


def _maximum_relative_token_norm_change(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    positions: tuple[int, ...],
) -> float:
    index = list(positions)
    source_norm = source[:, index].float().norm(dim=-1).clamp_min(1e-8)
    target_norm = target[:, index].float().norm(dim=-1)
    return float(
        ((target_norm - source_norm).abs() / source_norm).max().item()
    )


def _mean_curve(
    rows: list[dict[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row["cycle"]))].append(
            float(row[field])
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _window_auc(
    curve: dict[int, float],
    start: int,
    end: int,
) -> float | None:
    values = [curve[cycle] for cycle in range(start, end + 1) if cycle in curve]
    return float(np.mean(values)) if values else None


def _first_below_after(
    curve: dict[int, float],
    *,
    start: int,
    threshold: float,
) -> int | None:
    return next(
        (
            cycle
            for cycle in sorted(curve)
            if cycle >= start and curve[cycle] < threshold
        ),
        None,
    )


def _plot_results(
    *,
    rows: list[dict[str, Any]],
    onset_cycle: int,
    continuation_loops: int,
    out_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accuracy = _mean_curve(rows, field="accuracy")
    cycles = np.arange(1, continuation_loops + 1)
    colors = {
        "learned_J": "#1f77b4",
        "radial_answer_repeat": "#8c8c8c",
        "ln_direction_answer_a100_repeat": "#17becf",
        "ln_direction_all_a025_repeat": "#bcbd22",
        "ln_direction_all_a050_repeat": "#ff7f0e",
        "ln_direction_all_a075_repeat": "#d62728",
        "ln_direction_all_a100_repeat": "#2ca02c",
        "ln_direction_all_a100_once": "#9467bd",
        "ln_direction_all_a100_shuffled": "#8c564b",
        "exact_H7": "#111111",
    }

    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    dose_names = (
        "learned_J",
        "ln_direction_all_a025_repeat",
        "ln_direction_all_a050_repeat",
        "ln_direction_all_a075_repeat",
        "ln_direction_all_a100_repeat",
        "ln_direction_all_a100_shuffled",
        "exact_H7",
    )
    for name in dose_names:
        axis.plot(
            cycles,
            [accuracy[(name, int(cycle))] for cycle in cycles],
            color=colors[name],
            linewidth=1.8,
            label=name.replace("ln_direction_all_", "").replace("_repeat", ""),
        )
    axis.axvline(onset_cycle, color="#666666", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Continuation loop after H8",
        ylabel="Current-step accuracy",
        ylim=(-0.03, 1.03),
        title=(
            "LayerNorm-direction dose response with every token norm preserved"
        ),
    )
    axis.grid(alpha=0.18)
    axis.legend(ncol=4, fontsize=8)
    dose_path = out_dir / "angular_dose_accuracy.png"
    figure.savefig(dose_path, dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    scope_names = (
        "learned_J",
        "radial_answer_repeat",
        "ln_direction_answer_a100_repeat",
        "ln_direction_all_a100_repeat",
        "ln_direction_all_a100_once",
        "ln_direction_all_a100_shuffled",
        "exact_H7",
    )
    for name in scope_names:
        axis.plot(
            cycles,
            [accuracy[(name, int(cycle))] for cycle in cycles],
            color=colors[name],
            linewidth=1.8,
            label=name,
        )
    axis.axvline(onset_cycle, color="#666666", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Continuation loop after H8",
        ylabel="Current-step accuracy",
        ylim=(-0.03, 1.03),
        title="Radial vs angular patch; answer vs all; once vs repeated",
    )
    axis.grid(alpha=0.18)
    axis.legend(ncol=2, fontsize=7.5)
    scope_path = out_dir / "angular_scope_timing_accuracy.png"
    figure.savefig(scope_path, dpi=190)
    plt.close(figure)

    metrics = (
        ("answer_postJ_ln_cosine", "Answer post-J LN cosine to H7"),
        ("all_postJ_ln_cosine", "All-position post-J LN cosine to H7"),
        ("attention_update_norm_answer", "B2 attention answer-update norm"),
        ("mlp_update_norm_answer", "B2 MLP answer-update norm"),
    )
    metric_curves = {
        field: _mean_curve(rows, field=field) for field, _ in metrics
    }
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        sharex=True,
        constrained_layout=True,
    )
    diagnostic_names = (
        "learned_J",
        "ln_direction_all_a050_repeat",
        "ln_direction_all_a100_repeat",
        "ln_direction_all_a100_once",
        "ln_direction_all_a100_shuffled",
        "exact_H7",
    )
    for axis, (field, label) in zip(axes.ravel(), metrics, strict=True):
        for name in diagnostic_names:
            axis.plot(
                cycles,
                [
                    metric_curves[field][(name, int(cycle))]
                    for cycle in cycles
                ],
                color=colors[name],
                linewidth=1.5,
                label=name,
            )
        axis.axvline(onset_cycle, color="#666666", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.set_xlabel("Continuation loop after H8")
        axis.grid(alpha=0.18)
    axes[0, 1].legend(ncol=2, fontsize=7)
    circuit_path = out_dir / "angular_direction_and_updates.png"
    figure.savefig(circuit_path, dpi=190)
    plt.close(figure)
    return [dose_path.name, scope_path.name, circuit_path.name]


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
    if getattr(cfg, "inner_norm_style", None) != "pre_layernorm":
        raise ValueError("LayerNorm-direction experiment requires pre-LN")
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
    seen, unique_after_stage, total_training_draws = (
        reconstruct_primary_training_graphs(
            device=device,
            node_count=cfg.node_count,
        )
    )
    samples = _select_disjoint_unseen_eight_cycles(
        seen=seen,
        sample_count=sample_count,
        sample_seeds=sample_seeds,
    )
    jump = phase_positions[3] - phase_positions[2]
    condition_names = tuple(condition.name for condition in CONDITIONS)
    condition_index = {
        name: index for index, name in enumerate(condition_names)
    }
    aggregated: dict[tuple[int, str, int, str], SumCount] = defaultdict(
        SumCount
    )
    audit_max_norm_change: dict[str, float] = defaultdict(float)
    executor_gate_correct: dict[tuple[str, str], int] = defaultdict(int)
    executor_gate_count = 0
    replica_files = []

    for sample_seed in sample_seeds:
        successors_all, starts_all = _expand_all_starts(
            samples[int(sample_seed)],
            device=device,
        )
        example_count = int(successors_all.shape[0])
        correctness = np.zeros(
            (len(CONDITIONS), continuation_loops, example_count),
            dtype=np.bool_,
        )
        raw_metrics = np.full(
            (
                len(CONDITIONS),
                continuation_loops,
                len(RAW_METRIC_NAMES),
                example_count,
            ),
            np.nan,
            dtype=np.float32,
        )
        for offset in range(0, example_count, batch_size):
            successors = successors_all[offset : offset + batch_size]
            starts = starts_all[offset : offset + batch_size]
            batch_slice = slice(offset, offset + successors.shape[0])
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
            states = {name: initial.clone() for name in condition_names}
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
                shuffled_oracle = torch.roll(
                    oracle_interface,
                    shifts=cfg.node_count,
                    dims=0,
                )
                steps = {}
                captures = {}
                for condition in CONDITIONS:
                    step, capture = _run_condition_step(
                        condition=condition,
                        cycle=cycle,
                        onset_cycle=onset_cycle,
                        model=model,
                        state=states[condition.name],
                        loop_index=cfg.max_loops + cycle_index,
                        positions=positions,
                        answer_index=answer_index,
                        oracle_interface=oracle_interface,
                        shuffled_oracle=shuffled_oracle,
                        age_map=age_map,
                    )
                    steps[condition.name] = step
                    captures[condition.name] = capture

                if cycle == onset_cycle:
                    gate_names = (
                        "learned_J",
                        "radial_answer_repeat",
                        "ln_direction_answer_a100_repeat",
                        "ln_direction_all_a100_repeat",
                        "ln_direction_all_a100_shuffled",
                        "exact_H7",
                    )
                    zeros = torch.zeros(
                        successors.shape[0],
                        cfg.d_model,
                        device=device,
                        dtype=initial.dtype,
                    )
                    for name in gate_names:
                        executor_gate_correct[(name, "full_Block2")] += int(
                            steps[name]
                            .logits.argmax(dim=-1)
                            .eq(target)
                            .sum()
                            .item()
                        )
                        blocked, _ = _run_condition_step(
                            condition=CONDITION_BY_NAME[name],
                            cycle=cycle,
                            onset_cycle=onset_cycle,
                            model=model,
                            state=states[name],
                            loop_index=cfg.max_loops + cycle_index,
                            positions=positions,
                            answer_index=answer_index,
                            oracle_interface=oracle_interface,
                            shuffled_oracle=shuffled_oracle,
                            age_map=age_map,
                            attention_answer_override=zeros,
                            mlp_answer_override=zeros,
                        )
                        executor_gate_correct[
                            (name, "Block2_answer_updates_zero")
                        ] += int(
                            blocked.logits.argmax(dim=-1).eq(target).sum().item()
                        )
                    executor_gate_count += int(successors.shape[0])

                oracle_ln = model.blocks[1].ln_1(oracle_interface)
                next_states = {}
                for condition in CONDITIONS:
                    name = condition.name
                    step = steps[name]
                    correct = step.logits.argmax(dim=-1).eq(target)
                    correctness[
                        condition_index[name],
                        cycle_index,
                        batch_slice,
                    ] = correct.detach().cpu().numpy()
                    postj_ln = model.blocks[1].ln_1(step.block2_hidden_in)
                    metrics = {
                        "accuracy": correct.float(),
                        "answer_postJ_norm": _group_mean_norm(
                            step.block2_hidden_in,
                            (cfg.seq_len - 1,),
                        ),
                        "all_postJ_norm": _group_mean_norm(
                            step.block2_hidden_in,
                            tuple(range(cfg.seq_len)),
                        ),
                        "answer_postJ_ln_cosine": _group_mean_cosine(
                            postj_ln,
                            oracle_ln,
                            (cfg.seq_len - 1,),
                        ),
                        "all_postJ_ln_cosine": _group_mean_cosine(
                            postj_ln,
                            oracle_ln,
                            tuple(range(cfg.seq_len)),
                        ),
                        "attention_update_norm_answer": (
                            step.block2_attention_out[:, -1]
                            .float()
                            .norm(dim=-1)
                        ),
                        "mlp_update_norm_answer": (
                            step.block2_mlp_out[:, -1].float().norm(dim=-1)
                        ),
                        "output_norm_answer": (
                            step.state[:, -1].float().norm(dim=-1)
                        ),
                        "output_readout_cosine": F.cosine_similarity(
                            model.ln_final(step.state[:, -1]).float(),
                            model.ln_final(steps["exact_H7"].state[:, -1]).float(),
                            dim=-1,
                            eps=1e-8,
                        ),
                    }
                    for metric_name, values in metrics.items():
                        aggregated[
                            (
                                int(sample_seed),
                                name,
                                cycle,
                                metric_name,
                            )
                        ].add(values)
                    for metric_index, metric_name in enumerate(
                        RAW_METRIC_NAMES
                    ):
                        raw_metrics[
                            condition_index[name],
                            cycle_index,
                            metric_index,
                            batch_slice,
                        ] = metrics[metric_name].detach().cpu().numpy()

                    capture = captures[name]
                    if (
                        capture is not None
                        and bool(capture["applied"])
                        and condition.mode == "ln_direction"
                    ):
                        scope_positions = (
                            tuple(range(cfg.seq_len))
                            if condition.scope == "all"
                            else (cfg.seq_len - 1,)
                        )
                        norm_change = _maximum_relative_token_norm_change(
                            capture["mapped"],
                            capture["patched"],
                            positions=scope_positions,
                        )
                        audit_max_norm_change[name] = max(
                            audit_max_norm_change[name],
                            norm_change,
                        )
                    next_states[name] = step.state
                states = next_states

        replica_dir = out_dir / f"replica_{sample_seed}"
        replica_dir.mkdir(parents=True, exist_ok=True)
        raw_path = replica_dir / "raw_balanced_examples.npz"
        np.savez_compressed(
            raw_path,
            correctness=correctness,
            raw_metrics=raw_metrics,
            successors=successors_all.detach().cpu().numpy(),
            starts=starts_all.detach().cpu().numpy(),
            conditions=np.asarray(condition_names),
            cycles=np.arange(1, continuation_loops + 1),
            metric_names=np.asarray(RAW_METRIC_NAMES),
        )
        replica_files.append(
            {
                "sample_seed": int(sample_seed),
                "permutations": sample_count,
                "examples_all_starts": example_count,
                "raw_balanced_examples": str(raw_path.relative_to(out_dir)),
            }
        )

    rows: list[dict[str, Any]] = []
    for sample_seed in sample_seeds:
        for condition in CONDITIONS:
            for cycle in range(1, continuation_loops + 1):
                row = {
                    "sample_seed": int(sample_seed),
                    "condition": condition.name,
                    "cycle": cycle,
                }
                for metric_name in (
                    "accuracy",
                    "answer_postJ_norm",
                    "all_postJ_norm",
                    "answer_postJ_ln_cosine",
                    "all_postJ_ln_cosine",
                    "attention_update_norm_answer",
                    "mlp_update_norm_answer",
                    "output_norm_answer",
                    "output_readout_cosine",
                ):
                    row[metric_name] = aggregated[
                        (
                            int(sample_seed),
                            condition.name,
                            cycle,
                            metric_name,
                        )
                    ].mean
                rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "angular_rescue_by_loop.csv", rows)
    figure_files = _plot_results(
        rows=rows,
        onset_cycle=onset_cycle,
        continuation_loops=continuation_loops,
        out_dir=out_dir,
    )

    mean_accuracy = _mean_curve(rows, field="accuracy")
    curves = {
        name: {
            cycle: mean_accuracy[(name, cycle)]
            for cycle in range(1, continuation_loops + 1)
        }
        for name in condition_names
    }
    windows = (
        (1, onset_cycle - 1),
        (onset_cycle, min(64, continuation_loops)),
        (65, continuation_loops),
    )
    accuracy_windows = {
        name: {
            f"auc_{start}_{end}": _window_auc(curves[name], start, end)
            for start, end in windows
            if start <= end
        }
        for name in condition_names
    }
    lifespan = {
        name: {
            str(threshold): _first_below_after(
                curves[name],
                start=onset_cycle,
                threshold=threshold,
            )
            for threshold in (0.95, 0.90, 0.80, 0.50)
        }
        for name in condition_names
    }
    payload = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "J_artifact": str(j_artifact),
        "J_label": j_label,
        "loss_placement": (
            "no training; frozen D8L8 seed0 and frozen task-aware J; "
            "LayerNorm-direction causal interventions only"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "inner_norm_style": cfg.inner_norm_style,
        "readout_norm_style": cfg.readout_norm_style,
        "onset_cycle": onset_cycle,
        "dataset": {
            "distribution": (
                "strictly unseen single 8-cycle permutations; all 8 starts"
            ),
            "sample_count_per_replica": sample_count,
            "sample_seeds": [int(value) for value in sample_seeds],
            "replicas_are_disjoint": True,
            "training_graph_draws_reconstructed": total_training_draws,
            "unique_training_graphs_reconstructed": len(seen),
            "unique_after_training_stage": unique_after_stage,
        },
        "direction_definition": (
            "spherical interpolation of zero-mean feature directions; "
            "restore live token mean and centered norm, preserving raw L2"
        ),
        "conditions": [
            {
                "name": condition.name,
                "mode": condition.mode,
                "scope": condition.scope,
                "alpha": condition.alpha,
                "timing": condition.timing,
                "shuffled": condition.shuffled,
            }
            for condition in CONDITIONS
        ],
        "maximum_relative_token_norm_change": dict(audit_max_norm_change),
        "executor_gate_at_onset": {
            name: {
                variant: executor_gate_correct[(name, variant)]
                / executor_gate_count
                for variant in (
                    "full_Block2",
                    "Block2_answer_updates_zero",
                )
            }
            for name in (
                "learned_J",
                "radial_answer_repeat",
                "ln_direction_answer_a100_repeat",
                "ln_direction_all_a100_repeat",
                "ln_direction_all_a100_shuffled",
                "exact_H7",
            )
        },
        "accuracy_windows": accuracy_windows,
        "first_cycle_below_after_onset": lifespan,
        "replicas": replica_files,
        "claim_ledger": [
            {
                "claim": (
                    "LayerNorm-visible direction, rather than raw norm alone, "
                    "is causally sufficient to extend learned-J lifespan"
                ),
                "status": "tested",
                "evidence": (
                    "norm-preserving direction dose response, once/repeated "
                    "timing, and shuffled-oracle negative control"
                ),
                "revise_if": (
                    "correct direction does not outperform radial and "
                    "shuffled-direction controls"
                ),
            },
            {
                "claim": (
                    "directional repair restores Block2 answer routing/writing"
                ),
                "status": "tested",
                "evidence": (
                    "post-J LayerNorm cosine plus Block2 attention and MLP "
                    "answer-update norms"
                ),
                "revise_if": (
                    "accuracy changes without component-update recovery"
                ),
            },
        ],
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
            "trajectory": "angular_rescue_by_loop.csv",
            "figures": figure_files,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Norm-preserving LayerNorm-direction rejuvenation."
    )
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
