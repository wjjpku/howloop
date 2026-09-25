from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    Permutation,
    _expand_all_starts,
    cycle_type,
    reconstruct_primary_training_graphs,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
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


CONDITIONS = (
    "learned_J",
    "J_global_normmatched",
    "J_token_normmatched",
    "exact_H7",
    "no_J",
)
SITES = (
    "loop_input",
    "block1_update",
    "post_block1_pre_J",
    "J_update",
    "post_J_pre_Block2",
    "Block2_attention_update",
    "post_Block2_attention",
    "Block2_MLP_update",
    "loop_output",
)
METRICS = (
    "norm",
    "oracle_norm_ratio",
    "oracle_abs_norm_fraction_error",
    "oracle_cosine",
    "oracle_relative_l2",
)
RAW_SITES = ("post_J_pre_Block2", "loop_output")
RAW_GROUPS = ("all", "answer")
WINDOWS = ((1, 24), (25, 48), (49, 64), (65, 96))


@dataclass
class Moments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().float().flatten()
        if flat.numel() == 0:
            return
        self.count += int(flat.numel())
        self.total += float(flat.sum().item())
        self.total_square += float(flat.square().sum().item())
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _position_groups(node_count: int, seq_len: int) -> dict[str, tuple[int, ...]]:
    explicit = explicit_depth_position_groups(node_count)
    return {
        "all": tuple(range(seq_len)),
        "graph": explicit["graph"],
        "query_metadata": explicit["query_metadata"],
        "answer": explicit["answer"],
    }


def _site_values(state: torch.Tensor, step) -> dict[str, torch.Tensor]:
    pre_j = step.block2_hidden_pre_intervention
    post_j = step.block2_hidden_in
    return {
        "loop_input": state,
        "block1_update": pre_j - state,
        "post_block1_pre_J": pre_j,
        "J_update": post_j - pre_j,
        "post_J_pre_Block2": post_j,
        "Block2_attention_update": step.block2_attention_out,
        "post_Block2_attention": step.block2_residual_mid,
        "Block2_MLP_update": step.block2_mlp_out,
        "loop_output": step.state,
    }


def _group_metrics(
    value: torch.Tensor,
    oracle: torch.Tensor,
    positions: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    index = list(positions)
    selected = value[:, index].float()
    target = oracle[:, index].float()
    selected_norm = selected.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    mean_norm = selected_norm.mean(dim=1)
    mean_target_norm = target_norm.mean(dim=1).clamp_min(1e-8)
    return {
        "norm": mean_norm,
        "oracle_norm_ratio": mean_norm / mean_target_norm,
        "oracle_abs_norm_fraction_error": (
            (selected_norm - target_norm).abs()
            / target_norm.clamp_min(1e-8)
        ).mean(dim=1),
        "oracle_cosine": F.cosine_similarity(
            selected,
            target,
            dim=-1,
            eps=1e-8,
        ).mean(dim=1),
        "oracle_relative_l2": (
            (selected - target).flatten(1).norm(dim=1)
            / target.flatten(1).norm(dim=1).clamp_min(1e-8)
        ),
    }


def _global_normmatched(
    mapped: torch.Tensor,
    oracle: torch.Tensor,
) -> torch.Tensor:
    mapped_norm = mapped.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    target_norm = oracle.float().flatten(1).norm(dim=1)
    scale = (target_norm / mapped_norm).to(dtype=mapped.dtype)
    return mapped * scale[:, None, None]


def _token_normmatched(
    mapped: torch.Tensor,
    oracle: torch.Tensor,
) -> torch.Tensor:
    mapped_norm = mapped.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target_norm = oracle.float().norm(dim=-1, keepdim=True)
    return mapped * (target_norm / mapped_norm).to(dtype=mapped.dtype)


def _select_disjoint_unseen_eight_cycles(
    *,
    seen: set[Permutation],
    sample_count: int,
    sample_seeds: Sequence[int],
) -> dict[int, list[Permutation]]:
    available = sorted(
        permutation
        for permutation in itertools.permutations(range(8))
        if permutation not in seen and cycle_type(permutation) == (8,)
    )
    required = sample_count * len(sample_seeds)
    if len(available) < required:
        raise ValueError(
            f"need {required} disjoint unseen 8-cycles, have {len(available)}"
        )
    selected: dict[int, list[Permutation]] = {}
    remaining = available
    for seed in sample_seeds:
        shuffled = list(remaining)
        random.Random(seed).shuffle(shuffled)
        take = shuffled[:sample_count]
        selected[int(seed)] = take
        used = set(take)
        remaining = [item for item in remaining if item not in used]
    return selected


def _condition_step(
    *,
    condition: str,
    model,
    state: torch.Tensor,
    loop_index: int,
    positions: tuple[int, ...],
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
    if condition == "no_J":
        return run_one_loop(model, state, loop_index=loop_index)

    def transform(value: torch.Tensor) -> torch.Tensor:
        mapped = age_map(value)
        if condition == "learned_J":
            return mapped
        if condition == "J_global_normmatched":
            return _global_normmatched(mapped, oracle_interface)
        if condition == "J_token_normmatched":
            return _token_normmatched(mapped, oracle_interface)
        raise ValueError(f"unknown condition: {condition}")

    return run_one_loop(
        model,
        state,
        loop_index=loop_index,
        block2_position_transform=(positions, transform),
    )


def _metric_rows(
    storage: dict[tuple[str, int, str, str, str], Moments],
    *,
    replica_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (condition, cycle, site, group, metric), moments in sorted(
        storage.items()
    ):
        rows.append(
            {
                "replica_seed": replica_seed,
                "condition": condition,
                "cycle": cycle,
                "site": site,
                "position_group": group,
                "metric": metric,
                **moments.summary(),
            }
        )
    return rows


def _separation_rows(
    storage: dict[
        tuple[int, str, str, str, str, str],
        Moments,
    ],
    *,
    replica_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[
        tuple[int, str, str, str],
        dict[str, dict[str, float | int]],
    ] = defaultdict(dict)
    for (cycle, site, group, metric, outcome, _), moments in storage.items():
        grouped[(cycle, site, group, metric)][outcome] = moments.summary()
    for (cycle, site, group, metric), outcomes in sorted(grouped.items()):
        correct = outcomes.get("correct", Moments().summary())
        incorrect = outcomes.get("incorrect", Moments().summary())
        pooled_variance = (
            float(correct["std"]) ** 2 + float(incorrect["std"]) ** 2
        ) / 2
        standardized_difference = (
            (float(correct["mean"]) - float(incorrect["mean"]))
            / math.sqrt(max(pooled_variance, 1e-12))
            if int(correct["count"]) and int(incorrect["count"])
            else float("nan")
        )
        rows.append(
            {
                "replica_seed": replica_seed,
                "cycle": cycle,
                "site": site,
                "position_group": group,
                "metric": metric,
                "correct_count": correct["count"],
                "incorrect_count": incorrect["count"],
                "correct_mean": correct["mean"],
                "correct_std": correct["std"],
                "incorrect_mean": incorrect["mean"],
                "incorrect_std": incorrect["std"],
                "standardized_correct_minus_incorrect": (
                    standardized_difference
                ),
            }
        )
    return rows


@torch.no_grad()
def _run_replica(
    *,
    replica_seed: int,
    permutations: list[Permutation],
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    age_map,
    device: torch.device,
    batch_size: int,
    continuation_loops: int,
    operating_age: int,
    out_dir: Path,
) -> dict[str, Any]:
    successors_all, starts_all = _expand_all_starts(
        permutations,
        device=device,
    )
    example_count = int(successors_all.shape[0])
    jump = phase_positions[3] - phase_positions[2]
    groups = _position_groups(cfg.node_count, cfg.seq_len)
    group_names = tuple(groups)
    condition_to_index = {
        condition: index for index, condition in enumerate(CONDITIONS)
    }
    raw_site_to_index = {site: index for index, site in enumerate(RAW_SITES)}
    raw_group_to_index = {
        group: index for index, group in enumerate(RAW_GROUPS)
    }
    metric_to_index = {metric: index for index, metric in enumerate(METRICS)}
    correctness = np.zeros(
        (len(CONDITIONS), continuation_loops, example_count),
        dtype=np.bool_,
    )
    learned_raw = np.full(
        (
            continuation_loops,
            len(RAW_SITES),
            len(RAW_GROUPS),
            len(METRICS),
            example_count,
        ),
        np.nan,
        dtype=np.float32,
    )
    metric_storage: dict[
        tuple[str, int, str, str, str],
        Moments,
    ] = defaultdict(Moments)
    separation_storage: dict[
        tuple[int, str, str, str, str, str],
        Moments,
    ] = defaultdict(Moments)

    for offset in range(0, example_count, batch_size):
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
        batch_slice = slice(offset, offset + successors.shape[0])

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

            exact_step = _condition_step(
                condition="exact_H7",
                model=model,
                state=states["exact_H7"],
                loop_index=cfg.max_loops + cycle_index,
                positions=positions,
                oracle_interface=oracle_interface,
                age_map=age_map,
            )
            steps = {"exact_H7": exact_step}
            for condition in CONDITIONS:
                if condition == "exact_H7":
                    continue
                steps[condition] = _condition_step(
                    condition=condition,
                    model=model,
                    state=states[condition],
                    loop_index=cfg.max_loops + cycle_index,
                    positions=positions,
                    oracle_interface=oracle_interface,
                    age_map=age_map,
                )

            oracle_sites = _site_values(
                states["exact_H7"],
                exact_step,
            )
            for condition in CONDITIONS:
                step = steps[condition]
                correct = step.logits.argmax(dim=-1).eq(target)
                correctness[
                    condition_to_index[condition],
                    cycle_index,
                    batch_slice,
                ] = correct.detach().cpu().numpy()
                sites = _site_values(states[condition], step)
                for site in SITES:
                    for group in group_names:
                        metrics = _group_metrics(
                            sites[site],
                            oracle_sites[site],
                            groups[group],
                        )
                        for metric, values in metrics.items():
                            metric_storage[
                                (condition, cycle, site, group, metric)
                            ].add(values)
                            if condition == "learned_J":
                                separation_storage[
                                    (
                                        cycle,
                                        site,
                                        group,
                                        metric,
                                        "correct",
                                        "learned_J",
                                    )
                                ].add(values[correct])
                                separation_storage[
                                    (
                                        cycle,
                                        site,
                                        group,
                                        metric,
                                        "incorrect",
                                        "learned_J",
                                    )
                                ].add(values[~correct])
                                if site in RAW_SITES and group in RAW_GROUPS:
                                    learned_raw[
                                        cycle_index,
                                        raw_site_to_index[site],
                                        raw_group_to_index[group],
                                        metric_to_index[metric],
                                        batch_slice,
                                    ] = values.detach().cpu().numpy()
            states = {
                condition: steps[condition].state for condition in CONDITIONS
            }

    accuracy_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for cycle_index in range(continuation_loops):
            accuracy_rows.append(
                {
                    "replica_seed": replica_seed,
                    "condition": condition,
                    "cycle": cycle_index + 1,
                    "correct": int(
                        correctness[condition_index, cycle_index].sum()
                    ),
                    "count": example_count,
                    "accuracy": float(
                        correctness[condition_index, cycle_index].mean()
                    ),
                }
            )
    metric_rows = _metric_rows(
        metric_storage,
        replica_seed=replica_seed,
    )
    separation_rows = _separation_rows(
        separation_storage,
        replica_seed=replica_seed,
    )
    replica_dir = out_dir / f"replica_{replica_seed}"
    replica_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(replica_dir / "accuracy_by_loop.csv", accuracy_rows)
    _write_csv(replica_dir / "norm_metrics_by_loop.csv", metric_rows)
    _write_csv(
        replica_dir / "correct_vs_incorrect_norm_metrics.csv",
        separation_rows,
    )
    np.savez_compressed(
        replica_dir / "raw_balanced_examples.npz",
        correctness=correctness,
        learned_J_metrics=learned_raw,
        successors=successors_all.detach().cpu().numpy(),
        starts=starts_all.detach().cpu().numpy(),
        conditions=np.asarray(CONDITIONS),
        cycles=np.arange(1, continuation_loops + 1),
        raw_sites=np.asarray(RAW_SITES),
        raw_groups=np.asarray(RAW_GROUPS),
        metrics=np.asarray(METRICS),
    )
    return {
        "replica_seed": replica_seed,
        "permutations": len(permutations),
        "examples_all_starts": example_count,
        "cycle_type": [8],
        "accuracy_rows": accuracy_rows,
        "metric_rows": metric_rows,
        "separation_rows": separation_rows,
        "files": {
            "accuracy": str(
                (replica_dir / "accuracy_by_loop.csv").relative_to(out_dir)
            ),
            "norm_metrics": str(
                (replica_dir / "norm_metrics_by_loop.csv").relative_to(
                    out_dir
                )
            ),
            "correctness_separation": str(
                (
                    replica_dir / "correct_vs_incorrect_norm_metrics.csv"
                ).relative_to(out_dir)
            ),
            "raw_balanced_examples": str(
                (replica_dir / "raw_balanced_examples.npz").relative_to(
                    out_dir
                )
            ),
        },
    }


def _mean_rows(
    rows: Iterable[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    value: str,
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _plot_results(
    *,
    accuracy_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "learned_J": "#1f77b4",
        "J_global_normmatched": "#ff7f0e",
        "J_token_normmatched": "#2ca02c",
        "exact_H7": "#111111",
        "no_J": "#8c8c8c",
    }
    labels = {
        "learned_J": "learned J",
        "J_global_normmatched": "J + global norm match",
        "J_token_normmatched": "J + token norm match",
        "exact_H7": "exact H7 oracle",
        "no_J": "no J",
    }
    mean_accuracy = _mean_rows(
        accuracy_rows,
        keys=("condition", "cycle"),
        value="accuracy",
    )
    max_cycle = max(int(row["cycle"]) for row in accuracy_rows)
    cycles = np.arange(1, max_cycle + 1)
    figure, axis = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    for condition in CONDITIONS:
        values = [
            mean_accuracy[(condition, int(cycle))] for cycle in cycles
        ]
        axis.plot(
            cycles,
            values,
            color=colors[condition],
            label=labels[condition],
            linewidth=2 if condition != "no_J" else 1.2,
        )
    axis.axvspan(49, 64, color="#d62728", alpha=0.055)
    if max_cycle >= 65:
        axis.axvspan(65, max_cycle, color="#9467bd", alpha=0.045)
    axis.set(
        xlabel="Continuation loop after H8",
        ylabel="Current-step accuracy",
        ylim=(-0.03, 1.03),
        title="Does norm correction extend the learned-J lifespan?",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    accuracy_path = out_dir / "accuracy_and_norm_controls.png"
    figure.savefig(accuracy_path, dpi=190)
    plt.close(figure)

    mean_metric = _mean_rows(
        metric_rows,
        keys=(
            "condition",
            "cycle",
            "site",
            "position_group",
            "metric",
        ),
        value="mean",
    )
    selected_sites = (
        "loop_input",
        "post_block1_pre_J",
        "post_J_pre_Block2",
        "post_Block2_attention",
        "loop_output",
    )
    figure, axes = plt.subplots(
        len(selected_sites),
        2,
        figsize=(14, 15),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, site in enumerate(selected_sites):
        for column, group in enumerate(("all", "answer")):
            axis = axes[row_index, column]
            for condition in (
                "learned_J",
                "J_global_normmatched",
                "J_token_normmatched",
                "exact_H7",
            ):
                values = [
                    mean_metric[
                        (condition, int(cycle), site, group, "norm")
                    ]
                    for cycle in cycles
                ]
                axis.plot(
                    cycles,
                    values,
                    color=colors[condition],
                    label=labels[condition],
                    linewidth=1.6,
                )
            axis.set_title(f"{site}: {group}")
            axis.set_ylabel("Mean token L2 norm")
            axis.grid(alpha=0.18)
    for axis in axes[-1]:
        axis.set_xlabel("Continuation loop after H8")
    axes[0, 1].legend(fontsize=7.5, ncol=2)
    norm_path = out_dir / "hidden_norm_by_component_site.png"
    figure.savefig(norm_path, dpi=190)
    plt.close(figure)

    group_order = ("all", "graph", "query_metadata", "answer")
    figure, axes = plt.subplots(
        len(group_order),
        len(SITES),
        figsize=(29, 13),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, group in enumerate(group_order):
        for column, site in enumerate(SITES):
            axis = axes[row_index, column]
            for condition in CONDITIONS:
                values = [
                    mean_metric[
                        (condition, int(cycle), site, group, "norm")
                    ]
                    for cycle in cycles
                ]
                axis.plot(
                    cycles,
                    values,
                    color=colors[condition],
                    linewidth=(
                        1.5
                        if condition in ("learned_J", "exact_H7")
                        else 0.9
                    ),
                    alpha=0.9 if condition != "no_J" else 0.65,
                )
            if row_index == 0:
                axis.set_title(site.replace("_", " "), fontsize=8.5)
            if column == 0:
                axis.set_ylabel(f"{group}\nmean L2 norm", fontsize=8)
            if row_index == len(group_order) - 1:
                axis.set_xlabel("continuation loop", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.14)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[condition],
            linewidth=2,
            label=labels[condition],
        )
        for condition in CONDITIONS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(CONDITIONS),
        fontsize=9,
    )
    figure.suptitle(
        "All hidden-state and update norms: every component site × position role",
        fontsize=14,
    )
    all_norm_path = out_dir / "all_hidden_norm_trajectories.png"
    figure.savefig(all_norm_path, dpi=190)
    plt.close(figure)

    learned_accuracy = np.asarray(
        [mean_accuracy[("learned_J", int(cycle))] for cycle in cycles]
    )
    color_map = plt.get_cmap("viridis")
    color_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    figure, axes = plt.subplots(
        len(group_order),
        len(SITES),
        figsize=(29, 13),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, group in enumerate(group_order):
        for column, site in enumerate(SITES):
            axis = axes[row_index, column]
            learned_norm = np.asarray(
                [
                    mean_metric[
                        ("learned_J", int(cycle), site, group, "norm")
                    ]
                    for cycle in cycles
                ]
            )
            oracle_norm = np.asarray(
                [
                    mean_metric[
                        ("exact_H7", int(cycle), site, group, "norm")
                    ]
                    for cycle in cycles
                ]
            )
            axis.plot(
                cycles,
                oracle_norm,
                color="black",
                linestyle="--",
                linewidth=1.1,
                label="exact H7 norm",
            )
            axis.plot(
                cycles,
                learned_norm,
                color="#9e9e9e",
                linewidth=0.8,
                zorder=1,
            )
            axis.scatter(
                cycles,
                learned_norm,
                c=learned_accuracy,
                cmap=color_map,
                norm=color_norm,
                s=9,
                linewidths=0,
                zorder=2,
            )
            if row_index == 0:
                axis.set_title(site.replace("_", " "), fontsize=8.5)
            if column == 0:
                axis.set_ylabel(f"{group}\nmean L2 norm", fontsize=8)
            if row_index == len(group_order) - 1:
                axis.set_xlabel("continuation loop", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.14)
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axes.ravel().tolist(),
        shrink=0.78,
        pad=0.01,
    )
    color_bar.set_label("Learned-J current-step accuracy")
    figure.suptitle(
        (
            "Learned-J norm at every site; dot color is accuracy at the same "
            "loop (black dashed = exact H7)"
        ),
        fontsize=14,
    )
    colored_path = out_dir / "all_norms_accuracy_colored.png"
    figure.savefig(colored_path, dpi=190)
    plt.close(figure)

    split_group_paths = []
    for group in group_order:
        figure, axes = plt.subplots(
            3,
            3,
            figsize=(15, 11),
            sharex=True,
            constrained_layout=True,
        )
        for axis, site in zip(axes.ravel(), SITES, strict=True):
            learned_norm = np.asarray(
                [
                    mean_metric[
                        ("learned_J", int(cycle), site, group, "norm")
                    ]
                    for cycle in cycles
                ]
            )
            oracle_norm = np.asarray(
                [
                    mean_metric[
                        ("exact_H7", int(cycle), site, group, "norm")
                    ]
                    for cycle in cycles
                ]
            )
            axis.plot(
                cycles,
                oracle_norm,
                color="black",
                linestyle="--",
                linewidth=1.25,
            )
            axis.plot(
                cycles,
                learned_norm,
                color="#9e9e9e",
                linewidth=0.9,
                zorder=1,
            )
            axis.scatter(
                cycles,
                learned_norm,
                c=learned_accuracy,
                cmap=color_map,
                norm=color_norm,
                s=13,
                linewidths=0,
                zorder=2,
            )
            axis.set_title(site.replace("_", " "), fontsize=10)
            axis.set_xlabel("continuation loop")
            axis.set_ylabel("mean L2 norm")
            axis.grid(alpha=0.16)
        color_bar = figure.colorbar(
            plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
            ax=axes.ravel().tolist(),
            shrink=0.78,
            pad=0.01,
        )
        color_bar.set_label("Learned-J current-step accuracy")
        figure.suptitle(
            (
                f"Learned-J norm trajectories: {group} positions "
                "(black dashed = exact H7)"
            ),
            fontsize=14,
        )
        split_path = out_dir / f"norm_trajectories_{group}.png"
        figure.savefig(split_path, dpi=190)
        plt.close(figure)
        split_group_paths.append(split_path.name)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        sharex=True,
        constrained_layout=True,
    )
    for column, group in enumerate(("all", "answer")):
        for site, linestyle in (
            ("post_J_pre_Block2", "-"),
            ("post_Block2_attention", "--"),
            ("loop_output", ":"),
        ):
            norm_error = [
                mean_metric[
                    (
                        "learned_J",
                        int(cycle),
                        site,
                        group,
                        "oracle_abs_norm_fraction_error",
                    )
                ]
                for cycle in cycles
            ]
            cosine = [
                mean_metric[
                    (
                        "learned_J",
                        int(cycle),
                        site,
                        group,
                        "oracle_cosine",
                    )
                ]
                for cycle in cycles
            ]
            axes[0, column].plot(
                cycles,
                norm_error,
                linestyle=linestyle,
                linewidth=1.8,
                label=site,
            )
            axes[1, column].plot(
                cycles,
                cosine,
                linestyle=linestyle,
                linewidth=1.8,
                label=site,
            )
        axes[0, column].set_title(f"Learned J vs exact H7: {group}")
        axes[0, column].set_ylabel("Absolute norm fraction error")
        axes[1, column].set_ylabel("Hidden-state cosine")
        axes[1, column].set_xlabel("Continuation loop after H8")
        axes[0, column].grid(alpha=0.18)
        axes[1, column].grid(alpha=0.18)
    axes[0, 1].legend(fontsize=7.5)
    error_path = out_dir / "learned_J_norm_vs_direction_drift.png"
    figure.savefig(error_path, dpi=190)
    plt.close(figure)
    return [
        accuracy_path.name,
        norm_path.name,
        all_norm_path.name,
        colored_path.name,
        *split_group_paths,
        error_path.name,
    ]


def _window_mean(
    curve: dict[int, float],
    start: int,
    end: int,
) -> float | None:
    values = [curve[cycle] for cycle in range(start, end + 1) if cycle in curve]
    return float(np.mean(values)) if values else None


def _first_below(curve: dict[int, float], threshold: float) -> int | None:
    return next(
        (
            cycle
            for cycle in sorted(curve)
            if cycle > 8 and curve[cycle] < threshold
        ),
        None,
    )


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def _diagnostics(
    *,
    accuracy_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    accuracy = _mean_rows(
        accuracy_rows,
        keys=("condition", "cycle"),
        value="accuracy",
    )
    metrics = _mean_rows(
        metric_rows,
        keys=(
            "condition",
            "cycle",
            "site",
            "position_group",
            "metric",
        ),
        value="mean",
    )
    cycles = sorted(
        int(row["cycle"])
        for row in accuracy_rows
        if row["condition"] == "learned_J"
    )
    cycles = sorted(set(cycles))
    curves = {
        condition: {
            cycle: accuracy[(condition, cycle)] for cycle in cycles
        }
        for condition in CONDITIONS
    }
    windows = {}
    for condition in CONDITIONS:
        windows[condition] = {
            f"auc_{start}_{end}": _window_mean(
                curves[condition],
                start,
                min(end, cycles[-1]),
            )
            for start, end in WINDOWS
            if start <= cycles[-1]
        }
    learned_curve = curves["learned_J"]
    post_j_norm_error = [
        metrics[
            (
                "learned_J",
                cycle,
                "post_J_pre_Block2",
                "answer",
                "oracle_abs_norm_fraction_error",
            )
        ]
        for cycle in cycles
    ]
    post_j_cosine = [
        metrics[
            (
                "learned_J",
                cycle,
                "post_J_pre_Block2",
                "answer",
                "oracle_cosine",
            )
        ]
        for cycle in cycles
    ]
    learned_accuracy = [learned_curve[cycle] for cycle in cycles]
    late_start = min(49, cycles[-1])
    late_cycles = [cycle for cycle in cycles if cycle >= late_start]
    causal_rescue = {}
    for condition in ("J_global_normmatched", "J_token_normmatched"):
        differences = [
            curves[condition][cycle] - learned_curve[cycle]
            for cycle in late_cycles
        ]
        causal_rescue[condition] = {
            "mean_accuracy_delta_from_learned_J_late": float(
                np.mean(differences)
            ),
            "maximum_cycle_accuracy_delta_from_learned_J_late": float(
                np.max(differences)
            ),
        }
    return {
        "accuracy_windows": windows,
        "learned_J_first_cycle_below": {
            str(threshold): _first_below(learned_curve, threshold)
            for threshold in (0.95, 0.9, 0.8)
        },
        "cycle_level_correlations": {
            "accuracy_vs_postJ_answer_abs_norm_error": _correlation(
                learned_accuracy,
                post_j_norm_error,
            ),
            "accuracy_vs_postJ_answer_cosine": _correlation(
                learned_accuracy,
                post_j_cosine,
            ),
        },
        "norm_causal_rescue": causal_rescue,
        "selected_cycle_snapshots": {
            str(cycle): {
                "accuracy": learned_curve[cycle],
                "postJ_answer_norm_ratio": metrics[
                    (
                        "learned_J",
                        cycle,
                        "post_J_pre_Block2",
                        "answer",
                        "oracle_norm_ratio",
                    )
                ],
                "postJ_answer_abs_norm_error": metrics[
                    (
                        "learned_J",
                        cycle,
                        "post_J_pre_Block2",
                        "answer",
                        "oracle_abs_norm_fraction_error",
                    )
                ],
                "postJ_answer_cosine": metrics[
                    (
                        "learned_J",
                        cycle,
                        "post_J_pre_Block2",
                        "answer",
                        "oracle_cosine",
                    )
                ],
                "loop_output_answer_norm_ratio": metrics[
                    (
                        "learned_J",
                        cycle,
                        "loop_output",
                        "answer",
                        "oracle_norm_ratio",
                    )
                ],
                "loop_output_answer_cosine": metrics[
                    (
                        "learned_J",
                        cycle,
                        "loop_output",
                        "answer",
                        "oracle_cosine",
                    )
                ],
            }
            for cycle in (1, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96)
            if cycle in learned_curve
        },
    }


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
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
    shared_gpu: bool,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    replica_results = []
    all_accuracy_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_separation_rows: list[dict[str, Any]] = []
    for sample_seed in sample_seeds:
        result = _run_replica(
            replica_seed=int(sample_seed),
            permutations=samples[int(sample_seed)],
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            age_map=age_map,
            device=device,
            batch_size=batch_size,
            continuation_loops=continuation_loops,
            operating_age=operating_age,
            out_dir=out_dir,
        )
        all_accuracy_rows.extend(result.pop("accuracy_rows"))
        all_metric_rows.extend(result.pop("metric_rows"))
        all_separation_rows.extend(result.pop("separation_rows"))
        replica_results.append(result)
    _write_csv(out_dir / "accuracy_by_loop_all_replicas.csv", all_accuracy_rows)
    _write_csv(
        out_dir / "norm_metrics_by_loop_all_replicas.csv",
        all_metric_rows,
    )
    _write_csv(
        out_dir / "correct_vs_incorrect_all_replicas.csv",
        all_separation_rows,
    )
    figures = _plot_results(
        accuracy_rows=all_accuracy_rows,
        metric_rows=all_metric_rows,
        out_dir=out_dir,
    )
    diagnostics = _diagnostics(
        accuracy_rows=all_accuracy_rows,
        metric_rows=all_metric_rows,
    )
    payload = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "J_artifact": str(j_artifact),
        "J_label": j_label,
        "loss_placement": (
            "no training in this diagnostic; frozen backbone was trained "
            "with final-only CE at its sampled training horizon; task-aware "
            "J was previously trained with interface/state losses plus "
            "current-step graph CE"
        ),
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "operating_age": operating_age,
        "graph_steps_per_continuation_loop": (
            phase_positions[3] - phase_positions[2]
        ),
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
        "conditions": {
            "learned_J": "task-aware affine J before Block2 every loop",
            "J_global_normmatched": (
                "same J direction, rescaled to exact-H7 whole-interface norm"
            ),
            "J_token_normmatched": (
                "same J direction, each token rescaled to its exact-H7 norm"
            ),
            "exact_H7": "exact same-graph/current H7 interface every loop",
            "no_J": "uncontrolled overloop",
        },
        "sites": list(SITES),
        "position_groups": list(_position_groups(cfg.node_count, cfg.seq_len)),
        "replicas": replica_results,
        "diagnostics": diagnostics,
        "claim_ledger": [
            {
                "claim": (
                    "learned-J lifespan failure is primarily caused by "
                    "hidden-state norm drift"
                ),
                "status": "tested",
                "evidence": (
                    "norm/error timing plus global- and token-norm-matched "
                    "causal controls"
                ),
                "revise_if": (
                    "norm matching does not materially improve late-loop "
                    "accuracy while direction/error continues to drift"
                ),
            },
            {
                "claim": (
                    "directional/circuit-state drift remains after scale is "
                    "corrected"
                ),
                "status": "tested",
                "evidence": (
                    "cosine and relative-L2 trajectories at pre-Block2, "
                    "attention, MLP, and loop-output sites"
                ),
                "revise_if": (
                    "tokenwise norm matching restores exact-H7-like accuracy"
                ),
            },
        ],
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "shared_gpu": shared_gpu,
            "peak_cuda_reserved_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "accuracy": "accuracy_by_loop_all_replicas.csv",
            "norm_metrics": "norm_metrics_by_loop_all_replicas.csv",
            "correctness_separation": (
                "correct_vs_incorrect_all_replicas.csv"
            ),
            "figures": figures,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose task-aware J lifespan failure from hidden-state norms."
        )
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--continuation-loops", type=int, default=96)
    parser.add_argument("--operating-age", type=int, default=7)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=4.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
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
        physical_gpu=args.physical_gpu,
        prelaunch_used_mib=args.prelaunch_used_mib,
        prelaunch_free_mib=args.prelaunch_free_mib,
        declared_peak_gib=args.declared_peak_gib,
        reserve_gib=args.reserve_gib,
        shared_gpu=args.shared_gpu,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
