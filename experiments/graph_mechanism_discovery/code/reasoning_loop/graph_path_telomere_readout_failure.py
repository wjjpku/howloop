from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_telomere_age_drift_pca import (
    _condition_patch,
    run_age_step,
)
from reasoning_loop.graph_path_telomere_angular_rescue import (
    _select_disjoint_unseen_eight_cycles,
)
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
    reconstruct_primary_training_graphs,
)
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


SITE_NAMES = (
    "loop_input",
    "b1_output_preJ",
    "b2_input_postJ",
    "loop_output",
)
PATCH_COMPONENTS = ("B2H0_answer", "B2MLP_answer")
PATCH_MODES = ("young", "graph_shuffled", "start_shuffled", "zero")
DIRECT_COMPONENTS = (
    "B2H0",
    "B2H1",
    "B2H2",
    "B2H3",
    "B2AttentionTotal",
    "B2MLP",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _orbit_nodes(
    successors: torch.Tensor,
    origin: torch.Tensor,
) -> torch.Tensor:
    nodes = []
    current = origin
    for _ in range(successors.shape[1]):
        nodes.append(current)
        current = successors.gather(1, current[:, None]).squeeze(1)
    orbit = torch.stack(nodes, dim=1)
    if torch.sort(orbit, dim=1).values.ne(
        torch.arange(
            successors.shape[1],
            device=successors.device,
        )[None]
    ).any():
        raise ValueError("the selected graph is not one full cycle")
    return orbit


def _prediction_offsets_np(
    predictions: np.ndarray,
    orbit: np.ndarray,
) -> np.ndarray:
    matches = predictions[:, None] == orbit
    if not matches.any(axis=1).all():
        raise ValueError("prediction is outside the node orbit")
    return matches.argmax(axis=1)


def _centered_cosine_np(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_centered = left - left.mean(axis=-1, keepdims=True)
    right_centered = right - right.mean(axis=-1, keepdims=True)
    numerator = (left_centered * right_centered).sum(axis=-1)
    denominator = np.linalg.norm(left_centered, axis=-1) * np.linalg.norm(
        right_centered,
        axis=-1,
    )
    return numerator / np.maximum(denominator, 1e-12)


def _cycle_metrics(
    *,
    logits: np.ndarray,
    orbit: np.ndarray,
    cycle: int,
    young_phase_logits: np.ndarray,
) -> dict[str, float]:
    probabilities = _softmax_np(logits)
    prediction = logits.argmax(axis=-1)
    prediction_offset = _prediction_offsets_np(prediction, orbit)
    target_offset = cycle % orbit.shape[1]
    current_offset = (cycle - 1) % orbit.shape[1]
    target = orbit[:, target_offset]
    current = orbit[:, current_offset]
    row = np.arange(logits.shape[0])
    target_logits = logits[row, target]
    current_logits = logits[row, current]
    target_probability = probabilities[row, target]
    current_probability = probabilities[row, current]
    strongest_wrong = logits.copy()
    strongest_wrong[row, target] = -np.inf
    margin = target_logits - strongest_wrong.max(axis=-1)
    target_rank = 1 + (logits > target_logits[:, None]).sum(axis=-1)
    entropy = -(
        probabilities * np.log(np.maximum(probabilities, 1e-12))
    ).sum(axis=-1)
    top1_probability = probabilities.max(axis=-1)
    reference = young_phase_logits[:, (cycle - 1) % orbit.shape[1]]
    return {
        "accuracy": float(np.mean(prediction == target)),
        "current_accuracy": float(np.mean(prediction == current)),
        "target_probability": float(target_probability.mean()),
        "current_probability": float(current_probability.mean()),
        "top1_probability": float(top1_probability.mean()),
        "target_margin": float(margin.mean()),
        "target_rank": float(target_rank.mean()),
        "entropy": float(entropy.mean()),
        "normalized_entropy": float(
            entropy.mean() / math.log(logits.shape[-1])
        ),
        "centered_logit_cosine_to_young_same_phase": float(
            _centered_cosine_np(logits, reference).mean()
        ),
        "logit_distance_to_young_same_phase": float(
            np.linalg.norm(logits - reference, axis=-1).mean()
        ),
        "prediction_offset_mode": int(
            np.bincount(
                prediction_offset,
                minlength=orbit.shape[1],
            ).argmax()
        ),
        "prediction_error_mode": int(
            np.bincount(
                (prediction_offset - target_offset) % orbit.shape[1],
                minlength=orbit.shape[1],
            ).argmax()
        ),
    }


def _offset_rows(
    *,
    logits: np.ndarray,
    orbit: np.ndarray,
    cycle: int,
    site: str,
    replica: int | str,
) -> list[dict[str, Any]]:
    probabilities = _softmax_np(logits)
    prediction = logits.argmax(axis=-1)
    prediction_offset = _prediction_offsets_np(prediction, orbit)
    orbit_probability = np.take_along_axis(
        probabilities,
        orbit,
        axis=1,
    )
    target_offset = cycle % orbit.shape[1]
    rows = []
    for offset in range(orbit.shape[1]):
        rows.append(
            {
                "replica": replica,
                "cycle": cycle,
                "site": site,
                "semantic_offset": offset,
                "target_relative_error": (
                    offset - target_offset
                ) % orbit.shape[1],
                "top1_fraction": float(
                    np.mean(prediction_offset == offset)
                ),
                "mean_probability_mass": float(
                    orbit_probability[:, offset].mean()
                ),
            }
        )
    return rows


def _absolute_node_rows(
    *,
    logits: np.ndarray,
    targets: np.ndarray,
    cycle: int,
    replica: int | str,
) -> list[dict[str, Any]]:
    probabilities = _softmax_np(logits)
    predictions = logits.argmax(axis=-1)
    centered_logits = logits - logits.mean(axis=-1, keepdims=True)
    rows = []
    for node in range(logits.shape[-1]):
        rows.append(
            {
                "replica": replica,
                "cycle": cycle,
                "node": node,
                "target_fraction": float(np.mean(targets == node)),
                "top1_fraction": float(np.mean(predictions == node)),
                "mean_probability": float(probabilities[:, node].mean()),
                "mean_logit": float(logits[:, node].mean()),
                "mean_centered_logit": float(
                    centered_logits[:, node].mean()
                ),
            }
        )
    return rows


def _trace_site_state(trace, site: str) -> torch.Tensor:
    if site not in SITE_NAMES:
        raise ValueError(f"unknown trace site {site}")
    return getattr(trace, site)


def _head_projected_vector(
    *,
    block,
    context: torch.Tensor,
    head: int,
    answer_position: int,
) -> torch.Tensor:
    d_head = block.attn.d_head
    start = head * d_head
    stop = start + d_head
    weight = block.attn.out_proj.weight[:, start:stop]
    return context[:, head, answer_position] @ weight.T


def _direct_component_vectors(
    *,
    model,
    trace,
    answer_position: int,
) -> dict[str, torch.Tensor]:
    block = model.blocks[1]
    block_trace = trace.blocks[1]
    vectors = {
        f"B2H{head}": _head_projected_vector(
            block=block,
            context=block_trace.context,
            head=head,
            answer_position=answer_position,
        )
        for head in range(model.cfg.n_heads)
    }
    vectors["B2AttentionTotal"] = block_trace.attention_out[
        :, answer_position
    ]
    vectors["B2MLP"] = block_trace.mlp_out[:, answer_position]
    return vectors


def _direct_readout_rows(
    *,
    model,
    trace,
    orbit: torch.Tensor,
    target: torch.Tensor,
    cycle: int,
    replica: int,
    answer_position: int,
) -> list[dict[str, Any]]:
    if model.outer_norm is not None:
        raise ValueError(
            "direct residual subtraction assumes no post-stack outer norm"
        )
    base_logits = trace.logits.float()
    batch = torch.arange(base_logits.shape[0], device=base_logits.device)
    wrong_mask = F.one_hot(
        target,
        num_classes=base_logits.shape[-1],
    ).bool()
    strongest_wrong = base_logits.masked_fill(
        wrong_mask,
        float("-inf"),
    ).argmax(dim=-1)
    rows: list[dict[str, Any]] = []
    for component, vector in _direct_component_vectors(
        model=model,
        trace=trace,
        answer_position=answer_position,
    ).items():
        removed_state = trace.loop_output.clone()
        removed_state[:, answer_position] -= vector.to(
            removed_state.dtype
        )
        removed_logits = logits_from_raw_state(
            model,
            removed_state,
        ).float()
        delta = base_logits - removed_logits
        margin_contribution = (
            delta[batch, target] - delta[batch, strongest_wrong]
        )
        row: dict[str, Any] = {
            "replica": replica,
            "cycle": cycle,
            "component": component,
            "examples": base_logits.shape[0],
            "target_logit_contribution": float(
                delta[batch, target].mean()
            ),
            "target_margin_contribution": float(
                margin_contribution.mean()
            ),
            "direct_delta_logit_norm": float(
                delta.norm(dim=-1).mean()
            ),
        }
        for offset in range(orbit.shape[1]):
            row[f"orbit_offset_{offset}_logit_contribution"] = float(
                delta.gather(
                    1,
                    orbit[:, offset : offset + 1],
                ).mean()
            )
        rows.append(row)
    return rows


def _aggregate_weighted(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    weight: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for group_key, parts in sorted(groups.items()):
        total = sum(float(row[weight]) for row in parts)
        item = dict(zip(keys, group_key, strict=True))
        item[weight] = int(total)
        for field in parts[0]:
            if field in keys or field in {weight, "replica"}:
                continue
            values = [
                float(row[field]) * float(row[weight])
                for row in parts
                if row.get(field) not in (None, "")
            ]
            if values:
                item[field] = sum(values) / total
        result.append(item)
    return result


def _plot_actual_offsets(
    *,
    out_dir: Path,
    offset_fraction: np.ndarray,
) -> str:
    cycles = np.arange(1, offset_fraction.shape[0] + 1)
    error_fraction = np.zeros_like(offset_fraction)
    for cycle_index, cycle in enumerate(cycles):
        target = cycle % offset_fraction.shape[1]
        for offset in range(offset_fraction.shape[1]):
            error = (offset - target) % offset_fraction.shape[1]
            error_fraction[cycle_index, error] += offset_fraction[
                cycle_index,
                offset,
            ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        constrained_layout=True,
    )
    images = []
    images.append(
        axes[0].imshow(
            offset_fraction.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
            extent=(1, len(cycles), -0.5, 7.5),
        )
    )
    axes[0].plot(
        cycles,
        cycles % offset_fraction.shape[1],
        color="cyan",
        linewidth=1.2,
        label="correct offset",
    )
    axes[0].set(
        title="Actual readout: predicted orbit offset",
        ylabel="predicted offset from endpoint",
    )
    axes[0].legend(loc="upper right")
    images.append(
        axes[1].imshow(
            error_fraction.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
            extent=(1, len(cycles), -0.5, 7.5),
        )
    )
    axes[1].axhline(
        0,
        color="cyan",
        linewidth=1.2,
        label="correct",
    )
    axes[1].set(
        title="Actual readout: prediction error modulo 8",
        xlabel="continuation loop",
        ylabel="predicted - target (mod 8)",
    )
    axes[1].legend(loc="upper right")
    fig.colorbar(
        images[-1],
        ax=axes,
        label="top-1 fraction",
        shrink=0.9,
    )
    path = out_dir / "actual_readout_offset_heatmap.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_confidence(
    *,
    out_dir: Path,
    metrics: list[dict[str, Any]],
) -> str:
    cycles = np.array([int(row["cycle"]) for row in metrics])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(
        cycles,
        [row["accuracy"] for row in metrics],
        label="accuracy",
    )
    axes[0].plot(
        cycles,
        [row["target_probability"] for row in metrics],
        label="P(correct)",
    )
    axes[0].plot(
        cycles,
        [row["top1_probability"] for row in metrics],
        label="P(top1)",
    )
    axes[0].axhline(0.125, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("fraction / probability")
    axes[0].legend()
    axes[1].plot(
        cycles,
        [row["target_margin"] for row in metrics],
        color="tab:red",
        label="correct - strongest wrong",
    )
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set_ylabel("logit margin")
    axes[1].legend()
    axes[2].plot(
        cycles,
        [row["normalized_entropy"] for row in metrics],
        label="normalized entropy",
    )
    axes[2].plot(
        cycles,
        [
            row["centered_logit_cosine_to_young_same_phase"]
            for row in metrics
        ],
        label="logit cosine to young same phase",
    )
    axes[2].set(
        xlabel="continuation loop",
        ylabel="normalized value",
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.18)
    path = out_dir / "actual_readout_confidence.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_absolute_nodes(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    continuation_loops: int,
    node_count: int,
) -> str:
    top1 = np.zeros((continuation_loops, node_count), dtype=np.float64)
    probability = np.zeros_like(top1)
    centered_logit = np.zeros_like(top1)
    for row in rows:
        if row["replica"] != "pooled":
            continue
        cycle_index = int(row["cycle"]) - 1
        node = int(row["node"])
        top1[cycle_index, node] = float(row["top1_fraction"])
        probability[cycle_index, node] = float(row["mean_probability"])
        centered_logit[cycle_index, node] = float(
            row["mean_centered_logit"]
        )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        constrained_layout=True,
    )
    matrices = (top1, probability, centered_logit)
    titles = (
        "Absolute node top-1 frequency",
        "Absolute node mean probability",
        "Absolute node mean centered logit",
    )
    cmaps = ("magma", "magma", "coolwarm")
    for axis, matrix, title, cmap in zip(
        axes,
        matrices,
        titles,
        cmaps,
        strict=True,
    ):
        limit = (
            max(0.1, float(np.abs(matrix).max()))
            if cmap == "coolwarm"
            else None
        )
        image = axis.imshow(
            matrix.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-limit if limit is not None else 0,
            vmax=limit if limit is not None else 1,
            extent=(1, continuation_loops, -0.5, node_count - 0.5),
        )
        axis.set(title=title, ylabel="absolute node ID")
        fig.colorbar(image, ax=axis, shrink=0.85)
    axes[-1].set_xlabel("continuation loop")
    path = out_dir / "absolute_node_readout_bias.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_matched_stage_offsets(
    *,
    out_dir: Path,
    matched_cycles: np.ndarray,
    stage_fraction: np.ndarray,
) -> str:
    fig, axes = plt.subplots(
        1,
        len(SITE_NAMES),
        figsize=(16, 6),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for site_index, (axis, site) in enumerate(
        zip(axes, SITE_NAMES, strict=True)
    ):
        image = axis.imshow(
            stage_fraction[:, site_index].T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        axis.set_xticks(
            np.arange(len(matched_cycles)),
            labels=[str(int(value)) for value in matched_cycles],
            rotation=65,
        )
        axis.set_yticks(np.arange(8), labels=[str(i) for i in range(8)])
        axis.set_title(site)
        axis.set_xlabel("matched loop")
    axes[0].set_ylabel("decoded orbit offset (target = 0, current = 7)")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            label="top-1 fraction under final readout lens",
            shrink=0.85,
        )
    path = out_dir / "matched_stage_readout_offsets.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_direct_components(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    matched_cycles: np.ndarray,
) -> str:
    matrix = np.full(
        (len(DIRECT_COMPONENTS), len(matched_cycles)),
        np.nan,
    )
    for row_index, component in enumerate(DIRECT_COMPONENTS):
        for column, cycle in enumerate(matched_cycles):
            selected = [
                row
                for row in rows
                if row["component"] == component
                and int(row["cycle"]) == int(cycle)
            ]
            if selected:
                matrix[row_index, column] = float(
                    selected[0]["target_margin_contribution"]
                )
    limit = max(0.1, float(np.nanmax(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(
        np.arange(len(matched_cycles)),
        labels=[str(int(value)) for value in matched_cycles],
    )
    axis.set_yticks(
        np.arange(len(DIRECT_COMPONENTS)),
        labels=DIRECT_COMPONENTS,
    )
    axis.set(
        xlabel="matched continuation loop",
        title="Direct residual contribution to correct-vs-best-wrong margin",
    )
    fig.colorbar(image, ax=axis, label="logit-margin contribution")
    path = out_dir / "component_direct_readout_contribution.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_patch_offsets(
    *,
    out_dir: Path,
    patch_distributions: dict[
        tuple[int, str, str],
        np.ndarray,
    ],
    baseline_distributions: dict[int, np.ndarray],
) -> str:
    preferred_cycles = [
        cycle
        for cycle in (64, 96)
        if cycle in baseline_distributions
    ]
    cycles = (
        preferred_cycles
        if preferred_cycles
        else sorted(baseline_distributions)[-2:]
    )
    fig, axes = plt.subplots(
        len(cycles),
        len(PATCH_COMPONENTS),
        figsize=(14, 4.8 * max(1, len(cycles))),
        squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(8)
    for row, cycle in enumerate(cycles):
        for column, component in enumerate(PATCH_COMPONENTS):
            axis = axes[row, column]
            axis.plot(
                x,
                baseline_distributions[cycle],
                marker="o",
                label="baseline",
            )
            for mode, style in (
                ("young", "-"),
                ("graph_shuffled", "--"),
                ("start_shuffled", ":"),
                ("zero", "-."),
            ):
                key = (cycle, component, mode)
                if key not in patch_distributions:
                    continue
                axis.plot(
                    x,
                    patch_distributions[key],
                    linestyle=style,
                    marker=".",
                    label=mode,
                )
            axis.set(
                title=f"cycle {cycle}: {component}",
                xlabel="predicted orbit offset (target = 0)",
                ylabel="top-1 fraction",
                ylim=(-0.02, 1.02),
            )
            axis.grid(alpha=0.18)
            axis.legend(fontsize=8)
    path = out_dir / "young_patch_readout_offsets.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    j_artifact: Path,
    j_label: str,
    out_dir: Path,
    device_name: str,
    sample_count: int,
    sample_seeds: Sequence[int],
    batch_size: int,
    continuation_loops: int,
    matched_period: int,
    probe_cycles: Sequence[int],
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
    shared_gpu: bool,
) -> dict[str, Any]:
    from reasoning_loop.graph_path_loop import pick_device

    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
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
        or cfg.n_heads != 4
    ):
        raise ValueError("fixed to N8 D8L8, two blocks, four heads")
    if model.outer_norm is not None:
        raise ValueError("this audit expects the legacy model without outer norm")
    age_map, map_checkpoint = load_unit_j_map(
        j_artifact,
        label=j_label,
        device=device,
    )
    if map_checkpoint != str(checkpoint):
        raise ValueError("J and model checkpoints differ")
    groups = explicit_depth_position_groups(cfg.node_count)
    answer_position = groups["answer"][0]
    map_positions = intervention_groups(cfg.node_count)["all"]
    matched_cycles = np.arange(
        matched_period,
        continuation_loops + 1,
        matched_period,
        dtype=np.int64,
    )
    matched_lookup = {
        int(cycle): index for index, cycle in enumerate(matched_cycles)
    }
    probe_cycles = tuple(
        sorted(
            int(cycle)
            for cycle in probe_cycles
            if int(cycle) in matched_lookup
            and int(cycle) > matched_period
        )
    )

    seen, unique_after_stage, total_training_draws = (
        reconstruct_primary_training_graphs(
            device=device,
            node_count=cfg.node_count,
        )
    )
    selected = _select_disjoint_unseen_eight_cycles(
        seen=seen,
        sample_count=sample_count,
        sample_seeds=sample_seeds,
    )

    replica_payloads: list[dict[str, np.ndarray]] = []
    direct_raw_rows: list[dict[str, Any]] = []
    equivalence_max = defaultdict(float)
    for replica_index, sample_seed in enumerate(sample_seeds):
        successors_all, starts_all = _expand_all_starts(
            selected[int(sample_seed)],
            device=device,
        )
        example_count = successors_all.shape[0]
        actual_logits = np.zeros(
            (example_count, continuation_loops, cfg.node_count),
            dtype=np.float32,
        )
        matched_site_logits = np.zeros(
            (
                example_count,
                len(matched_cycles),
                len(SITE_NAMES),
                cfg.node_count,
            ),
            dtype=np.float32,
        )
        patch_logits = np.full(
            (
                example_count,
                len(probe_cycles),
                len(PATCH_COMPONENTS),
                len(PATCH_MODES),
                cfg.node_count,
            ),
            np.nan,
            dtype=np.float32,
        )
        for offset in range(0, example_count, batch_size):
            successors = successors_all[offset : offset + batch_size]
            starts = starts_all[offset : offset + batch_size]
            batch_n = successors.shape[0]
            endpoint = advance_nodes(
                successors,
                starts,
                steps=cfg.max_depth,
            )
            orbit = _orbit_nodes(successors, endpoint)
            state = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=8,
                phase_position=8,
            )
            donor_trace = None
            for cycle in range(1, continuation_loops + 1):
                target = orbit[:, cycle % cfg.node_count]
                trace = run_age_step(
                    model=model,
                    state=state,
                    loop_index=cfg.max_loops + cycle - 1,
                    age_map=age_map,
                    map_positions=map_positions,
                )
                batch_slice = slice(offset, offset + batch_n)
                actual_logits[
                    batch_slice,
                    cycle - 1,
                ] = trace.logits.float().cpu().numpy()
                if cycle in matched_lookup:
                    matched_index = matched_lookup[cycle]
                    for site_index, site in enumerate(SITE_NAMES):
                        site_logits = logits_from_raw_state(
                            model,
                            _trace_site_state(trace, site),
                        )
                        matched_site_logits[
                            batch_slice,
                            matched_index,
                            site_index,
                        ] = site_logits.float().cpu().numpy()
                    direct_raw_rows.extend(
                        _direct_readout_rows(
                            model=model,
                            trace=trace,
                            orbit=orbit,
                            target=target,
                            cycle=cycle,
                            replica=replica_index,
                            answer_position=answer_position,
                        )
                    )
                if cycle == matched_period:
                    donor_trace = trace
                if cycle in probe_cycles:
                    if donor_trace is None:
                        raise RuntimeError("young matched donor is missing")
                    probe_index = probe_cycles.index(cycle)
                    for component_index, component in enumerate(
                        PATCH_COMPONENTS
                    ):
                        for mode_index, mode in enumerate(PATCH_MODES):
                            changed = run_age_step(
                                model=model,
                                state=state,
                                loop_index=cfg.max_loops + cycle - 1,
                                age_map=age_map,
                                map_positions=map_positions,
                                patch=_condition_patch(
                                    label=component,
                                    mode=mode,
                                    donor_trace=donor_trace,
                                    groups=groups,
                                    graph_stride=cfg.node_count,
                                ),
                            )
                            patch_logits[
                                batch_slice,
                                probe_index,
                                component_index,
                                mode_index,
                            ] = changed.logits.float().cpu().numpy()
                if replica_index == 0 and offset == 0 and cycle == 1:
                    equivalence_max["actual_output"] = float(
                        (
                            trace.logits
                            - logits_from_raw_state(
                                model,
                                trace.loop_output,
                            )
                        ).abs().max()
                    )
                    projected_sum = sum(
                        _direct_component_vectors(
                            model=model,
                            trace=trace,
                            answer_position=answer_position,
                        )[f"B2H{head}"]
                        for head in range(cfg.n_heads)
                    )
                    bias = model.blocks[1].attn.out_proj.bias
                    if bias is not None:
                        projected_sum = projected_sum + bias
                    equivalence_max["head_sum_to_attention"] = float(
                        (
                            projected_sum
                            - trace.blocks[1].attention_out[
                                :, answer_position
                            ]
                        ).abs().max()
                    )
                state = trace.loop_output

        orbit_all = _orbit_nodes(successors_all, starts_all).cpu().numpy()
        replica_dir = out_dir / f"replica_{int(sample_seed)}"
        replica_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            replica_dir / "readout_raw_balanced.npz",
            successors=successors_all.cpu().numpy(),
            starts=starts_all.cpu().numpy(),
            orbit=orbit_all,
            cycles=np.arange(1, continuation_loops + 1),
            matched_cycles=matched_cycles,
            site_names=np.array(SITE_NAMES),
            actual_logits=actual_logits,
            matched_site_logits=matched_site_logits,
            probe_cycles=np.array(probe_cycles),
            patch_components=np.array(PATCH_COMPONENTS),
            patch_modes=np.array(PATCH_MODES),
            patch_logits=patch_logits,
        )
        replica_payloads.append(
            {
                "successors": successors_all.cpu().numpy(),
                "starts": starts_all.cpu().numpy(),
                "orbit": orbit_all,
                "actual_logits": actual_logits,
                "matched_site_logits": matched_site_logits,
                "patch_logits": patch_logits,
            }
        )

    if max(equivalence_max.values(), default=0.0) > 2e-5:
        raise RuntimeError(f"readout trace equivalence failed: {equivalence_max}")

    pooled_orbit = np.concatenate(
        [item["orbit"] for item in replica_payloads],
        axis=0,
    )
    pooled_actual = np.concatenate(
        [item["actual_logits"] for item in replica_payloads],
        axis=0,
    )
    pooled_sites = np.concatenate(
        [item["matched_site_logits"] for item in replica_payloads],
        axis=0,
    )
    pooled_patch = np.concatenate(
        [item["patch_logits"] for item in replica_payloads],
        axis=0,
    )

    metrics_by_replica: list[dict[str, Any]] = []
    actual_offset_rows: list[dict[str, Any]] = []
    absolute_node_rows: list[dict[str, Any]] = []
    for replica_index, item in enumerate(replica_payloads):
        young_phase = item["actual_logits"][:, : cfg.node_count]
        for cycle in range(1, continuation_loops + 1):
            metrics_by_replica.append(
                {
                    "replica": replica_index,
                    "cycle": cycle,
                    **_cycle_metrics(
                        logits=item["actual_logits"][:, cycle - 1],
                        orbit=item["orbit"],
                        cycle=cycle,
                        young_phase_logits=young_phase,
                    ),
                }
            )
            actual_offset_rows.extend(
                _offset_rows(
                    logits=item["actual_logits"][:, cycle - 1],
                    orbit=item["orbit"],
                    cycle=cycle,
                    site="actual_loop_output",
                    replica=replica_index,
                )
            )
            absolute_node_rows.extend(
                _absolute_node_rows(
                    logits=item["actual_logits"][:, cycle - 1],
                    targets=item["orbit"][:, cycle % cfg.node_count],
                    cycle=cycle,
                    replica=replica_index,
                )
            )

    pooled_young_phase = pooled_actual[:, : cfg.node_count]
    pooled_metrics = [
        {
            "replica": "pooled",
            "cycle": cycle,
            **_cycle_metrics(
                logits=pooled_actual[:, cycle - 1],
                orbit=pooled_orbit,
                cycle=cycle,
                young_phase_logits=pooled_young_phase,
            ),
        }
        for cycle in range(1, continuation_loops + 1)
    ]
    for cycle in range(1, continuation_loops + 1):
        actual_offset_rows.extend(
            _offset_rows(
                logits=pooled_actual[:, cycle - 1],
                orbit=pooled_orbit,
                cycle=cycle,
                site="actual_loop_output",
                replica="pooled",
            )
        )
        absolute_node_rows.extend(
            _absolute_node_rows(
                logits=pooled_actual[:, cycle - 1],
                targets=pooled_orbit[:, cycle % cfg.node_count],
                cycle=cycle,
                replica="pooled",
            )
        )

    matched_stage_rows: list[dict[str, Any]] = []
    matched_stage_offset_rows: list[dict[str, Any]] = []
    stage_fraction = np.zeros(
        (len(matched_cycles), len(SITE_NAMES), cfg.node_count),
        dtype=np.float64,
    )
    for matched_index, cycle in enumerate(matched_cycles):
        for site_index, site in enumerate(SITE_NAMES):
            logits = pooled_sites[:, matched_index, site_index]
            matched_stage_rows.append(
                {
                    "cycle": int(cycle),
                    "site": site,
                    **_cycle_metrics(
                        logits=logits,
                        orbit=pooled_orbit,
                        cycle=int(cycle),
                        young_phase_logits=np.repeat(
                            pooled_sites[
                                :, 0, site_index
                            ][:, None],
                            cfg.node_count,
                            axis=1,
                        ),
                    ),
                }
            )
            parts = _offset_rows(
                logits=logits,
                orbit=pooled_orbit,
                cycle=int(cycle),
                site=site,
                replica="pooled",
            )
            matched_stage_offset_rows.extend(parts)
            stage_fraction[matched_index, site_index] = [
                row["top1_fraction"] for row in parts
            ]

    direct_rows = _aggregate_weighted(
        direct_raw_rows,
        keys=("cycle", "component"),
        weight="examples",
    )
    patch_metric_rows: list[dict[str, Any]] = []
    patch_offset_rows: list[dict[str, Any]] = []
    patch_distributions: dict[tuple[int, str, str], np.ndarray] = {}
    baseline_distributions: dict[int, np.ndarray] = {}
    for probe_index, cycle in enumerate(probe_cycles):
        baseline_logits = pooled_actual[:, cycle - 1]
        baseline_parts = _offset_rows(
            logits=baseline_logits,
            orbit=pooled_orbit,
            cycle=cycle,
            site="baseline",
            replica="pooled",
        )
        baseline_distributions[cycle] = np.array(
            [row["top1_fraction"] for row in baseline_parts]
        )
        for component_index, component in enumerate(PATCH_COMPONENTS):
            for mode_index, mode in enumerate(PATCH_MODES):
                logits = pooled_patch[
                    :,
                    probe_index,
                    component_index,
                    mode_index,
                ]
                patch_metric_rows.append(
                    {
                        "cycle": cycle,
                        "component": component,
                        "mode": mode,
                        **_cycle_metrics(
                            logits=logits,
                            orbit=pooled_orbit,
                            cycle=cycle,
                            young_phase_logits=np.repeat(
                                pooled_actual[
                                    :, matched_period - 1
                                ][:, None],
                                cfg.node_count,
                                axis=1,
                            ),
                        ),
                    }
                )
                parts = _offset_rows(
                    logits=logits,
                    orbit=pooled_orbit,
                    cycle=cycle,
                    site=f"{component}:{mode}",
                    replica="pooled",
                )
                patch_offset_rows.extend(parts)
                patch_distributions[(cycle, component, mode)] = np.array(
                    [row["top1_fraction"] for row in parts]
                )

    actual_offset_fraction = np.zeros(
        (continuation_loops, cfg.node_count),
        dtype=np.float64,
    )
    pooled_actual_offsets = [
        row
        for row in actual_offset_rows
        if row["replica"] == "pooled"
    ]
    for row in pooled_actual_offsets:
        actual_offset_fraction[
            int(row["cycle"]) - 1,
            int(row["semantic_offset"]),
        ] = float(row["top1_fraction"])

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "actual_readout_metrics_by_replica.csv",
        metrics_by_replica,
    )
    _write_csv(out_dir / "actual_readout_metrics.csv", pooled_metrics)
    _write_csv(
        out_dir / "actual_readout_offset_distribution.csv",
        actual_offset_rows,
    )
    _write_csv(
        out_dir / "absolute_node_readout.csv",
        absolute_node_rows,
    )
    _write_csv(
        out_dir / "matched_stage_readout_metrics.csv",
        matched_stage_rows,
    )
    _write_csv(
        out_dir / "matched_stage_offset_distribution.csv",
        matched_stage_offset_rows,
    )
    _write_csv(
        out_dir / "component_direct_readout.csv",
        direct_rows,
    )
    _write_csv(out_dir / "young_patch_readout_metrics.csv", patch_metric_rows)
    _write_csv(
        out_dir / "young_patch_offset_distribution.csv",
        patch_offset_rows,
    )
    figures = {
        "actual_offsets": _plot_actual_offsets(
            out_dir=out_dir,
            offset_fraction=actual_offset_fraction,
        ),
        "actual_confidence": _plot_confidence(
            out_dir=out_dir,
            metrics=pooled_metrics,
        ),
        "absolute_nodes": _plot_absolute_nodes(
            out_dir=out_dir,
            rows=absolute_node_rows,
            continuation_loops=continuation_loops,
            node_count=cfg.node_count,
        ),
        "matched_stage_offsets": _plot_matched_stage_offsets(
            out_dir=out_dir,
            matched_cycles=matched_cycles,
            stage_fraction=stage_fraction,
        ),
        "direct_components": _plot_direct_components(
            out_dir=out_dir,
            rows=direct_rows,
            matched_cycles=matched_cycles,
        ),
        "patch_offsets": _plot_patch_offsets(
            out_dir=out_dir,
            patch_distributions=patch_distributions,
            baseline_distributions=baseline_distributions,
        ),
    }

    selected_cycles = [
        cycle
        for cycle in (8, 48, 56, 64, 72, 80, 96)
        if cycle <= continuation_loops
    ]
    metric_lookup = {
        int(row["cycle"]): row for row in pooled_metrics
    }
    payload = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "J_artifact": str(j_artifact),
        "J_label": j_label,
        "readout_definition": (
            "actual readout is LN_final(answer position 28) -> unembed -> "
            "node logits 0..7; intermediate-site results reuse this fixed "
            "head as a descriptive logit lens, not a trained probe"
        ),
        "unembedding_row_norms": (
            model.unembed.weight[: cfg.node_count]
            .float()
            .norm(dim=-1)
            .cpu()
            .tolist()
        ),
        "loss_placement": (
            "frozen final-only D8L8 seed0; no training in this audit"
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_loops": continuation_loops,
        "random_baseline": 1.0 / cfg.node_count,
        "dataset": {
            "replica_seeds": list(sample_seeds),
            "strictly_unseen_single_8_cycles_per_replica": sample_count,
            "all_starts_per_graph": cfg.node_count,
            "examples_per_replica": sample_count * cfg.node_count,
            "fixed_answer_position": answer_position,
            "primary_training_draws": total_training_draws,
            "unique_primary_training_graphs": unique_after_stage,
        },
        "selected_actual_readout_metrics": {
            str(cycle): metric_lookup[cycle]
            for cycle in selected_cycles
        },
        "matched_stage_readout_metrics": matched_stage_rows,
        "component_direct_readout": direct_rows,
        "young_patch_readout_metrics": patch_metric_rows,
        "equivalence_max_abs_error": dict(equivalence_max),
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "shared": shared_gpu,
            "peak_reserved_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "claim_ledger": [
            {
                "claim": (
                    "late learned-J failure is a confident semantic phase "
                    "error rather than unstructured readout collapse"
                ),
                "status": "tested descriptively by full logit and orbit-offset "
                "distributions",
                "evidence": (
                    "actual_readout_metrics.csv and "
                    "actual_readout_offset_distribution.csv"
                ),
            },
            {
                "claim": (
                    "the answer is already linearly readable before Block 2"
                ),
                "status": "localization only",
                "evidence": (
                    "matched-stage fixed-head logit lens; causal use requires "
                    "separate interventions"
                ),
            },
            {
                "claim": (
                    "young B2 H0 / MLP restore the semantic readout target"
                ),
                "status": "component-level causal role under matched patching",
                "evidence": (
                    "young patch versus same-graph wrong-start, "
                    "cross-graph same-target, and zero controls"
                ),
            },
            {
                "claim": "this is the complete or unique readout circuit",
                "status": "not established",
                "evidence_needed": (
                    "joint necessity, circuit-only/complement tests, and "
                    "alternative-path search"
                ),
            },
        ],
        "files": {
            "actual_metrics": "actual_readout_metrics.csv",
            "actual_offsets": "actual_readout_offset_distribution.csv",
            "absolute_nodes": "absolute_node_readout.csv",
            "stage_metrics": "matched_stage_readout_metrics.csv",
            "stage_offsets": "matched_stage_offset_distribution.csv",
            "direct_components": "component_direct_readout.csv",
            "patch_metrics": "young_patch_readout_metrics.csv",
            "patch_offsets": "young_patch_offset_distribution.csv",
            "figures": figures,
            "raw_replica_pattern": "replica_*/readout_raw_balanced.npz",
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
            "Semantic and component-level readout audit for learned-J "
            "late-horizon failure."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--j-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument(
        "--sample-seeds",
        type=int,
        nargs="+",
        default=(20260811, 20260812),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--continuation-loops", type=int, default=96)
    parser.add_argument("--matched-period", type=int, default=8)
    parser.add_argument(
        "--probe-cycles",
        type=int,
        nargs="+",
        default=(48, 64, 80, 96),
    )
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=2.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        j_artifact=args.j_artifact,
        j_label=args.j_label,
        out_dir=args.out_dir,
        device_name=args.device,
        sample_count=args.sample_count,
        sample_seeds=args.sample_seeds,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        matched_period=args.matched_period,
        probe_cycles=args.probe_cycles,
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
