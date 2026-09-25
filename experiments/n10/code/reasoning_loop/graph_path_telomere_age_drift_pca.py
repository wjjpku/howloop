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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    _attention_parts,
    _attention_pattern,
    _project_context,
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import TransformerBlock, pick_device
from reasoning_loop.graph_path_telomere_angular_rescue import (
    _select_disjoint_unseen_eight_cycles,
)
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    _expand_all_starts,
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
from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map
from reasoning_loop.graph_path_temporal_intervention import (
    logits_from_raw_state,
)


STATE_SITES = (
    "loop_input",
    "b1_output_preJ",
    "b2_input_postJ",
    "loop_output",
)
COMPONENT_LABELS = (
    "B1H0_graph",
    "B1H1_graph",
    "B1H2_graph",
    "B1H3_graph",
    "B1MLP_graph",
    "B2H0_answer",
    "B2H1_answer",
    "B2H2_answer",
    "B2H3_answer",
    "B2MLP_answer",
)


@dataclass
class BlockTrace:
    hidden_in: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    pattern: torch.Tensor
    context: torch.Tensor
    attention_out: torch.Tensor
    residual_mid: torch.Tensor
    mlp_hidden: torch.Tensor
    mlp_out: torch.Tensor
    hidden_out: torch.Tensor


@dataclass
class AgeStepTrace:
    loop_input: torch.Tensor
    b1_output_preJ: torch.Tensor
    b2_input_postJ: torch.Tensor
    loop_output: torch.Tensor
    logits: torch.Tensor
    blocks: tuple[BlockTrace, BlockTrace]


@dataclass(frozen=True)
class ComponentPatch:
    block: int
    component: str
    positions: tuple[int, ...]
    mode: str
    donor: torch.Tensor | None = None
    head: int | None = None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _centered_direction(value: torch.Tensor) -> torch.Tensor:
    centered = value.float() - value.float().mean(dim=-1, keepdim=True)
    return F.normalize(centered, dim=-1, eps=1e-8)


def _margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    mask = F.one_hot(target, num_classes=logits.shape[-1]).bool()
    strongest_wrong = logits.masked_fill(mask, float("-inf")).max(dim=-1).values
    return correct - strongest_wrong


def _apply_patch(
    value: torch.Tensor,
    patch: ComponentPatch | None,
    *,
    block: int,
    component: str,
) -> torch.Tensor:
    if patch is None or patch.block != block or patch.component != component:
        return value
    result = value.clone()
    index = list(patch.positions)
    if component == "head_context":
        if patch.head is None:
            raise ValueError("head_context patch requires a head")
        replacement = (
            torch.zeros_like(result[:, patch.head, index])
            if patch.mode == "zero"
            else patch.donor
        )
        if replacement is None:
            raise ValueError("nonzero patch requires a donor")
        result[:, patch.head, index] = replacement.to(dtype=result.dtype)
        return result
    if component == "mlp_out":
        replacement = (
            torch.zeros_like(result[:, index])
            if patch.mode == "zero"
            else patch.donor
        )
        if replacement is None:
            raise ValueError("nonzero patch requires a donor")
        result[:, index] = replacement.to(dtype=result.dtype)
        return result
    raise ValueError(f"unsupported component patch: {component}")


@torch.no_grad()
def run_age_step(
    *,
    model,
    state: torch.Tensor,
    loop_index: int,
    age_map,
    map_positions: tuple[int, ...],
    patch: ComponentPatch | None = None,
) -> AgeStepTrace:
    if model.block_style != "legacy" or len(model.blocks) != 2:
        raise ValueError("age drift experiment requires two legacy blocks")
    x = state
    loop_input = state
    traces: list[BlockTrace] = []
    b1_output_pre_j: torch.Tensor | None = None
    b2_input_post_j: torch.Tensor | None = None
    for block_index in model.active_block_indices(loop_index):
        block = model.blocks[block_index]
        if not isinstance(block, TransformerBlock):
            raise TypeError("legacy TransformerBlock required")
        if block_index == 1:
            b1_output_pre_j = x
            x = x.clone()
            positions = list(map_positions)
            x[:, positions] = age_map(x[:, positions]).to(dtype=x.dtype)
            b2_input_post_j = x
        hidden_in = x
        q, k, v = _attention_parts(block.attn, block.ln_1(x))
        pattern = _attention_pattern(q, k)
        context = torch.matmul(pattern, v)
        context = _apply_patch(
            context,
            patch,
            block=block_index,
            component="head_context",
        )
        attention_out = _project_context(block.attn, context)
        if block.inner_norm_style == "ouro_sandwich_rms":
            attention_out = block.attn_out_norm(attention_out)
        residual_mid = x + attention_out
        normalized = block.ln_2(residual_mid)
        mlp_hidden = block.mlp[1](block.mlp[0](normalized))
        mlp_out = block.mlp[3](block.mlp[2](mlp_hidden))
        if block.inner_norm_style == "ouro_sandwich_rms":
            mlp_out = block.mlp_out_norm(mlp_out)
        mlp_out = _apply_patch(
            mlp_out,
            patch,
            block=block_index,
            component="mlp_out",
        )
        x = residual_mid + mlp_out
        if block.residual_projector is not None:
            x = hidden_in + block.residual_projector(
                hidden_in,
                x - hidden_in,
            )
        traces.append(
            BlockTrace(
                hidden_in=hidden_in,
                q=q,
                k=k,
                v=v,
                pattern=pattern,
                context=context,
                attention_out=attention_out,
                residual_mid=residual_mid,
                mlp_hidden=mlp_hidden,
                mlp_out=mlp_out,
                hidden_out=x,
            )
        )
    if model.outer_norm is not None:
        x = model.outer_norm(x)
    if b1_output_pre_j is None or b2_input_post_j is None or len(traces) != 2:
        raise RuntimeError("two-block trace was not produced")
    return AgeStepTrace(
        loop_input=loop_input,
        b1_output_preJ=b1_output_pre_j,
        b2_input_postJ=b2_input_post_j,
        loop_output=x,
        logits=logits_from_raw_state(model, x),
        blocks=(traces[0], traces[1]),
    )


def _head_projected_rms(
    *,
    block: TransformerBlock,
    trace: BlockTrace,
    head: int,
    positions: tuple[int, ...],
) -> float:
    d_head = block.attn.d_head
    start = head * d_head
    stop = start + d_head
    context = trace.context[:, head, list(positions)].float()
    weight = block.attn.out_proj.weight[:, start:stop].float()
    projected = context @ weight.T
    return float(projected.square().mean().sqrt())


def _attention_entropy(
    pattern: torch.Tensor,
    *,
    head: int,
    positions: tuple[int, ...],
) -> float:
    selected = pattern[:, head, list(positions)].float().clamp_min(1e-12)
    entropy = -(selected * selected.log()).sum(dim=-1)
    return float(entropy.mean())


def _correct_destination_metrics(
    pattern: torch.Tensor,
    *,
    head: int,
    answer_position: int,
    current: torch.Tensor,
) -> tuple[float, float]:
    destination = 3 + 3 * current
    batch = torch.arange(pattern.shape[0], device=pattern.device)
    row = pattern[:, head, answer_position]
    mass = row[batch, destination]
    hit = row.argmax(dim=-1).eq(destination)
    return float(mass.mean()), float(hit.float().mean())


def _trace_state(trace: AgeStepTrace, site: str) -> torch.Tensor:
    if site not in STATE_SITES:
        raise ValueError(f"unknown state site {site}")
    return getattr(trace, site)


def _component_spec(
    label: str,
    *,
    groups: dict[str, tuple[int, ...]],
) -> tuple[int, str, tuple[int, ...], int | None]:
    if label.startswith("B1H"):
        return 0, "head_context", groups["graph"], int(label[3])
    if label == "B1MLP_graph":
        return 0, "mlp_out", groups["graph"], None
    if label.startswith("B2H"):
        return 1, "head_context", groups["answer"], int(label[3])
    if label == "B2MLP_answer":
        return 1, "mlp_out", groups["answer"], None
    raise ValueError(f"unknown component label {label}")


def _component_donor(
    trace: AgeStepTrace,
    *,
    block: int,
    component: str,
    positions: tuple[int, ...],
    head: int | None,
    shuffle_mode: str,
    graph_stride: int,
) -> torch.Tensor:
    source = trace.blocks[block]
    if component == "head_context":
        if head is None:
            raise ValueError("head donor requires head")
        value = source.context[:, head, list(positions)]
    elif component == "mlp_out":
        value = source.mlp_out[:, list(positions)]
    else:
        raise ValueError(f"unsupported donor component {component}")
    if shuffle_mode == "graph":
        value = torch.roll(value, shifts=graph_stride, dims=0)
    elif shuffle_mode == "start":
        if value.shape[0] % graph_stride:
            raise ValueError("start shuffle requires complete graph groups")
        grouped = value.reshape(
            value.shape[0] // graph_stride,
            graph_stride,
            *value.shape[1:],
        )
        value = torch.roll(grouped, shifts=1, dims=1).reshape_as(value)
    elif shuffle_mode != "none":
        raise ValueError(f"unknown donor shuffle mode {shuffle_mode}")
    return value


def _pca_fit(
    values: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    # values: [trajectory, age, feature]. Remove per-trajectory cycle-8
    # content before fitting a shared age-drift basis.
    tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    delta = tensor - tensor[:, :1]
    flat = delta.flatten(0, 1)
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = centered.T @ centered
    covariance /= max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    total = eigenvalues.sum().clamp_min(1e-12)
    explained = eigenvalues / total
    scores = (centered @ eigenvectors[:, :8]).reshape(
        tensor.shape[0],
        tensor.shape[1],
        -1,
    )
    cumulative = explained.cumsum(dim=0)
    dim90 = int(torch.searchsorted(cumulative, 0.90).item() + 1)
    return {
        "mean": mean.detach().cpu().numpy(),
        "components": eigenvectors[:, :16].T.detach().cpu().numpy(),
        "eigenvalues": eigenvalues.detach().cpu().numpy(),
        "explained": explained.detach().cpu().numpy(),
        "scores": scores.detach().cpu().numpy(),
        "dim90": dim90,
    }


def _pca_replica_pc1(
    values: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
    delta = tensor - tensor[:, :1]
    flat = delta.flatten(0, 1)
    centered = flat - flat.mean(dim=0)
    covariance = centered.T @ centered
    _, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors[:, -1].detach().cpu().numpy()


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _trajectory_diagnostics(values: np.ndarray) -> dict[str, float]:
    delta = values - values[:, :1]
    final = delta[:, -1]
    global_direction = final.mean(axis=0)
    norms = np.linalg.norm(final, axis=-1)
    valid = norms > 1e-8
    cosines = (
        (final[valid] @ global_direction)
        / (
            norms[valid] * max(np.linalg.norm(global_direction), 1e-8)
        )
        if valid.any()
        else np.asarray([np.nan])
    )
    steps = np.diff(values, axis=1)
    step_norm = np.linalg.norm(steps, axis=-1, keepdims=True)
    unit = steps / np.maximum(step_norm, 1e-8)
    adjacent = (unit[:, 1:] * unit[:, :-1]).sum(axis=-1)
    shared_fraction = float(
        np.square(global_direction).sum()
        / max(float(np.square(final).sum(axis=-1).mean()), 1e-12)
    )
    return {
        "final_drift_norm_mean": float(norms.mean()),
        "final_to_global_direction_cosine_mean": float(np.nanmean(cosines)),
        "adjacent_step_direction_cosine_mean": float(
            np.nanmean(adjacent)
        ),
        "shared_final_drift_energy_fraction": shared_fraction,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _condition_patch(
    *,
    label: str,
    mode: str,
    donor_trace: AgeStepTrace,
    groups: dict[str, tuple[int, ...]],
    graph_stride: int,
) -> ComponentPatch:
    block, component, positions, head = _component_spec(
        label,
        groups=groups,
    )
    donor = None
    if mode in {"young", "graph_shuffled", "start_shuffled"}:
        donor = _component_donor(
            donor_trace,
            block=block,
            component=component,
            positions=positions,
            head=head,
            shuffle_mode=(
                "graph"
                if mode == "graph_shuffled"
                else "start"
                if mode == "start_shuffled"
                else "none"
            ),
            graph_stride=graph_stride,
        )
    return ComponentPatch(
        block=block,
        component=component,
        positions=positions,
        mode="zero" if mode == "zero" else "patch",
        donor=donor,
        head=head,
    )


def _plot_fixed_pca(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    pca_payloads: dict[str, dict[str, Any]],
    accuracies: np.ndarray,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, site in zip(axes.flat, STATE_SITES, strict=True):
        scores = pca_payloads[f"{site}:raw"]["scores"][0]
        scatter = axis.scatter(
            scores[:, 0],
            scores[:, 1],
            c=cycles,
            cmap="viridis",
            s=55,
            edgecolor="black",
            linewidth=0.3,
        )
        axis.plot(scores[:, 0], scores[:, 1], color="#555555", alpha=0.55)
        for index, cycle in enumerate(cycles):
            axis.annotate(
                str(int(cycle)),
                (scores[index, 0], scores[index, 1]),
                fontsize=7,
                xytext=(3, 2),
                textcoords="offset points",
            )
        evr = pca_payloads[f"{site}:raw"]["explained"]
        axis.set_title(
            f"{site}: PC1 {evr[0]:.2f}, PC2 {evr[1]:.2f}"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.18)
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.82)
    colorbar.set_label("continuation cycle (same current every 8)")
    fig.suptitle(
        "Fixed graph/start/current/answer-position age trajectory\n"
        f"matched-cycle accuracy: {', '.join(f'{v:.2f}' for v in accuracies)}"
    )
    path = out_dir / "fixed_graph_answer_pca.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_pca_variance(
    *,
    out_dir: Path,
    pca_payloads: dict[str, dict[str, Any]],
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, kind in zip(axes, ("raw", "ln_direction"), strict=True):
        for site in STATE_SITES:
            evr = pca_payloads[f"{site}:{kind}"]["explained"]
            axis.plot(
                np.arange(1, 17),
                np.cumsum(evr[:16]),
                marker="o",
                markersize=3,
                label=site,
            )
        axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("number of PCs")
        axis.set_ylabel("cumulative explained variance")
        axis.set_title(kind)
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    path = out_dir / "pca_explained_variance.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_drift_accuracy(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    pooled_states: dict[str, np.ndarray],
    accuracies: np.ndarray,
) -> str:
    fig, axis = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    for site in STATE_SITES:
        values = pooled_states[site]
        distance = np.linalg.norm(values - values[:, :1], axis=-1).mean(axis=0)
        axis.plot(cycles, distance, marker="o", label=site)
    axis.set_xlabel("continuation cycle (same current every 8)")
    axis.set_ylabel("mean raw distance from cycle 8")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper left", fontsize=8)
    accuracy_axis = axis.twinx()
    accuracy_axis.plot(
        cycles,
        accuracies,
        color="black",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        label="accuracy",
    )
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_ylim(0, 1.05)
    path = out_dir / "drift_distance_and_accuracy.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_component_activation(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    values: np.ndarray,
    accuracies: np.ndarray,
) -> str:
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    z = (values - mean) / np.maximum(scale, 1e-8)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7.5),
        gridspec_kw={"height_ratios": [1, 4]},
        constrained_layout=True,
    )
    axes[0].plot(cycles, accuracies, color="black", marker="o")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.18)
    image = axes[1].imshow(
        z,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
    )
    axes[1].set_xticks(np.arange(len(cycles)), labels=cycles)
    axes[1].set_yticks(
        np.arange(len(COMPONENT_LABELS)),
        labels=COMPONENT_LABELS,
    )
    axes[1].set_xlabel("continuation cycle, matched graph/current")
    axes[1].set_title("Component projected-output RMS (row-wise z-score)")
    fig.colorbar(image, ax=axes[1], shrink=0.85, label="z-score")
    path = out_dir / "component_activation_age_heatmap.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_neuron_reuse(
    *,
    out_dir: Path,
    cycles: np.ndarray,
    rows: list[dict[str, Any]],
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for block, label in ((1, "B1 graph"), (2, "B2 answer")):
        selected = [row for row in rows if int(row["block"]) == block]
        axes[0].plot(
            cycles,
            [float(row["profile_cosine_to_cycle8"]) for row in selected],
            marker="o",
            label=label,
        )
        axes[1].plot(
            cycles,
            [float(row["top64_jaccard_to_cycle8"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[0].set_title("MLP activation-profile cosine")
    axes[1].set_title("Top-64 neuron Jaccard")
    for axis in axes:
        axis.set_xlabel("continuation cycle")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.18)
        axis.legend()
    path = out_dir / "mlp_neuron_reuse.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path.name


def _plot_component_patch(
    *,
    out_dir: Path,
    patch_rows: list[dict[str, Any]],
    probe_cycles: Sequence[int],
) -> str:
    graph_matrix = np.full(
        (len(COMPONENT_LABELS), len(probe_cycles)),
        np.nan,
        dtype=np.float64,
    )
    start_matrix = np.full(
        (len(COMPONENT_LABELS), len(probe_cycles)),
        np.nan,
        dtype=np.float64,
    )
    for row_index, label in enumerate(COMPONENT_LABELS):
        for column, cycle in enumerate(probe_cycles):
            parts = [
                row
                for row in patch_rows
                if row["component"] == label
                and int(row["cycle"]) == int(cycle)
            ]
            young = [row for row in parts if row["mode"] == "young"]
            graph_shuffled = [
                row for row in parts if row["mode"] == "graph_shuffled"
            ]
            start_shuffled = [
                row for row in parts if row["mode"] == "start_shuffled"
            ]
            if young and graph_shuffled:
                graph_matrix[row_index, column] = np.mean(
                    [float(row["margin_gain"]) for row in young]
                ) - np.mean(
                    [float(row["margin_gain"]) for row in graph_shuffled]
                )
            if young and start_shuffled:
                start_matrix[row_index, column] = np.mean(
                    [float(row["margin_gain"]) for row in young]
                ) - np.mean(
                    [float(row["margin_gain"]) for row in start_shuffled]
                )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    limit = max(
        0.1,
        float(
            np.nanmax(
                np.abs(np.concatenate([graph_matrix, start_matrix], axis=1))
            )
        ),
    )
    for axis, matrix, title in (
        (
            axes[0],
            graph_matrix,
            "young - graph-shuffled (same start/target)",
        ),
        (
            axes[1],
            start_matrix,
            "young - start-shuffled (same graph)",
        ),
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xticks(
            np.arange(len(probe_cycles)),
            labels=[str(value) for value in probe_cycles],
        )
        axis.set_yticks(
            np.arange(len(COMPONENT_LABELS)),
            labels=COMPONENT_LABELS,
        )
        axis.set_xlabel("continuation cycle")
        axis.set_title(title)
    fig.colorbar(
        image,
        ax=axes,
        label="specific margin recovery",
        shrink=0.85,
    )
    path = out_dir / "component_patch_specificity.png"
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
    top_k: int,
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.08")
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
        raise ValueError("experiment is fixed to N8 D8L8, two blocks, four heads")
    age_map, map_checkpoint = load_unit_j_map(
        j_artifact,
        label=j_label,
        device=device,
    )
    if map_checkpoint != str(checkpoint):
        raise ValueError("J and model checkpoints differ")
    groups = explicit_depth_position_groups(cfg.node_count)
    map_positions = intervention_groups(cfg.node_count)["all"]
    answer_position = groups["answer"][0]
    matched_cycles = np.arange(
        matched_period,
        continuation_loops + 1,
        matched_period,
        dtype=np.int64,
    )
    probe_cycles = tuple(
        sorted(
            cycle
            for cycle in {int(value) for value in probe_cycles}
            if cycle in set(int(value) for value in matched_cycles)
            and cycle > matched_period
        )
    )
    if len(matched_cycles) < 3:
        raise ValueError("need at least three matched cycles")
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

    replica_states: dict[int, dict[str, np.ndarray]] = {}
    replica_correctness: dict[int, np.ndarray] = {}
    head_metric_sum: dict[tuple[int, int, int, int, str], float] = defaultdict(float)
    head_metric_count: dict[tuple[int, int, int, int, str], int] = defaultdict(int)
    mlp_out_sum: dict[tuple[int, int, int], float] = defaultdict(float)
    mlp_out_count: dict[tuple[int, int, int], int] = defaultdict(int)
    neuron_profile_sum: dict[tuple[int, int, int], torch.Tensor] = {}
    neuron_profile_count: dict[tuple[int, int, int], int] = defaultdict(int)
    patch_rows: list[dict[str, Any]] = []
    equivalence_max = defaultdict(float)
    j_rows: list[dict[str, Any]] = []

    for replica_index, sample_seed in enumerate(sample_seeds):
        successors_all, starts_all = _expand_all_starts(
            samples[int(sample_seed)],
            device=device,
        )
        example_count = successors_all.shape[0]
        correctness = np.zeros(
            (continuation_loops, example_count),
            dtype=np.bool_,
        )
        state_arrays = {
            site: np.zeros(
                (
                    example_count,
                    len(matched_cycles),
                    cfg.d_model,
                ),
                dtype=np.float32,
            )
            for site in STATE_SITES
        }
        matched_lookup = {
            int(cycle): index
            for index, cycle in enumerate(matched_cycles)
        }
        for offset in range(0, example_count, batch_size):
            successors = successors_all[offset : offset + batch_size]
            starts = starts_all[offset : offset + batch_size]
            batch_n = successors.shape[0]
            endpoint = advance_nodes(
                successors,
                starts,
                steps=cfg.max_depth,
            )
            state = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=8,
                phase_position=8,
            )
            donor_trace: AgeStepTrace | None = None
            previous_matched_pre_j: torch.Tensor | None = None
            for cycle in range(1, continuation_loops + 1):
                loop_index = cfg.max_loops + cycle - 1
                current = advance_nodes(
                    successors,
                    endpoint,
                    steps=cycle - 1,
                )
                target = advance_nodes(
                    successors,
                    endpoint,
                    steps=cycle,
                )
                trace = run_age_step(
                    model=model,
                    state=state,
                    loop_index=loop_index,
                    age_map=age_map,
                    map_positions=map_positions,
                )
                prediction = trace.logits.argmax(dim=-1)
                correctness[
                    cycle - 1,
                    offset : offset + batch_n,
                ] = prediction.eq(target).cpu().numpy()

                if replica_index == 0 and offset == 0 and cycle == 1:
                    reference = run_one_loop(
                        model,
                        state,
                        loop_index=loop_index,
                        block2_position_transform=(
                            map_positions,
                            age_map,
                        ),
                    )
                    equivalence_max["state"] = float(
                        (trace.loop_output - reference.state).abs().max()
                    )
                    equivalence_max["logits"] = float(
                        (trace.logits - reference.logits).abs().max()
                    )
                    equivalence_max["block2_input"] = float(
                        (
                            trace.b2_input_postJ
                            - reference.block2_hidden_in
                        ).abs().max()
                    )
                    equivalence_max["block2_context"] = float(
                        (
                            trace.blocks[1].context
                            - reference.block2_head_context
                        ).abs().max()
                    )
                    equivalence_max["block2_mlp"] = float(
                        (
                            trace.blocks[1].mlp_out
                            - reference.block2_mlp_out
                        ).abs().max()
                    )

                # Aggregate component activity at every effective loop site.
                scopes = (groups["graph"], groups["answer"])
                for block_index, scope in enumerate(scopes):
                    block_trace = trace.blocks[block_index]
                    block = model.blocks[block_index]
                    for head in range(cfg.n_heads):
                        metrics = {
                            "projected_rms": _head_projected_rms(
                                block=block,
                                trace=block_trace,
                                head=head,
                                positions=scope,
                            ),
                            "context_rms": float(
                                block_trace.context[
                                    :, head, list(scope)
                                ].float().square().mean().sqrt()
                            ),
                            "attention_entropy": _attention_entropy(
                                block_trace.pattern,
                                head=head,
                                positions=scope,
                            ),
                        }
                        if block_index == 1:
                            mass, hit = _correct_destination_metrics(
                                block_trace.pattern,
                                head=head,
                                answer_position=answer_position,
                                current=current,
                            )
                            metrics["correct_destination_mass"] = mass
                            metrics["correct_destination_hit"] = hit
                        for name, value in metrics.items():
                            key = (
                                replica_index,
                                cycle,
                                block_index,
                                head,
                                name,
                            )
                            head_metric_sum[key] += value * batch_n
                            head_metric_count[key] += batch_n
                    selected_mlp_out = block_trace.mlp_out[
                        :, list(scope)
                    ].float()
                    mlp_key = (replica_index, cycle, block_index)
                    mlp_out_sum[mlp_key] += (
                        float(selected_mlp_out.square().mean().sqrt())
                        * batch_n
                    )
                    mlp_out_count[mlp_key] += batch_n
                    profile = block_trace.mlp_hidden[
                        :, list(scope)
                    ].float().abs().sum(dim=(0, 1)).cpu()
                    if mlp_key not in neuron_profile_sum:
                        neuron_profile_sum[mlp_key] = torch.zeros_like(profile)
                    neuron_profile_sum[mlp_key] += profile
                    neuron_profile_count[mlp_key] += batch_n * len(scope)

                if cycle in matched_lookup:
                    age_index = matched_lookup[cycle]
                    batch_slice = slice(offset, offset + batch_n)
                    for site in STATE_SITES:
                        state_arrays[site][
                            batch_slice,
                            age_index,
                        ] = (
                            _trace_state(trace, site)[:, answer_position]
                            .float()
                            .cpu()
                            .numpy()
                        )
                    pre_j = trace.b1_output_preJ[:, answer_position].float()
                    post_j = trace.b2_input_postJ[:, answer_position].float()
                    if previous_matched_pre_j is not None:
                        drift = pre_j - previous_matched_pre_j
                        update = post_j - pre_j
                        cosine = F.cosine_similarity(
                            update,
                            -drift,
                            dim=-1,
                        )
                        j_rows.append(
                            {
                                "replica": replica_index,
                                "cycle": cycle,
                                "batch_offset": offset,
                                "examples": batch_n,
                                "J_update_norm": float(
                                    update.norm(dim=-1).mean()
                                ),
                                "preJ_age_step_norm": float(
                                    drift.norm(dim=-1).mean()
                                ),
                                "J_alignment_against_age_drift": float(
                                    cosine.mean()
                                ),
                            }
                        )
                    previous_matched_pre_j = pre_j.detach()

                if cycle == matched_period:
                    donor_trace = trace

                if cycle in probe_cycles:
                    if donor_trace is None:
                        raise RuntimeError("early matched donor was not captured")
                    baseline_accuracy = float(prediction.eq(target).float().mean())
                    baseline_margin = float(_margin(trace.logits, target).mean())
                    for label in COMPONENT_LABELS:
                        for mode in (
                            "young",
                            "graph_shuffled",
                            "start_shuffled",
                            "zero",
                        ):
                            patch = _condition_patch(
                                label=label,
                                mode=mode,
                                donor_trace=donor_trace,
                                groups=groups,
                                graph_stride=cfg.node_count,
                            )
                            changed = run_age_step(
                                model=model,
                                state=state,
                                loop_index=loop_index,
                                age_map=age_map,
                                map_positions=map_positions,
                                patch=patch,
                            )
                            changed_accuracy = float(
                                changed.logits.argmax(dim=-1)
                                .eq(target)
                                .float()
                                .mean()
                            )
                            changed_margin = float(
                                _margin(changed.logits, target).mean()
                            )
                            patch_rows.append(
                                {
                                    "replica": replica_index,
                                    "sample_seed": int(sample_seed),
                                    "batch_offset": offset,
                                    "examples": batch_n,
                                    "cycle": cycle,
                                    "component": label,
                                    "mode": mode,
                                    "baseline_accuracy": baseline_accuracy,
                                    "changed_accuracy": changed_accuracy,
                                    "accuracy_gain": (
                                        changed_accuracy - baseline_accuracy
                                    ),
                                    "baseline_margin": baseline_margin,
                                    "changed_margin": changed_margin,
                                    "margin_gain": (
                                        changed_margin - baseline_margin
                                    ),
                                }
                            )
                state = trace.loop_output
        replica_states[int(sample_seed)] = state_arrays
        replica_correctness[int(sample_seed)] = correctness
        replica_dir = out_dir / f"replica_{int(sample_seed)}"
        replica_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            replica_dir / "fixed_content_answer_states.npz",
            successors=successors_all.cpu().numpy(),
            starts=starts_all.cpu().numpy(),
            matched_cycles=matched_cycles,
            correctness=correctness,
            **state_arrays,
        )

    if max(equivalence_max.values(), default=0.0) > 2e-5:
        raise RuntimeError(f"custom trace failed equivalence: {equivalence_max}")

    # Aggregate behavior and components across replicas.
    accuracy_by_cycle = np.stack(
        [
            replica_correctness[int(seed)].mean(axis=1)
            for seed in sample_seeds
        ],
        axis=0,
    )
    accuracy_mean = accuracy_by_cycle.mean(axis=0)
    matched_accuracy = accuracy_mean[matched_cycles - 1]
    component_rows: list[dict[str, Any]] = []
    matched_component_values = np.zeros(
        (len(COMPONENT_LABELS), len(matched_cycles)),
        dtype=np.float64,
    )
    for cycle_index, cycle in enumerate(matched_cycles):
        component_index = 0
        for block_index in range(2):
            for head in range(cfg.n_heads):
                values = []
                row: dict[str, Any] = {
                    "cycle": int(cycle),
                    "block": block_index + 1,
                    "component": f"H{head}",
                    "kind": "head",
                }
                for metric in (
                    "projected_rms",
                    "context_rms",
                    "attention_entropy",
                    "correct_destination_mass",
                    "correct_destination_hit",
                ):
                    replica_values = []
                    for replica_index in range(len(sample_seeds)):
                        key = (
                            replica_index,
                            int(cycle),
                            block_index,
                            head,
                            metric,
                        )
                        if head_metric_count.get(key, 0):
                            replica_values.append(
                                head_metric_sum[key]
                                / head_metric_count[key]
                            )
                    if replica_values:
                        row[f"{metric}_mean"] = float(
                            np.mean(replica_values)
                        )
                        row[f"{metric}_std"] = float(
                            np.std(replica_values)
                        )
                component_rows.append(row)
                matched_component_values[
                    component_index,
                    cycle_index,
                ] = float(row["projected_rms_mean"])
                component_index += 1
            mlp_values = [
                mlp_out_sum[(replica_index, int(cycle), block_index)]
                / mlp_out_count[(replica_index, int(cycle), block_index)]
                for replica_index in range(len(sample_seeds))
            ]
            row = {
                "cycle": int(cycle),
                "block": block_index + 1,
                "component": "MLP",
                "kind": "mlp",
                "projected_rms_mean": float(np.mean(mlp_values)),
                "projected_rms_std": float(np.std(mlp_values)),
            }
            component_rows.append(row)
            matched_component_values[
                component_index,
                cycle_index,
            ] = row["projected_rms_mean"]
            component_index += 1

    # The ordering above is B1 heads, B1 MLP, B2 heads, B2 MLP.
    if component_index != len(COMPONENT_LABELS):
        raise RuntimeError("component ordering mismatch")

    neuron_rows: list[dict[str, Any]] = []
    for block_index in range(2):
        profiles = []
        for cycle in matched_cycles:
            replica_profiles = []
            for replica_index in range(len(sample_seeds)):
                key = (replica_index, int(cycle), block_index)
                replica_profiles.append(
                    neuron_profile_sum[key].numpy()
                    / neuron_profile_count[key]
                )
            profiles.append(np.mean(replica_profiles, axis=0))
        reference = profiles[0]
        reference_top = set(
            np.argsort(reference)[-min(top_k, reference.size) :].tolist()
        )
        for cycle, profile in zip(matched_cycles, profiles, strict=True):
            top = set(
                np.argsort(profile)[-min(top_k, profile.size) :].tolist()
            )
            neuron_rows.append(
                {
                    "cycle": int(cycle),
                    "block": block_index + 1,
                    "scope": "graph" if block_index == 0 else "answer",
                    "profile_cosine_to_cycle8": _safe_cosine(
                        profile,
                        reference,
                    ),
                    "top64_jaccard_to_cycle8": (
                        len(top & reference_top)
                        / len(top | reference_top)
                    ),
                    "mean_abs_activation": float(profile.mean()),
                    "top_neuron_ids": " ".join(
                        str(value) for value in sorted(top)
                    ),
                }
            )

    pooled_states = {
        site: np.concatenate(
            [
                replica_states[int(seed)][site]
                for seed in sample_seeds
            ],
            axis=0,
        )
        for site in STATE_SITES
    }
    pca_payloads: dict[str, dict[str, Any]] = {}
    pca_rows: list[dict[str, Any]] = []
    pca_artifact: dict[str, np.ndarray] = {}
    for site in STATE_SITES:
        for kind in ("raw", "ln_direction"):
            values = pooled_states[site]
            if kind == "ln_direction":
                tensor = torch.from_numpy(values)
                values = _centered_direction(tensor).numpy()
            payload = _pca_fit(values, device=device)
            pca_payloads[f"{site}:{kind}"] = payload
            replica_pc1 = []
            for seed in sample_seeds:
                replica_values = replica_states[int(seed)][site]
                if kind == "ln_direction":
                    replica_values = _centered_direction(
                        torch.from_numpy(replica_values)
                    ).numpy()
                replica_pc1.append(
                    _pca_replica_pc1(replica_values, device=device)
                )
            pc1_stability = (
                abs(_safe_cosine(replica_pc1[0], replica_pc1[1]))
                if len(replica_pc1) >= 2
                else float("nan")
            )
            diagnostics = _trajectory_diagnostics(values)
            weight = age_map.weight.detach().cpu().numpy()
            pc1 = payload["components"][0]
            transformed_pc1 = pc1 @ weight
            row = {
                "site": site,
                "representation": kind,
                "evr_pc1": float(payload["explained"][0]),
                "evr_pc2": float(payload["explained"][1]),
                "evr_top4": float(payload["explained"][:4].sum()),
                "evr_top8": float(payload["explained"][:8].sum()),
                "dim90": int(payload["dim90"]),
                "pc1_replica_abs_cosine": pc1_stability,
                "pc1_J_gain": float(
                    np.linalg.norm(transformed_pc1)
                    / max(np.linalg.norm(pc1), 1e-8)
                ),
                "pc1_J_direction_cosine": _safe_cosine(
                    pc1,
                    transformed_pc1,
                ),
                **diagnostics,
            }
            pca_rows.append(row)
            key = f"{site}_{kind}"
            pca_artifact[f"{key}_mean"] = payload["mean"]
            pca_artifact[f"{key}_components"] = payload["components"]
            pca_artifact[f"{key}_explained"] = payload["explained"]

    # Use the pooled basis but display one strictly fixed graph/start/current.
    fixed_payloads: dict[str, dict[str, Any]] = {}
    first_seed = int(sample_seeds[0])
    for site in STATE_SITES:
        for kind in ("raw", "ln_direction"):
            pooled = pca_payloads[f"{site}:{kind}"]
            fixed_values = replica_states[first_seed][site][:1]
            if kind == "ln_direction":
                fixed_values = _centered_direction(
                    torch.from_numpy(fixed_values)
                ).numpy()
            delta = fixed_values - fixed_values[:, :1]
            flat = delta.reshape(-1, cfg.d_model)
            scores = (
                (flat - pooled["mean"]) @ pooled["components"][:8].T
            ).reshape(1, len(matched_cycles), -1)
            fixed_payloads[f"{site}:{kind}"] = {
                **pooled,
                "scores": scores,
            }

    # J contraction relative to the fixed-content early state.
    j_effect_rows: list[dict[str, Any]] = []
    pre = pooled_states["b1_output_preJ"]
    post = pooled_states["b2_input_postJ"]
    for age_index, cycle in enumerate(matched_cycles):
        pre_distance = np.linalg.norm(
            pre[:, age_index] - pre[:, 0],
            axis=-1,
        )
        post_distance = np.linalg.norm(
            post[:, age_index] - post[:, 0],
            axis=-1,
        )
        j_effect_rows.append(
            {
                "cycle": int(cycle),
                "preJ_distance_from_cycle8": float(pre_distance.mean()),
                "postJ_distance_from_cycle8": float(post_distance.mean()),
                "J_drift_contraction_ratio": float(
                    post_distance.mean()
                    / max(pre_distance.mean(), 1e-8)
                ),
                "accuracy": float(matched_accuracy[age_index]),
            }
        )

    # Aggregate causal patch rows over batches/replicas.
    patch_aggregate_rows: list[dict[str, Any]] = []
    for cycle in probe_cycles:
        for label in COMPONENT_LABELS:
            for mode in (
                "young",
                "graph_shuffled",
                "start_shuffled",
                "zero",
            ):
                parts = [
                    row
                    for row in patch_rows
                    if int(row["cycle"]) == cycle
                    and row["component"] == label
                    and row["mode"] == mode
                ]
                weights = np.asarray(
                    [int(row["examples"]) for row in parts],
                    dtype=np.float64,
                )
                patch_aggregate_rows.append(
                    {
                        "cycle": cycle,
                        "component": label,
                        "mode": mode,
                        "batches": len(parts),
                        "examples": int(weights.sum()),
                        **{
                            field: float(
                                np.average(
                                    [float(row[field]) for row in parts],
                                    weights=weights,
                                )
                            )
                            for field in (
                                "baseline_accuracy",
                                "changed_accuracy",
                                "accuracy_gain",
                                "baseline_margin",
                                "changed_margin",
                                "margin_gain",
                            )
                        },
                    }
                )

    # Component-accuracy associations are descriptive only.
    association_rows: list[dict[str, Any]] = []
    for component_index, label in enumerate(COMPONENT_LABELS):
        activation = matched_component_values[component_index]
        association_rows.append(
            {
                "component": label,
                "activation_accuracy_pearson": _pearson(
                    activation,
                    matched_accuracy,
                ),
                "cycle96_over_cycle8_activation": float(
                    activation[-1] / max(activation[0], 1e-8)
                ),
                "cycle8_activation": float(activation[0]),
                "cycle96_activation": float(activation[-1]),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "pca_basis.npz", **pca_artifact)
    _write_csv(out_dir / "pca_summary.csv", pca_rows)
    _write_csv(out_dir / "component_activation_by_age.csv", component_rows)
    _write_csv(out_dir / "mlp_neuron_reuse.csv", neuron_rows)
    _write_csv(out_dir / "component_patch_raw.csv", patch_rows)
    _write_csv(
        out_dir / "component_patch_aggregate.csv",
        patch_aggregate_rows,
    )
    _write_csv(out_dir / "J_drift_alignment_batches.csv", j_rows)
    _write_csv(out_dir / "J_drift_effect.csv", j_effect_rows)
    _write_csv(out_dir / "component_accuracy_association.csv", association_rows)
    figures = {
        "fixed_pca": _plot_fixed_pca(
            out_dir=out_dir,
            cycles=matched_cycles,
            pca_payloads=fixed_payloads,
            accuracies=matched_accuracy,
        ),
        "pca_variance": _plot_pca_variance(
            out_dir=out_dir,
            pca_payloads=pca_payloads,
        ),
        "drift_accuracy": _plot_drift_accuracy(
            out_dir=out_dir,
            cycles=matched_cycles,
            pooled_states=pooled_states,
            accuracies=matched_accuracy,
        ),
        "component_activation": _plot_component_activation(
            out_dir=out_dir,
            cycles=matched_cycles,
            values=matched_component_values,
            accuracies=matched_accuracy,
        ),
        "neuron_reuse": _plot_neuron_reuse(
            out_dir=out_dir,
            cycles=matched_cycles,
            rows=neuron_rows,
        ),
        "component_patch": _plot_component_patch(
            out_dir=out_dir,
            patch_rows=patch_aggregate_rows,
            probe_cycles=probe_cycles,
        ),
    }

    payload: dict[str, Any] = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "J_artifact": str(j_artifact),
        "J_label": j_label,
        "J_definition": (
            "model-specific full affine map applied to all 29 pre-Block2 "
            "positions; test-time rollout reads current hidden state only"
        ),
        "loss_placement": (
            "frozen final-only D8L8 seed0; no training in this analysis"
        ),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_loops": continuation_loops,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_mlp": cfg.d_mlp,
        "dataset": {
            "strictly_unseen_single_8_cycles_per_replica": sample_count,
            "all_starts_per_graph": cfg.node_count,
            "replica_seeds": [int(value) for value in sample_seeds],
            "examples_per_replica": sample_count * cfg.node_count,
            "primary_fixed_position": answer_position,
            "fixed_content_rule": (
                "within each graph/start trajectory compare cycles separated "
                "by 8, so graph, current, and absolute token position match"
            ),
            "primary_training_draws": total_training_draws,
            "unique_primary_training_graphs": unique_after_stage,
        },
        "matched_cycles": matched_cycles.tolist(),
        "probe_cycles": list(probe_cycles),
        "accuracy": {
            "mean_by_cycle": accuracy_mean.tolist(),
            "replica_by_cycle": accuracy_by_cycle.tolist(),
            "matched_cycle_mean": {
                str(int(cycle)): float(value)
                for cycle, value in zip(
                    matched_cycles,
                    matched_accuracy,
                    strict=True,
                )
            },
        },
        "equivalence_max_abs_error": dict(equivalence_max),
        "pca": pca_rows,
        "J_drift_effect": j_effect_rows,
        "component_accuracy_association": association_rows,
        "component_patch": patch_aggregate_rows,
        "claim_ledger": [
            {
                "claim": (
                    "learned-J late failure follows a reproducible "
                    "fixed-content residual drift"
                ),
                "status": "supported only if PCA direction and drift metrics "
                "replicate across sample seeds",
                "evidence": "pca_summary.csv and raw replica NPZ files",
            },
            {
                "claim": (
                    "specific head/MLP activation changes are causal "
                    "contributors to late failure"
                ),
                "status": "candidate causal roles, not a complete circuit",
                "evidence": (
                    "same-trajectory young component patch versus "
                    "cross-graph shuffled and zero controls"
                ),
            },
            {
                "claim": "PCA components are an age subspace",
                "status": "not established",
                "evidence_needed": (
                    "no-oracle intervention along PCs plus graph/current "
                    "disentanglement"
                ),
            },
        ],
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "peak_reserved_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
            "shared": False,
        },
        "files": {
            "pca": "pca_summary.csv",
            "component_activation": "component_activation_by_age.csv",
            "neuron_reuse": "mlp_neuron_reuse.csv",
            "component_patch": "component_patch_aggregate.csv",
            "J_drift": "J_drift_effect.csv",
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
            "Fixed-content PCA and component-activation audit of learned-J "
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
        default=(16, 32, 48, 56, 64, 80, 96),
    )
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=7.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
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
        top_k=args.top_k,
        physical_gpu=args.physical_gpu,
        prelaunch_used_mib=args.prelaunch_used_mib,
        prelaunch_free_mib=args.prelaunch_free_mib,
        declared_peak_gib=args.declared_peak_gib,
        reserve_gib=args.reserve_gib,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
