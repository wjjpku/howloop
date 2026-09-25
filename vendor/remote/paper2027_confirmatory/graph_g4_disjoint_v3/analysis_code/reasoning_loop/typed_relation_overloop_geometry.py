from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.typed_relation_circuit import load_checkpoint
from reasoning_loop.typed_relation_composition import (
    RelationBatch,
    TypedRelationModel,
    TypedRelationWorkspaceCell,
    make_relation_batch,
)
from reasoning_loop.typed_relation_train import pick_device


def decode(model: TypedRelationModel, state: torch.Tensor) -> torch.Tensor:
    return model.readout(model.readout_norm(state))


def mean_margin(logits: torch.Tensor, target: torch.Tensor) -> float:
    correct = logits.gather(1, target[:, None]).squeeze(1)
    wrong = logits.masked_fill(
        torch.nn.functional.one_hot(target, logits.shape[1]).bool(), -torch.inf
    ).max(dim=1).values
    return float((correct - wrong).mean().item())


def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=1)
    return {
        "accuracy": float(prediction.eq(target).float().mean().item()),
        "margin": mean_margin(logits, target),
    }


def _full_visibility(batch: RelationBatch, model: TypedRelationModel) -> torch.Tensor:
    return torch.ones(
        (batch.batch_size, 2 * model.cfg.node_count),
        dtype=torch.bool,
        device=batch.query.device,
    )


def cell_step(
    model: TypedRelationModel,
    state: torch.Tensor,
    static_edges: torch.Tensor,
    visible_edges: torch.Tensor,
) -> torch.Tensor:
    if model.shared_cell is None:
        raise ValueError("overloop analysis requires a shared looped cell")
    return model.shared_cell(state, static_edges, visible_edges)[0]


@torch.no_grad()
def unroll(
    model: TypedRelationModel,
    batch: RelationBatch,
    *,
    max_loops: int,
) -> list[torch.Tensor]:
    static_edges = model.encode_edges(batch)
    visible = _full_visibility(batch, model)
    state = model._initial_workspace(batch.query)
    states = [state]
    for _ in range(max_loops):
        state = cell_step(model, state, static_edges, visible)
        states.append(state)
    return states


def random_orthogonal(
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    matrix = torch.randn(
        (dimension, dimension), device=device, dtype=dtype, generator=generator
    )
    q, r = torch.linalg.qr(matrix)
    sign = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0).to(dtype)
    return q * sign.unsqueeze(0)


def readout_classification_jacobian(
    model: TypedRelationModel,
    state: torch.Tensor,
) -> torch.Tensor:
    """Jacobian of centered logits with respect to the pre-readout state."""
    norm = model.readout_norm
    if not isinstance(norm, torch.nn.LayerNorm) or norm.normalized_shape != (
        model.cfg.d_model,
    ):
        raise ValueError("analysis expects a one-dimensional LayerNorm readout")
    dimension = state.shape[1]
    centered = state - state.mean(dim=1, keepdim=True)
    variance = centered.square().mean(dim=1, keepdim=True)
    inv_std = torch.rsqrt(variance + norm.eps)
    normalized = centered * inv_std
    eye = torch.eye(dimension, device=state.device, dtype=state.dtype)
    centering = eye - torch.full_like(eye, 1.0 / dimension)
    normalization_jacobian = inv_std[:, None] * (
        centering.unsqueeze(0)
        - normalized[:, :, None] * normalized[:, None, :] / dimension
    )
    gamma = (
        norm.weight
        if norm.elementwise_affine
        else torch.ones(dimension, device=state.device, dtype=state.dtype)
    )
    linear = model.readout.weight * gamma.unsqueeze(0)
    jacobian = torch.einsum("kd,bdj->bkj", linear, normalization_jacobian)
    return jacobian - jacobian.mean(dim=1, keepdim=True)


def project_readout_sensitive(
    jacobian: torch.Tensor,
    vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if jacobian.ndim != 3 or vector.shape != (
        jacobian.shape[0],
        jacobian.shape[2],
    ):
        raise ValueError("jacobian/vector shapes are incompatible")
    _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    threshold = (
        singular.max(dim=1, keepdim=True).values
        * max(jacobian.shape[1:])
        * torch.finfo(jacobian.dtype).eps
    )
    active = singular > threshold
    coefficients = torch.einsum("bkd,bd->bk", vh, vector) * active
    sensitive = torch.einsum("bk,bkd->bd", coefficients, vh)
    return sensitive, vector - sensitive


def energy_fraction(part: torch.Tensor, whole: torch.Tensor) -> float:
    numerator = part.square().sum(dim=1)
    denominator = whole.square().sum(dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean().item())


def _normalized(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-12)


def prediction_retention(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    target: torch.Tensor,
) -> float:
    before_correct = before_logits.argmax(dim=1).eq(target)
    if not before_correct.any():
        return float("nan")
    after_correct = after_logits.argmax(dim=1).eq(target)
    return float(after_correct[before_correct].float().mean().item())


@torch.no_grad()
def analyze_checkpoint(
    checkpoint: Path,
    *,
    examples: int,
    max_loops: int,
    rotations: int,
    seed: int,
    device: torch.device,
    family: str,
) -> dict[str, list[dict[str, Any]]]:
    model, payload = load_checkpoint(checkpoint, device)
    if model.shared_cell is None or int(payload["train_loops"]) != 2:
        raise ValueError("checkpoint must be a two-loop shared-cell model")
    composition_order = payload.get("composition_order", "f_then_g")
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    target = batch.targets[:, 1]
    static_edges = model.encode_edges(batch)
    visible = _full_visibility(batch, model)
    states = unroll(model, batch, max_loops=max_loops)
    model_seed = int(payload["seed"])
    common = {
        "family": family,
        "model_seed": model_seed,
        "checkpoint_step": int(payload["step"]),
        "node_count": model.cfg.node_count,
        "d_model": model.cfg.d_model,
        "isotropic_rank_fraction": (model.cfg.node_count - 1) / model.cfg.d_model,
        "checkpoint": str(checkpoint.resolve()),
    }

    trajectories: list[dict[str, Any]] = []
    for loop_index, state in enumerate(states[1:], start=1):
        trajectories.append(
            {
                **common,
                "condition": "trained_cell",
                "replicate": 0,
                "loop": loop_index,
                **metrics(decode(model, state), target),
            }
        )

    no_edge_state = states[2]
    trajectories.append(
        {
            **common,
            "condition": "trained_cell_no_edges",
            "replicate": 0,
            "loop": 2,
            **metrics(decode(model, no_edge_state), target),
        }
    )
    no_edges = torch.zeros_like(visible)
    for loop_index in range(3, max_loops + 1):
        no_edge_state = cell_step(
            model, no_edge_state, static_edges, no_edges
        )
        trajectories.append(
            {
                **common,
                "condition": "trained_cell_no_edges",
                "replicate": 0,
                "loop": loop_index,
                **metrics(decode(model, no_edge_state), target),
            }
        )

    # A decoder-valid coordinate rotation preserves D2 logits exactly.  The
    # matched transition conjugates the cell; the mismatched transition does not.
    for rotation_index in range(rotations):
        rotation_generator = torch.Generator(device=device).manual_seed(
            seed + 10_000 * (model_seed + 1) + rotation_index
        )
        rotation = random_orthogonal(
            model.cfg.d_model,
            device=device,
            dtype=states[2].dtype,
            generator=rotation_generator,
        )
        rotated_start = states[2] @ rotation.T
        mismatched = rotated_start
        matched = rotated_start
        for condition in ("rotated_misaligned", "rotated_matched"):
            trajectories.append(
                {
                    **common,
                    "condition": condition,
                    "replicate": rotation_index,
                    "loop": 2,
                    **metrics(decode(model, rotated_start @ rotation), target),
                }
            )
        for loop_index in range(3, max_loops + 1):
            mismatched = cell_step(model, mismatched, static_edges, visible)
            matched_original = cell_step(
                model, matched @ rotation, static_edges, visible
            )
            matched = matched_original @ rotation.T
            trajectories.append(
                {
                    **common,
                    "condition": "rotated_misaligned",
                    "replicate": rotation_index,
                    "loop": loop_index,
                    **metrics(decode(model, mismatched @ rotation), target),
                }
            )
            trajectories.append(
                {
                    **common,
                    "condition": "rotated_matched",
                    "replicate": rotation_index,
                    "loop": loop_index,
                    **metrics(decode(model, matched @ rotation), target),
                }
            )

    # Random-cell and isotropic controls start from the already-solved D2 state.
    for replicate in range(rotations):
        with torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else []
        ):
            torch.manual_seed(seed + 91_000 + 100 * model_seed + replicate)
            random_cell = TypedRelationWorkspaceCell(model.cfg).to(device).eval()
        raw_state = states[2]
        matched_norm_state = states[2]
        isotropic_state = states[2]
        for condition in (
            "random_cell_raw",
            "random_cell_norm_matched",
            "isotropic_norm_matched",
        ):
            trajectories.append(
                {
                    **common,
                    "condition": condition,
                    "replicate": replicate,
                    "loop": 2,
                    **metrics(decode(model, states[2]), target),
                }
            )
        random_generator = torch.Generator(device=device).manual_seed(
            seed + 93_000 + 100 * model_seed + replicate
        )
        for loop_index in range(3, max_loops + 1):
            raw_state = random_cell(raw_state, static_edges, visible)[0]
            random_candidate = random_cell(
                matched_norm_state, static_edges, visible
            )[0]
            trained_candidate = cell_step(
                model, matched_norm_state, static_edges, visible
            )
            random_delta = random_candidate - matched_norm_state
            target_norm = (trained_candidate - matched_norm_state).norm(
                dim=1, keepdim=True
            )
            matched_norm_state = (
                matched_norm_state + _normalized(random_delta) * target_norm
            )
            trained_isotropic_candidate = cell_step(
                model, isotropic_state, static_edges, visible
            )
            isotropic_norm = (
                trained_isotropic_candidate - isotropic_state
            ).norm(dim=1, keepdim=True)
            direction = torch.randn(
                isotropic_state.shape,
                device=device,
                dtype=isotropic_state.dtype,
                generator=random_generator,
            )
            isotropic_state = isotropic_state + _normalized(direction) * isotropic_norm
            for condition, state in (
                ("random_cell_raw", raw_state),
                ("random_cell_norm_matched", matched_norm_state),
                ("isotropic_norm_matched", isotropic_state),
            ):
                trajectories.append(
                    {
                        **common,
                        "condition": condition,
                        "replicate": replicate,
                        "loop": loop_index,
                        **metrics(decode(model, state), target),
                    }
                )

    geometry: list[dict[str, Any]] = []
    geometry_generator = torch.Generator(device=device).manual_seed(
        seed + 77_000 + model_seed
    )
    for loop_index in range(1, max_loops + 1):
        before, after = states[loop_index - 1], states[loop_index]
        delta = after - before
        jacobian = readout_classification_jacobian(model, before)
        sensitive, null = project_readout_sensitive(jacobian, delta)
        random_direction = torch.randn(
            delta.shape,
            device=device,
            dtype=delta.dtype,
            generator=geometry_generator,
        )
        random_delta = _normalized(random_direction) * delta.norm(
            dim=1, keepdim=True
        )
        random_sensitive, _ = project_readout_sensitive(jacobian, random_delta)
        before_logits = decode(model, before)
        after_logits = decode(model, after)
        target_logit = before_logits.gather(1, target[:, None]).squeeze(1)
        wrong_index = before_logits.masked_fill(
            torch.nn.functional.one_hot(target, before_logits.shape[1]).bool(),
            -torch.inf,
        ).argmax(dim=1)
        batch_index = torch.arange(target.shape[0], device=target.device)
        margin_gradient = (
            jacobian[batch_index, target] - jacobian[batch_index, wrong_index]
        )
        gradient_norm = margin_gradient.norm(dim=1).clamp_min(1e-12)
        local_margin = target_logit - before_logits[batch_index, wrong_index]
        signed_margin_step = (margin_gradient * delta).sum(dim=1) / gradient_norm
        row_only_logits = decode(model, before + sensitive)
        null_only_logits = decode(model, before + null)
        random_logits = decode(model, before + random_delta)
        geometry.append(
            {
                **common,
                "transition": f"D{loop_index - 1}->D{loop_index}",
                "loop": loop_index,
                "delta_norm": float(delta.norm(dim=1).mean().item()),
                "state_norm": float(before.norm(dim=1).mean().item()),
                "sensitive_energy_fraction": energy_fraction(sensitive, delta),
                "null_energy_fraction": energy_fraction(null, delta),
                "random_sensitive_energy_fraction": energy_fraction(
                    random_sensitive, random_delta
                ),
                "before_accuracy": metrics(before_logits, target)["accuracy"],
                "after_accuracy": metrics(after_logits, target)["accuracy"],
                "row_only_accuracy": metrics(row_only_logits, target)["accuracy"],
                "null_only_accuracy": metrics(null_only_logits, target)["accuracy"],
                "random_same_norm_accuracy": metrics(random_logits, target)["accuracy"],
                "margin_change": mean_margin(after_logits, target)
                - mean_margin(before_logits, target),
                "centered_logit_change": float(
                    (
                        (after_logits - after_logits.mean(dim=1, keepdim=True))
                        - (before_logits - before_logits.mean(dim=1, keepdim=True))
                    )
                    .norm(dim=1)
                    .mean()
                    .item()
                ),
                "local_boundary_distance": float(
                    (local_margin / gradient_norm).mean().item()
                ),
                "signed_sensitive_step": float(signed_margin_step.mean().item()),
                "absolute_sensitive_step": float(
                    signed_margin_step.abs().mean().item()
                ),
            }
        )

    state_dependence: list[dict[str, Any]] = []
    alphas = (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    for alpha in alphas:
        start = states[1] + alpha * (states[2] - states[1])
        after_one = cell_step(model, start, static_edges, visible)
        after_two = cell_step(model, after_one, static_edges, visible)
        cycle_delta = after_two - start
        jacobian = readout_classification_jacobian(model, start)
        sensitive, _ = project_readout_sensitive(jacobian, cycle_delta)
        start_logits = decode(model, start)
        after_logits = decode(model, after_two)
        state_dependence.append(
            {
                **common,
                "experiment": "h1_h2_interpolation",
                "coordinate": alpha,
                "direction": "interpolation",
                "start_accuracy": metrics(start_logits, target)["accuracy"],
                "after_two_accuracy": metrics(after_logits, target)["accuracy"],
                "retention": prediction_retention(start_logits, after_logits, target),
                "start_margin": mean_margin(start_logits, target),
                "after_two_margin": mean_margin(after_logits, target),
                "cycle_delta_norm": float(cycle_delta.norm(dim=1).mean().item()),
                "sensitive_energy_fraction": energy_fraction(
                    sensitive, cycle_delta
                ),
            }
        )

    # Equal-radius perturbations around the solved D2 state separate local
    # readout-sensitive and readout-null directions.
    h2 = states[2]
    h4 = states[4]
    base_scale = (h4 - h2).norm(dim=1, keepdim=True).clamp_min(1e-6)
    jacobian_h2 = readout_classification_jacobian(model, h2)
    perturb_generator = torch.Generator(device=device).manual_seed(
        seed + 81_000 + model_seed
    )
    random_vector = torch.randn(
        h2.shape,
        device=device,
        dtype=h2.dtype,
        generator=perturb_generator,
    )
    row_vector, null_vector = project_readout_sensitive(
        jacobian_h2, random_vector
    )
    directions = {
        "readout_sensitive": _normalized(row_vector),
        "readout_null": _normalized(null_vector),
        "isotropic": _normalized(random_vector),
    }
    for radius in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        for direction_name, direction in directions.items():
            start = h2 + radius * base_scale * direction
            after_one = cell_step(model, start, static_edges, visible)
            after_two = cell_step(model, after_one, static_edges, visible)
            cycle_delta = after_two - start
            local_jacobian = readout_classification_jacobian(model, start)
            sensitive, _ = project_readout_sensitive(local_jacobian, cycle_delta)
            start_logits = decode(model, start)
            after_logits = decode(model, after_two)
            state_dependence.append(
                {
                    **common,
                    "experiment": "d2_local_perturbation",
                    "coordinate": radius,
                    "direction": direction_name,
                    "start_accuracy": metrics(start_logits, target)["accuracy"],
                    "after_two_accuracy": metrics(after_logits, target)["accuracy"],
                    "retention": prediction_retention(
                        start_logits, after_logits, target
                    ),
                    "start_margin": mean_margin(start_logits, target),
                    "after_two_margin": mean_margin(after_logits, target),
                    "cycle_delta_norm": float(
                        cycle_delta.norm(dim=1).mean().item()
                    ),
                    "sensitive_energy_fraction": energy_fraction(
                        sensitive, cycle_delta
                    ),
                }
            )

    invariance_error = max(
        abs(
            row["accuracy"]
            - next(
                baseline["accuracy"]
                for baseline in trajectories
                if baseline["condition"] == "trained_cell"
                and baseline["loop"] == row["loop"]
            )
        )
        for row in trajectories
        if row["condition"] == "rotated_matched"
    )
    diagnostics = [
        {
            **common,
            "d2_rotation_logit_max_abs_error": float(
                max(
                    (
                        decode(model, states[2])
                        - decode(
                            model,
                            (states[2] @ random_orthogonal(
                                model.cfg.d_model,
                                device=device,
                                dtype=states[2].dtype,
                                generator=torch.Generator(device=device).manual_seed(
                                    seed + 123_456 + model_seed
                                ),
                            ).T)
                            @ random_orthogonal(
                                model.cfg.d_model,
                                device=device,
                                dtype=states[2].dtype,
                                generator=torch.Generator(device=device).manual_seed(
                                    seed + 123_456 + model_seed
                                ),
                            ),
                        )
                    )
                    .abs()
                    .max()
                    .item(),
                    0.0,
                )
            ),
            "matched_rotation_accuracy_max_abs_error": invariance_error,
        }
    ]
    return {
        "trajectories": trajectories,
        "geometry": geometry,
        "state_dependence": state_dependence,
        "diagnostics": diagnostics,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_mean(
    rows: Iterable[dict[str, Any]],
    keys: Sequence[str],
    value: str,
) -> dict[tuple[Any, ...], tuple[float, float]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        number = float(row[value])
        if math.isfinite(number):
            groups[tuple(row[key] for key in keys)].append(number)
    result: dict[tuple[Any, ...], tuple[float, float]] = {}
    for key, values in groups.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        result[key] = (mean, std / math.sqrt(len(values)))
    return result


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "PingFang SC",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def plot_trajectories(rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plots()
    selected = [
        row
        for row in rows
        if row["family"] == "learned_readout"
        and row["condition"]
        in {
            "trained_cell",
            "trained_cell_no_edges",
            "rotated_misaligned",
            "random_cell_norm_matched",
            "isotropic_norm_matched",
        }
    ]
    grouped = _group_mean(selected, ("condition", "loop"), "accuracy")
    labels = {
        "trained_cell": "训练后的共享 block",
        "trained_cell_no_edges": "训练 block（遮掉全部关系边）",
        "rotated_misaligned": "decoder 可读、转移坐标错位",
        "random_cell_norm_matched": "随机 block（同范数）",
        "isotropic_norm_matched": "随机 residual（同范数）",
    }
    colors = {
        "trained_cell": "#C62828",
        "trained_cell_no_edges": "#2E7D32",
        "rotated_misaligned": "#1565C0",
        "random_cell_norm_matched": "#6A1B9A",
        "isotropic_norm_matched": "#546E7A",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for condition, label in labels.items():
        points = sorted(
            (key[1], value)
            for key, value in grouped.items()
            if key[0] == condition
        )
        if not points:
            continue
        x = np.asarray([point[0] for point in points])
        y = np.asarray([point[1][0] for point in points])
        sem = np.asarray([point[1][1] for point in points])
        ax.plot(x, y, marker="o", linewidth=2.2, label=label, color=colors[condition])
        ax.fill_between(x, y - sem, y + sem, color=colors[condition], alpha=0.14)
    ax.axvline(2, color="black", linestyle="--", linewidth=1, alpha=0.55)
    ax.axhline(1 / 8, color="gray", linestyle=":", linewidth=1)
    ax.set(
        xlabel="有效 loop / readout 深度",
        ylabel="终点答案准确率",
        ylim=(0.0, 1.04),
        title="训练后的继续计算比随机方向更容易改坏答案",
    )
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_readout_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plots()
    baseline = [row for row in rows if row["condition"] == "trained_cell"]
    grouped = _group_mean(baseline, ("family", "loop"), "accuracy")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for family, color, label in (
        ("learned_readout", "#C62828", "学习 decoder"),
        ("frozen_random_readout", "#1565C0", "冻结随机 decoder"),
    ):
        points = sorted(
            (key[1], value)
            for key, value in grouped.items()
            if key[0] == family
        )
        if not points:
            continue
        x = np.asarray([point[0] for point in points])
        y = np.asarray([point[1][0] for point in points])
        sem = np.asarray([point[1][1] for point in points])
        ax.plot(x, y, marker="o", linewidth=2.4, color=color, label=label)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15)
    ax.axvline(2, color="black", linestyle="--", linewidth=1, alpha=0.55)
    ax.axhline(1 / 8, color="gray", linestyle=":", linewidth=1)
    ax.set(
        xlabel="有效 loop / readout 深度",
        ylabel="终点答案准确率",
        ylim=(0.0, 1.04),
        title="与输入 embedding 无关的 decoder 仍出现 overloop 鲁棒性",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plots()
    selected = [row for row in rows if row["family"] == "learned_readout"]
    energy = _group_mean(selected, ("loop",), "sensitive_energy_fraction")
    random_energy = _group_mean(
        selected, ("loop",), "random_sensitive_energy_fraction"
    )
    accuracies = {
        key: _group_mean(selected, ("loop",), key)
        for key in (
            "after_accuracy",
            "null_only_accuracy",
            "row_only_accuracy",
            "random_same_norm_accuracy",
        )
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    x = np.asarray(sorted(key[0] for key in energy))
    y = np.asarray([energy[(int(value),)][0] for value in x])
    axes[0].plot(
        x, y, color="#C62828", marker="o", linewidth=2.3, label="训练 residual"
    )
    axes[0].plot(
        x,
        [random_energy[(int(value),)][0] for value in x],
        color="#546E7A",
        marker="o",
        linestyle="--",
        linewidth=1.8,
        label="同范数随机 residual",
    )
    axes[0].axvline(2, color="black", linestyle="--", linewidth=1, alpha=0.55)
    axes[0].set(
        xlabel="到达 D_t 的状态转移",
        ylabel="residual 能量比例",
        ylim=(0.0, 1.0),
        title="落入局部 readout 敏感子空间的能量",
    )
    axes[0].legend(frameon=False, fontsize=8)
    label_map = {
        "after_accuracy": ("完整 residual", "#C62828"),
        "null_only_accuracy": ("只加 readout-null 分量", "#EF6C00"),
        "row_only_accuracy": ("只加敏感分量", "#1565C0"),
        "random_same_norm_accuracy": ("同范数随机 residual", "#546E7A"),
    }
    for key, grouped in accuracies.items():
        points = sorted((group[0], value[0]) for group, value in grouped.items())
        axes[1].plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=2,
            label=label_map[key][0],
            color=label_map[key][1],
        )
    axes[1].axvline(2, color="black", linestyle="--", linewidth=1, alpha=0.55)
    axes[1].axhline(1 / 8, color="gray", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="到达 D_t 的状态转移",
        ylabel="干预后的终点准确率",
        ylim=(0.0, 1.04),
        title="residual 的因果分解",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_state_dependence(rows: list[dict[str, Any]], path: Path) -> None:
    _configure_plots()
    selected = [row for row in rows if row["family"] == "learned_readout"]
    interpolation = [
        row for row in selected if row["experiment"] == "h1_h2_interpolation"
    ]
    perturbation = [
        row for row in selected if row["experiment"] == "d2_local_perturbation"
    ]
    start = _group_mean(interpolation, ("coordinate",), "start_accuracy")
    after = _group_mean(interpolation, ("coordinate",), "after_two_accuracy")
    sensitive = _group_mean(
        interpolation, ("coordinate",), "sensitive_energy_fraction"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    alpha = sorted(key[0] for key in start)
    axes[0].plot(
        alpha,
        [start[(value,)][0] for value in alpha],
        marker="o",
        color="#1565C0",
        label="F² 之前",
    )
    axes[0].plot(
        alpha,
        [after[(value,)][0] for value in alpha],
        marker="o",
        color="#C62828",
        label="F² 之后",
    )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    axes[0].set(
        xlabel="α in h = h₁ + α(h₂-h₁)",
        ylabel="终点答案准确率",
        ylim=(0.0, 1.04),
        title="鲁棒性由当前状态决定",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(
        alpha,
        [sensitive[(value,)][0] for value in alpha],
        marker="o",
        color="#6A1B9A",
    )
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.55)
    axes[1].set(
        xlabel="α in h = h₁ + α(h₂-h₁)",
        ylabel="F² residual 的敏感能量比例",
        ylim=(0.0, 1.0),
        title="靠近/越过答案状态时，更新场发生变化",
    )
    directions = ("readout_sensitive", "isotropic", "readout_null")
    radii = sorted({float(row["coordinate"]) for row in perturbation})
    after_group = _group_mean(
        perturbation, ("direction", "coordinate"), "after_two_accuracy"
    )
    matrix = np.asarray(
        [
            [after_group[(direction, radius)][0] for radius in radii]
            for direction in directions
        ]
    )
    image = axes[2].imshow(
        matrix,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[2].set_xticks(range(len(radii)), [str(radius) for radius in radii])
    axes[2].set_yticks(
        range(len(directions)), ["readout 敏感", "各向同性", "readout-null"]
    )
    axes[2].set(
        xlabel="扰动半径 / 自然 ||F²(h₂)-h₂||",
        title="F² 后准确率（越红越鲁棒）",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axes[2].text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.3 or value > 0.75 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summarize(
    trajectories: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
    state_dependence: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    def mean_where(
        rows: list[dict[str, Any]], value: str, **filters: Any
    ) -> float | None:
        values = [
            float(row[value])
            for row in rows
            if all(row.get(key) == expected for key, expected in filters.items())
            and math.isfinite(float(row[value]))
        ]
        return statistics.mean(values) if values else None

    return {
        "learned_readout_accuracy": {
            f"D{loop}": mean_where(
                trajectories,
                "accuracy",
                family="learned_readout",
                condition="trained_cell",
                loop=loop,
            )
            for loop in range(2, 9)
        },
        "frozen_random_readout_accuracy": {
            f"D{loop}": mean_where(
                trajectories,
                "accuracy",
                family="frozen_random_readout",
                condition="trained_cell",
                loop=loop,
            )
            for loop in range(2, 9)
        },
        "coordinate_intervention_D4": {
            condition: mean_where(
                trajectories,
                "accuracy",
                family="learned_readout",
                condition=condition,
                loop=4,
            )
            for condition in (
                "trained_cell",
                "rotated_misaligned",
                "rotated_matched",
                "random_cell_norm_matched",
                "isotropic_norm_matched",
            )
        },
        "residual_geometry_D2_to_D4": {
            f"D{loop - 1}->D{loop}": {
                "sensitive_energy_fraction": mean_where(
                    geometry,
                    "sensitive_energy_fraction",
                    family="learned_readout",
                    loop=loop,
                ),
                "actual_accuracy": mean_where(
                    geometry,
                    "after_accuracy",
                    family="learned_readout",
                    loop=loop,
                ),
                "random_same_norm_accuracy": mean_where(
                    geometry,
                    "random_same_norm_accuracy",
                    family="learned_readout",
                    loop=loop,
                ),
            }
            for loop in (3, 4)
        },
        "interpolation": {
            str(alpha): {
                "start_accuracy": mean_where(
                    state_dependence,
                    "start_accuracy",
                    family="learned_readout",
                    experiment="h1_h2_interpolation",
                    coordinate=alpha,
                ),
                "after_F2_accuracy": mean_where(
                    state_dependence,
                    "after_two_accuracy",
                    family="learned_readout",
                    experiment="h1_h2_interpolation",
                    coordinate=alpha,
                ),
                "sensitive_energy_fraction": mean_where(
                    state_dependence,
                    "sensitive_energy_fraction",
                    family="learned_readout",
                    experiment="h1_h2_interpolation",
                    coordinate=alpha,
                ),
            }
            for alpha in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
        },
        "diagnostics": diagnostics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal geometry tests for two-loop typed-relation models."
    )
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-checkpoints", nargs="*", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--max-loops", type=int, default=8)
    parser.add_argument("--rotations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=82_271)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.examples < 8 or args.max_loops < 4 or args.rotations < 1:
        parser.error("examples, max-loops, and rotations are too small")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    all_results: dict[str, list[dict[str, Any]]] = {
        "trajectories": [],
        "geometry": [],
        "state_dependence": [],
        "diagnostics": [],
    }
    for family, checkpoints in (
        ("learned_readout", args.checkpoints),
        ("frozen_random_readout", args.frozen_checkpoints),
    ):
        for index, checkpoint in enumerate(checkpoints):
            result = analyze_checkpoint(
                checkpoint,
                examples=args.examples,
                max_loops=args.max_loops,
                rotations=args.rotations,
                seed=args.seed + 1000 * index,
                device=device,
                family=family,
            )
            for key, rows in result.items():
                all_results[key].extend(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in all_results.items():
        _write_csv(args.out_dir / f"{key}.csv", rows)
    plot_trajectories(all_results["trajectories"], args.out_dir / "overloop_controls.png")
    plot_readout_comparison(
        all_results["trajectories"], args.out_dir / "decoder_readout_comparison.png"
    )
    plot_geometry(all_results["geometry"], args.out_dir / "residual_geometry.png")
    plot_state_dependence(
        all_results["state_dependence"], args.out_dir / "state_dependence.png"
    )
    summary = summarize(
        all_results["trajectories"],
        all_results["geometry"],
        all_results["state_dependence"],
        all_results["diagnostics"],
    )
    summary.update(
        {
            "device": str(device),
            "examples_per_checkpoint": args.examples,
            "max_loops": args.max_loops,
            "rotation_replicates": args.rotations,
            "learned_checkpoints": [str(path.resolve()) for path in args.checkpoints],
            "frozen_checkpoints": [
                str(path.resolve()) for path in args.frozen_checkpoints
            ],
        }
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
