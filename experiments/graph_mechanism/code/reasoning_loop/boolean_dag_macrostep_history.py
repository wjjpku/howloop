from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.boolean_dag_data import BooleanDAGBatch, BooleanDAGConfig, make_boolean_dag_batch
from reasoning_loop.boolean_dag_macrostep import (
    DepthConditionedBooleanDAGTransformer,
    macrostep_targets,
)
from reasoning_loop.boolean_dag_macrostep_eval import realized_depth_from_predictions
from reasoning_loop.boolean_dag_model import BooleanDAGModelConfig
from reasoning_loop.graph_path_loop import pick_device, set_seed


SEMANTIC_DEPTH = 4
HISTORIES: tuple[tuple[int, ...], ...] = (
    (4,),
    (1, 3),
    (2, 2),
    (3, 1),
    (1, 1, 1, 1),
)
PATCH_PAIRS: tuple[tuple[tuple[int, ...], tuple[int, ...], str], ...] = (
    ((1, 3), (4,), "good_to_cold"),
    ((4,), (1, 3), "cold_to_good"),
    ((1, 3), (3, 1), "same_loop_forward"),
)
EVAL_SEED = 20_260_724
MASK_SEED = 20_260_725


def _history_label(history: tuple[int, ...]) -> str:
    return "+".join(str(value) for value in history)


def run_history(
    model: DepthConditionedBooleanDAGTransformer,
    batch: BooleanDAGBatch,
    history: tuple[int, ...],
) -> torch.Tensor:
    if not history or sum(history) != SEMANTIC_DEPTH:
        raise ValueError(f"history must be nonempty and sum to {SEMANTIC_DEPTH}")
    state = model.encode(batch)
    for increment in history:
        state = model.apply_macro_step(
            state,
            torch.full(
                (batch.batch_size,),
                increment,
                dtype=torch.long,
                device=state.device,
            ),
        )
    return state


def _masked_example_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    denominator = mask.sum(dim=1)
    if denominator.eq(0).any():
        raise ValueError("every example must contain at least one selected node")
    return (values * mask).sum(dim=1) / denominator


def _target_margin(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    correct = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    target_one_hot = F.one_hot(targets, num_classes=logits.shape[-1]).bool()
    strongest_other = logits.masked_fill(target_one_hot, -torch.inf).max(dim=-1).values
    return _masked_example_mean((correct - strongest_other).float(), mask)


def readout_metrics(
    model: DepthConditionedBooleanDAGTransformer,
    state: torch.Tensor,
    batch: BooleanDAGBatch,
    target_depth: int,
) -> dict[str, torch.Tensor]:
    if not 0 <= target_depth <= model.data_cfg.eval_max_depth:
        raise ValueError("target_depth is outside the configured graph range")
    logits, normalized_state = model._readout(state)
    predictions = logits.argmax(dim=-1)
    depths = torch.full(
        (batch.batch_size, 1),
        target_depth,
        dtype=torch.long,
        device=state.device,
    )
    targets = macrostep_targets(batch, depths)[:, 0]
    correct = predictions.eq(targets)
    realized, _ = realized_depth_from_predictions(
        predictions,
        batch,
        max_depth=model.data_cfg.eval_max_depth,
    )
    all_nodes = torch.ones_like(correct)
    return {
        "logits": logits,
        "normalized_state": normalized_state,
        "predictions": predictions,
        "targets": targets,
        "state_accuracy": correct.float().mean(dim=1),
        "exact_state_accuracy": correct.all(dim=1).float(),
        "realized_depth": realized.float(),
        "target_margin": _target_margin(logits, targets, all_nodes),
    }


def future_metrics(
    model: DepthConditionedBooleanDAGTransformer,
    state: torch.Tensor,
    batch: BooleanDAGBatch,
    semantic_depth: int,
    increment: int,
) -> dict[str, torch.Tensor]:
    if increment not in {1, 2, 3, 4}:
        raise ValueError("increment must lie in 1 through 4")
    target_depth = semantic_depth + increment
    next_state = model.apply_macro_step(
        state,
        torch.full(
            (batch.batch_size,),
            increment,
            dtype=torch.long,
            device=state.device,
        ),
    )
    metrics = readout_metrics(model, next_state, batch, target_depth)
    frontier = batch.levels.gt(semantic_depth) & batch.levels.le(target_depth)
    frontier_correct = metrics["predictions"].eq(metrics["targets"])
    metrics.update(
        {
            "state": next_state,
            "frontier_mask": frontier,
            "frontier_accuracy": _masked_example_mean(
                frontier_correct.float(),
                frontier,
            ),
            "frontier_margin": _target_margin(
                metrics["logits"],
                metrics["targets"],
                frontier,
            ),
            "realized_progress": metrics["realized_depth"] - semantic_depth,
        }
    )
    return metrics


def _matched_random_mask(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = torch.rand(
        reference.shape,
        device=reference.device,
        generator=generator,
    )
    result = torch.zeros_like(reference)
    for row in range(reference.shape[0]):
        count = int(reference[row].sum())
        if count:
            selected = scores[row].topk(count).indices
            result[row, selected] = True
    return result


def history_region_masks(
    batch: BooleanDAGBatch,
    *,
    semantic_depth: int,
    increment: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    if semantic_depth + increment > int(batch.depths.min()):
        raise ValueError("the requested next frontier exceeds at least one graph")
    masks = {
        "all": torch.ones_like(batch.levels, dtype=torch.bool),
        "resolved": batch.levels.le(semantic_depth),
        "current_frontier": batch.levels.eq(semantic_depth),
        "unresolved": batch.levels.gt(semantic_depth),
        "next_frontier": batch.levels.gt(semantic_depth)
        & batch.levels.le(semantic_depth + increment),
    }
    for name in ("resolved", "current_frontier", "unresolved", "next_frontier"):
        masks[f"random_matched_{name}"] = _matched_random_mask(
            masks[name],
            generator=generator,
        )
    return masks


def patch_state(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if receiver.shape != donor.shape or receiver.shape[:2] != mask.shape:
        raise ValueError("receiver, donor, and mask shapes are incompatible")
    return torch.where(mask[:, :, None], donor, receiver)


def normalized_recovery(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    patched: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    denominator = donor - receiver
    result = (patched - receiver) / denominator
    return torch.where(denominator.abs() > eps, result, torch.full_like(result, torch.nan))


@dataclass
class _Accumulator:
    values: dict[str, list[float]]

    def __init__(self) -> None:
        self.values = {}

    def add(self, **values: float) -> None:
        for name, value in values.items():
            self.values.setdefault(name, []).append(float(value))

    def means(self) -> dict[str, float]:
        return {
            name: float(np.nanmean(values)) if not np.isnan(values).all() else float("nan")
            for name, values in self.values.items()
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _update_batch_fingerprint(hasher: Any, batch: BooleanDAGBatch) -> None:
    for name, value in sorted(batch.__dict__.items()):
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().to("cpu").contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())


def _aggregate_recovery(row: dict[str, Any], *, eps: float = 1e-8) -> float:
    denominator = row["donor_frontier_margin"] - row["receiver_frontier_margin"]
    if abs(denominator) <= eps:
        return float("nan")
    return (row["patched_frontier_margin"] - row["receiver_frontier_margin"]) / denominator


def _cosine_mean(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first.float(), second.float(), dim=-1).mean())


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    difference = (first.float() - second.float()).norm(dim=-1)
    scale = 0.5 * (first.float().norm(dim=-1) + second.float().norm(dim=-1))
    return float((difference / scale.clamp_min(1e-8)).mean())


def _plot_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: list[str],
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    value_format: str,
) -> Any:
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set(
        title=title,
        xticks=range(len(labels)),
        xticklabels=labels,
        yticks=range(len(labels)),
        yticklabels=labels,
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < (vmin + vmax) / 2 else "black",
            )
    return image


def _plot_representation(
    path: Path,
    pair_rows: list[dict[str, Any]],
    condition: str,
) -> None:
    labels = [_history_label(history) for history in HISTORIES]
    index = {label: idx for idx, label in enumerate(labels)}
    raw = np.eye(len(labels))
    future = np.zeros_like(raw)
    future_counts = np.zeros_like(raw)
    for row in pair_rows:
        left = index[row["history_a"]]
        right = index[row["history_b"]]
        if row["next_increment"] == 0:
            raw[left, right] = raw[right, left] = row["raw_state_cosine"]
        else:
            value = row["future_prediction_disagreement"]
            future[left, right] += value
            future[right, left] += value
            future_counts[left, right] += 1
            future_counts[right, left] += 1
    future = np.divide(future, future_counts, out=np.zeros_like(future), where=future_counts > 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor("white")
    first = _plot_matrix(
        axes[0],
        raw,
        labels,
        title="Raw recurrent-state cosine",
        vmin=-0.1,
        vmax=1,
        cmap="viridis",
        value_format=".2f",
    )
    second = _plot_matrix(
        axes[1],
        future,
        labels,
        title="Future prediction disagreement",
        vmin=0,
        vmax=max(0.01, float(future.max())),
        cmap="magma",
        value_format=".2f",
    )
    fig.colorbar(first, ax=axes[0], shrink=0.8)
    fig.colorbar(second, ax=axes[1], shrink=0.8)
    fig.suptitle(f"History-state comparison: {condition}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _plot_future(
    path: Path,
    future_rows: list[dict[str, Any]],
    condition: str,
) -> None:
    labels = [_history_label(history) for history in HISTORIES]
    error = np.full((len(labels), 4), np.nan)
    accuracy = np.full_like(error, np.nan)
    for row in future_rows:
        history = labels.index(row["history"])
        increment = row["increment"] - 1
        error[history, increment] = row["realized_progress_error"]
        accuracy[history, increment] = row["frontier_accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.patch.set_facecolor("white")
    for ax, matrix, title, cmap, vmin, vmax, fmt in (
        (axes[0], error, "Realized progress minus request", "RdBu_r", -4, 4, "+.1f"),
        (axes[1], accuracy, "Future-frontier accuracy", "viridis", 0, 1, ".2f"),
    ):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set(
            title=title,
            xlabel="next instruction d",
            ylabel="history to semantic depth 4",
            xticks=range(4),
            xticklabels=[1, 2, 3, 4],
            yticks=range(len(labels)),
            yticklabels=labels,
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    format(matrix[row, column], fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"Future transitions from equal target depth: {condition}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _plot_patch_recovery(
    path: Path,
    patch_rows: list[dict[str, Any]],
    condition: str,
) -> None:
    regions = [
        "resolved",
        "random_matched_resolved",
        "current_frontier",
        "random_matched_current_frontier",
        "unresolved",
        "random_matched_unresolved",
        "next_frontier",
        "random_matched_next_frontier",
        "all",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    for ax, direction in zip(axes, ("good_to_cold", "cold_to_good")):
        selected = [row for row in patch_rows if row["direction"] == direction]
        matrix = np.full((len(regions), 4), np.nan)
        for row in selected:
            matrix[regions.index(row["region"]), row["increment"] - 1] = row[
                "aggregate_normalized_margin_recovery"
            ]
        finite = matrix[np.isfinite(matrix)]
        limit = max(1.0, float(np.abs(finite).max())) if finite.size else 1.0
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(
            title=direction.replace("_", " "),
            xlabel="next instruction d",
            ylabel="donor-patched node region",
            xticks=range(4),
            xticklabels=[1, 2, 3, 4],
            yticks=range(len(regions)),
            yticklabels=regions,
        )
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.8, label="normalized margin recovery")
    fig.suptitle(f"History-state patching: {condition}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


@torch.no_grad()
def evaluate_history_interchange(
    *,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 510,
    batches: int = 8,
    amp: bool = True,
) -> dict[str, Any]:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size must be a positive even number")
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    data_cfg = BooleanDAGConfig(**checkpoint_data["data_config"])
    model_cfg = BooleanDAGModelConfig(**checkpoint_data["model_config"])
    if data_cfg.eval_max_depth < SEMANTIC_DEPTH + 4:
        raise ValueError("the graph configuration cannot realize all future instructions")
    model = DepthConditionedBooleanDAGTransformer(
        data_cfg,
        model_cfg,
        use_instruction=checkpoint_data["use_instruction"],
    ).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    set_seed(EVAL_SEED)
    data_generator = torch.Generator(device=device).manual_seed(EVAL_SEED)
    mask_generator = torch.Generator(device=device).manual_seed(MASK_SEED)

    state_accumulators = {_history_label(history): _Accumulator() for history in HISTORIES}
    future_accumulators = {
        (_history_label(history), increment): _Accumulator()
        for history in HISTORIES
        for increment in range(1, 5)
    }
    pair_accumulators = {
        (_history_label(left), _history_label(right), increment): _Accumulator()
        for left, right in itertools.combinations(HISTORIES, 2)
        for increment in range(5)
    }
    patch_accumulators = {
        (direction, _history_label(donor), _history_label(receiver), increment, region): _Accumulator()
        for donor, receiver, direction in PATCH_PAIRS
        for increment in range(1, 5)
        for region in (
            "all",
            "resolved",
            "current_frontier",
            "unresolved",
            "next_frontier",
            "random_matched_resolved",
            "random_matched_current_frontier",
            "random_matched_unresolved",
            "random_matched_next_frontier",
        )
    }
    self_patch_max = 0.0
    whole_patch_max = 0.0
    same_loop_state_max = 0.0
    graph_hasher = hashlib.sha256()

    for _ in range(batches):
        batch = make_boolean_dag_batch(
            data_cfg,
            batch_size,
            device,
            depths=torch.full(
                (batch_size,),
                data_cfg.eval_max_depth,
                device=device,
            ),
            generator=data_generator,
        )
        _update_batch_fingerprint(graph_hasher, batch)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            states = {history: run_history(model, batch, history) for history in HISTORIES}
            state_metrics = {
                history: readout_metrics(model, state, batch, SEMANTIC_DEPTH)
                for history, state in states.items()
            }
            futures = {
                (history, increment): future_metrics(
                    model,
                    state,
                    batch,
                    SEMANTIC_DEPTH,
                    increment,
                )
                for history, state in states.items()
                for increment in range(1, 5)
            }

            same_loop_states = [states[history] for history in ((1, 3), (2, 2), (3, 1))]
            for left, right in itertools.combinations(same_loop_states, 2):
                same_loop_state_max = max(
                    same_loop_state_max,
                    float((left - right).abs().max()),
                )

            for history in HISTORIES:
                metrics = state_metrics[history]
                state_accumulators[_history_label(history)].add(
                    state_accuracy=float(metrics["state_accuracy"].mean()),
                    exact_state_accuracy=float(metrics["exact_state_accuracy"].mean()),
                    realized_depth=float(metrics["realized_depth"].mean()),
                    target_margin=float(metrics["target_margin"].mean()),
                )
                for increment in range(1, 5):
                    result = futures[(history, increment)]
                    future_accumulators[(_history_label(history), increment)].add(
                        state_accuracy=float(result["state_accuracy"].mean()),
                        exact_state_accuracy=float(result["exact_state_accuracy"].mean()),
                        frontier_accuracy=float(result["frontier_accuracy"].mean()),
                        frontier_margin=float(result["frontier_margin"].mean()),
                        realized_depth=float(result["realized_depth"].mean()),
                        realized_progress=float(result["realized_progress"].mean()),
                    )

            for left, right in itertools.combinations(HISTORIES, 2):
                left_metrics = state_metrics[left]
                right_metrics = state_metrics[right]
                base_values = {
                    "raw_state_cosine": _cosine_mean(states[left], states[right]),
                    "normalized_state_cosine": _cosine_mean(
                        left_metrics["normalized_state"],
                        right_metrics["normalized_state"],
                    ),
                    "raw_state_relative_l2": _relative_l2(states[left], states[right]),
                    "prediction_disagreement": float(
                        left_metrics["predictions"].ne(right_metrics["predictions"]).float().mean()
                    ),
                }
                pair_accumulators[
                    (_history_label(left), _history_label(right), 0)
                ].add(
                    **base_values,
                    future_prediction_disagreement=float("nan"),
                    future_frontier_disagreement=float("nan"),
                )
                for increment in range(1, 5):
                    left_future = futures[(left, increment)]
                    right_future = futures[(right, increment)]
                    frontier = left_future["frontier_mask"]
                    pair_accumulators[
                        (_history_label(left), _history_label(right), increment)
                    ].add(
                        **base_values,
                        future_prediction_disagreement=float(
                            left_future["predictions"]
                            .ne(right_future["predictions"])
                            .float()
                            .mean()
                        ),
                        future_frontier_disagreement=float(
                            _masked_example_mean(
                                left_future["predictions"]
                                .ne(right_future["predictions"])
                                .float(),
                                frontier,
                            ).mean()
                        ),
                    )

            reference_state = states[(1, 3)]
            for increment in range(1, 5):
                self_patched = patch_state(
                    reference_state,
                    reference_state,
                    torch.ones_like(batch.levels, dtype=torch.bool),
                )
                self_result = future_metrics(
                    model,
                    self_patched,
                    batch,
                    SEMANTIC_DEPTH,
                    increment,
                )
                self_patch_max = max(
                    self_patch_max,
                    float(
                        (
                            self_result["logits"]
                            - futures[((1, 3), increment)]["logits"]
                        )
                        .abs()
                        .max()
                    ),
                )

            for donor_history, receiver_history, direction in PATCH_PAIRS:
                donor_state = states[donor_history]
                receiver_state = states[receiver_history]
                for increment in range(1, 5):
                    donor_result = futures[(donor_history, increment)]
                    receiver_result = futures[(receiver_history, increment)]
                    masks = history_region_masks(
                        batch,
                        semantic_depth=SEMANTIC_DEPTH,
                        increment=increment,
                        generator=mask_generator,
                    )
                    donor_margin = donor_result["frontier_margin"]
                    receiver_margin = receiver_result["frontier_margin"]
                    for region, mask in masks.items():
                        patched_state = patch_state(receiver_state, donor_state, mask)
                        patched_result = future_metrics(
                            model,
                            patched_state,
                            batch,
                            SEMANTIC_DEPTH,
                            increment,
                        )
                        patched_margin = patched_result["frontier_margin"]
                        recovery = normalized_recovery(
                            receiver_margin,
                            donor_margin,
                            patched_margin,
                        )
                        random_reference_overlap = float("nan")
                        if region.startswith("random_matched_"):
                            reference_mask = masks[region.removeprefix("random_matched_")]
                            overlap = (mask & reference_mask).sum(dim=1).float()
                            random_reference_overlap = float(
                                (overlap / reference_mask.sum(dim=1).clamp_min(1)).mean()
                            )
                        if region == "all":
                            whole_patch_max = max(
                                whole_patch_max,
                                float(
                                    (
                                        patched_result["logits"] - donor_result["logits"]
                                    )
                                    .abs()
                                    .max()
                                ),
                            )
                        patch_accumulators[
                            (
                                direction,
                                _history_label(donor_history),
                                _history_label(receiver_history),
                                increment,
                                region,
                            )
                        ].add(
                            donor_frontier_accuracy=float(
                                donor_result["frontier_accuracy"].mean()
                            ),
                            receiver_frontier_accuracy=float(
                                receiver_result["frontier_accuracy"].mean()
                            ),
                            patched_frontier_accuracy=float(
                                patched_result["frontier_accuracy"].mean()
                            ),
                            donor_frontier_margin=float(donor_margin.mean()),
                            receiver_frontier_margin=float(receiver_margin.mean()),
                            patched_frontier_margin=float(patched_margin.mean()),
                            mean_examplewise_normalized_margin_recovery=float(
                                recovery[torch.isfinite(recovery)].mean()
                            )
                            if torch.isfinite(recovery).any()
                            else float("nan"),
                            patched_node_fraction=float(mask.float().mean()),
                            random_reference_overlap_fraction=random_reference_overlap,
                            prediction_disagreement_from_receiver=float(
                                patched_result["predictions"]
                                .ne(receiver_result["predictions"])
                                .float()
                                .mean()
                            ),
                            prediction_disagreement_from_donor=float(
                                patched_result["predictions"]
                                .ne(donor_result["predictions"])
                                .float()
                                .mean()
                            ),
                        )

    condition = checkpoint_data["condition"]
    history_rows = [
        {"condition": condition, "history": label, **accumulator.means()}
        for label, accumulator in state_accumulators.items()
    ]
    future_rows = []
    for (history, increment), accumulator in future_accumulators.items():
        row = {
            "condition": condition,
            "history": history,
            "history_length": history.count("+") + 1,
            "semantic_depth": SEMANTIC_DEPTH,
            "increment": increment,
            "target_depth": SEMANTIC_DEPTH + increment,
            **accumulator.means(),
        }
        row["realized_progress_error"] = row["realized_progress"] - increment
        future_rows.append(row)
    pair_rows = [
        {
            "condition": condition,
            "history_a": history_a,
            "history_b": history_b,
            "next_increment": increment,
            **accumulator.means(),
        }
        for (history_a, history_b, increment), accumulator in pair_accumulators.items()
    ]
    patch_rows = [
        {
            "condition": condition,
            "direction": direction,
            "donor_history": donor,
            "receiver_history": receiver,
            "increment": increment,
            "region": region,
            **accumulator.means(),
        }
        for (direction, donor, receiver, increment, region), accumulator in patch_accumulators.items()
    ]
    for row in patch_rows:
        row["aggregate_normalized_margin_recovery"] = _aggregate_recovery(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "history_state_metrics.csv", history_rows)
    _write_csv(out_dir / "history_pair_metrics.csv", pair_rows)
    _write_csv(out_dir / "future_transition_metrics.csv", future_rows)
    _write_csv(out_dir / "state_patch_metrics.csv", patch_rows)
    _plot_representation(out_dir / "representation_similarity.png", pair_rows, condition)
    _plot_future(out_dir / "future_transition.png", future_rows, condition)
    _plot_patch_recovery(out_dir / "state_patch_recovery.png", patch_rows, condition)

    primary_rows = [row for row in patch_rows if row["direction"] == "good_to_cold"]
    causal_regions = {"resolved", "current_frontier", "unresolved", "next_frontier"}
    random_regions = {f"random_matched_{name}" for name in causal_regions}
    summary = {
        "task_version": checkpoint_data.get("task_version"),
        "checkpoint": str(checkpoint),
        "condition": condition,
        "use_instruction": checkpoint_data["use_instruction"],
        "checkpoint_step": checkpoint_data.get("step"),
        "batch_size": batch_size,
        "batches": batches,
        "example_count": batch_size * batches,
        "semantic_depth": SEMANTIC_DEPTH,
        "histories": [_history_label(history) for history in HISTORIES],
        "graph_seed": EVAL_SEED,
        "graph_manifest_sha256": graph_hasher.hexdigest(),
        "mask_seed": MASK_SEED,
        "cuda_max_memory_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else None
        ),
        "self_patch_max_abs_logit_difference": self_patch_max,
        "whole_patch_max_abs_logit_difference_from_donor": whole_patch_max,
        "same_loop_raw_state_max_abs_difference": same_loop_state_max,
        "same_loop_future_prediction_disagreement_mean": float(
            np.mean(
                [
                    row["future_prediction_disagreement"]
                    for row in pair_rows
                    if row["next_increment"] > 0
                    and row["history_a"] in {"1+3", "2+2", "3+1"}
                    and row["history_b"] in {"1+3", "2+2", "3+1"}
                ]
            )
        ),
        "future_progress_mae_by_history": {
            history: float(
                np.mean(
                    [
                        abs(row["realized_progress_error"])
                        for row in future_rows
                        if row["history"] == history
                    ]
                )
            )
            for history in [_history_label(value) for value in HISTORIES]
        },
        "primary_patch_causal_region_recovery_mean": float(
            np.nanmean(
                [
                    row["aggregate_normalized_margin_recovery"]
                    for row in primary_rows
                    if row["region"] in causal_regions
                ]
            )
        ),
        "primary_patch_random_region_recovery_mean": float(
            np.nanmean(
                [
                    row["aggregate_normalized_margin_recovery"]
                    for row in primary_rows
                    if row["region"] in random_regions
                ]
            )
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze equal-depth history states.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=510)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_history_interchange(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=pick_device(args.device),
        batch_size=args.batch_size,
        batches=args.batches,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
