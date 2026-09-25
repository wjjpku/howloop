from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    name, checkpoint = text.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initial_state(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
) -> torch.Tensor:
    state = model.token_embed(tokens)
    if model.block_style == "legacy":
        state = state + model.pos_embed.unsqueeze(0)
    return state


def masked_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, int]:
    count = int(valid.sum())
    if count == 0:
        return float("nan"), 0
    return float(prediction[valid].eq(target[valid]).float().mean()), count


def mean_relative_distance(
    current: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    if current.shape != reference.shape:
        raise ValueError("current and reference must have identical shapes")
    current_flat = current.float().flatten(1)
    reference_flat = reference.float().flatten(1)
    return float(
        (current_flat - reference_flat)
        .norm(dim=1)
        .div(reference_flat.norm(dim=1).clamp_min(eps))
        .mean()
    )


def mean_cosine(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    if current.shape != reference.shape:
        raise ValueError("current and reference must have identical shapes")
    return float(
        F.cosine_similarity(
            current.float().flatten(1),
            reference.float().flatten(1),
            dim=1,
        ).mean()
    )


def earliest_stable_loop(
    endpoint_accuracies: list[float],
    *,
    trained_loops: int,
    threshold: float = 0.95,
) -> int:
    if len(endpoint_accuracies) <= trained_loops:
        raise ValueError("endpoint_accuracies must include loop zero")
    for loop in range(1, trained_loops + 1):
        if all(
            value >= threshold
            for value in endpoint_accuracies[loop : trained_loops + 1]
        ):
            return loop
    return trained_loops


def feature_groups(cfg: GraphPathConfig) -> dict[str, tuple[int, ...]]:
    groups = explicit_depth_position_groups(cfg.node_count)
    return {
        "graph": groups["graph"],
        "registers": (
            groups["start"] + groups["depth"] + groups["answer"]
        ),
        "answer": groups["answer"],
        "all": tuple(range(cfg.seq_len)),
    }


def probe_features(
    model: LoopedGraphPathTransformer,
    state: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    return {
        "answer_raw": state[:, groups["answer"][0]].float(),
        "answer_readout": model.ln_final(
            state[:, groups["answer"][0]]
        ).float(),
        "registers_flat": state[:, list(groups["registers"])].float().flatten(1),
        "all_mean": state.float().mean(dim=1),
    }


def fit_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, torch.Tensor]:
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("ridge inputs must have [sample, feature] and [sample]")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("ridge sample counts do not match")
    x = features.float()
    y = targets.float()
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-5)
    x = (x - mean) / scale
    y_mean = y.mean()
    y_centered = y - y_mean
    dimension = x.shape[1]
    gram = x.T @ x
    regularizer = ridge * x.shape[0] * torch.eye(
        dimension, device=x.device, dtype=x.dtype
    )
    weight = torch.linalg.solve(gram + regularizer, x.T @ y_centered)
    return {
        "mean": mean,
        "scale": scale,
        "weight": weight,
        "target_mean": y_mean,
    }


def ridge_predict(
    model: dict[str, torch.Tensor],
    features: torch.Tensor,
) -> torch.Tensor:
    normalized = (
        features.float() - model["mean"]
    ) / model["scale"]
    return normalized @ model["weight"] + model["target_mean"]


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.float()
    target = target.float()
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1e-12)
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    correlation = (
        (centered_prediction * centered_target).sum()
        / (
            centered_prediction.square().sum().sqrt()
            * centered_target.square().sum().sqrt()
        ).clamp_min(1e-12)
    )
    return {
        "r2": float(1.0 - residual / total),
        "mae": float((prediction - target).abs().mean()),
        "correlation": float(correlation),
        "rounded_accuracy": float(
            prediction.round().eq(target).float().mean()
        ),
    }


def nearest_centroid_accuracy(
    train_features: torch.Tensor,
    train_ages: torch.Tensor,
    test_features: torch.Tensor,
    test_ages: torch.Tensor,
) -> float:
    ages = train_ages.unique(sorted=True)
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    centroids = torch.stack(
        [train[train_ages.eq(age)].mean(dim=0) for age in ages]
    )
    distance = torch.cdist(test.float(), centroids.float())
    prediction = ages[distance.argmin(dim=1)]
    return float(prediction.eq(test_ages).float().mean())


def summarize_accumulator(
    accumulators: list[dict[str, float]],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for loop, bucket in enumerate(accumulators):
        batches = max(1.0, bucket["batch_count"])
        endpoint_count = max(1.0, bucket["endpoint_count"])
        continuation_count = bucket["continuation_count"]
        row: dict[str, float | int] = {"loop": loop}
        for key in (
            "argmax_change_rate",
            "js_divergence",
            "state_norm_all",
            "update_rel_all",
            "cosine_prev_all",
            "period2_rel_all",
            "period4_rel_all",
            "answer_norm_raw",
            "answer_update_rel",
            "answer_cosine_prev",
            "readout_norm",
            "readout_update_rel",
            "readout_cosine_prev",
            "graph_update_rel",
            "registers_update_rel",
            "distance_to_h8_all",
            "distance_to_h8_readout",
        ):
            row[key] = bucket[key] / batches
        row["endpoint_accuracy"] = (
            bucket["endpoint_correct"] / endpoint_count
        )
        row["endpoint_probability"] = (
            bucket["endpoint_probability"] / endpoint_count
        )
        row["continued_twohop_accuracy"] = (
            bucket["continuation_correct"] / continuation_count
            if continuation_count
            else float("nan")
        )
        row["continued_twohop_probability"] = (
            bucket["continuation_probability"] / continuation_count
            if continuation_count
            else float("nan")
        )
        row["continuation_valid_count"] = int(continuation_count)
        rows.append(row)
    return rows


@torch.no_grad()
def long_dynamics(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    batch_size: int,
    batches: int,
    loops: int,
    seed: int,
) -> list[dict[str, float | int]]:
    keys = (
        "batch_count",
        "endpoint_correct",
        "endpoint_count",
        "endpoint_probability",
        "continuation_correct",
        "continuation_count",
        "continuation_probability",
        "argmax_change_rate",
        "js_divergence",
        "state_norm_all",
        "update_rel_all",
        "cosine_prev_all",
        "period2_rel_all",
        "period4_rel_all",
        "answer_norm_raw",
        "answer_update_rel",
        "answer_cosine_prev",
        "readout_norm",
        "readout_update_rel",
        "readout_cosine_prev",
        "graph_update_rel",
        "registers_update_rel",
        "distance_to_h8_all",
        "distance_to_h8_readout",
    )
    accumulators = [{key: 0.0 for key in keys} for _ in range(loops + 1)]
    groups = feature_groups(cfg)
    set_seed(seed)
    for _ in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=max(cfg.max_depth, 2 * loops),
        )
        endpoint = targets[:, cfg.max_depth - 1]
        state = initial_state(model, tokens)
        history: deque[torch.Tensor] = deque(maxlen=5)
        previous_logits: torch.Tensor | None = None
        anchor_h8: torch.Tensor | None = None
        anchor_h8_readout: torch.Tensor | None = None
        for loop in range(loops + 1):
            logits = logits_from_raw_state(model, state).float()
            probability = logits.softmax(dim=-1)
            prediction = logits.argmax(dim=-1)
            readout = model.ln_final(
                state[:, groups["answer"][0]]
            ).float()
            bucket = accumulators[loop]
            bucket["batch_count"] += 1
            bucket["endpoint_correct"] += float(
                prediction.eq(endpoint).sum()
            )
            bucket["endpoint_count"] += endpoint.numel()
            bucket["endpoint_probability"] += float(
                probability.gather(1, endpoint[:, None]).sum()
            )
            continued_position = 2 * loop
            continued_target = (
                start
                if continued_position == 0
                else targets[:, continued_position - 1]
            )
            valid = continued_target.ne(endpoint)
            if loop == cfg.max_depth // 2:
                valid = torch.ones_like(valid)
            continued_accuracy, valid_count = masked_accuracy(
                prediction, continued_target, valid
            )
            if valid_count:
                bucket["continuation_correct"] += (
                    continued_accuracy * valid_count
                )
                bucket["continuation_count"] += valid_count
                bucket["continuation_probability"] += float(
                    probability[valid]
                    .gather(1, continued_target[valid, None])
                    .sum()
                )
            bucket["state_norm_all"] += float(
                state.float().flatten(1).norm(dim=1).mean()
            )
            bucket["answer_norm_raw"] += float(
                state[:, groups["answer"][0]].float().norm(dim=1).mean()
            )
            bucket["readout_norm"] += float(readout.norm(dim=1).mean())
            if history:
                previous = history[-1]
                previous_readout = model.ln_final(
                    previous[:, groups["answer"][0]]
                ).float()
                bucket["update_rel_all"] += mean_relative_distance(
                    state, previous
                )
                bucket["cosine_prev_all"] += mean_cosine(state, previous)
                bucket["answer_update_rel"] += mean_relative_distance(
                    state[:, list(groups["answer"])],
                    previous[:, list(groups["answer"])],
                )
                bucket["answer_cosine_prev"] += mean_cosine(
                    state[:, list(groups["answer"])],
                    previous[:, list(groups["answer"])],
                )
                bucket["readout_update_rel"] += mean_relative_distance(
                    readout, previous_readout
                )
                bucket["readout_cosine_prev"] += mean_cosine(
                    readout, previous_readout
                )
                bucket["graph_update_rel"] += mean_relative_distance(
                    state[:, list(groups["graph"])],
                    previous[:, list(groups["graph"])],
                )
                bucket["registers_update_rel"] += mean_relative_distance(
                    state[:, list(groups["registers"])],
                    previous[:, list(groups["registers"])],
                )
                previous_probability = previous_logits.softmax(dim=-1)
                mixture = 0.5 * (probability + previous_probability)
                js = 0.5 * (
                    F.kl_div(
                        mixture.log(),
                        probability,
                        reduction="batchmean",
                    )
                    + F.kl_div(
                        mixture.log(),
                        previous_probability,
                        reduction="batchmean",
                    )
                )
                bucket["js_divergence"] += float(js)
                bucket["argmax_change_rate"] += float(
                    prediction.ne(previous_logits.argmax(dim=-1))
                    .float()
                    .mean()
                )
            else:
                for key in (
                    "update_rel_all",
                    "cosine_prev_all",
                    "answer_update_rel",
                    "answer_cosine_prev",
                    "readout_update_rel",
                    "readout_cosine_prev",
                    "graph_update_rel",
                    "registers_update_rel",
                    "js_divergence",
                    "argmax_change_rate",
                ):
                    bucket[key] += float("nan")
            if len(history) >= 2:
                bucket["period2_rel_all"] += mean_relative_distance(
                    state, history[-2]
                )
            else:
                bucket["period2_rel_all"] += float("nan")
            if len(history) >= 4:
                bucket["period4_rel_all"] += mean_relative_distance(
                    state, history[-4]
                )
            else:
                bucket["period4_rel_all"] += float("nan")
            if loop == cfg.max_loops:
                anchor_h8 = state.clone()
                anchor_h8_readout = readout.clone()
            if anchor_h8 is not None and anchor_h8_readout is not None:
                bucket["distance_to_h8_all"] += mean_relative_distance(
                    state, anchor_h8
                )
                bucket["distance_to_h8_readout"] += mean_relative_distance(
                    readout, anchor_h8_readout
                )
            else:
                bucket["distance_to_h8_all"] += float("nan")
                bucket["distance_to_h8_readout"] += float("nan")
            history.append(state)
            previous_logits = logits
            if loop < loops:
                state = apply_shared_stack(
                    model, state, loop_index=loop
                )
    return summarize_accumulator(accumulators)


@torch.no_grad()
def collect_age_features(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    ages: list[int],
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    groups = feature_groups(cfg)
    collected: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("answer_raw", "answer_readout", "registers_flat", "all_mean")
    }
    set_seed(seed)
    for _ in range(batches):
        tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        state = initial_state(model, tokens)
        states = {0: state}
        for loop in range(1, max(ages) + 1):
            state = apply_shared_stack(model, state, loop_index=loop - 1)
            if loop in ages:
                states[loop] = state
        by_feature: dict[str, list[torch.Tensor]] = {
            key: [] for key in collected
        }
        for age in ages:
            features = probe_features(model, states[age], groups)
            for key, value in features.items():
                by_feature[key].append(value.cpu())
        for key, values in by_feature.items():
            collected[key].append(torch.stack(values, dim=1))
    return {
        key: torch.cat(values, dim=0)
        for key, values in collected.items()
    }


def flatten_age_dataset(
    feature_tensor: torch.Tensor,
    ages: list[int],
    *,
    residualize_within_example: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_tensor.ndim != 3:
        raise ValueError("feature tensor must have [example, age, feature]")
    features = feature_tensor.float()
    if residualize_within_example:
        features = features - features.mean(dim=1, keepdim=True)
    targets = torch.as_tensor(ages, dtype=torch.float32).repeat(
        features.shape[0]
    )
    return features.flatten(0, 1), targets


def age_probe_rows(
    *,
    train_features: dict[str, torch.Tensor],
    test_features: dict[str, torch.Tensor],
    ages: list[int],
    ridge: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    raw_models: dict[str, dict[str, torch.Tensor]] = {}
    for feature_name in train_features:
        for residualized in (False, True):
            train_x, train_y = flatten_age_dataset(
                train_features[feature_name],
                ages,
                residualize_within_example=residualized,
            )
            test_x, test_y = flatten_age_dataset(
                test_features[feature_name],
                ages,
                residualize_within_example=residualized,
            )
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            ridge_model = fit_ridge(train_x, train_y, ridge=ridge)
            prediction = ridge_predict(ridge_model, test_x)
            metrics = regression_metrics(prediction, test_y)
            centroid_accuracy = nearest_centroid_accuracy(
                train_x,
                train_y,
                test_x,
                test_y,
            )
            rows.append(
                {
                    "feature": feature_name,
                    "residualized_within_example": residualized,
                    "ages": "-".join(map(str, ages)),
                    "chance_rounded_accuracy": 1.0 / len(ages),
                    "nearest_centroid_accuracy": centroid_accuracy,
                    **metrics,
                }
            )
            if not residualized:
                raw_models[feature_name] = {
                    key: value.detach()
                    for key, value in ridge_model.items()
                }
    return rows, raw_models


@torch.no_grad()
def age_extrapolation_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    ridge_models: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    batch_size: int,
    loops: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups = feature_groups(cfg)
    set_seed(seed)
    tokens, _, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=cfg.max_depth,
    )
    state = initial_state(model, tokens)
    rows: list[dict[str, Any]] = []
    for loop in range(loops + 1):
        features = probe_features(model, state, groups)
        for feature_name, ridge_model in ridge_models.items():
            prediction = ridge_predict(
                ridge_model,
                features[feature_name],
            )
            rows.append(
                {
                    "loop": loop,
                    "feature": feature_name,
                    "predicted_age_mean": float(prediction.mean()),
                    "predicted_age_std": float(
                        prediction.std(unbiased=False)
                    ),
                }
            )
        if loop < loops:
            state = apply_shared_stack(model, state, loop_index=loop)
    return rows


def plot_dynamics(
    all_rows: dict[str, list[dict[str, Any]]],
    *,
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    for name, rows in all_rows.items():
        selected_rows = [
            row for row in rows if int(row["loop"]) >= 4
        ]
        loops = [int(row["loop"]) for row in selected_rows]
        loop8_norm = float(rows[8]["state_norm_all"])
        axes[0, 0].plot(
            loops,
            [float(row["endpoint_accuracy"]) for row in selected_rows],
            label=name,
        )
        axes[0, 1].plot(
            loops,
            [float(row["update_rel_all"]) for row in selected_rows],
            label=name,
        )
        axes[1, 0].plot(
            loops,
            [
                float(row["state_norm_all"]) / loop8_norm
                for row in selected_rows
            ],
            label=name,
        )
        axes[1, 1].plot(
            loops,
            [
                float(row["readout_cosine_prev"])
                for row in selected_rows
            ],
            label=name,
        )
    axes[0, 0].set_ylabel("endpoint accuracy")
    axes[0, 1].set_ylabel(r"$||h_t-h_{t-1}||/||h_{t-1}||$")
    axes[1, 0].set_ylabel(r"$||h_t||/||h_8||$")
    axes[1, 1].set_ylabel("readout-state cosine to previous")
    for axis in axes.flat:
        axis.axvline(8, color="black", linestyle="--", alpha=0.35)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("macro loop")
    axes[1, 1].set_xlabel("macro loop")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylim(-1.03, 1.03)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Hidden-state telomere dynamics: output stability vs state motion")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_age_extrapolation(
    all_rows: dict[str, list[dict[str, Any]]],
    *,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharex=True)
    features = ("answer_readout", "registers_flat")
    for axis, feature in zip(axes, features, strict=True):
        for name, rows in all_rows.items():
            selected = [row for row in rows if row["feature"] == feature]
            axis.plot(
                [row["loop"] for row in selected],
                [row["predicted_age_mean"] for row in selected],
                label=name,
            )
        axis.axvline(8, color="black", linestyle="--", alpha=0.35)
        axis.set_title(feature)
        axis.set_xlabel("macro loop")
        axis.set_ylabel("linear age prediction")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Extrapolation of age probes trained only inside the trained horizon")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    loops: int,
    probe_batch_size: int,
    probe_batches: int,
    ridge: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("analysis expects D8L8 with two shared physical blocks")
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    dynamics = long_dynamics(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        batches=batches,
        loops=loops,
        seed=seed,
    )
    write_csv(run_dir / "long_dynamics.csv", dynamics)
    endpoint_accuracies = [
        float(row["endpoint_accuracy"]) for row in dynamics
    ]
    plateau_start = earliest_stable_loop(
        endpoint_accuracies,
        trained_loops=cfg.max_loops,
    )
    plateau_ages = list(range(plateau_start, cfg.max_loops + 1))
    train_features = collect_age_features(
        model=model,
        cfg=cfg,
        device=device,
        ages=plateau_ages,
        batch_size=probe_batch_size,
        batches=probe_batches,
        seed=seed + 100,
    )
    test_features = collect_age_features(
        model=model,
        cfg=cfg,
        device=device,
        ages=plateau_ages,
        batch_size=probe_batch_size,
        batches=probe_batches,
        seed=seed + 200,
    )
    probe_rows, ridge_models = age_probe_rows(
        train_features=train_features,
        test_features=test_features,
        ages=plateau_ages,
        ridge=ridge,
        device=device,
    )
    write_csv(run_dir / "plateau_age_probe.csv", probe_rows)
    extrapolation = age_extrapolation_rows(
        model=model,
        cfg=cfg,
        ridge_models=ridge_models,
        device=device,
        batch_size=probe_batch_size,
        loops=loops,
        seed=seed + 300,
    )
    write_csv(run_dir / "age_extrapolation.csv", extrapolation)
    endpoint_above_90 = [
        int(row["loop"])
        for row in dynamics
        if float(row["endpoint_accuracy"]) >= 0.90
    ]
    last_above_90 = max(endpoint_above_90) if endpoint_above_90 else -1
    final_row = dynamics[-1]
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": asdict(cfg),
        "trained_loops": cfg.max_loops,
        "physical_shared_blocks": cfg.n_layers,
        "trained_effective_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_loops": loops,
        "examples": batch_size * batches,
        "plateau_start_loop": plateau_start,
        "plateau_ages_used_for_probe": plateau_ages,
        "endpoint_accuracy_loop8": endpoint_accuracies[cfg.max_loops],
        "last_loop_with_endpoint_accuracy_at_least_0p90": last_above_90,
        "endpoint_accuracy_loop200": float(final_row["endpoint_accuracy"]),
        "state_update_relative_loop8": float(
            dynamics[cfg.max_loops]["update_rel_all"]
        ),
        "state_update_relative_loop200": float(
            final_row["update_rel_all"]
        ),
        "state_norm_loop8": float(
            dynamics[cfg.max_loops]["state_norm_all"]
        ),
        "state_norm_loop200": float(final_row["state_norm_all"]),
        "readout_cosine_prev_loop200": float(
            final_row["readout_cosine_prev"]
        ),
        "periodicity_loop200": {
            "period1_relative_distance": float(
                final_row["update_rel_all"]
            ),
            "period2_relative_distance": float(
                final_row["period2_rel_all"]
            ),
            "period4_relative_distance": float(
                final_row["period4_rel_all"]
            ),
        },
        "age_probe": probe_rows,
        "claim_ledger": {
            "output_fixed_point_through_200": (
                float(final_row["endpoint_accuracy"]) >= 0.90
            ),
            "hidden_fixed_point_at_200": (
                float(final_row["update_rel_all"]) <= 0.01
                and float(final_row["distance_to_h8_all"]) <= 0.05
            ),
            "projective_readout_fixed_point_at_200": (
                float(final_row["readout_cosine_prev"]) >= 0.9999
                and float(final_row["endpoint_accuracy"]) >= 0.90
            ),
            "state_norm_bounded_relative_to_loop8": (
                0.90
                <= float(final_row["state_norm_all"])
                / float(dynamics[cfg.max_loops]["state_norm_all"])
                <= 1.10
            ),
            "period2_cycle_at_200": (
                float(final_row["period2_rel_all"])
                <= 0.25 * float(final_row["update_rel_all"])
            ),
            "period4_cycle_at_200": (
                float(final_row["period4_rel_all"])
                <= 0.25 * float(final_row["update_rel_all"])
            ),
            "plateau_age_linearly_decodable": any(
                row["feature"] == "answer_readout"
                and not row["residualized_within_example"]
                and float(row["r2"]) >= 0.80
                for row in probe_rows
            ),
            "plateau_age_survives_within_example_content_removal": any(
                row["feature"] == "answer_readout"
                and row["residualized_within_example"]
                and float(row["r2"]) >= 0.80
                for row in probe_rows
            ),
        },
        "cuda_peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 2**30
            if device.type == "cuda"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, dynamics, extrapolation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Separate output arrest, hidden-state convergence, periodicity, "
            "and age decodability in D8L8 graph-path checkpoints."
        )
    )
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--loops", type=int, default=200)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries: dict[str, Any] = {}
    dynamics: dict[str, list[dict[str, Any]]] = {}
    extrapolations: dict[str, list[dict[str, Any]]] = {}
    for name, checkpoint in args.run:
        summary, model_dynamics, extrapolation = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            loops=args.loops,
            probe_batch_size=args.probe_batch_size,
            probe_batches=args.probe_batches,
            ridge=args.ridge,
            seed=args.seed,
        )
        summaries[name] = summary
        dynamics[name] = model_dynamics
        extrapolations[name] = extrapolation
        print(f"done {name}", flush=True)
    plot_dynamics(
        dynamics, path=args.out_dir / "combined_telomere_dynamics.png"
    )
    plot_age_extrapolation(
        extrapolations,
        path=args.out_dir / "combined_age_extrapolation.png",
    )
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
