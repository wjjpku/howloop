"""Natural-state and task interventions for seven age-specific affine J maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_transition_matrices import (
    affine_metrics,
    fit_affine,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    evaluate_random_trajectories,
)


AGES = tuple(range(2, 9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=826001)
    parser.add_argument("--state-train-examples", type=int, default=512)
    parser.add_argument("--state-test-examples", type=int, default=256)
    parser.add_argument("--state-batch-size", type=int, default=64)
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument("--affine-ridge", type=float, default=1e-3)
    parser.add_argument("--task-trajectories", type=int, default=56)
    parser.add_argument("--task-batch-size", type=int, default=64)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.045)
    return parser.parse_args()


def load_arrays(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if payload.get("map_architecture") == "shared_diagonal_stage_lora":
        diagonal = np.diag(state["shared_diagonal_scale"].double().numpy())
        shared = (
            state["shared_A"].double().numpy()
            @ state["shared_B"].double().numpy()
        )
        weights = {
            age: diagonal
            + shared
            + state[f"stage_A.{age}"].double().numpy()
            @ state[f"stage_B.{age}"].double().numpy()
            for age in AGES
        }
        shared_bias = state["shared_bias"].double().numpy()
        biases = {age: shared_bias.copy() for age in AGES}
    else:
        weights = {
            age: state[f"maps.{age}.weight"].double().numpy() for age in AGES
        }
        biases = {
            age: state[f"maps.{age}.bias"].double().numpy() for age in AGES
        }
    return weights, biases, payload


def affine_delta(weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.vstack((weight - np.eye(weight.shape[0]), bias[None, :]))


def arrays_from_delta(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimension = delta.shape[1]
    return np.eye(dimension) + delta[:dimension], delta[dimension]


def truncated(value: np.ndarray, rank: int) -> np.ndarray:
    if rank == 0:
        return np.zeros_like(value)
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def polar_factors(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return right-polar factors value = rotation @ deformation."""
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    rotation = left @ right
    deformation = (right.T * singular) @ right
    return rotation, deformation


def make_bank(
    weights: dict[int, np.ndarray],
    biases: dict[int, np.ndarray],
    *,
    dimension: int,
    device: torch.device,
) -> AgeSpecificJBank:
    bank = AgeSpecificJBank(
        dimension=dimension, rank=dimension, map_architecture="full_affine"
    ).to(device)
    with torch.no_grad():
        for age in AGES:
            bank.maps[str(age)].weight.copy_(
                torch.as_tensor(weights[age], device=device, dtype=torch.float32)
            )
            bank.maps[str(age)].bias.copy_(
                torch.as_tensor(biases[age], device=device, dtype=torch.float32)
            )
    return bank.frozen()


def build_variants(
    weights: dict[int, np.ndarray], biases: dict[int, np.ndarray]
) -> dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]]:
    dimension = next(iter(weights.values())).shape[0]
    identity = np.eye(dimension)
    deltas = {age: affine_delta(weights[age], biases[age]) for age in AGES}
    mean_delta = np.mean(list(deltas.values()), axis=0)
    mean_weight, mean_bias = arrays_from_delta(mean_delta)
    mean_flat = mean_delta.reshape(-1)
    step = {
        age: float(deltas[age].reshape(-1) @ mean_flat / (mean_flat @ mean_flat))
        for age in AGES
    }
    residuals = {age: deltas[age] - step[age] * mean_delta for age in AGES}
    simple_residuals = {age: deltas[age] - mean_delta for age in AGES}
    variants: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = {}

    def add(name: str, maps: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        variants[name] = (
            {age: maps[age][0] for age in AGES},
            {age: maps[age][1] for age in AGES},
        )

    add("full", {age: (weights[age], biases[age]) for age in AGES})
    add("shared_mean", {age: (mean_weight, mean_bias) for age in AGES})
    add(
        "scalar_generator",
        {age: arrays_from_delta(step[age] * mean_delta) for age in AGES},
    )
    add("no_bias", {age: (weights[age], np.zeros(dimension)) for age in AGES})
    add("bias_only", {age: (identity, biases[age]) for age in AGES})
    add(
        "stage_weight_shared_bias",
        {age: (weights[age], mean_bias) for age in AGES},
    )
    add(
        "shared_weight_stage_bias",
        {age: (mean_weight, biases[age]) for age in AGES},
    )
    cyclic = {age: AGES[(index + 1) % len(AGES)] for index, age in enumerate(AGES)}
    add(
        "cyclic_stage_residual",
        {
            age: arrays_from_delta(mean_delta + simple_residuals[cyclic[age]])
            for age in AGES
        },
    )
    reverse = {age: 10 - age for age in AGES}
    add(
        "reverse_stage_residual",
        {
            age: arrays_from_delta(mean_delta + simple_residuals[reverse[age]])
            for age in AGES
        },
    )
    for scale in (0.25, 0.5, 0.75, 1.25):
        add(
            f"stage_residual_scale_{scale:g}",
            {
                age: arrays_from_delta(mean_delta + scale * simple_residuals[age])
                for age in AGES
            },
        )
    polar = {age: polar_factors(weights[age]) for age in AGES}
    add(
        "polar_rotation_plus_bias",
        {age: (polar[age][0], biases[age]) for age in AGES},
    )
    add(
        "polar_deformation_plus_bias",
        {age: (polar[age][1], biases[age]) for age in AGES},
    )
    for rank in (1, 2, 4, 8, 16, 32, 64, 128):
        add(
            f"full_delta_rank_{rank}",
            {age: arrays_from_delta(truncated(deltas[age], rank)) for age in AGES},
        )
    for rank in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        add(
            f"mean_plus_stage_residual_rank_{rank}",
            {
                age: arrays_from_delta(
                    step[age] * mean_delta + truncated(residuals[age], rank)
                )
                for age in AGES
            },
        )
    return variants


@torch.no_grad()
def fit_geometric_variants(
    states: dict[int, torch.Tensor], ridge: float
) -> dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]]:
    """Fit non-task baselines from aligned natural hidden states only."""
    backward_weights: dict[int, np.ndarray] = {}
    backward_biases: dict[int, np.ndarray] = {}
    inverse_weights: dict[int, np.ndarray] = {}
    inverse_biases: dict[int, np.ndarray] = {}
    for age in AGES:
        backward_weight, backward_bias = fit_affine(
            states[age], states[age - 1], ridge
        )
        forward_weight, forward_bias = fit_affine(
            states[age - 1], states[age], ridge
        )
        backward_weights[age] = backward_weight.cpu().double().numpy()
        backward_biases[age] = backward_bias.cpu().double().numpy()
        forward_inverse = np.linalg.pinv(
            forward_weight.cpu().double().numpy(), rcond=1e-6
        )
        inverse_weights[age] = forward_inverse
        inverse_biases[age] = -forward_bias.cpu().double().numpy() @ forward_inverse
    return {
        "direct_hidden_regression": (backward_weights, backward_biases),
        "fitted_forward_pseudoinverse": (inverse_weights, inverse_biases),
    }


@torch.no_grad()
def collect_states(
    *, model, cfg, phase_positions: list[int], device: torch.device,
    examples: int, batch_size: int, seed: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    if examples % batch_size:
        raise ValueError("state examples must be divisible by batch size")
    set_seed(seed)
    chunks = {age: [] for age in range(1, 9)}
    successor_chunks = []
    current_chunks = []
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        current = path_targets[:, cfg.max_depth - 1]
        successor_chunks.append(successors)
        current_chunks.append(current)
        for age in chunks:
            chunks[age].append(
                _aligned_state_at_age(
                    model=model, cfg=cfg, successors=successors, current=current,
                    age=age, phase_position=phase_positions[age]
                ).float()
            )
    return (
        {age: torch.cat(values) for age, values in chunks.items()},
        torch.cat(successor_chunks),
        torch.cat(current_chunks),
    )


@torch.no_grad()
def fit_age_probe(states: dict[int, torch.Tensor], ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([states[age][:, -1, :] for age in range(1, 9)]).float()
    y = torch.cat(
        [torch.full((states[age].shape[0],), float(age), device=x.device) for age in range(1, 9)]
    )
    x_mean, y_mean = x.mean(0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    covariance = xc.T @ xc / x.shape[0]
    cross = xc.T @ yc / x.shape[0]
    scale = torch.diagonal(covariance).mean().clamp_min(1e-8)
    weight = torch.linalg.solve(
        covariance + ridge * scale * torch.eye(x.shape[1], device=x.device), cross
    )
    bias = y_mean - x_mean @ weight
    return weight, bias


def state_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = prediction.float().reshape(-1, prediction.shape[-1])
    t = target.float().reshape(-1, target.shape[-1])
    residual = p - t
    centered = t - t.mean(0, keepdim=True)
    return {
        "relative_error": float(residual.norm() / t.norm().clamp_min(1e-12)),
        "r2": float(1.0 - residual.square().sum() / centered.square().sum().clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(p, t, dim=-1).mean()),
    }


@torch.no_grad()
def age_probe_rows(
    *, states: dict[int, torch.Tensor], current: torch.Tensor,
    probe_weight: torch.Tensor, probe_bias: torch.Tensor,
    variant_banks: dict[str, AgeSpecificJBank], model, positions: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    all_predictions = []
    all_targets = []
    rows = []
    selected_variants = (
        "full", "shared_mean", "no_bias", "stage_weight_shared_bias",
        "shared_weight_stage_bias", "cyclic_stage_residual",
    )
    for age in range(1, 9):
        prediction = states[age][:, -1, :] @ probe_weight + probe_bias
        all_predictions.append(prediction)
        all_targets.append(torch.full_like(prediction, float(age)))
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    centered = targets - targets.mean()
    probe_summary = {
        "r2": float(1.0 - (predictions - targets).square().sum() / centered.square().sum()),
        "mae": float((predictions - targets).abs().mean()),
        "rounded_age_accuracy": float(predictions.round().clamp(1, 8).eq(targets).float().mean()),
    }
    for age in AGES:
        source = states[age]
        before_age = source[:, -1, :] @ probe_weight + probe_bias
        for variant in selected_variants:
            output = variant_banks[variant].rollback(
                source, source_age=age, positions=positions
            )
            after_age = output[:, -1, :] @ probe_weight + probe_bias
            logits = logits_from_raw_state(model, output)
            rows.append(
                {
                    "source_age": age,
                    "variant": variant,
                    "predicted_age_before": float(before_age.mean()),
                    "predicted_age_after": float(after_age.mean()),
                    "predicted_age_shift": float((after_age - before_age).mean()),
                    "target_younger_age_mae": float((after_age - (age - 1)).abs().mean()),
                    "current_readout_accuracy": float(
                        logits.argmax(-1).eq(current).float().mean()
                    ),
                }
            )
    return rows, probe_summary


@torch.no_grad()
def inverse_rows(
    *, train_states: dict[int, torch.Tensor], test_states: dict[int, torch.Tensor],
    successors: torch.Tensor, current: torch.Tensor, model, cfg,
    phase_positions: list[int], bank: AgeSpecificJBank,
    positions: tuple[int, ...], ridge: float,
) -> list[dict[str, Any]]:
    rows = []
    loop = run_one_loop.__wrapped__
    predecessor = successors.argsort(dim=1).gather(1, current[:, None]).squeeze(1)
    for age in AGES:
        backward_weight, backward_bias = fit_affine(
            train_states[age], train_states[age - 1], ridge
        )
        forward_weight, forward_bias = fit_affine(
            train_states[age - 1], train_states[age], ridge
        )
        source, target = test_states[age], test_states[age - 1]
        learned = bank.rollback(source, source_age=age, positions=positions)
        direct = source @ backward_weight + backward_bias
        learned_metrics = state_metrics(learned, target)
        direct_metrics = state_metrics(direct, target)
        forward_then_j = (target @ forward_weight + forward_bias)
        forward_then_j = bank.rollback(
            forward_then_j, source_age=age, positions=positions
        )
        j_then_forward = learned @ forward_weight + forward_bias
        inverse_left = state_metrics(forward_then_j, target)
        inverse_right = state_metrics(j_then_forward, source)

        learned_weight = bank.maps[str(age)].weight.float()
        learned_bias = bank.maps[str(age)].bias.float()
        identity = torch.eye(cfg.d_model, device=source.device)
        ambient_gj_weight = forward_weight @ learned_weight
        ambient_gj_bias = forward_bias @ learned_weight + learned_bias
        ambient_jg_weight = learned_weight @ forward_weight
        ambient_jg_bias = learned_bias @ forward_weight + forward_bias

        f_state = loop(model, target, loop_index=age - 1).state
        next_current = advance_nodes(successors, current, steps=1)
        jf_state = bank.rollback(f_state, source_age=age, positions=positions)
        jf_target = _aligned_state_at_age(
            model=model, cfg=cfg, successors=successors, current=next_current,
            age=age - 1, phase_position=phase_positions[age - 1]
        )
        jf_metrics = state_metrics(jf_state, jf_target)
        j_state = learned
        fj_state = loop(model, j_state, loop_index=age - 1).state
        fj_target = _aligned_state_at_age(
            model=model, cfg=cfg, successors=successors, current=next_current,
            age=age, phase_position=phase_positions[age]
        )
        fj_metrics = state_metrics(fj_state, fj_target)
        j_logits = logits_from_raw_state(model, learned)
        rows.append(
            {
                "source_age": age,
                **{f"learned_to_young_{key}": value for key, value in learned_metrics.items()},
                **{f"direct_regression_{key}": value for key, value in direct_metrics.items()},
                **{f"regression_G_then_J_{key}": value for key, value in inverse_left.items()},
                **{f"J_then_regression_G_{key}": value for key, value in inverse_right.items()},
                **{f"actual_F_then_J_{key}": value for key, value in jf_metrics.items()},
                **{f"J_then_actual_F_{key}": value for key, value in fj_metrics.items()},
                "ambient_GJ_weight_identity_relative_error": float(
                    (ambient_gj_weight - identity).norm() / identity.norm()
                ),
                "ambient_GJ_bias_relative_to_target_scale": float(
                    ambient_gj_bias.norm() / target.reshape(-1, cfg.d_model).norm(dim=-1).mean()
                ),
                "ambient_JG_weight_identity_relative_error": float(
                    (ambient_jg_weight - identity).norm() / identity.norm()
                ),
                "ambient_JG_bias_relative_to_source_scale": float(
                    ambient_jg_bias.norm() / source.reshape(-1, cfg.d_model).norm(dim=-1).mean()
                ),
                "J_readout_current_accuracy": float(
                    j_logits.argmax(-1).eq(current).float().mean()
                ),
                "J_readout_predecessor_accuracy": float(
                    j_logits.argmax(-1).eq(predecessor).float().mean()
                ),
                "actual_F_then_J_readout_accuracy": float(
                    logits_from_raw_state(model, jf_state).argmax(-1).eq(next_current).float().mean()
                ),
                "J_then_actual_F_readout_accuracy": float(
                    logits_from_raw_state(model, fj_state).argmax(-1).eq(next_current).float().mean()
                ),
            }
        )
    return rows


@torch.no_grad()
def manifold_rows(
    *, states: dict[int, torch.Tensor], bank: AgeSpecificJBank,
    mean_bank: AgeSpecificJBank, cyclic_bank: AgeSpecificJBank,
    positions: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    generator = torch.Generator(device=states[1].device).manual_seed(827001)
    for age in AGES:
        source = states[age]
        full = bank.rollback(source, source_age=age, positions=positions)
        shared = mean_bank.rollback(source, source_age=age, positions=positions)
        cyclic = cyclic_bank.rollback(source, source_age=age, positions=positions)
        target = states[age - 1]
        residual_effect = full - shared
        full_displacement = full - source
        random = torch.randn(
            source.shape, generator=generator, device=source.device, dtype=source.dtype
        )
        random = random * source.std(dim=(0, 1), keepdim=True) + source.mean(
            dim=(0, 1), keepdim=True
        )
        random_full = bank.rollback(random, source_age=age, positions=positions)
        random_shared = mean_bank.rollback(random, source_age=age, positions=positions)
        rows.append(
            {
                "source_age": age,
                **{f"full_{key}": value for key, value in state_metrics(full, target).items()},
                **{f"shared_{key}": value for key, value in state_metrics(shared, target).items()},
                **{f"cyclic_{key}": value for key, value in state_metrics(cyclic, target).items()},
                "natural_stage_residual_effect_fraction": float(
                    residual_effect.norm() / full_displacement.norm().clamp_min(1e-12)
                ),
                "randomized_stage_residual_effect_fraction": float(
                    (random_full - random_shared).norm()
                    / (random_full - random).norm().clamp_min(1e-12)
                ),
                "natural_to_random_residual_amplification": float(
                    (residual_effect.norm() / full_displacement.norm().clamp_min(1e-12))
                    / (
                        (random_full - random_shared).norm()
                        / (random_full - random).norm().clamp_min(1e-12)
                    ).clamp_min(1e-12)
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_specs() -> tuple[dict[str, Any], ...]:
    return (
        dict(split="single_J", max_total=1, max_run=1, minimum=1,
             required=1, mandatory=True, offset=1),
        dict(split="focused_mixture_a", max_total=5, max_run=5, minimum=None,
             required=0, mandatory=True, offset=2),
        dict(split="focused_mixture_b", max_total=5, max_run=5, minimum=None,
             required=0, mandatory=True, offset=3),
        dict(split="hard_boundary_T05_R5", max_total=5, max_run=5, minimum=5,
             required=5, mandatory=False, offset=4),
    )


@torch.no_grad()
def evaluate_variants(
    *, variants: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]],
    model, cfg, phase_positions: list[int], positions: tuple[int, ...],
    device: torch.device, trajectories: int, batch_size: int, seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for variant_index, (variant, (weights, biases)) in enumerate(variants.items()):
        bank = make_bank(weights, biases, dimension=cfg.d_model, device=device)
        for spec in task_specs():
            _, summary = evaluate_random_trajectories(
                model=model, cfg=cfg, bank=bank,
                phase_positions=phase_positions, positions=positions,
                device=device, trajectories=trajectories, batch_size=batch_size,
                max_extra_backs=None, max_total_backs=spec["max_total"],
                max_consecutive_backs=spec["max_run"],
                minimum_total_backs=spec["minimum"],
                required_consecutive_backs=spec["required"],
                schedule_mandatory_j=spec["mandatory"],
                seed=seed + spec["offset"], split=spec["split"],
                conditions=("learned",),
            )
            row = dict(summary[0])
            row.update(variant=variant, variant_index=variant_index)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del bank
    return rows


def plot_task(rows: list[dict[str, Any]], path: Path) -> None:
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["split"]] = row["accuracy_mean"]
    key_variants = [
        "full", "direct_hidden_regression", "fitted_forward_pseudoinverse",
        "shared_mean", "scalar_generator", "no_bias", "bias_only",
        "stage_weight_shared_bias", "shared_weight_stage_bias",
        "cyclic_stage_residual", "polar_rotation_plus_bias",
        "polar_deformation_plus_bias",
    ]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    mix = lambda item: np.mean(
        [item["focused_mixture_a"], item["focused_mixture_b"]]
    )
    axes[0].barh(
        key_variants[::-1], [mix(by_variant[name]) for name in key_variants[::-1]]
    )
    axes[0].set_title("mixed trajectory accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].axvline(mix(by_variant["full"]), color="black", linestyle="--")

    delta_ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    axes[1].plot(
        delta_ranks,
        [mix(by_variant[f"full_delta_rank_{rank}"]) for rank in delta_ranks],
        "o-", label="truncate full [W-I;b]",
    )
    residual_ranks = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    axes[1].plot(
        residual_ranks,
        [mix(by_variant[f"mean_plus_stage_residual_rank_{rank}"]) for rank in residual_ranks],
        "o-", label="mean + stage residual",
    )
    axes[1].axhline(mix(by_variant["full"]), color="black", linestyle="--")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("task rank curve")
    axes[1].set_xlabel("retained rank")
    axes[1].legend()

    scales = [0, 0.25, 0.5, 0.75, 1, 1.25]
    names = {
        0: "shared_mean", 0.25: "stage_residual_scale_0.25",
        0.5: "stage_residual_scale_0.5", 0.75: "stage_residual_scale_0.75",
        1: "full", 1.25: "stage_residual_scale_1.25",
    }
    axes[2].plot(scales, [mix(by_variant[names[value]]) for value in scales], "o-", label="mixed")
    axes[2].plot(
        scales,
        [by_variant[names[value]]["hard_boundary_T05_R5"] for value in scales],
        "o-", label="five consecutive",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("stage-residual causal dose")
    axes[2].set_xlabel("stage residual scale")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_state(age_rows: list[dict[str, Any]], inverse: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    variants = ["full", "shared_mean", "no_bias", "cyclic_stage_residual"]
    for variant in variants:
        selected = [row for row in age_rows if row["variant"] == variant]
        axes[0].plot(
            [row["source_age"] for row in selected],
            [row["predicted_age_shift"] for row in selected],
            "o-", label=variant,
        )
    axes[0].axhline(-1, color="black", linestyle="--")
    axes[0].set_title("linear age-probe shift")
    axes[0].set_xlabel("source age")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        AGES, [row["learned_to_young_r2"] for row in inverse], "o-", label="learned J"
    )
    axes[1].plot(
        AGES, [row["direct_regression_r2"] for row in inverse], "o-", label="hidden regression"
    )
    axes[1].plot(
        AGES, [row["actual_F_then_J_r2"] for row in inverse], "o-", label="actual F then J"
    )
    axes[1].set_title("natural-state geometry")
    axes[1].set_xlabel("source age")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        AGES, [row["J_readout_current_accuracy"] for row in inverse], "o-", label="current"
    )
    axes[2].plot(
        AGES, [row["J_readout_predecessor_accuracy"] for row in inverse], "o-", label="predecessor"
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("what graph node J reads out")
    axes[2].set_xlabel("source age")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = tuple(range(cfg.seq_len))
    weights, biases, bank_payload = load_arrays(args.bank_artifact)
    variants = build_variants(weights, biases)
    primary_bank = make_bank(weights, biases, dimension=cfg.d_model, device=device)
    train_states, _, _ = collect_states(
        model=model, cfg=cfg, phase_positions=phase_positions, device=device,
        examples=args.state_train_examples, batch_size=args.state_batch_size,
        seed=args.seed + 100,
    )
    test_states, test_successors, test_current = collect_states(
        model=model, cfg=cfg, phase_positions=phase_positions, device=device,
        examples=args.state_test_examples, batch_size=args.state_batch_size,
        seed=args.seed + 200,
    )
    variants.update(fit_geometric_variants(train_states, args.affine_ridge))
    key_banks = {
        name: make_bank(*variants[name], dimension=cfg.d_model, device=device)
        for name in (
            "full", "shared_mean", "no_bias", "stage_weight_shared_bias",
            "shared_weight_stage_bias", "cyclic_stage_residual",
        )
    }
    probe_weight, probe_bias = fit_age_probe(train_states, args.probe_ridge)
    age_rows_data, probe_summary = age_probe_rows(
        states=test_states, current=test_current, probe_weight=probe_weight,
        probe_bias=probe_bias, variant_banks=key_banks, model=model,
        positions=positions,
    )
    inverse = inverse_rows(
        train_states=train_states, test_states=test_states,
        successors=test_successors, current=test_current, model=model, cfg=cfg,
        phase_positions=phase_positions, bank=primary_bank, positions=positions,
        ridge=args.affine_ridge,
    )
    manifold = manifold_rows(
        states=test_states, bank=primary_bank,
        mean_bank=key_banks["shared_mean"],
        cyclic_bank=key_banks["cyclic_stage_residual"], positions=positions,
    )
    task_rows_data = evaluate_variants(
        variants=variants, model=model, cfg=cfg,
        phase_positions=phase_positions, positions=positions, device=device,
        trajectories=args.task_trajectories, batch_size=args.task_batch_size,
        seed=820001,
    )
    write_csv(args.out_dir / "age_probe_interventions.csv", age_rows_data)
    write_csv(args.out_dir / "inverse_and_cycle_metrics.csv", inverse)
    write_csv(args.out_dir / "natural_manifold_metrics.csv", manifold)
    write_csv(args.out_dir / "task_intervention_accuracy.csv", task_rows_data)
    plot_task(task_rows_data, args.out_dir / "task_interventions.png")
    plot_state(age_rows_data, inverse, args.out_dir / "age_probe_and_inverse.png")
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "backbone_loss_placement": "final-only CE at loop 8",
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "bank_curriculum": bank_payload.get("curriculum"),
        "age_probe": probe_summary,
        "state_train_examples": args.state_train_examples,
        "state_test_examples": args.state_test_examples,
        "task_trajectories_per_split": args.task_trajectories,
        "task_examples_per_trajectory": args.task_batch_size,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
