from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_global_rejuvenator import (
    PairDataset,
    collect_evaluation_batch,
    collect_pair_dataset,
    evaluate_commutator,
    evaluate_mapping,
    evaluate_repeated_reset,
    predict_pair_dataset,
    select_and_refit_map,
    to_answer_map,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_rejuvenator_commutator import (
    AffineAnswerMap,
    apply_answer_map,
    random_orientation_control,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


CONDITIONS = ("pre_layernorm_warmup500", "post_layernorm_warmup2000")
CONDITION_LABELS = {
    "pre_layernorm_warmup500": "Pre-LN, warmup 500",
    "post_layernorm_warmup2000": "Post-LN, warmup 2000",
}
CANDIDATE = {
    "reference_age": 1,
    "reference_path_before": 1,
    "programmed_jump": 1,
}
RIDGE_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
EXPECTED_POSITIONS = (1, 2, 3, 4, 5, 6, 7, 8)


def spectral_summary(matrix: torch.Tensor) -> dict[str, Any]:
    """Summarize a square map without depending on SciPy on the GPU host."""
    matrix64 = matrix.detach().cpu().double()
    eigenvalues = torch.linalg.eigvals(matrix64)
    real = eigenvalues.real.numpy()
    imaginary = eigenvalues.imag.numpy()
    modulus = eigenvalues.abs().numpy()
    sign, logabsdet = torch.linalg.slogdet(matrix64)
    singular_values = torch.linalg.svdvals(matrix64)

    limit = max(1.05, float(modulus.max()) * 1.05)
    grid = np.linspace(-limit, limit, 181)
    xx, yy = np.meshgrid(grid, grid)
    # Isotropic Scott-bandwidth KDE.  Only the peak is used; the complete
    # density field is intentionally not retained in the output artifact.
    scale = max(
        float(np.sqrt(0.5 * (np.var(real) + np.var(imaginary)))),
        limit / 90.0,
    )
    bandwidth = max(scale * len(real) ** (-1.0 / 6.0), limit / 90.0)
    density = np.zeros(xx.shape, dtype=np.float64)
    for value_real, value_imaginary in zip(real, imaginary, strict=True):
        distance2 = (
            (xx - value_real) ** 2 + (yy - value_imaginary) ** 2
        ) / (bandwidth**2)
        density += np.exp(-0.5 * distance2)
    peak_index = np.unravel_index(int(np.argmax(density)), density.shape)
    peak_real = float(xx[peak_index])
    peak_imag = float(yy[peak_index])
    peak = peak_real + 1j * peak_imag

    return {
        "dimension": int(matrix.shape[0]),
        "spectral_radius": float(modulus.max()),
        "minimum_eigenvalue_modulus": float(modulus.min()),
        "log_absolute_determinant": float(logabsdet),
        "determinant_sign": float(sign),
        "log_absolute_determinant_per_dimension": float(
            logabsdet / matrix.shape[0]
        ),
        "geometric_mean_eigenvalue_modulus": float(
            torch.exp(logabsdet / matrix.shape[0])
        ),
        "eigenvalue_centroid_real": float(real.mean()),
        "eigenvalue_centroid_imag": float(imaginary.mean()),
        "median_eigenvalue_real": float(np.median(real)),
        "median_eigenvalue_modulus": float(np.median(modulus)),
        "kde_peak_real": peak_real,
        "kde_peak_imag": peak_imag,
        "kde_peak_distance_to_one": float(abs(peak - 1.0)),
        "fraction_eigenvalues_within_0p1_of_one": float(
            np.mean(np.abs(eigenvalues.numpy() - 1.0) < 0.1)
        ),
        "fraction_eigenvalues_within_0p2_of_one": float(
            np.mean(np.abs(eigenvalues.numpy() - 1.0) < 0.2)
        ),
        "fraction_eigenvalue_modulus_within_0p1_of_one": float(
            np.mean(np.abs(modulus - 1.0) < 0.1)
        ),
        "maximum_singular_value": float(singular_values.max()),
        "minimum_singular_value": float(singular_values.min()),
        "condition_number": float(
            singular_values.max() / singular_values.min().clamp_min(1e-15)
        ),
        "eigenvalues_real": [float(value) for value in real],
        "eigenvalues_imag": [float(value) for value in imaginary],
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_directory(root: Path, seed: int, *, postnorm: bool) -> Path:
    family = f"D8_L8_postnorm_seed{seed}" if postnorm else f"D8_L8_seed{seed}"
    return root / family / f"graphpath_N8_D8_d256_B2_L8_seed{seed}"


def checkpoint_path(root: Path, seed: int, *, postnorm: bool) -> Path:
    return run_directory(root, seed, postnorm=postnorm) / "best.pt"


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    keys: Sequence[str],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    result: list[dict[str, Any]] = []
    for group_key, selected in sorted(grouped.items(), key=lambda item: str(item[0])):
        output = dict(zip(keys, group_key, strict=True))
        output["replications"] = len(selected)
        for field in fields:
            values = np.asarray([float(row[field]) for row in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                output[f"{field}_mean"] = float(finite.mean())
                output[f"{field}_min"] = float(finite.min())
                output[f"{field}_max"] = float(finite.max())
            else:
                output[f"{field}_mean"] = float("nan")
                output[f"{field}_min"] = float("nan")
                output[f"{field}_max"] = float("nan")
        result.append(output)
    return result


def strict_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    endpoint: torch.Tensor,
) -> tuple[float, int]:
    valid = target.ne(endpoint)
    count = int(valid.sum())
    if not count:
        return float("nan"), 0
    return float(prediction[valid].eq(target[valid]).float().mean()), count


@torch.no_grad()
def rollout_batch_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    start: torch.Tensor,
    condition: str,
    seed: int,
    batch_index: int,
    maximum_loop: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_targets = torch.cat([start[:, None], targets], dim=1)
    endpoint = targets[:, cfg.max_depth - 1]
    state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    previous_answer = state[:, -1].float()
    predictions: list[torch.Tensor] = []
    cached: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for loop_index in range(maximum_loop):
        state = apply_shared_stack(model, state, loop_index=loop_index)
        answer = state[:, -1].float()
        prediction = logits_from_raw_state(model, state).argmax(dim=-1)
        predictions.append(prediction)
        continued_position = loop_index + 1
        continued = all_targets[:, continued_position]
        continued_strict, strict_count = strict_accuracy(
            prediction, continued, endpoint
        )
        delta = (answer - previous_answer).square().sum(dim=-1).sqrt()
        scale = previous_answer.square().sum(dim=-1).sqrt().clamp_min(1e-8)
        cached.append(
            {
                "condition": condition,
                "seed": seed,
                "batch_index": batch_index,
                "loop": loop_index + 1,
                "sample_count": int(tokens.shape[0]),
                "final_target_accuracy": float(
                    prediction.eq(endpoint).float().mean()
                ),
                "continued_target_accuracy": float(
                    prediction.eq(continued).float().mean()
                ),
                "continued_target_strict_accuracy": continued_strict,
                "continued_target_strict_count": strict_count,
                "answer_rms": float(answer.square().mean(dim=-1).sqrt().mean()),
                "answer_within_std": float(answer.std(dim=-1).mean()),
                "answer_mean_absolute": float(answer.mean(dim=-1).abs().mean()),
                "relative_adjacent_loop_change": float((delta / scale).mean()),
            }
        )
        if loop_index < cfg.max_loops:
            for position in range(cfg.max_depth + 1):
                position_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "batch_index": batch_index,
                        "loop": loop_index + 1,
                        "path_position": position,
                        "accuracy": float(
                            prediction.eq(all_targets[:, position]).float().mean()
                        ),
                    }
                )
        previous_answer = answer
    loop8_prediction = predictions[cfg.max_loops - 1]
    for row, prediction in zip(cached, predictions, strict=True):
        row["prediction_stability_to_loop8"] = float(
            prediction.eq(loop8_prediction).float().mean()
        )
    return cached, position_rows


@torch.no_grad()
def analyze_natural_rollout(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    condition: str,
    seed: int,
    device: torch.device,
    maximum_loop: int,
    batch_size: int,
    batches: int,
    data_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(data_seed)
    rows: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    for batch_index in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=maximum_loop,
        )
        batch_rows, batch_positions = rollout_batch_rows(
            model=model,
            cfg=cfg,
            tokens=tokens,
            targets=targets,
            start=start,
            condition=condition,
            seed=seed,
            batch_index=batch_index,
            maximum_loop=maximum_loop,
        )
        rows.extend(batch_rows)
        positions.extend(batch_positions)
    return rows, positions


def trajectory_summary(
    position_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate = aggregate_rows(
        position_rows,
        keys=("condition", "seed", "loop", "path_position"),
        fields=("accuracy",),
    )
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in aggregate:
        key = (str(row["condition"]), int(row["seed"]), int(row["loop"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (condition, seed, loop), selected in sorted(grouped.items()):
        best = max(selected, key=lambda row: float(row["accuracy_mean"]))
        expected = EXPECTED_POSITIONS[loop - 1]
        expected_row = next(row for row in selected if row["path_position"] == expected)
        summaries.append(
            {
                "condition": condition,
                "seed": seed,
                "loop": loop,
                "expected_path_position": expected,
                "expected_position_accuracy": expected_row["accuracy_mean"],
                "best_path_position": int(best["path_position"]),
                "best_position_accuracy": best["accuracy_mean"],
                "best_position_matches_expected": int(
                    best["path_position"]
                ) == expected,
                "expected_accuracy_ge_0p9": float(
                    expected_row["accuracy_mean"]
                ) >= 0.9,
                "trajectory_compatible": int(best["path_position"]) == expected,
            }
        )
    return summaries


def age_balanced_validation_metrics(
    validation: PairDataset,
    matrix: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, Any]:
    prediction = predict_pair_dataset(validation, matrix, bias)
    output: dict[str, Any] = {}
    per_age = []
    for age in sorted(int(value) for value in validation.source_age.unique()):
        mask = validation.source_age.eq(age)
        target = validation.target[mask].float()
        residual = prediction[mask].float() - target
        denominator = (
            target - target.mean(dim=0, keepdim=True)
        ).square().mean().clamp_min(1e-12)
        value = float(residual.square().mean() / denominator)
        output[f"h{age}_to_h{age - 1}_relative_mse"] = value
        per_age.append(value)
    output["age_balanced_relative_mse"] = float(np.mean(per_age))
    return output


def zero_map(d_model: int, device: torch.device) -> AffineAnswerMap:
    return AffineAnswerMap(
        update_matrix=torch.zeros(d_model, d_model, device=device),
        bias=torch.zeros(d_model, device=device),
    )


@torch.no_grad()
def oracle_reset_row(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: Any,
) -> dict[str, Any]:
    oracle = batch.reset_oracles[8]
    post = apply_shared_stack(model, oracle, loop_index=cfg.max_loops)
    current = advance_nodes(batch.successors, batch.start, steps=8)
    target = advance_nodes(batch.successors, batch.start, steps=9)
    strict, count = strict_accuracy(
        logits_from_raw_state(model, post).argmax(dim=-1),
        target,
        current,
    )
    return {
        "condition": "aligned_oracle_h2",
        "source_age": 8,
        "application_count": 0,
        "sample_count": int(current.shape[0]),
        "reset_post_target_accuracy": strict,
        "strict_count": count,
    }


@torch.no_grad()
def repeated_length_rollout(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    answer_map: AffineAnswerMap,
    condition: str,
    model_condition: str,
    seed: int,
    eval_seed: int,
    maximum_cycle: int,
) -> list[dict[str, Any]]:
    state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    for loop_index in range(cfg.max_loops):
        state = apply_shared_stack(model, state, loop_index=loop_index)
    endpoint = targets[:, cfg.max_depth - 1]
    for _ in range(cfg.max_loops - 2):
        state = apply_answer_map(state, answer_map)
    rows: list[dict[str, Any]] = []
    for cycle in range(1, maximum_cycle + 1):
        state = apply_shared_stack(model, state, loop_index=cfg.max_loops)
        target = targets[:, cfg.max_depth + cycle - 1]
        prediction = logits_from_raw_state(model, state).argmax(dim=-1)
        strict, count = strict_accuracy(prediction, target, endpoint)
        rows.append(
            {
                "model_condition": model_condition,
                "seed": seed,
                "eval_seed": eval_seed,
                "intervention": condition,
                "extra_cycle": cycle,
                "effective_path_position": cfg.max_depth + cycle,
                "sample_count": int(tokens.shape[0]),
                "continued_target_accuracy": float(
                    prediction.eq(target).float().mean()
                ),
                "continued_target_strict_accuracy": strict,
                "continued_target_strict_count": count,
                "endpoint_accuracy": float(
                    prediction.eq(endpoint).float().mean()
                ),
            }
        )
        if cycle < maximum_cycle:
            state = apply_answer_map(state, answer_map)
    return rows


def fit_and_evaluate_j(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    condition: str,
    seed: int,
    device: torch.device,
    out_dir: Path,
    calibration_samples: int,
    validation_samples: int,
    collection_batch_size: int,
    evaluation_batch_size: int,
    evaluation_seeds: Sequence[int],
    maximum_rejuvenated_cycle: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    calibration = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=CANDIDATE,
        sample_count=calibration_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=7_700_000 + seed,
    )
    validation = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=CANDIDATE,
        sample_count=validation_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=7_710_000 + seed,
    )
    matrix, bias, selection_rows, ridge = select_and_refit_map(
        calibration=calibration,
        validation=validation,
        use_bias=False,
        ridge_grid=RIDGE_GRID,
    )
    validation_metrics = age_balanced_validation_metrics(
        validation, matrix, bias
    )
    spectrum = spectral_summary(matrix)
    summary: dict[str, Any] = {
        "condition": condition,
        "seed": seed,
        "selected_ridge": ridge,
        **validation_metrics,
        **{
            key: value
            for key, value in spectrum.items()
            if not key.startswith("eigenvalues_")
        },
        "eigenvalues_real": spectrum["eigenvalues_real"],
        "eigenvalues_imag": spectrum["eigenvalues_imag"],
    }
    artifact_path = out_dir / f"global_J_{condition}_seed{seed}.pt"
    torch.save(
        {
            "condition": condition,
            "seed": seed,
            "candidate": CANDIDATE,
            "matrix": matrix,
            "bias": bias,
            "selected_ridge": ridge,
            "validation_metrics": validation_metrics,
            "spectrum": spectrum,
        },
        artifact_path,
    )
    selection_output = [
        {"condition": condition, "seed": seed, **row}
        for row in selection_rows
    ]

    learned = to_answer_map(matrix, bias, device=device)
    identity = zero_map(cfg.d_model, device)
    random_control = random_orientation_control(
        learned, seed=7_720_000 + seed
    )
    mapping_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    for eval_seed in evaluation_seeds:
        batch = collect_evaluation_batch(
            model=model,
            cfg=cfg,
            candidate=CANDIDATE,
            batch_size=evaluation_batch_size,
            device=device,
            seed=eval_seed + 100 * seed,
            maximum_commutator_age=8,
            reset_source_ages=(8,),
        )
        for age in range(3, 9):
            mapping_rows.append(
                {
                    "model_condition": condition,
                    "seed": seed,
                    "eval_seed": eval_seed,
                    **evaluate_mapping(
                        model=model,
                        cfg=cfg,
                        candidate=CANDIDATE,
                        batch=batch,
                        answer_map=learned,
                        condition="global_linear_256x256",
                        source_age=age,
                    ),
                }
            )
            commutator_rows.append(
                {
                    "model_condition": condition,
                    "seed": seed,
                    "eval_seed": eval_seed,
                    **evaluate_commutator(
                        model=model,
                        cfg=cfg,
                        candidate=CANDIDATE,
                        batch=batch,
                        answer_map=learned,
                        condition="global_linear_256x256",
                        source_age=age,
                    ),
                }
            )
        for intervention, answer_map in (
            ("learned_global_J", learned),
            ("identity_no_rejuvenation", identity),
            ("random_orientation_matched_update_norm", random_control),
        ):
            reset_rows.append(
                {
                    "model_condition": condition,
                    "seed": seed,
                    "eval_seed": eval_seed,
                    **evaluate_repeated_reset(
                        model=model,
                        cfg=cfg,
                        candidate=CANDIDATE,
                        batch=batch,
                        answer_map=answer_map,
                        condition=intervention,
                        source_age=8,
                    ),
                }
            )
        reset_rows.append(
            {
                "model_condition": condition,
                "seed": seed,
                "eval_seed": eval_seed,
                **oracle_reset_row(model=model, cfg=cfg, batch=batch),
            }
        )

        set_seed(eval_seed + 1_000 + 100 * seed)
        tokens, targets, _, _ = fixed_depth_batch(
            cfg,
            evaluation_batch_size,
            device,
            path_positions=cfg.max_depth + 2 * maximum_rejuvenated_cycle,
        )
        for intervention, answer_map in (
            ("learned_global_J", learned),
            ("identity_no_rejuvenation", identity),
            ("random_orientation_matched_update_norm", random_control),
        ):
            length_rows.extend(
                repeated_length_rollout(
                    model=model,
                    cfg=cfg,
                    tokens=tokens,
                    targets=targets,
                    answer_map=answer_map,
                    condition=intervention,
                    model_condition=condition,
                    seed=seed,
                    eval_seed=eval_seed,
                    maximum_cycle=maximum_rejuvenated_cycle,
                )
            )
    return summary, selection_output, mapping_rows, commutator_rows, reset_rows + length_rows


def plot_natural_overloop(rows: Sequence[dict[str, Any]], path: Path) -> None:
    aggregates = aggregate_rows(
        rows,
        keys=("condition", "loop"),
        fields=(
            "final_target_accuracy",
            "continued_target_strict_accuracy",
            "prediction_stability_to_loop8",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharex=True, sharey=True)
    fields = (
        ("final_target_accuracy", "Accuracy on trained $f^8$ target"),
        ("continued_target_strict_accuracy", "Accuracy on continued $f^L$ target"),
        ("prediction_stability_to_loop8", "Prediction agreement with loop 8"),
    )
    colors = {
        "pre_layernorm_warmup500": "#2563eb",
        "post_layernorm_warmup2000": "#dc2626",
    }
    for axis, (field, title) in zip(axes, fields, strict=True):
        for condition in CONDITIONS:
            selected = [row for row in aggregates if row["condition"] == condition]
            x = np.asarray([row["loop"] for row in selected])
            mean = np.asarray([row[f"{field}_mean"] for row in selected])
            low = np.asarray([row[f"{field}_min"] for row in selected])
            high = np.asarray([row[f"{field}_max"] for row in selected])
            axis.plot(x, mean, color=colors[condition], label=CONDITION_LABELS[condition])
            axis.fill_between(x, low, high, color=colors[condition], alpha=0.14)
        axis.axvline(8, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("recurrent loop")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("accuracy / agreement")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_state_dynamics(rows: Sequence[dict[str, Any]], path: Path) -> None:
    aggregates = aggregate_rows(
        rows,
        keys=("condition", "loop"),
        fields=("answer_rms", "answer_within_std", "relative_adjacent_loop_change"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), sharex=True)
    fields = (
        ("answer_rms", "Answer-state RMS"),
        ("answer_within_std", "Within-state standard deviation"),
        ("relative_adjacent_loop_change", "Relative adjacent-loop change"),
    )
    colors = {
        "pre_layernorm_warmup500": "#2563eb",
        "post_layernorm_warmup2000": "#dc2626",
    }
    for axis, (field, title) in zip(axes, fields, strict=True):
        for condition in CONDITIONS:
            selected = [row for row in aggregates if row["condition"] == condition]
            x = np.asarray([row["loop"] for row in selected])
            mean = np.asarray([row[f"{field}_mean"] for row in selected])
            low = np.asarray([row[f"{field}_min"] for row in selected])
            high = np.asarray([row[f"{field}_max"] for row in selected])
            axis.plot(x, mean, color=colors[condition], label=CONDITION_LABELS[condition])
            axis.fill_between(x, low, high, color=colors[condition], alpha=0.14)
        axis.axvline(8, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("recurrent loop")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("state statistic")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_postnorm_spectra(summaries: Sequence[dict[str, Any]], path: Path) -> None:
    selected = sorted(
        (
            row
            for row in summaries
            if row["condition"] == "post_layernorm_warmup2000"
        ),
        key=lambda row: int(row["seed"]),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14.6, 9.3), dpi=220)
    limit = max(
        1.05,
        max(float(row["spectral_radius"]) for row in selected) * 1.06,
    )
    theta = np.linspace(0, 2 * np.pi, 1000)
    for axis, row in zip(axes.ravel(), selected):
        real = np.asarray(row["eigenvalues_real"])
        imaginary = np.asarray(row["eigenvalues_imag"])
        modulus = np.hypot(real, imaginary)
        axis.plot(np.cos(theta), np.sin(theta), "--", color="#9ca3af", linewidth=0.9)
        axis.scatter(
            real,
            imaginary,
            c=modulus,
            cmap="viridis",
            vmin=0,
            vmax=limit,
            s=17,
            alpha=0.78,
            linewidths=0,
        )
        axis.scatter(
            [row["kde_peak_real"]],
            [row["kde_peak_imag"]],
            marker="X",
            s=85,
            color="#dc2626",
            edgecolor="white",
            linewidth=0.5,
        )
        axis.scatter([1], [0], marker="*", s=100, color="#f59e0b", edgecolor="#78350f")
        axis.axhline(0, color="#d1d5db", linewidth=0.6)
        axis.axvline(0, color="#d1d5db", linewidth=0.6)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_title(
            f"Post-LN seed {row['seed']}\n"
            f"peak={row['kde_peak_real']:.3f}{row['kde_peak_imag']:+.3f}i, "
            f"median |λ|={row['median_eigenvalue_modulus']:.3f}"
        )
    for axis in axes.ravel()[len(selected) :]:
        axis.set_visible(False)
    figure.supxlabel(r"$\operatorname{Re}(\lambda)$")
    figure.supylabel(r"$\operatorname{Im}(\lambda)$")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_spectrum_comparison(summaries: Sequence[dict[str, Any]], path: Path) -> None:
    fields = (
        ("kde_peak_real", "KDE peak real part"),
        ("median_eigenvalue_modulus", r"Median $|\lambda|$"),
        ("log_absolute_determinant_per_dimension", r"$\log|\det J| / d$"),
        ("spectral_radius", "Spectral radius"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))
    colors = ("#2563eb", "#dc2626")
    for axis, (field, title) in zip(axes, fields, strict=True):
        values = [
            [float(row[field]) for row in summaries if row["condition"] == condition]
            for condition in CONDITIONS
        ]
        axis.boxplot(values, tick_labels=("Pre-LN", "Post-LN"), widths=0.55)
        for index, (condition_values, color) in enumerate(zip(values, colors, strict=True), 1):
            jitter = np.linspace(-0.08, 0.08, len(condition_values))
            axis.scatter(
                index + jitter,
                condition_values,
                color=color,
                s=28,
                zorder=3,
            )
        if field in {"kde_peak_real", "median_eigenvalue_modulus"}:
            axis.axhline(1, color="#f59e0b", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_rejuvenated_rollout(rows: Sequence[dict[str, Any]], path: Path) -> None:
    length_rows = [row for row in rows if "extra_cycle" in row]
    aggregate = aggregate_rows(
        length_rows,
        keys=("model_condition", "intervention", "extra_cycle"),
        fields=("continued_target_strict_accuracy",),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True, sharey=True)
    intervention_styles = {
        "learned_global_J": ("#059669", "-"),
        "identity_no_rejuvenation": ("#6b7280", "--"),
        "random_orientation_matched_update_norm": ("#7c3aed", ":"),
    }
    for axis, model_condition in zip(axes, CONDITIONS, strict=True):
        for intervention, (color, linestyle) in intervention_styles.items():
            selected = [
                row
                for row in aggregate
                if row["model_condition"] == model_condition
                and row["intervention"] == intervention
            ]
            x = np.asarray([row["extra_cycle"] for row in selected])
            mean = np.asarray(
                [row["continued_target_strict_accuracy_mean"] for row in selected]
            )
            low = np.asarray(
                [row["continued_target_strict_accuracy_min"] for row in selected]
            )
            high = np.asarray(
                [row["continued_target_strict_accuracy_max"] for row in selected]
            )
            axis.plot(x, mean, color=color, linestyle=linestyle, label=intervention)
            if intervention == "learned_global_J":
                axis.fill_between(x, low, high, color=color, alpha=0.14)
        axis.set_title(CONDITION_LABELS[model_condition])
        axis.set_xlabel("additional two-hop execution cycle")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("strict continued-path accuracy")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-root", type=Path, required=True)
    parser.add_argument("--pre-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--maximum-natural-loop", type=int, default=64)
    parser.add_argument("--natural-batch-size", type=int, default=512)
    parser.add_argument("--natural-batches", type=int, default=2)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--collection-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=[7_730_001, 7_730_002]
    )
    parser.add_argument("--maximum-rejuvenated-cycle", type=int, default=200)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    rollout_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    j_summaries: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    commutator_rows: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for condition, root, postnorm in (
        ("pre_layernorm_warmup500", args.pre_root, False),
        ("post_layernorm_warmup2000", args.post_root, True),
    ):
        for seed in args.seeds:
            checkpoint = checkpoint_path(root, seed, postnorm=postnorm)
            print(f"loading {condition} seed {seed}: {checkpoint}", flush=True)
            model, cfg, payload = load_checkpoint(checkpoint, device)
            if (
                cfg.node_count,
                cfg.max_depth,
                cfg.d_model,
                cfg.n_layers,
                cfg.max_loops,
            ) != (8, 8, 256, 2, 8):
                raise ValueError(f"unexpected D8L8 config for {checkpoint}: {cfg}")
            expected_norm = "post_layernorm" if postnorm else "pre_layernorm"
            if cfg.inner_norm_style != expected_norm:
                raise ValueError(
                    f"expected {expected_norm}, got {cfg.inner_norm_style}"
                )
            run_rows, run_positions = analyze_natural_rollout(
                model=model,
                cfg=cfg,
                condition=condition,
                seed=seed,
                device=device,
                maximum_loop=args.maximum_natural_loop,
                batch_size=args.natural_batch_size,
                batches=args.natural_batches,
                data_seed=7_740_000 + seed,
            )
            rollout_rows.extend(run_rows)
            position_rows.extend(run_positions)
            print(f"natural rollout complete: {condition} seed {seed}", flush=True)

            summary, selected, mappings, commutators, causal = fit_and_evaluate_j(
                model=model,
                cfg=cfg,
                condition=condition,
                seed=seed,
                device=device,
                out_dir=args.out_dir,
                calibration_samples=args.calibration_samples,
                validation_samples=args.validation_samples,
                collection_batch_size=args.collection_batch_size,
                evaluation_batch_size=args.evaluation_batch_size,
                evaluation_seeds=args.evaluation_seeds,
                maximum_rejuvenated_cycle=args.maximum_rejuvenated_cycle,
            )
            j_summaries.append(summary)
            selection_rows.extend(selected)
            mapping_rows.extend(mappings)
            commutator_rows.extend(commutators)
            causal_rows.extend(causal)
            manifests.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": int(payload.get("step", -1)),
                    "inner_norm_style": cfg.inner_norm_style,
                    "physical_blocks": cfg.n_layers,
                    "trained_loops": cfg.max_loops,
                    "trained_effective_depth": cfg.n_layers * cfg.max_loops,
                }
            )
            print(f"global J complete: {condition} seed {seed}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    trajectories = trajectory_summary(position_rows)
    if not all(bool(row["trajectory_compatible"]) for row in trajectories):
        failed = [row for row in trajectories if not row["trajectory_compatible"]]
        print(
            "one-step intermediate readout is not shared by every checkpoint; "
            f"recording this as a result ({len(failed)} failed loop checks)",
            flush=True,
        )

    rollout_aggregate = aggregate_rows(
        rollout_rows,
        keys=("condition", "loop"),
        fields=(
            "final_target_accuracy",
            "continued_target_accuracy",
            "continued_target_strict_accuracy",
            "prediction_stability_to_loop8",
            "answer_rms",
            "answer_within_std",
            "answer_mean_absolute",
            "relative_adjacent_loop_change",
        ),
    )
    spectrum_aggregate = aggregate_rows(
        j_summaries,
        keys=("condition",),
        fields=(
            "age_balanced_relative_mse",
            "spectral_radius",
            "median_eigenvalue_modulus",
            "kde_peak_real",
            "kde_peak_imag",
            "kde_peak_distance_to_one",
            "log_absolute_determinant",
            "log_absolute_determinant_per_dimension",
            "geometric_mean_eigenvalue_modulus",
            "condition_number",
        ),
    )
    length_only = [row for row in causal_rows if "extra_cycle" in row]
    length_aggregate = aggregate_rows(
        length_only,
        keys=("model_condition", "intervention", "extra_cycle"),
        fields=(
            "continued_target_accuracy",
            "continued_target_strict_accuracy",
            "endpoint_accuracy",
        ),
    )

    write_csv(args.out_dir / "natural_rollout_rows.csv", rollout_rows)
    write_csv(args.out_dir / "natural_rollout_aggregate.csv", rollout_aggregate)
    write_csv(args.out_dir / "path_position_rows.csv", position_rows)
    write_csv(args.out_dir / "trajectory_summary.csv", trajectories)
    write_csv(
        args.out_dir / "J_summary.csv",
        [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("eigenvalues_")
            }
            for row in j_summaries
        ],
    )
    write_csv(args.out_dir / "J_spectrum_aggregate.csv", spectrum_aggregate)
    write_csv(args.out_dir / "J_ridge_selection.csv", selection_rows)
    write_csv(args.out_dir / "J_mapping_rows.csv", mapping_rows)
    write_csv(args.out_dir / "J_commutator_rows.csv", commutator_rows)
    write_csv(
        args.out_dir / "J_reset_rows.csv",
        [row for row in causal_rows if "extra_cycle" not in row],
    )
    write_csv(args.out_dir / "rejuvenated_length_rows.csv", length_only)
    write_csv(args.out_dir / "rejuvenated_length_aggregate.csv", length_aggregate)

    plot_natural_overloop(
        rollout_rows, args.out_dir / "natural_overloop_comparison.png"
    )
    plot_state_dynamics(
        rollout_rows, args.out_dir / "answer_state_dynamics.png"
    )
    plot_postnorm_spectra(
        j_summaries, args.out_dir / "postnorm_J_complex_spectra.png"
    )
    plot_spectrum_comparison(
        j_summaries, args.out_dir / "pre_vs_post_J_spectrum.png"
    )
    plot_rejuvenated_rollout(
        causal_rows, args.out_dir / "rejuvenated_rollout_200cycles.png"
    )

    payload = {
        "design": {
            "task": "Graph Path D8L8 final-loss-only",
            "candidate": CANDIDATE,
            "J_fit": (
                "one bias-free global 256x256 row-vector map per checkpoint, "
                "jointly fit on aligned h3->h2,...,h8->h7 pairs"
            ),
            "natural_maximum_loop": args.maximum_natural_loop,
            "maximum_rejuvenated_cycle": args.maximum_rejuvenated_cycle,
            "natural_examples_per_seed": args.natural_batch_size
            * args.natural_batches,
            "calibration_samples_per_age": args.calibration_samples,
            "validation_samples_per_age": args.validation_samples,
            "evaluation_examples_per_seed": args.evaluation_batch_size
            * len(args.evaluation_seeds),
            "claim_boundary": (
                "Pre-vs-Post training comparison also changes warmup 500->2000; "
                "matrix spectrum is descriptive, while repeated rollout is causal."
            ),
        },
        "manifest": manifests,
        "trajectory_summary": trajectories,
        "J_summaries": j_summaries,
        "J_spectrum_aggregate": spectrum_aggregate,
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"analysis complete: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
