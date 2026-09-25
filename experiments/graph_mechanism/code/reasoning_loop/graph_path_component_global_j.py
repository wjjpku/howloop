from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_global_rejuvenator import (
    TRAIN_SOURCE_AGES,
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
from reasoning_loop.graph_path_postnorm_multiseed_analysis import (
    age_balanced_validation_metrics,
    spectral_summary,
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


CANDIDATE = {
    "reference_age": 2,
    "reference_path_before": 4,
    "programmed_jump": 2,
}
RIDGE_GRID = (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
PRIMARY_TRAIN_AGES = TRAIN_SOURCE_AGES
ACTIVE_TRAIN_AGES = (3, 4)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=CHECKPOINT")
    name, checkpoint = value.split("=", 1)
    if not name or not checkpoint:
        raise argparse.ArgumentTypeError("model must be NAME=CHECKPOINT")
    return name, Path(checkpoint)


def zero_map(d_model: int, device: torch.device) -> AffineAnswerMap:
    return AffineAnswerMap(
        update_matrix=torch.zeros(d_model, d_model, device=device),
        bias=torch.zeros(d_model, device=device),
    )


def opposite_direction_map(answer_map: AffineAnswerMap) -> AffineAnswerMap:
    """Return h - (J(h) - h), with the same update norm as J."""
    return AffineAnswerMap(
        update_matrix=-answer_map.update_matrix,
        bias=-answer_map.bias,
    )


def strict_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *excluded: torch.Tensor,
) -> tuple[float, int]:
    valid = torch.ones_like(target, dtype=torch.bool)
    for value in excluded:
        valid &= target.ne(value)
    count = int(valid.sum())
    if not count:
        return float("nan"), 0
    return float(prediction[valid].eq(target[valid]).float().mean()), count


def rejuvenated_target_position(cycle: int, *, max_depth: int = 8, jump: int = 2) -> int:
    if cycle < 0:
        raise ValueError("cycle must be non-negative")
    return max_depth + jump * cycle


@torch.no_grad()
def natural_overloop_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    model_name: str,
    device: torch.device,
    maximum_loop: int,
    batch_size: int,
    data_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for data_seed in data_seeds:
        set_seed(data_seed)
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=2 * maximum_loop,
        )
        endpoint = advance_nodes(successors, start, steps=cfg.max_depth)
        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        cache: list[tuple[torch.Tensor, dict[str, Any]]] = []
        previous_answer = state[:, -1].float()
        for loop_index in range(maximum_loop):
            state = apply_shared_stack(model, state, loop_index=loop_index)
            answer = state[:, -1].float()
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            continued_position = 2 * (loop_index + 1)
            continued = advance_nodes(
                successors, start, steps=continued_position
            )
            supervised_position = min(cfg.max_depth, continued_position)
            supervised = advance_nodes(
                successors, start, steps=supervised_position
            )
            strict_current, current_count = strict_accuracy(
                prediction, continued, supervised
            )
            strict_endpoint, endpoint_count = strict_accuracy(
                prediction, continued, endpoint
            )
            delta = (answer - previous_answer).square().sum(dim=-1).sqrt()
            scale = previous_answer.square().sum(dim=-1).sqrt().clamp_min(1e-8)
            cache.append(
                (
                    prediction,
                    {
                        "model": model_name,
                        "data_seed": data_seed,
                        "loop": loop_index + 1,
                        "sample_count": batch_size,
                        "expected_supervised_position": supervised_position,
                        "continued_position": continued_position,
                        "supervised_target_accuracy": float(
                            prediction.eq(supervised).float().mean()
                        ),
                        "endpoint_accuracy": float(
                            prediction.eq(endpoint).float().mean()
                        ),
                        "continued_target_accuracy": float(
                            prediction.eq(continued).float().mean()
                        ),
                        "continued_strict_current_accuracy": strict_current,
                        "continued_strict_current_count": current_count,
                        "continued_strict_endpoint_accuracy": strict_endpoint,
                        "continued_strict_endpoint_count": endpoint_count,
                        "answer_rms": float(
                            answer.square().mean(dim=-1).sqrt().mean()
                        ),
                        "relative_adjacent_loop_change": float(
                            (delta / scale).mean()
                        ),
                    },
                )
            )
            previous_answer = answer
        loop8_prediction = cache[cfg.max_loops - 1][0]
        for prediction, row in cache:
            row["prediction_stability_to_loop8"] = float(
                prediction.eq(loop8_prediction).float().mean()
            )
            rows.append(row)
    return rows


@torch.no_grad()
def oracle_answer_reset_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    batch: Any,
    model_name: str,
    eval_seed: int,
) -> list[dict[str, Any]]:
    source = batch.base_states[8]
    oracle = batch.reset_oracles[8]
    answer_patched = source.clone()
    answer_patched[:, -1] = oracle[:, -1]
    answer_post = apply_shared_stack(
        model, answer_patched, loop_index=cfg.max_loops
    )
    full_post = apply_shared_stack(model, oracle, loop_index=cfg.max_loops)
    current = advance_nodes(batch.successors, batch.start, steps=8)
    target = advance_nodes(batch.successors, batch.start, steps=10)
    rows = []
    for intervention, state, post in (
        ("aligned_oracle_answer_token_h2", answer_patched, answer_post),
        ("aligned_oracle_full_state_h2", oracle, full_post),
    ):
        pre_prediction = logits_from_raw_state(model, state).argmax(dim=-1)
        prediction = logits_from_raw_state(model, post).argmax(dim=-1)
        post_accuracy, count = strict_accuracy(
            prediction, target, current
        )
        rows.append(
            {
                "model": model_name,
                "eval_seed": eval_seed,
                "condition": intervention,
                "source_age": 8,
                "application_count": 0,
                "sample_count": int(source.shape[0]),
                "reset_pre_current_accuracy": float(
                    pre_prediction.eq(current).float().mean()
                ),
                "reset_pre_post_target_accuracy": float(
                    pre_prediction.eq(target).float().mean()
                ),
                "reset_post_target_accuracy": post_accuracy,
                "strict_count": count,
            }
        )
    return rows


@torch.no_grad()
def rejuvenated_rollout_rows(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    tokens: torch.Tensor,
    successors: torch.Tensor,
    start: torch.Tensor,
    answer_map: AffineAnswerMap,
    intervention: str,
    model_name: str,
    eval_seed: int,
    maximum_cycle: int,
) -> list[dict[str, Any]]:
    state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    for loop_index in range(cfg.max_loops):
        state = apply_shared_stack(model, state, loop_index=loop_index)
    for _ in range(cfg.max_loops - 2):
        state = apply_answer_map(state, answer_map)

    endpoint = advance_nodes(successors, start, steps=cfg.max_depth)
    rows = []
    for cycle in range(1, maximum_cycle + 1):
        current_position = rejuvenated_target_position(cycle - 1)
        target_position = rejuvenated_target_position(cycle)
        current = advance_nodes(successors, start, steps=current_position)
        target = advance_nodes(successors, start, steps=target_position)
        pre_prediction = logits_from_raw_state(model, state).argmax(dim=-1)
        state = apply_shared_stack(model, state, loop_index=cfg.max_loops)
        prediction = logits_from_raw_state(model, state).argmax(dim=-1)
        strict_current, current_count = strict_accuracy(
            prediction, target, current
        )
        strict_endpoint, endpoint_count = strict_accuracy(
            prediction, target, endpoint
        )
        strict_both, both_count = strict_accuracy(
            prediction, target, current, endpoint
        )
        rows.append(
            {
                "model": model_name,
                "eval_seed": eval_seed,
                "intervention": intervention,
                "extra_cycle": cycle,
                "current_path_position": current_position,
                "target_path_position": target_position,
                "sample_count": int(tokens.shape[0]),
                "pre_current_accuracy": float(
                    pre_prediction.eq(current).float().mean()
                ),
                "pre_target_accuracy": float(
                    pre_prediction.eq(target).float().mean()
                ),
                "continued_target_accuracy": float(
                    prediction.eq(target).float().mean()
                ),
                "continued_strict_current_accuracy": strict_current,
                "continued_strict_current_count": current_count,
                "continued_strict_endpoint_accuracy": strict_endpoint,
                "continued_strict_endpoint_count": endpoint_count,
                "continued_doubly_strict_accuracy": strict_both,
                "continued_doubly_strict_count": both_count,
                "endpoint_accuracy": float(
                    prediction.eq(endpoint).float().mean()
                ),
                "answer_rms": float(
                    state[:, -1].float().square().mean(dim=-1).sqrt().mean()
                ),
            }
        )
        if cycle < maximum_cycle:
            state = apply_answer_map(state, answer_map)
    return rows


def mean_rows(
    rows: Sequence[dict[str, Any]],
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        output = dict(zip(keys, group, strict=True))
        output["replications"] = len(selected)
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in selected], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            output[f"{metric}_mean"] = (
                float(finite.mean()) if finite.size else float("nan")
            )
            output[f"{metric}_min"] = (
                float(finite.min()) if finite.size else float("nan")
            )
            output[f"{metric}_max"] = (
                float(finite.max()) if finite.size else float("nan")
            )
        result.append(output)
    return result


def fit_map(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    source_ages: Sequence[int],
    calibration_samples: int,
    validation_samples: int,
    collection_batch_size: int,
    calibration_seed: int,
    validation_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    calibration = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=CANDIDATE,
        sample_count=calibration_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=calibration_seed,
        source_ages=source_ages,
    )
    validation = collect_pair_dataset(
        model=model,
        cfg=cfg,
        candidate=CANDIDATE,
        sample_count=validation_samples,
        batch_size=collection_batch_size,
        device=device,
        seed=validation_seed,
        source_ages=source_ages,
    )
    matrix, bias, selection, ridge = select_and_refit_map(
        calibration=calibration,
        validation=validation,
        use_bias=False,
        ridge_grid=RIDGE_GRID,
    )
    metrics = age_balanced_validation_metrics(validation, matrix, bias)
    metrics["selected_ridge"] = ridge
    return matrix, bias, selection, metrics


def validate_model_config(
    cfg: GraphPathConfig,
    model: LoopedGraphPathTransformer,
    checkpoint: Path,
) -> None:
    actual = (
        cfg.node_count,
        cfg.max_depth,
        cfg.d_model,
        cfg.n_layers,
        cfg.max_loops,
        cfg.inner_norm_style,
    )
    expected = (8, 8, 256, 2, 8, "pre_layernorm")
    if actual != expected:
        raise ValueError(f"unexpected model config for {checkpoint}: {actual}")
    if any(
        model.active_block_indices(loop) != (0, 1)
        for loop in range(65)
    ):
        raise ValueError("the recurrent shared unit changes across loop index")


def plot_spectra(summaries: Sequence[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(
        1, len(summaries), figsize=(5.0 * len(summaries), 4.7), squeeze=False
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    for axis, summary in zip(axes[0], summaries, strict=True):
        real = np.asarray(summary["eigenvalues_real"])
        imaginary = np.asarray(summary["eigenvalues_imag"])
        axis.scatter(real, imaginary, s=13, alpha=0.68, color="#2563eb")
        axis.plot(np.cos(theta), np.sin(theta), "--", color="#9ca3af", linewidth=1)
        axis.scatter(
            [summary["kde_peak_real"]],
            [summary["kde_peak_imag"]],
            marker="x",
            s=70,
            color="#dc2626",
            label="KDE peak",
        )
        axis.axhline(0.0, color="#d1d5db", linewidth=0.8)
        axis.axvline(0.0, color="#d1d5db", linewidth=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{summary['model']}\n"
            f"peak={summary['kde_peak_real']:.3f}, "
            f"log|det|={summary['log_absolute_determinant']:.1f}"
        )
        axis.set_xlabel("Re(λ)")
        axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel("Im(λ)")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_length_curves(rows: Sequence[dict[str, Any]], path: Path) -> None:
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    aggregate = mean_rows(
        rows,
        keys=("model", "intervention", "extra_cycle"),
        metrics=("continued_doubly_strict_accuracy",),
    )
    figure, axes = plt.subplots(
        1, len(models), figsize=(5.2 * len(models), 4.5), sharex=True, sharey=True,
        squeeze=False,
    )
    colors = {
        "global_all_ages": "#2563eb",
        "active_ages_only": "#16a34a",
        "identity_no_rejuvenation": "#6b7280",
        "random_orientation_matched_update_norm": "#dc2626",
        "opposite_direction_matched_update_norm": "#d97706",
    }
    for axis, model_name in zip(axes[0], models, strict=True):
        for intervention in colors:
            selected = [
                row
                for row in aggregate
                if row["model"] == model_name
                and row["intervention"] == intervention
            ]
            if not selected:
                continue
            selected.sort(key=lambda row: int(row["extra_cycle"]))
            axis.plot(
                [row["extra_cycle"] for row in selected],
                [row["continued_doubly_strict_accuracy_mean"] for row in selected],
                label=intervention,
                color=colors[intervention],
                linewidth=1.5,
            )
        axis.axhline(0.9, color="#111827", linestyle=":", linewidth=0.9)
        axis.axhline(0.125, color="#9ca3af", linestyle="--", linewidth=0.9)
        axis.set_title(model_name)
        axis.set_xlabel("additional two-hop F cycles")
        axis.grid(alpha=0.22)
    axes[0, 0].set_ylabel("doubly-strict continued accuracy")
    axes[0, -1].legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_natural_overloop(rows: Sequence[dict[str, Any]], path: Path) -> None:
    aggregate = mean_rows(
        rows,
        keys=("model", "loop"),
        metrics=(
            "endpoint_accuracy",
            "continued_strict_endpoint_accuracy",
            "prediction_stability_to_loop8",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True, sharey=True)
    metrics = (
        ("endpoint_accuracy_mean", "trained endpoint f^8"),
        ("continued_strict_endpoint_accuracy_mean", "continued f^(2L), strict"),
        ("prediction_stability_to_loop8_mean", "agreement with loop 8"),
    )
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for model_name in dict.fromkeys(str(row["model"]) for row in aggregate):
            selected = [row for row in aggregate if row["model"] == model_name]
            selected.sort(key=lambda row: int(row["loop"]))
            axis.plot(
                [row["loop"] for row in selected],
                [row[metric] for row in selected],
                label=model_name,
                linewidth=1.5,
            )
        axis.set_title(title)
        axis.set_xlabel("recurrent loop")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_model(
    *,
    model_name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    calibration_samples: int,
    validation_samples: int,
    collection_batch_size: int,
    evaluation_batch_size: int,
    evaluation_seeds: Sequence[int],
    maximum_natural_loop: int,
    maximum_rejuvenated_cycle: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint, device)
    validate_model_config(cfg, model, checkpoint)
    model.eval()

    natural_rows = natural_overloop_rows(
        model=model,
        cfg=cfg,
        model_name=model_name,
        device=device,
        maximum_loop=maximum_natural_loop,
        batch_size=evaluation_batch_size,
        data_seeds=evaluation_seeds,
    )
    print(f"{model_name}: natural overloop complete", flush=True)

    global_matrix, global_bias, global_selection, global_fit = fit_map(
        model=model,
        cfg=cfg,
        device=device,
        source_ages=PRIMARY_TRAIN_AGES,
        calibration_samples=calibration_samples,
        validation_samples=validation_samples,
        collection_batch_size=collection_batch_size,
        calibration_seed=8_100_001,
        validation_seed=8_110_001,
    )
    active_matrix, active_bias, active_selection, active_fit = fit_map(
        model=model,
        cfg=cfg,
        device=device,
        source_ages=ACTIVE_TRAIN_AGES,
        calibration_samples=calibration_samples,
        validation_samples=validation_samples,
        collection_batch_size=collection_batch_size,
        calibration_seed=8_120_001,
        validation_seed=8_130_001,
    )
    print(f"{model_name}: global and active-age maps fit", flush=True)

    global_map = to_answer_map(global_matrix, global_bias, device=device)
    active_map = to_answer_map(active_matrix, active_bias, device=device)
    identity = zero_map(cfg.d_model, device)
    random_control = random_orientation_control(global_map, seed=8_140_001)
    opposite = opposite_direction_map(global_map)

    spectrum = spectral_summary(global_matrix)
    active_spectrum = spectral_summary(active_matrix)
    j_summary: dict[str, Any] = {
        "model": model_name,
        **global_fit,
        **spectrum,
    }
    active_summary: dict[str, Any] = {
        "model": model_name,
        "map": "active_ages_only",
        **active_fit,
        **active_spectrum,
    }

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "candidate": CANDIDATE,
            "global_all_ages": {
                "source_ages": list(PRIMARY_TRAIN_AGES),
                "matrix": global_matrix,
                "bias": global_bias,
                "fit": global_fit,
                "spectrum": spectrum,
            },
            "active_ages_only": {
                "source_ages": list(ACTIVE_TRAIN_AGES),
                "matrix": active_matrix,
                "bias": active_bias,
                "fit": active_fit,
                "spectrum": active_spectrum,
            },
        },
        model_out / "rejuvenators.pt",
    )

    selection_rows = [
        {"model": model_name, "map": "global_all_ages", **row}
        for row in global_selection
    ] + [
        {"model": model_name, "map": "active_ages_only", **row}
        for row in active_selection
    ]
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
            seed=eval_seed,
            maximum_commutator_age=12,
            reset_source_ages=(8, 16, 32, 64),
        )
        for age in range(3, 13):
            mapping_rows.append(
                {
                    "model": model_name,
                    "eval_seed": eval_seed,
                    **evaluate_mapping(
                        model=model,
                        cfg=cfg,
                        candidate=CANDIDATE,
                        batch=batch,
                        answer_map=global_map,
                        condition="global_linear_256x256",
                        source_age=age,
                    ),
                }
            )
            commutator_rows.append(
                {
                    "model": model_name,
                    "eval_seed": eval_seed,
                    **evaluate_commutator(
                        model=model,
                        cfg=cfg,
                        candidate=CANDIDATE,
                        batch=batch,
                        answer_map=global_map,
                        condition="global_linear_256x256",
                        source_age=age,
                    ),
                }
            )
        for age in range(3, 9):
            mapping_rows.append(
                {
                    "model": model_name,
                    "eval_seed": eval_seed,
                    **evaluate_mapping(
                        model=model,
                        cfg=cfg,
                        candidate=CANDIDATE,
                        batch=batch,
                        answer_map=active_map,
                        condition="active_ages_only",
                        source_age=age,
                    ),
                }
            )
        for intervention, answer_map in (
            ("global_all_ages", global_map),
            ("active_ages_only", active_map),
            ("identity_no_rejuvenation", identity),
            ("random_orientation_matched_update_norm", random_control),
            ("opposite_direction_matched_update_norm", opposite),
        ):
            for source_age in (8, 16, 32, 64):
                reset_rows.append(
                    {
                        "model": model_name,
                        "eval_seed": eval_seed,
                        **evaluate_repeated_reset(
                            model=model,
                            cfg=cfg,
                            candidate=CANDIDATE,
                            batch=batch,
                            answer_map=answer_map,
                            condition=intervention,
                            source_age=source_age,
                        ),
                    }
                )
        reset_rows.extend(
            oracle_answer_reset_rows(
                model=model,
                cfg=cfg,
                batch=batch,
                model_name=model_name,
                eval_seed=eval_seed,
            )
        )

        set_seed(eval_seed + 10_000)
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            evaluation_batch_size,
            device,
            path_positions=rejuvenated_target_position(
                maximum_rejuvenated_cycle
            ),
        )
        for intervention, answer_map in (
            ("global_all_ages", global_map),
            ("active_ages_only", active_map),
            ("identity_no_rejuvenation", identity),
            ("random_orientation_matched_update_norm", random_control),
            ("opposite_direction_matched_update_norm", opposite),
        ):
            length_rows.extend(
                rejuvenated_rollout_rows(
                    model=model,
                    cfg=cfg,
                    tokens=tokens,
                    successors=successors,
                    start=start,
                    answer_map=answer_map,
                    intervention=intervention,
                    model_name=model_name,
                    eval_seed=eval_seed,
                    maximum_cycle=maximum_rejuvenated_cycle,
                )
            )
        print(f"{model_name}: evaluation seed {eval_seed} complete", flush=True)

    write_csv(model_out / "natural_overloop_rows.csv", natural_rows)
    write_csv(model_out / "J_ridge_selection.csv", selection_rows)
    write_csv(model_out / "J_mapping_rows.csv", mapping_rows)
    write_csv(model_out / "J_commutator_rows.csv", commutator_rows)
    write_csv(model_out / "J_reset_rows.csv", reset_rows)
    write_csv(model_out / "rejuvenated_length_rows.csv", length_rows)
    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": asdict(cfg),
        "candidate": CANDIDATE,
        "loss_placement": (
            "component intermediate plus final" if "component" in model_name
            else "final loop 8 only"
        ),
        "shared_unit": "two physical pre-LN blocks per recurrent cycle",
        "trained_loops": cfg.max_loops,
        "effective_depth": cfg.max_loops * cfg.n_layers,
        "global_fit": global_fit,
        "global_spectrum": spectrum,
        "active_fit": active_fit,
        "active_spectrum": active_spectrum,
    }
    (model_out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {
        "summary": summary,
        "j_summary": j_summary,
        "active_summary": active_summary,
        "natural_rows": natural_rows,
        "selection_rows": selection_rows,
        "mapping_rows": mapping_rows,
        "commutator_rows": commutator_rows,
        "reset_rows": reset_rows,
        "length_rows": length_rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare global rejuvenation maps in component-supervised D8L8."
    )
    parser.add_argument(
        "--model", action="append", type=parse_model_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-samples", type=int, default=4096)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--collection-batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument(
        "--evaluation-seeds", type=int, nargs="+", default=(8_200_001, 8_200_002)
    )
    parser.add_argument("--maximum-natural-loop", type=int, default=64)
    parser.add_argument("--maximum-rejuvenated-cycle", type=int, default=200)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)

    outputs = []
    for model_name, checkpoint in args.model:
        print(f"loading {model_name}: {checkpoint}", flush=True)
        outputs.append(
            run_model(
                model_name=model_name,
                checkpoint=checkpoint,
                out_dir=args.out_dir,
                device=device,
                calibration_samples=args.calibration_samples,
                validation_samples=args.validation_samples,
                collection_batch_size=args.collection_batch_size,
                evaluation_batch_size=args.evaluation_batch_size,
                evaluation_seeds=args.evaluation_seeds,
                maximum_natural_loop=args.maximum_natural_loop,
                maximum_rejuvenated_cycle=args.maximum_rejuvenated_cycle,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    natural_rows = [row for output in outputs for row in output["natural_rows"]]
    selection_rows = [row for output in outputs for row in output["selection_rows"]]
    mapping_rows = [row for output in outputs for row in output["mapping_rows"]]
    commutator_rows = [
        row for output in outputs for row in output["commutator_rows"]
    ]
    reset_rows = [row for output in outputs for row in output["reset_rows"]]
    length_rows = [row for output in outputs for row in output["length_rows"]]
    j_summaries = [output["j_summary"] for output in outputs]

    write_csv(args.out_dir / "natural_overloop_rows.csv", natural_rows)
    write_csv(args.out_dir / "J_ridge_selection.csv", selection_rows)
    write_csv(args.out_dir / "J_mapping_rows.csv", mapping_rows)
    write_csv(args.out_dir / "J_commutator_rows.csv", commutator_rows)
    write_csv(args.out_dir / "J_reset_rows.csv", reset_rows)
    write_csv(args.out_dir / "rejuvenated_length_rows.csv", length_rows)
    write_csv(
        args.out_dir / "J_summary.csv",
        [
            {
                key: value
                for key, value in summary.items()
                if not key.startswith("eigenvalues_")
            }
            for summary in j_summaries
        ],
    )
    length_aggregate = mean_rows(
        length_rows,
        keys=("model", "intervention", "extra_cycle"),
        metrics=(
            "pre_current_accuracy",
            "pre_target_accuracy",
            "continued_target_accuracy",
            "continued_strict_current_accuracy",
            "continued_strict_endpoint_accuracy",
            "continued_doubly_strict_accuracy",
            "endpoint_accuracy",
            "answer_rms",
        ),
    )
    write_csv(args.out_dir / "rejuvenated_length_aggregate.csv", length_aggregate)
    plot_spectra(j_summaries, args.out_dir / "global_J_complex_spectra.png")
    plot_length_curves(length_rows, args.out_dir / "rejuvenated_200cycle_curves.png")
    plot_natural_overloop(natural_rows, args.out_dir / "natural_overloop.png")

    manifest = {
        "device": str(device),
        "candidate": CANDIDATE,
        "primary_train_ages": list(PRIMARY_TRAIN_AGES),
        "active_train_ages": list(ACTIVE_TRAIN_AGES),
        "calibration_samples_per_age": args.calibration_samples,
        "validation_samples_per_age": args.validation_samples,
        "evaluation_batch_size": args.evaluation_batch_size,
        "evaluation_seeds": list(args.evaluation_seeds),
        "maximum_natural_loop": args.maximum_natural_loop,
        "maximum_rejuvenated_cycle": args.maximum_rejuvenated_cycle,
        "models": [output["summary"] for output in outputs],
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"analysis complete: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
