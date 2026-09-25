"""Audit what a scalar linear loop-age readout sees in the D8L8 residual stream.

This is a representation/localisation analysis.  It tests whether independently
fit token-group probes share one residual direction and analytically decomposes
the trained loop-boundary J's effect on that scalar.  It deliberately does not
call a high-R2 probe a causal circuit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_transition_matrices import cache_natural_states
from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import explicit_depth_position_groups
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the scalar loop-age readout T(h).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--heldout-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--bootstrap-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _features_targets(
    states: list[torch.Tensor],
    positions: tuple[int, ...],
    *,
    example_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for age in range(2, len(states)):
        value = states[age]
        if example_indices is not None:
            value = value.index_select(0, example_indices)
        value = value[:, list(positions), :].reshape(-1, value.shape[-1]).float()
        features.append(value)
        targets.append(torch.full((value.shape[0],), float(age), device=value.device))
    return torch.cat(features), torch.cat(targets)


def fit_probe_xy(
    x: torch.Tensor, y: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    x_mean, y_mean = x.mean(dim=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    covariance = xc.transpose(0, 1) @ xc / x.shape[0]
    cross = xc.transpose(0, 1) @ yc / x.shape[0]
    scale = torch.diagonal(covariance).mean().clamp_min(1e-8)
    weight = torch.linalg.solve(
        covariance + ridge * scale * torch.eye(x.shape[1], device=x.device), cross
    )
    return weight, y_mean - x_mean @ weight


def fit_probe(
    states: list[torch.Tensor],
    positions: tuple[int, ...],
    ridge: float,
    *,
    example_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fit_probe_xy(
        *_features_targets(states, positions, example_indices=example_indices), ridge
    )


def predict_by_age(
    states: list[torch.Tensor],
    positions: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> list[torch.Tensor]:
    return [
        states[age][:, list(positions), :].float() @ weight + bias
        for age in range(2, len(states))
    ]


def r2_by_age(predictions: list[torch.Tensor]) -> float:
    pred = torch.cat([value.reshape(-1) for value in predictions])
    target = torch.cat(
        [
            torch.full_like(value.reshape(-1), float(age))
            for age, value in enumerate(predictions, start=2)
        ]
    )
    return float(
        1.0
        - (pred - target).square().sum()
        / (target - target.mean()).square().sum().clamp_min(1e-12)
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left @ right)
        / (left.norm() * right.norm()).clamp_min(1e-12)
    )


def calibrate_scalar(
    source_scores: torch.Tensor, target_age: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    centered = source_scores - source_scores.mean()
    target_centered = target_age - target_age.mean()
    scale = (centered @ target_centered) / centered.square().sum().clamp_min(1e-12)
    offset = target_age.mean() - scale * source_scores.mean()
    return scale, offset


def affine_from_j(operator: DiagonalIdentityLoRAJ) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.diag(operator.diagonal_scale.float()) + operator.A.float() @ operator.B.float()
    return weight, operator.bias.float()


def draw_probe_geometry(
    out_dir: Path,
    probes: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cosine_rows: list[dict[str, Any]],
    cross_rows: list[dict[str, Any]],
) -> None:
    groups = list(probes)
    weights = torch.stack([probes[group][0].cpu() for group in groups])
    cos_matrix = np.full((len(groups), len(groups)), np.nan)
    for row in cosine_rows:
        cos_matrix[groups.index(row["source_group"]), groups.index(row["target_group"])] = row["cosine"]
    cross_matrix = np.full_like(cos_matrix, np.nan)
    for row in cross_rows:
        cross_matrix[groups.index(row["source_group"]), groups.index(row["target_group"])] = row["heldout_r2"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), constrained_layout=True)
    limit = float(weights.abs().quantile(0.995))
    im = axes[0].imshow(weights.numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[0].set_yticks(range(len(groups)), groups)
    axes[0].set_xlabel("residual dimension")
    axes[0].set_title("Independently fitted T directions")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for axis, matrix, title, vmin in [
        (axes[1], cos_matrix, "Pairwise cosine of T directions", -1.0),
        (axes[2], cross_matrix, "Cross-position transfer R2", 0.0),
    ]:
        image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=1.0)
        axis.set_xticks(range(len(groups)), groups, rotation=35, ha="right")
        axis.set_yticks(range(len(groups)), groups)
        axis.set_xlabel("target group")
        axis.set_ylabel("source group")
        axis.set_title(title)
        for i in range(len(groups)):
            for j in range(len(groups)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.savefig(out_dir / "01_T_weights_and_transfer.png", dpi=220)
    plt.close(fig)


def draw_j_decomposition(
    out_dir: Path,
    decomposition_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
) -> None:
    groups = sorted({row["group"] for row in decomposition_rows})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    answer = [row for row in decomposition_rows if row["group"] == "answer"]
    ages = sorted({int(row["age"]) for row in answer})
    for component in ["diagonal", "low_rank", "bias", "total"]:
        selected = [row for row in answer if row["component"] == component]
        axes[0].plot(ages, [row["mean_delta_age"] for row in selected], marker="o", label=component)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_title("How J changes answer-token T(h)")
    axes[0].set_xlabel("native loop age")
    axes[0].set_ylabel("age-score change")
    axes[0].set_xticks(ages)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    x = np.arange(len(groups))
    axes[1].bar(
        x - 0.18,
        [next(row["mean_step_score"] for row in drift_rows if row["group"] == group) for group in groups],
        width=0.36,
        label="native one-loop change",
    )
    axes[1].bar(
        x + 0.18,
        [
            np.mean([row["mean_delta_age"] for row in decomposition_rows if row["group"] == group and row["component"] == "total"])
            for group in groups
        ],
        width=0.36,
        label="mean J change",
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(x, groups, rotation=25, ha="right")
    axes[1].set_ylabel("age-score change")
    axes[1].set_title("Native aging versus J rejuvenation")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "02_J_effect_on_T.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.calibration_examples % args.batch_size or args.heldout_examples % args.batch_size:
        raise ValueError("example counts must be divisible by batch-size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    artifact_checkpoint, positions, operators, artifact_payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("this audit expects a position-shared J applied to all tokens")
    operator = operators[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ):
        raise TypeError("expected a diagonal plus low-rank J")
    calibration = cache_natural_states(
        model=model,
        cfg=cfg,
        device=device,
        examples=args.calibration_examples,
        batch_size=args.batch_size,
    )
    heldout = cache_natural_states(
        model=model,
        cfg=cfg,
        device=device,
        examples=args.heldout_examples,
        batch_size=args.batch_size,
    )
    base_groups = explicit_depth_position_groups(cfg.node_count)
    groups = {
        "answer": base_groups["answer"],
        "start": base_groups["start"],
        "depth": base_groups["depth"],
        "graph": base_groups["graph"],
        "all": tuple(range(cfg.seq_len)),
    }
    probes = {
        group: fit_probe(calibration, group_positions, args.ridge)
        for group, group_positions in groups.items()
    }
    probe_rows: list[dict[str, Any]] = []
    for group, group_positions in groups.items():
        weight, bias = probes[group]
        square = weight.square()
        sorted_square = torch.sort(square, descending=True).values
        probe_rows.append(
            {
                "group": group,
                "heldout_r2": r2_by_age(predict_by_age(heldout, group_positions, weight, bias)),
                "weight_norm": float(weight.norm()),
                "effective_dimensions": float(square.sum().square() / square.square().sum().clamp_min(1e-12)),
                "top_16_energy_fraction": float(sorted_square[:16].sum() / square.sum()),
                "top_48_energy_fraction": float(sorted_square[:48].sum() / square.sum()),
            }
        )
    cosine_rows = [
        {
            "source_group": source,
            "target_group": target,
            "cosine": cosine(probes[source][0], probes[target][0]),
        }
        for source in groups
        for target in groups
    ]
    cross_rows: list[dict[str, Any]] = []
    for source, (weight, _) in probes.items():
        for target, target_positions in groups.items():
            calibration_x, calibration_y = _features_targets(calibration, target_positions)
            scale, offset = calibrate_scalar(calibration_x @ weight, calibration_y)
            heldout_x, heldout_y = _features_targets(heldout, target_positions)
            prediction = scale * (heldout_x @ weight) + offset
            uncalibrated_prediction = heldout_x @ weight + probes[source][1]
            r2 = 1.0 - (prediction - heldout_y).square().sum() / (
                heldout_y - heldout_y.mean()
            ).square().sum().clamp_min(1e-12)
            uncalibrated_r2 = 1.0 - (
                uncalibrated_prediction - heldout_y
            ).square().sum() / (heldout_y - heldout_y.mean()).square().sum().clamp_min(1e-12)
            cross_rows.append(
                {
                    "source_group": source,
                    "target_group": target,
                    "calibration_scale": float(scale),
                    "calibration_offset": float(offset),
                    "heldout_r2": float(r2),
                    "uncalibrated_heldout_r2": float(uncalibrated_r2),
                }
            )

    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    stability_rows: list[dict[str, Any]] = []
    for group in ["answer", "all"]:
        full_weight = probes[group][0]
        for repeat in range(args.bootstrap_repeats):
            indices = torch.randperm(
                args.calibration_examples, generator=generator, device=device
            )[: args.calibration_examples // 2]
            weight, bias = fit_probe(
                calibration, groups[group], args.ridge, example_indices=indices
            )
            stability_rows.append(
                {
                    "group": group,
                    "repeat": repeat,
                    "cosine_to_full_probe": cosine(weight, full_weight),
                    "heldout_r2": r2_by_age(
                        predict_by_age(heldout, groups[group], weight, bias)
                    ),
                }
            )

    j_weight, j_bias = affine_from_j(operator)
    diagonal_delta = operator.diagonal_scale.float() - 1.0
    decomposition_rows: list[dict[str, Any]] = []
    drift_stage_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for group, group_positions in groups.items():
        weight = probes[group][0]
        per_stage: list[float] = []
        for age in range(2, cfg.max_loops):
            source = heldout[age][:, list(group_positions), :].float()
            target = heldout[age + 1][:, list(group_positions), :].float()
            delta = target - source
            mean_vector = delta.reshape(-1, cfg.d_model).mean(dim=0)
            scores = delta @ weight
            per_stage.append(float(scores.mean()))
            drift_stage_rows.append(
                {
                    "group": group,
                    "source_age": age,
                    "target_age": age + 1,
                    "mean_score_change": float(scores.mean()),
                    "std_score_change": float(scores.std(unbiased=False)),
                    "positive_fraction": float((scores > 0).float().mean()),
                    "cosine_probe_with_mean_drift": cosine(weight, mean_vector),
                }
            )
        drift_rows.append({"group": group, "mean_step_score": float(np.mean(per_stage))})
        for age in range(2, cfg.max_loops + 1):
            state = heldout[age][:, list(group_positions), :].float()
            components = {
                "diagonal": (state * diagonal_delta) @ weight,
                "low_rank": ((state @ operator.A.float()) @ operator.B.float()) @ weight,
                "bias": torch.full(
                    state.shape[:-1], float(j_bias @ weight), device=device
                ),
            }
            components["total"] = (state @ (j_weight - torch.eye(cfg.d_model, device=device)) + j_bias) @ weight
            for component, score in components.items():
                decomposition_rows.append(
                    {
                        "group": group,
                        "age": age,
                        "component": component,
                        "mean_delta_age": float(score.mean()),
                        "std_delta_age": float(score.std(unbiased=False)),
                    }
                )

    low_rank = operator.A.float() @ operator.B.float()
    _, singular, right_t = torch.linalg.svd(low_rank, full_matrices=False)
    numerical_rank = int((singular > singular.max() * 1e-6).sum())
    mode_rows: list[dict[str, Any]] = []
    for group, (weight, _) in probes.items():
        projection_fraction = float(
            (right_t[:numerical_rank] @ weight).square().sum()
            / weight.square().sum().clamp_min(1e-12)
        )
        for mode in range(numerical_rank):
            mode_rows.append(
                {
                    "group": group,
                    "mode": mode,
                    "singular_value": float(singular[mode]),
                    "signed_output_sensitivity": float(
                        singular[mode] * (right_t[mode] @ weight)
                    ),
                    "probe_energy_in_low_rank_output_span": projection_fraction,
                }
            )

    feedback_rows: list[dict[str, Any]] = []
    identity = torch.eye(cfg.d_model, device=device)
    for group, group_positions in groups.items():
        weight, bias = probes[group]
        correction_direction = (j_weight - identity) @ weight
        calibration_x, calibration_age = _features_targets(
            calibration, group_positions
        )
        heldout_x, heldout_age = _features_targets(heldout, group_positions)
        native_calibration = calibration_x @ weight + bias
        native_heldout = heldout_x @ weight + bias
        delta_calibration = calibration_x @ correction_direction + j_bias @ weight
        delta_heldout = heldout_x @ correction_direction + j_bias @ weight
        feedback_scale, feedback_offset = calibrate_scalar(
            native_calibration, delta_calibration
        )
        delta_prediction = feedback_scale * native_heldout + feedback_offset
        feedback_r2 = 1.0 - (delta_prediction - delta_heldout).square().sum() / (
            delta_heldout - delta_heldout.mean()
        ).square().sum().clamp_min(1e-12)
        age_scale, age_offset = calibrate_scalar(
            calibration_x @ correction_direction, calibration_age
        )
        age_prediction = age_scale * (heldout_x @ correction_direction) + age_offset
        correction_age_r2 = 1.0 - (age_prediction - heldout_age).square().sum() / (
            heldout_age - heldout_age.mean()
        ).square().sum().clamp_min(1e-12)
        feedback_rows.append(
            {
                "group": group,
                "cosine_correction_direction_to_T": cosine(
                    correction_direction, weight
                ),
                "feedback_slope_deltaT_per_T": float(feedback_scale),
                "feedback_intercept": float(feedback_offset),
                "heldout_r2_deltaT_from_T": float(feedback_r2),
                "heldout_r2_age_from_correction_signal": float(correction_age_r2),
                "bias_contribution_to_deltaT": float(j_bias @ weight),
            }
        )

    shuffled_x, shuffled_y = _features_targets(calibration, groups["answer"])
    shuffled_y = shuffled_y[torch.randperm(shuffled_y.numel(), generator=generator, device=device)]
    shuffled_weight, shuffled_bias = fit_probe_xy(shuffled_x, shuffled_y, args.ridge)
    negative_control_r2 = r2_by_age(
        predict_by_age(heldout, groups["answer"], shuffled_weight, shuffled_bias)
    )

    torch.save(
        {
            "checkpoint": str(args.checkpoint),
            "groups": groups,
            "probes": {
                group: {"weight": weight.cpu(), "bias": bias.cpu()}
                for group, (weight, bias) in probes.items()
            },
        },
        args.out_dir / "T_probe_weights.pt",
    )
    write_csv(args.out_dir / "probe_metrics.csv", probe_rows)
    write_csv(args.out_dir / "probe_cosines.csv", cosine_rows)
    write_csv(args.out_dir / "cross_position_transfer.csv", cross_rows)
    write_csv(args.out_dir / "probe_stability.csv", stability_rows)
    write_csv(args.out_dir / "native_stage_drift.csv", drift_stage_rows)
    write_csv(args.out_dir / "J_T_decomposition.csv", decomposition_rows)
    write_csv(args.out_dir / "J_low_rank_modes.csv", mode_rows)
    write_csv(args.out_dir / "J_scalar_feedback.csv", feedback_rows)
    draw_probe_geometry(args.out_dir, probes, cosine_rows, cross_rows)
    draw_j_decomposition(args.out_dir, decomposition_rows, drift_rows)

    answer_probe = next(row for row in probe_rows if row["group"] == "answer")
    all_probe = next(row for row in probe_rows if row["group"] == "all")
    all_to_groups = [row for row in cross_rows if row["source_group"] == "all"]
    cross_off_diagonal = [
        row["heldout_r2"]
        for row in cross_rows
        if row["source_group"] != row["target_group"]
    ]
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": artifact_payload.get("backbone_loss_description", "not recorded"),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "operator_placement": artifact_payload.get("placement"),
        "answer_probe_r2": answer_probe["heldout_r2"],
        "all_position_probe_r2": all_probe["heldout_r2"],
        "cross_position_transfer_r2_mean": float(np.mean(cross_off_diagonal)),
        "cross_position_transfer_r2_min": float(np.min(cross_off_diagonal)),
        "strict_shared_T_r2_by_group": {
            row["target_group"]: row["uncalibrated_heldout_r2"]
            for row in all_to_groups
        },
        "recalibrated_shared_direction_r2_by_group": {
            row["target_group"]: row["heldout_r2"] for row in all_to_groups
        },
        "shuffled_age_negative_control_r2": negative_control_r2,
        "low_rank_numerical_rank": numerical_rank,
        "J_scalar_feedback": feedback_rows,
        "probe_metrics": probe_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "CLAIM_LEDGER.md").write_text(
        "# Claim ledger\n\n"
        "- Linear age probe and cross-position transfer: localisation evidence only.\n"
        "- Analytic diagonal/low-rank/bias decomposition: exact for "
        "T(J(h))-T(h), but only relative to the fitted scalar T.\n"
        "- No causal circuit claim: a direction-selective intervention with "
        "matched random controls is still required.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
