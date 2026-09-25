"""Test whether native loop-transition products support a telomere-like account.

The object of study is the frozen D8L8 seed0 trajectory, not a backward
regression.  We fit same-position forward maps h_t -> h_(t+1), compose them,
then ask whether their products accumulate a common contraction and whether
the trained loop-boundary J counteracts the associated scalar age readout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_transition_matrices import (
    affine_metrics,
    cache_natural_states,
    fit_affine,
)
from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_functional_circuit import explicit_depth_position_groups
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test forward-transition products for a telomere-like state variable."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--heldout-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260802)
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


def flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, value.shape[-1]).float()


def compose_affine(
    first: tuple[torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return second(first(x)) for row-vector affine maps xW+b."""
    first_weight, first_bias = first
    second_weight, second_bias = second
    return (
        first_weight @ second_weight,
        first_bias @ second_weight + second_bias,
    )


def apply_affine(
    state: torch.Tensor, affine: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    weight, bias = affine
    return state.float() @ weight + bias


def affine_from_j(operator: DiagonalIdentityLoRAJ) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.diag(operator.diagonal_scale.float()) + operator.A.float() @ operator.B.float()
    return weight, operator.bias.float()


def relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction - target).norm() / target.norm().clamp_min(1e-12))


def fit_scalar_age_probe(
    states: list[torch.Tensor],
    positions: tuple[int, ...],
    ridge: float,
    *,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for age in range(2, len(states)):
        value = states[age][:, list(positions), :].reshape(-1, states[age].shape[-1]).float()
        if normalize:
            value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        features.append(value)
        targets.append(torch.full((value.shape[0],), float(age), device=value.device))
    x, y = torch.cat(features), torch.cat(targets)
    x_mean, y_mean = x.mean(dim=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    covariance = xc.transpose(0, 1) @ xc / x.shape[0]
    cross = xc.transpose(0, 1) @ yc / x.shape[0]
    scale = torch.diagonal(covariance).mean().clamp_min(1e-8)
    weight = torch.linalg.solve(
        covariance + ridge * scale * torch.eye(x.shape[1], device=x.device), cross
    )
    bias = y_mean - x_mean @ weight
    return weight, bias


def scalar_score(
    state: torch.Tensor,
    positions: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    value = state[:, list(positions), :].float()
    if normalize:
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return value @ weight + bias


def scalar_r2(values: list[torch.Tensor]) -> float:
    predictions = torch.cat([value.reshape(-1) for value in values])
    targets = torch.cat(
        [torch.full_like(value.reshape(-1), float(age)) for age, value in enumerate(values, start=2)]
    )
    return float(1.0 - (predictions - targets).square().sum() / (targets - targets.mean()).square().sum())


def draw_product_figure(
    *,
    out_dir: Path,
    product_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    ages = [int(row["target_age"]) for row in product_rows]
    axes[0].plot(ages, [row["product_relative_rmse"] for row in product_rows], marker="o", label="product of one-step fits")
    axes[0].plot(ages, [row["direct_relative_rmse"] for row in product_rows], marker="o", label="direct h2→hk fit")
    axes[0].set_title("Does the product reproduce the native trajectory?")
    axes[0].set_xlabel("target age k in h2→hk")
    axes[0].set_ylabel("held-out relative RMSE")
    axes[0].set_xticks(ages)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for quantile, label in [(0.1, "10th percentile"), (0.5, "median"), (0.9, "90th percentile")]:
        selected = [row for row in spectrum_rows if abs(float(row["quantile"]) - quantile) < 1e-9]
        axes[1].plot(
            [int(row["target_age"]) for row in selected],
            [float(row["singular_value"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[1].axhline(1.0, color="black", lw=0.8, linestyle="--")
    axes[1].set_yscale("log")
    axes[1].set_title("Singular values of cumulative forward products")
    axes[1].set_xlabel("target age k in h2→hk")
    axes[1].set_ylabel("singular value")
    axes[1].set_xticks(ages)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_dir / "01_forward_products.png", dpi=220)
    plt.close(fig)


def draw_common_contraction_figure(
    *,
    out_dir: Path,
    common_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for subset, label in [("finally_contracting", "directions contracting by h8"), ("finally_expanding", "other directions")]:
        selected = [row for row in common_rows if row["subset"] == subset]
        axes[0].plot(
            [int(row["target_age"]) for row in selected],
            [float(row["median_gain"]) for row in selected],
            marker="o",
            label=label,
        )
    axes[0].axhline(1.0, color="black", lw=0.8, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_title("Fixed h2 directions under cumulative products")
    axes[0].set_xlabel("target age k in h2→hk")
    axes[0].set_ylabel("median directional gain")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    selected = [row for row in common_rows if row["subset"] == "finally_contracting"]
    axes[1].bar(
        [str(row["target_age"]) for row in selected],
        [float(row["monotone_fraction"]) for row in selected],
        color="#4477AA",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Fraction of final-contracting directions\nstill below their preceding gain")
    axes[1].set_xlabel("target age")
    axes[1].set_ylabel("fraction")
    fig.savefig(out_dir / "02_common_contraction.png", dpi=220)
    plt.close(fig)


def draw_scalar_figure(
    *,
    out_dir: Path,
    probe_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    groups = ["answer", "start", "depth", "graph"]
    x = np.arange(len(groups))
    for offset, normalized, label in [(-0.18, False, "raw hidden state"), (0.18, True, "norm-normalized state")]:
        values = [
            next(
                float(row["r2"])
                for row in probe_rows
                if row["group"] == group and bool(row["normalized"]) == normalized
            )
            for group in groups
        ]
        axes[0].bar(x + offset, values, width=0.34, label=label)
    axes[0].set_xticks(x, groups)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title("Linear age readout, held-out R²")
    axes[0].set_ylabel("R²")
    axes[0].legend(fontsize=8)
    answer_native = [
        row for row in trajectory_rows if row["group"] == "answer" and row["condition"] == "native"
    ]
    answer_j = [
        row for row in trajectory_rows if row["group"] == "answer" and row["condition"] == "after_J"
    ]
    axes[1].plot([int(row["age"]) for row in answer_native], [float(row["mean_score"]) for row in answer_native], marker="o", label="native h_t")
    axes[1].plot([int(row["age"]) for row in answer_j], [float(row["mean_score"]) for row in answer_j], marker="o", label="J(h_t)")
    axes[1].set_title("Answer-token scalar age score")
    axes[1].set_xlabel("native age t")
    axes[1].set_ylabel("linear score (calibrated to age)")
    axes[1].set_xticks(range(2, 9))
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    for group in groups:
        selected = [row for row in trajectory_rows if row["group"] == group and row["condition"] == "J_delta"]
        axes[2].plot(
            [int(row["age"]) for row in selected],
            [float(row["mean_score"]) for row in selected],
            marker="o",
            label=group,
        )
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_title("Change in age score caused by J")
    axes[2].set_xlabel("native age t")
    axes[2].set_ylabel("mean[T(J(h_t)) − T(h_t)]")
    axes[2].set_xticks(range(2, 9))
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.savefig(out_dir / "03_scalar_age_and_J.png", dpi=220)
    plt.close(fig)


def write_report(
    *,
    out_dir: Path,
    product_rows: list[dict[str, Any]],
    common_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> None:
    last = product_rows[-1]
    answer_probe = next(
        row for row in probe_rows if row["group"] == "answer" and not bool(row["normalized"])
    )
    answer_j = [
        row for row in trajectory_rows if row["group"] == "answer" and row["condition"] == "J_delta"
    ]
    contraction = [row for row in common_rows if row["subset"] == "finally_contracting"]
    text = """# Forward-product test of the telomere hypothesis

## Question

We call the effect telomere-like only if normal loop updates form a cumulative,
directionally coherent age drift and the fixed loop-boundary J moves against
that drift.  This analysis uses only same-position **forward** fits
`F_t: h_t -> h_(t+1)`; no backward regression R is part of the claim.

## Results

- The composed h2->h8 map has held-out relative RMSE {product_rmse:.4f}; a
  separately fit direct h2->h8 affine map has {direct_rmse:.4f}.  Their gap
  measures how much local linearisation error accumulates under composition.
- The answer token has raw linear age-readout R² {answer_r2:.4f}.  The
  norm-normalized control in `age_probe_metrics.csv` distinguishes a direction
  code from a pure norm clock.
- For the answer age score, J's change across h2..h8 ranges from {j_min:.4f}
  to {j_max:.4f}.  A consistently negative value would be direct evidence
  that J makes this scalar younger; a mixed sign would reject a one-scalar
  telomere account.
- `02_common_contraction.png` uses the **natural threshold 1** (contracting
  vs non-contracting singular gain), not a chosen PCA dimension.  It tests
  whether directions that contract in the full h2->h8 product are already
  contracting through the prefix products.

## Evidence boundary

These are representation-dynamics tests, not yet a causal circuit result.
To make a circuit claim, the next experiment must selectively perturb the
identified scalar/subspace at the loop boundary, use magnitude-matched random
directions, and measure held-out task readout recovery.
""".format(
        product_rmse=float(last["product_relative_rmse"]),
        direct_rmse=float(last["direct_relative_rmse"]),
        answer_r2=float(answer_probe["r2"]),
        j_min=min(float(row["mean_score"]) for row in answer_j),
        j_max=max(float(row["mean_score"]) for row in answer_j),
    )
    (out_dir / "REPORT_CN.md").write_text(text, encoding="utf-8")


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.calibration_examples % args.batch_size or args.heldout_examples % args.batch_size:
        raise ValueError("example counts must divide batch-size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    set_seed(args.seed)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    artifact_checkpoint, positions, modules, _ = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("J artifact checkpoint does not match model checkpoint")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("expected J to operate on every token position")
    operator = modules[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ):
        raise TypeError("expected diagonal-low-rank J")
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

    forward: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for age in range(2, cfg.max_loops):
        forward[age] = fit_affine(calibration[age], calibration[age + 1], args.ridge)

    identity = torch.eye(cfg.d_model, device=device)
    accumulated = (identity, torch.zeros(cfg.d_model, device=device))
    products: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    product_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for target_age in range(3, cfg.max_loops + 1):
        accumulated = compose_affine(accumulated, forward[target_age - 1])
        products[target_age] = accumulated
        product_metrics = affine_metrics(
            heldout[2], heldout[target_age], accumulated[0], accumulated[1]
        )
        direct = fit_affine(calibration[2], calibration[target_age], args.ridge)
        direct_metrics = affine_metrics(heldout[2], heldout[target_age], direct[0], direct[1])
        composed_prediction = apply_affine(heldout[2], accumulated)
        direct_prediction = apply_affine(heldout[2], direct)
        product_rows.append(
            {
                "target_age": target_age,
                "product_relative_rmse": product_metrics["relative_rmse"],
                "product_r2": product_metrics["r2"],
                "direct_relative_rmse": direct_metrics["relative_rmse"],
                "direct_r2": direct_metrics["r2"],
                "product_vs_direct_relative_difference": relative_error(
                    composed_prediction, direct_prediction
                ),
                "weight_condition_number": float(torch.linalg.cond(accumulated[0])),
            }
        )
        singular = torch.linalg.svdvals(accumulated[0])
        for quantile in [0.1, 0.5, 0.9]:
            spectrum_rows.append(
                {
                    "target_age": target_age,
                    "quantile": quantile,
                    "singular_value": float(torch.quantile(singular, quantile)),
                }
            )

    # Common fixed directions: use the input singular vectors of the full h2->h8
    # product and separate them by the non-arbitrary gain threshold of 1.
    final_weight = products[cfg.max_loops][0]
    U, final_singular, _ = torch.linalg.svd(final_weight, full_matrices=False)
    contracting = final_singular < 1.0
    if not bool(contracting.any()) or bool(contracting.all()):
        raise RuntimeError("full product did not contain both contracting and non-contracting directions")
    common_rows: list[dict[str, Any]] = []
    prior_gains: dict[str, torch.Tensor] = {}
    for target_age, (weight, _) in products.items():
        for mask, subset in [(contracting, "finally_contracting"), (~contracting, "finally_expanding")]:
            gains = (U[:, mask].transpose(0, 1) @ weight).norm(dim=-1)
            previous = prior_gains.get(subset)
            monotone = 1.0 if previous is None else float((gains <= previous).float().mean())
            common_rows.append(
                {
                    "target_age": target_age,
                    "subset": subset,
                    "directions": int(mask.sum()),
                    "median_gain": float(gains.median()),
                    "mean_gain": float(gains.mean()),
                    "monotone_fraction": monotone,
                }
            )
            prior_gains[subset] = gains

    groups_all = explicit_depth_position_groups(cfg.node_count)
    groups = {key: groups_all[key] for key in ["answer", "start", "depth", "graph"]}
    j_affine = affine_from_j(operator)
    probe_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for group, group_positions in groups.items():
        raw_weight, raw_bias = fit_scalar_age_probe(
            calibration, group_positions, args.ridge, normalize=False
        )
        normalized_weight, normalized_bias = fit_scalar_age_probe(
            calibration, group_positions, args.ridge, normalize=True
        )
        for normalized, weight, fitted_bias in [
            (False, raw_weight, raw_bias),
            (True, normalized_weight, normalized_bias),
        ]:
            scores = [
                scalar_score(
                    heldout[age], group_positions, weight, fitted_bias, normalize=normalized
                )
                for age in range(2, cfg.max_loops + 1)
            ]
            probe_rows.append(
                {
                    "group": group,
                    "normalized": normalized,
                    "r2": scalar_r2(scores),
                }
            )
        for age in range(2, cfg.max_loops + 1):
            native = scalar_score(
                heldout[age], group_positions, raw_weight, raw_bias, normalize=False
            )
            after_j = scalar_score(
                apply_affine(heldout[age], j_affine),
                group_positions,
                raw_weight,
                raw_bias,
                normalize=False,
            )
            for condition, value in [("native", native), ("after_J", after_j), ("J_delta", after_j - native)]:
                trajectory_rows.append(
                    {
                        "group": group,
                        "age": age,
                        "condition": condition,
                        "mean_score": float(value.mean()),
                        "std_score": float(value.std(unbiased=False)),
                    }
                )

    write_csv(args.out_dir / "forward_product_metrics.csv", product_rows)
    write_csv(args.out_dir / "product_spectrum.csv", spectrum_rows)
    write_csv(args.out_dir / "common_contraction.csv", common_rows)
    write_csv(args.out_dir / "age_probe_metrics.csv", probe_rows)
    write_csv(args.out_dir / "age_scalar_trajectory.csv", trajectory_rows)
    draw_product_figure(out_dir=args.out_dir, product_rows=product_rows, spectrum_rows=spectrum_rows)
    draw_common_contraction_figure(out_dir=args.out_dir, common_rows=common_rows)
    draw_scalar_figure(
        out_dir=args.out_dir, probe_rows=probe_rows, trajectory_rows=trajectory_rows
    )
    write_report(
        out_dir=args.out_dir,
        product_rows=product_rows,
        common_rows=common_rows,
        probe_rows=probe_rows,
        trajectory_rows=trajectory_rows,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "seed": args.seed,
        "calibration_examples": args.calibration_examples,
        "heldout_examples": args.heldout_examples,
        "ridge": args.ridge,
        "forward_product_rows": product_rows,
        "age_probe_rows": probe_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
