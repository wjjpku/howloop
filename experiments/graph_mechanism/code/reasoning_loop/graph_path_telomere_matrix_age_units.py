"""Calibrate non-uniform loop age in units of a long-rollout-trained J.

No rejuvenation map is regressed from hidden-state pairs.  The only J is the
canonical CE-only, long-rollout-trained loop-boundary controller.  For each
native stage and one/two true F updates, search how many exact matrix powers
of J best restore the same-age interface at the advanced graph current.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_power_schedule import (
    affine_from_j,
    affine_powers,
    apply_affine_positions,
)
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure loop age in trained-J matrix units.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-j-power", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260807)
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


def homogeneous(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Row-vector affine map as one exact (d+1)x(d+1) linear matrix."""
    dimension = weight.shape[0]
    result = torch.zeros(
        dimension + 1, dimension + 1, device=weight.device, dtype=weight.dtype
    )
    result[:dimension, :dimension] = weight
    result[dimension, :dimension] = bias
    result[dimension, dimension] = 1.0
    return result


def state_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    residual = prediction.float() - target.float()
    target_centered = target.float() - target.float().mean(dim=(0, 1), keepdim=True)
    return {
        "relative_rmse": float(residual.norm() / target.float().norm().clamp_min(1e-12)),
        "centered_r2": float(
            1.0
            - residual.square().sum()
            / target_centered.square().sum().clamp_min(1e-12)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                prediction.float().reshape(-1, prediction.shape[-1]),
                target.float().reshape(-1, target.shape[-1]),
                dim=-1,
            ).mean()
        ),
    }


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    cfg: Any,
    phase_positions: list[int],
    powers: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    examples: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch-size")
    set_seed(seed)
    totals: dict[tuple[int, int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    ages = range(2, 7)
    horizons = (1, 2)
    positions = tuple(range(cfg.seq_len))
    for _ in range(examples // batch_size):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        for start_age in ages:
            for horizon in horizons:
                state = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=endpoint,
                    age=start_age,
                    phase_position=phase_positions[start_age],
                )
                for step_index in range(horizon):
                    state = run_one_loop(
                        model,
                        state,
                        loop_index=cfg.max_loops + step_index,
                    ).state
                advanced_current = advance_nodes(successors, endpoint, steps=horizon)
                same_age_target = _aligned_state_at_age(
                    model=model,
                    cfg=cfg,
                    successors=successors,
                    current=advanced_current,
                    age=start_age,
                    phase_position=phase_positions[start_age],
                )
                next_target = advance_nodes(successors, advanced_current, steps=1)
                oracle_step = run_one_loop(
                    model,
                    same_age_target,
                    loop_index=cfg.max_loops + horizon,
                )
                oracle_correct = int(
                    oracle_step.logits.argmax(dim=-1).eq(next_target).sum().item()
                )
                for power, affine in powers.items():
                    corrected = apply_affine_positions(state, positions, affine)
                    metrics = state_metrics(corrected, same_age_target)
                    future = run_one_loop(
                        model,
                        corrected,
                        loop_index=cfg.max_loops + horizon,
                    )
                    oracle_logits = oracle_step.logits.float()
                    future_logits = future.logits.float()
                    logit_residual = future_logits - oracle_logits
                    key = (start_age, horizon, power)
                    for name, value in metrics.items():
                        totals[key][name] += value * batch_size
                    totals[key]["future_correct"] += int(
                        future.logits.argmax(dim=-1).eq(next_target).sum().item()
                    )
                    totals[key]["future_logit_squared_error"] += float(
                        logit_residual.square().sum()
                    )
                    totals[key]["future_oracle_logit_squared_norm"] += float(
                        oracle_logits.square().sum()
                    )
                    totals[key]["future_prediction_agreement"] += int(
                        future.logits.argmax(dim=-1)
                        .eq(oracle_step.logits.argmax(dim=-1))
                        .sum()
                        .item()
                    )
                    totals[key]["oracle_future_correct"] += oracle_correct
                    totals[key]["examples"] += batch_size
    rows: list[dict[str, Any]] = []
    for (start_age, horizon, power), values in sorted(totals.items()):
        count = values["examples"]
        rows.append(
            {
                "start_age": start_age,
                "F_updates": horizon,
                "J_power": power,
                "target_age_after_rejuvenation": start_age,
                "relative_rmse": values["relative_rmse"] / count,
                "centered_r2": values["centered_r2"] / count,
                "cosine": values["cosine"] / count,
                "future_accuracy": values["future_correct"] / count,
                "oracle_future_accuracy": values["oracle_future_correct"] / count,
                "future_relative_logit_rmse_to_oracle": float(
                    np.sqrt(
                        values["future_logit_squared_error"]
                        / max(values["future_oracle_logit_squared_norm"], 1e-12)
                    )
                ),
                "future_prediction_agreement_with_oracle": values[
                    "future_prediction_agreement"
                ]
                / count,
                "examples": int(count),
            }
        )
    return rows


def best_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["start_age"]), int(row["F_updates"]))].append(row)
    best: list[dict[str, Any]] = []
    for (age, horizon), selected in sorted(grouped.items()):
        hidden = min(selected, key=lambda row: float(row["relative_rmse"]))
        causal = min(
            selected,
            key=lambda row: (
                float(row["future_relative_logit_rmse_to_oracle"]),
                float(row["relative_rmse"]),
            ),
        )
        best.append(
            {
                "start_age": age,
                "F_updates": horizon,
                "best_hidden_J_power": hidden["J_power"],
                "best_hidden_relative_rmse": hidden["relative_rmse"],
                "best_hidden_future_accuracy": hidden["future_accuracy"],
                "best_causal_J_power": causal["J_power"],
                "best_causal_future_accuracy": causal["future_accuracy"],
                "best_causal_relative_rmse": causal["relative_rmse"],
                "best_causal_future_relative_logit_rmse": causal[
                    "future_relative_logit_rmse_to_oracle"
                ],
                "best_causal_prediction_agreement": causal[
                    "future_prediction_agreement_with_oracle"
                ],
                "oracle_future_accuracy": causal["oracle_future_accuracy"],
            }
        )
    lookup = {
        (int(row["start_age"]), int(row["F_updates"])): row for row in best
    }
    additivity: list[dict[str, Any]] = []
    for age in range(2, 7):
        one = lookup[(age, 1)]
        next_one = lookup.get((age + 1, 1))
        two = lookup[(age, 2)]
        if next_one is None:
            continue
        additivity.append(
            {
                "start_age": age,
                "one_step_hidden_units": one["best_hidden_J_power"],
                "next_one_step_hidden_units": next_one["best_hidden_J_power"],
                "predicted_two_step_units": int(one["best_hidden_J_power"])
                + int(next_one["best_hidden_J_power"]),
                "observed_two_step_units": two["best_hidden_J_power"],
                "hidden_additivity_holds": int(one["best_hidden_J_power"])
                + int(next_one["best_hidden_J_power"])
                == int(two["best_hidden_J_power"]),
                "one_step_causal_units": one["best_causal_J_power"],
                "next_one_step_causal_units": next_one["best_causal_J_power"],
                "predicted_two_step_causal_units": int(one["best_causal_J_power"])
                + int(next_one["best_causal_J_power"]),
                "observed_two_step_causal_units": two["best_causal_J_power"],
                "causal_additivity_holds": int(one["best_causal_J_power"])
                + int(next_one["best_causal_J_power"])
                == int(two["best_causal_J_power"]),
            }
        )
    return best, additivity


def draw(out_dir: Path, rows: list[dict[str, Any]], best: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for horizon, style in [(1, "-"), (2, "--")]:
        for age in range(2, 7):
            selected = [
                row
                for row in rows
                if row["start_age"] == age and row["F_updates"] == horizon
            ]
            axes[0].plot(
                [row["J_power"] for row in selected],
                [row["relative_rmse"] for row in selected],
                linestyle=style,
                marker="o",
                label=f"H{age}, F^{horizon}",
            )
    axes[0].set_xlabel("number of trained-J age units")
    axes[0].set_ylabel("relative RMSE to same-age aligned interface")
    axes[0].set_title("How many J units undo each F transition?")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    ages = sorted({int(row["start_age"]) for row in best})
    for horizon, offset, label in [(1, -0.18, "one F"), (2, 0.18, "two F")]:
        selected = [row for row in best if row["F_updates"] == horizon]
        axes[1].bar(
            np.arange(len(ages)) + offset,
            [row["best_hidden_J_power"] for row in selected],
            width=0.36,
            label=label,
        )
    axes[1].set_xticks(np.arange(len(ages)), [f"H{age}" for age in ages])
    axes[1].set_ylabel("best integer J power")
    axes[1].set_title("Loop age increment is stage dependent")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "01_matrix_age_units.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, modules, payload = load_task_lora_modules(
        args.artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("J and backbone checkpoint differ")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("expected canonical all-position J")
    operator = modules[args.label]
    if not isinstance(operator, DiagonalIdentityLoRAJ):
        raise TypeError("expected diagonal plus low-rank trained J")
    weight, bias = affine_from_j(operator)
    powers = affine_powers(weight, bias, args.max_j_power)
    H = homogeneous(weight, bias)
    homogeneous_errors: list[dict[str, Any]] = []
    for power, (power_weight, power_bias) in powers.items():
        H_power = torch.linalg.matrix_power(H, power)
        expected = homogeneous(power_weight, power_bias)
        homogeneous_errors.append(
            {
                "J_power": power,
                "relative_matrix_error": float(
                    (H_power - expected).norm() / expected.norm().clamp_min(1e-12)
                ),
            }
        )
    phase = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(x) for x in phase["trajectory_positions_including_initial"]]
    rows = evaluate(
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        powers=powers,
        device=device,
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    best, additivity = best_rows(rows)
    write_csv(args.out_dir / "age_unit_grid.csv", rows)
    write_csv(args.out_dir / "best_age_units.csv", best)
    write_csv(args.out_dir / "age_unit_additivity.csv", additivity)
    write_csv(args.out_dir / "homogeneous_matrix_check.csv", homogeneous_errors)
    draw(args.out_dir, rows, best)
    torch.save(
        {
            "checkpoint": str(args.checkpoint),
            "operator_label": args.label,
            "homogeneous_J": H.cpu(),
            "homogeneous_J_powers": {
                power: torch.linalg.matrix_power(H, power).cpu()
                for power in powers
            },
        },
        args.out_dir / "J_homogeneous_powers.pt",
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": payload.get("backbone_loss_description", "not recorded"),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "J_training": "CE-only long-rollout trained; no hidden-pair affine regression in this experiment",
        "age_definition": "one unit is one application of the trained homogeneous J matrix",
        "examples": args.examples,
        "best_age_units": best,
        "additivity": additivity,
        "homogeneous_matrix_errors": homogeneous_errors,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
