"""Compare true affine powers of a loop-boundary J under overloop schedules.

The central matched-dose comparison is

    J, F, J, F, ...                 (J1_every1)
    J^2, F, F, J^2, F, F, ...      (J2_every2_front)

where F is one full reuse of all shared transformer blocks.  J^k is composed
algebraically and applied once, so this is not merely calling the module k
times in Python.  Phase-offset and token-position controls separate timing
from total rejuvenation dose.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import explicit_depth_position_groups
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)


@dataclass(frozen=True)
class Condition:
    label: str
    power: int = 0
    period: int = 1
    phase: int = 1
    start_cycle: int = 1
    token_group: str = "all"
    adaptive_target: float | None = None

    def scheduled_power(self, cycle: int) -> int:
        if self.adaptive_target is not None or self.power == 0 or cycle < self.start_cycle:
            return 0
        return self.power if (cycle - self.phase) % self.period == 0 else 0


def conditions() -> tuple[Condition, ...]:
    return (
        Condition("no_J"),
        Condition("J1_every1_all", power=1),
        # Dose-mismatched controls retained to explain the old J_every_2 result.
        Condition("J1_every2_all", power=1, period=2, phase=1),
        Condition("J1_every4_all", power=1, period=4, phase=1),
        # Equal asymptotic dose: one algebraic power-k burst every k F updates.
        Condition("J2_every2_front_all", power=2, period=2, phase=1),
        Condition("J2_every2_phase2_all", power=2, period=2, phase=2),
        Condition("J2_after_two_F_all", power=2, period=2, phase=1, start_cycle=3),
        Condition("J3_every3_front_all", power=3, period=3, phase=1),
        Condition("J3_after_three_F_all", power=3, period=3, phase=1, start_cycle=4),
        Condition("J4_every4_front_all", power=4, period=4, phase=1),
        Condition("J4_after_four_F_all", power=4, period=4, phase=1, start_cycle=5),
        # Spatial interventions: same temporal schedule, different token subsets.
        Condition("J1_every1_query_answer", power=1, token_group="query_answer"),
        Condition("J1_every1_graph", power=1, token_group="graph"),
        Condition("J1_every1_answer", power=1, token_group="answer"),
        # T is measured before F; target roughly seven means F should return it to eight.
        Condition("T_adaptive_target7_all", token_group="all", adaptive_target=7.0),
        Condition("T_adaptive_target6_all", token_group="all", adaptive_target=6.0),
    )


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


def affine_from_j(operator: DiagonalIdentityLoRAJ) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.diag(operator.diagonal_scale.float())
        + operator.A.float() @ operator.B.float(),
        operator.bias.float(),
    )


def affine_powers(
    weight: torch.Tensor, bias: torch.Tensor, maximum_power: int
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Return exact J^k for row maps J(h)=hW+b."""
    identity = torch.eye(weight.shape[0], device=weight.device)
    powers = {0: (identity, torch.zeros_like(bias))}
    current_weight, current_bias = identity, torch.zeros_like(bias)
    for power in range(1, maximum_power + 1):
        current_weight = current_weight @ weight
        current_bias = current_bias @ weight + bias
        powers[power] = (current_weight.clone(), current_bias.clone())
    return powers


def apply_affine_positions(
    state: torch.Tensor,
    positions: tuple[int, ...],
    affine: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    result = state.clone()
    selected = result[:, list(positions), :].float()
    result[:, list(positions), :] = selected @ affine[0] + affine[1]
    return result


def apply_per_example_power(
    state: torch.Tensor,
    positions: tuple[int, ...],
    requested_power: torch.Tensor,
    powers: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    result = state.clone()
    for power in range(1, max(powers) + 1):
        mask = requested_power == power
        if bool(mask.any()):
            selected = result[mask][:, list(positions), :].float()
            updated = selected @ powers[power][0] + powers[power][1]
            batch_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            for position_index, position in enumerate(positions):
                result[batch_indices, position, :] = updated[:, position_index, :]
    return result


def load_T_probe(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    probe = payload["probes"]["all"]
    return probe["weight"].to(device).float(), probe["bias"].to(device).float()


def T_score(
    state: torch.Tensor,
    positions: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return (state[:, list(positions), :].float() @ weight + bias).mean(dim=1)


def _segment(values: list[float], start: int, stop: int) -> float | None:
    selected = values[start:stop]
    return float(np.mean(selected)) if selected else None


def _rolling_last(values: list[float], threshold: float, window: int = 8) -> int:
    if len(values) < window:
        return 0
    rolling = np.convolve(np.asarray(values), np.ones(window) / window, mode="valid")
    valid = np.nonzero(rolling >= threshold)[0]
    return int(valid[-1] + window) if valid.size else 0


def token_groups(cfg: Any) -> dict[str, tuple[int, ...]]:
    base = explicit_depth_position_groups(cfg.node_count)
    return {
        "all": tuple(range(cfg.seq_len)),
        "graph": base["graph"],
        "answer": base["answer"],
        "query_answer": base["query_metadata"] + base["answer"],
    }


@torch.no_grad()
def evaluate(
    *,
    model: torch.nn.Module,
    cfg: Any,
    powers: dict[int, tuple[torch.Tensor, torch.Tensor]],
    T_weight: torch.Tensor,
    T_bias: torch.Tensor,
    phase_positions: list[int],
    device: torch.device,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    specs = conditions()
    groups = token_groups(cfg)
    labels = tuple(spec.label for spec in specs) + ("exact_H7_every_loop",)
    correct = {label: torch.zeros(continuation_loops, dtype=torch.long) for label in labels}
    T_sum = {
        spec.label: torch.zeros(continuation_loops, dtype=torch.float64)
        for spec in specs
    }
    norm_sum = {
        spec.label: torch.zeros(continuation_loops, dtype=torch.float64)
        for spec in specs
    }
    dose_sum = {
        spec.label: torch.zeros(continuation_loops, dtype=torch.float64)
        for spec in specs
    }
    total = 0
    jump = phase_positions[3] - phase_positions[2]
    for _ in range(batches):
        _, path_targets, successors, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        endpoint = path_targets[:, cfg.max_depth - 1]
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        states = {spec.label: initial.clone() for spec in specs}
        for cycle in range(1, continuation_loops + 1):
            loop_index = cfg.max_loops + cycle - 1
            current = advance_nodes(successors, endpoint, steps=jump * (cycle - 1))
            target = advance_nodes(successors, endpoint, steps=jump * cycle)
            for spec in specs:
                state = states[spec.label]
                selected_positions = groups[spec.token_group]
                if spec.adaptive_target is not None:
                    score_before = T_score(state, groups["all"], T_weight, T_bias)
                    requested = torch.round(score_before - spec.adaptive_target).long().clamp(0, max(powers))
                    state = apply_per_example_power(state, selected_positions, requested, powers)
                    dose_sum[spec.label][cycle - 1] += float(requested.sum())
                else:
                    power = spec.scheduled_power(cycle)
                    if power:
                        state = apply_affine_positions(state, selected_positions, powers[power])
                    dose_sum[spec.label][cycle - 1] += power * batch_size
                step = run_one_loop(model, state, loop_index=loop_index)
                correct[spec.label][cycle - 1] += step.logits.argmax(dim=-1).eq(target).sum().cpu()
                states[spec.label] = step.state
                T_sum[spec.label][cycle - 1] += float(
                    T_score(step.state, groups["all"], T_weight, T_bias).sum()
                )
                norm_sum[spec.label][cycle - 1] += float(
                    step.state.float().norm(dim=-1).mean(dim=-1).sum()
                )
            exact = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=current,
                age=7,
                phase_position=phase_positions[7],
            )
            exact_step = run_one_loop(model, exact, loop_index=loop_index)
            correct["exact_H7_every_loop"][cycle - 1] += exact_step.logits.argmax(dim=-1).eq(target).sum().cpu()
        total += batch_size
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        accuracy = (correct[label].float() / total).tolist()
        for cycle, value in enumerate(accuracy, start=1):
            row: dict[str, Any] = {
                "condition": label,
                "cycle": cycle,
                "accuracy": value,
                "examples": total,
            }
            if label != "exact_H7_every_loop":
                row.update(
                    {
                        "mean_T_age": float(T_sum[label][cycle - 1] / total),
                        "mean_token_norm": float(norm_sum[label][cycle - 1] / total),
                        "mean_applied_J_power": float(dose_sum[label][cycle - 1] / total),
                    }
                )
            curve_rows.append(row)
        summary_rows.append(
            {
                "condition": label,
                "auc_1_24": _segment(accuracy, 0, 24),
                "auc_25_48": _segment(accuracy, 24, 48),
                "auc_49_64": _segment(accuracy, 48, 64),
                "auc_65_96": _segment(accuracy, 64, 96),
                "auc_97_128": _segment(accuracy, 96, 128),
                "final_accuracy": accuracy[-1],
                "last_rolling8_at_least_90": _rolling_last(accuracy, 0.9),
                "last_rolling8_at_least_50": _rolling_last(accuracy, 0.5),
                "mean_J_power_per_cycle": None
                if label == "exact_H7_every_loop"
                else float(dose_sum[label].sum() / (total * continuation_loops)),
            }
        )
    return curve_rows, summary_rows, {
        "examples": total,
        "continuation_loops": continuation_loops,
        "jump_per_loop": jump,
        "seed": seed,
    }


def draw_figures(out_dir: Path, curve_rows: list[dict[str, Any]]) -> None:
    labels = list(dict.fromkeys(row["condition"] for row in curve_rows))
    matrix = np.asarray(
        [
            [row["accuracy"] for row in curve_rows if row["condition"] == label]
            for label in labels
        ]
    )
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("continuation cycle (zero-indexed in image)")
    axis.set_title("Overloop accuracy: affine-power schedule x cycle")
    fig.colorbar(image, ax=axis, label="accuracy")
    fig.savefig(out_dir / "01_accuracy_heatmap.png", dpi=220)
    plt.close(fig)


def power_geometry_rows(
    powers: dict[int, tuple[torch.Tensor, torch.Tensor]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimension = powers[0][0].shape[0]
    identity = torch.eye(dimension, device=powers[0][0].device)
    for power in sorted(powers):
        if power == 0:
            continue
        weight, bias = powers[power]
        singular = torch.linalg.svdvals(weight)
        rows.append(
            {
                "power": power,
                "spectral_radius": float(torch.linalg.eigvals(weight).abs().max()),
                "largest_singular_value": float(singular.max()),
                "median_singular_value": float(singular.median()),
                "smallest_singular_value": float(singular.min()),
                "condition_number": float(torch.linalg.cond(weight)),
                "bias_norm": float(bias.norm()),
                "weight_minus_identity_frobenius": float((weight - identity).norm()),
            }
        )
    return rows
    selected = [
        "no_J",
        "J1_every1_all",
        "J1_every2_all",
        "J2_every2_front_all",
        "J2_every2_phase2_all",
        "J2_after_two_F_all",
        "J3_every3_front_all",
        "J4_every4_front_all",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), constrained_layout=True)
    for label in selected:
        rows = [row for row in curve_rows if row["condition"] == label]
        axes[0].plot([row["cycle"] for row in rows], [row["accuracy"] for row in rows], label=label)
        if label != "no_J":
            axes[1].plot([row["cycle"] for row in rows], [row.get("mean_T_age", np.nan) for row in rows], label=label)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("continuation cycle")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Matched and mismatched J dose schedules")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("continuation cycle")
    axes[1].set_ylabel("shared T age score after F")
    axes[1].set_title("What the shared age readout sees")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.savefig(out_dir / "02_schedule_curves.png", dpi=220)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit algebraic J-power overloop schedules.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--T-probe-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--continuation-loops", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    return parser.parse_args(argv)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    artifact_checkpoint, positions, operators, artifact_payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if artifact_checkpoint != str(args.checkpoint):
        raise ValueError("operator and backbone checkpoints differ")
    if artifact_payload.get("placement") != "loop_boundary":
        raise ValueError("power schedules require a loop-boundary J")
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("expected the canonical all-position J")
    operator = operators[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ):
        raise TypeError("expected diagonal plus low-rank J")
    powers = affine_powers(*affine_from_j(operator), maximum_power=4)
    geometry_rows = power_geometry_rows(powers)
    T_weight, T_bias = load_T_probe(args.T_probe_artifact, device)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value) for value in phase_payload["trajectory_positions_including_initial"]
    ]
    curve_rows, summary_rows, evaluation = evaluate(
        model=model,
        cfg=cfg,
        powers=powers,
        T_weight=T_weight,
        T_bias=T_bias,
        phase_positions=phase_positions,
        device=device,
        batch_size=args.batch_size,
        batches=args.batches,
        continuation_loops=args.continuation_loops,
        seed=args.seed,
    )
    write_csv(args.out_dir / "power_schedule_curves.csv", curve_rows)
    write_csv(args.out_dir / "power_schedule_summary.csv", summary_rows)
    write_csv(args.out_dir / "J_power_geometry.csv", geometry_rows)
    draw_figures(args.out_dir, curve_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": artifact_payload.get("backbone_loss_description", "not recorded"),
        "trained_loop_count": cfg.max_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "operator_placement": "loop_boundary before one full F update",
        "algebraic_power_definition": "J^k(h)=h W^k + b(I+W+...+W^(k-1))",
        "evaluation": evaluation,
        "conditions": [spec.__dict__ for spec in conditions()],
        "J_power_geometry": geometry_rows,
        "metrics": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
