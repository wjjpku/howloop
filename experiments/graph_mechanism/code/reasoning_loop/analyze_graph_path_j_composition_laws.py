"""Compare exact, diagonal-gated residual, and pure-residual J composition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    ROLLBACK_SOURCE_AGES,
)


MODES = ("sequential", "exact_product", "diag_gated_residual", "pure_residual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=828001)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def load_bank(path: Path, *, dimension: int, device: torch.device) -> tuple[AgeSpecificJBank, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=payload.get("map_architecture", "diagonal_lora"),
    )
    bank.load_state_dict(payload["state_dict"])
    return bank.to(device).frozen(), payload


def state_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction.float().reshape(-1, prediction.shape[-1])
    target = target.float().reshape(-1, target.shape[-1])
    residual = prediction - target
    centered = target - target.mean(0, keepdim=True)
    return {
        "relative_error": float(residual.norm() / target.norm().clamp_min(1e-12)),
        "r2": float(1 - residual.square().sum() / centered.square().sum().clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(prediction, target, dim=-1).mean()),
    }


def exact_product(
    affines: list[tuple[torch.Tensor, torch.Tensor]], *, dimension: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.eye(dimension, device=device)
    bias = torch.zeros(dimension, device=device)
    for next_weight, next_bias in affines:
        bias = bias @ next_weight + next_bias
        weight = weight @ next_weight
    return weight, bias


def pure_residual(
    affines: list[tuple[torch.Tensor, torch.Tensor]], *, dimension: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(dimension, device=device)
    weight = identity + sum((value[0] - identity for value in affines), start=torch.zeros_like(identity))
    bias = sum((value[1] for value in affines), start=torch.zeros(dimension, device=device))
    return weight, bias


def diag_gated_residual(
    *, bank: AgeSpecificJBank, ages: list[int], dimension: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """First-order expansion around diagonal paths; omit all U_i U_j terms."""
    if bank.map_architecture != "shared_diagonal_stage_lora":
        raise TypeError("diagonal-gated residual requires shared_diagonal_stage_lora")
    diagonals = [torch.diag(bank.shared_diagonal_scale.float()) for _ in ages]
    updates = [bank.affine(age)[0] - diagonal for age, diagonal in zip(ages, diagonals, strict=True)]
    identity = torch.eye(dimension, device=device)
    prefixes = [identity]
    for diagonal in diagonals:
        prefixes.append(prefixes[-1] @ diagonal)
    suffixes = [identity for _ in range(len(ages) + 1)]
    for index in range(len(ages) - 1, -1, -1):
        suffixes[index] = diagonals[index] @ suffixes[index + 1]
    weight = prefixes[-1].clone()
    for index, update in enumerate(updates):
        weight = weight + prefixes[index] @ update @ suffixes[index + 1]
    shared_bias = bank.shared_bias.float()
    bias = torch.zeros(dimension, device=device)
    for index in range(len(ages)):
        bias = bias + shared_bias @ suffixes[index + 1]
    return weight, bias


def apply_affine(state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (state.float() @ weight + bias).to(state.dtype)


@torch.no_grad()
def forward_steps(model, state: torch.Tensor, *, start_age: int, steps: int) -> torch.Tensor:
    loop = run_one_loop.__wrapped__
    for offset in range(steps):
        state = loop(model, state, loop_index=start_age + offset).state
    return state


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for mode in MODES[1:]:
        selected = [row for row in rows if row["mode"] == mode]
        by_step = {}
        for row in selected:
            by_step.setdefault(row["rollback_steps"], []).append(row)
        steps = sorted(by_step)
        axes[0].plot(
            steps,
            [np.mean([item["relative_error_to_sequential"] for item in by_step[step]]) for step in steps],
            "o-", label=mode,
        )
        axes[1].plot(
            steps,
            [np.mean([item["current_readout_accuracy"] for item in by_step[step]]) for step in steps],
            "o-", label=mode,
        )
        axes[2].plot(
            steps,
            [np.mean([item["roundtrip_task_accuracy"] for item in by_step[step]]) for step in steps],
            "o-", label=mode,
        )
    axes[0].set_yscale("symlog", linthresh=1e-7)
    axes[0].set_title("fused state vs sequential J")
    axes[0].set_ylabel("relative error")
    axes[1].set_title("readout immediately after rollback")
    axes[1].set_ylabel("accuracy")
    axes[2].set_title("rollback then same number of F steps")
    axes[2].set_ylabel("task accuracy")
    for axis in axes:
        axis.set_xlabel("rollback steps")
        axis.legend(fontsize=8)
        if axis is not axes[0]:
            axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    phase = json.loads(args.phase_summary.read_text())
    phase_positions = [int(value) for value in phase["trajectory_positions_including_initial"]]
    jump = phase_positions[2] - phase_positions[1]
    bank, bank_payload = load_bank(args.bank_artifact, dimension=cfg.d_model, device=device)
    if bank.map_architecture != "shared_diagonal_stage_lora":
        raise TypeError("experiment expects the compact shared/stage bank")
    rows: list[dict[str, Any]] = []
    for source_age in ROLLBACK_SOURCE_AGES:
        max_steps = min(args.max_steps, source_age - 1)
        if max_steps < 1:
            continue
        storage = {
            steps: {mode: [] for mode in MODES} | {"natural": [], "current": [], "successors": []}
            for steps in range(1, max_steps + 1)
        }
        for _ in range(args.examples // args.batch_size):
            _, path_targets, successors, _ = fixed_depth_batch(
                cfg, args.batch_size, device, path_positions=cfg.max_depth
            )
            current = path_targets[:, cfg.max_depth - 1]
            source = _aligned_state_at_age(
                model=model, cfg=cfg, successors=successors, current=current,
                age=source_age, phase_position=phase_positions[source_age],
            )
            sequential = source
            affines: list[tuple[torch.Tensor, torch.Tensor]] = []
            ages: list[int] = []
            for steps in range(1, max_steps + 1):
                rollback_age = source_age - steps + 1
                ages.append(rollback_age)
                affines.append(bank.affine(rollback_age))
                sequential = bank.rollback(
                    sequential, source_age=rollback_age, positions=tuple(range(cfg.seq_len))
                )
                exact_weight, exact_bias = exact_product(
                    affines, dimension=cfg.d_model, device=device
                )
                gated_weight, gated_bias = diag_gated_residual(
                    bank=bank, ages=ages, dimension=cfg.d_model, device=device
                )
                residual_weight, residual_bias = pure_residual(
                    affines, dimension=cfg.d_model, device=device
                )
                outputs = {
                    "sequential": sequential,
                    "exact_product": apply_affine(source, exact_weight, exact_bias),
                    "diag_gated_residual": apply_affine(source, gated_weight, gated_bias),
                    "pure_residual": apply_affine(source, residual_weight, residual_bias),
                }
                natural = _aligned_state_at_age(
                    model=model, cfg=cfg, successors=successors, current=current,
                    age=source_age - steps,
                    phase_position=phase_positions[source_age - steps],
                )
                for mode, output in outputs.items():
                    storage[steps][mode].append(output.cpu())
                storage[steps]["natural"].append(natural.cpu())
                storage[steps]["current"].append(current.cpu())
                storage[steps]["successors"].append(successors.cpu())
        model_cpu = model.cpu()
        for steps, values in storage.items():
            sequential_all = torch.cat(values["sequential"])
            natural_all = torch.cat(values["natural"])
            current_all = torch.cat(values["current"])
            successors_all = torch.cat(values["successors"])
            roundtrip_target = advance_nodes(successors_all, current_all, steps=steps * jump)
            for mode in MODES:
                output = torch.cat(values[mode])
                relative_to_sequential = state_metrics(output, sequential_all)
                relative_to_natural = state_metrics(output, natural_all)
                readout = logits_from_raw_state(model_cpu, output).argmax(-1)
                roundtrip = forward_steps(
                    model_cpu, output, start_age=source_age - steps, steps=steps
                )
                roundtrip_readout = logits_from_raw_state(model_cpu, roundtrip).argmax(-1)
                rows.append(
                    {
                        "source_age": source_age,
                        "target_age": source_age - steps,
                        "rollback_steps": steps,
                        "mode": mode,
                        "relative_error_to_sequential": relative_to_sequential["relative_error"],
                        "r2_to_sequential": relative_to_sequential["r2"],
                        "relative_error_to_natural": relative_to_natural["relative_error"],
                        "r2_to_natural": relative_to_natural["r2"],
                        "current_readout_accuracy": float(readout.eq(current_all).float().mean()),
                        "roundtrip_task_accuracy": float(roundtrip_readout.eq(roundtrip_target).float().mean()),
                        "examples": args.examples,
                    }
                )
        model.to(device)
    write_csv(args.out_dir / "composition_law_metrics.csv", rows)
    plot(rows, args.out_dir / "composition_laws.png")
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_curriculum": bank_payload.get("curriculum"),
        "shared_rank": bank.rank,
        "stage_rank": bank.stage_rank,
        "examples_per_source_age": args.examples,
        "max_rollback_steps": args.max_steps,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
