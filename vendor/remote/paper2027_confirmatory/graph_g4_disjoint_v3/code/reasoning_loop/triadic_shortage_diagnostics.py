from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.triadic_shortage import (
    TriadicShortageConfig,
    TriadicShortageModel,
    all_triples,
    hybrid_sum_target,
    make_visibility_mask,
)
from reasoning_loop.triadic_shortage_train import evaluate, pick_device


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def evaluate_reset(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    *,
    condition: str,
    seed: int = 61_001,
) -> dict[str, Any]:
    if operands.shape[0] != labels.shape[0]:
        raise ValueError("operands and labels must contain the same number of examples")
    device = _model_device(model)
    operands = operands.to(device)
    labels = labels.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    visibility = make_visibility_mask(
        condition,
        operands.shape[0],
        model.cfg.loops,
        device,
        generator=generator,
    )
    model.eval()
    baseline_logits, _ = model(operands, visibility)
    reset_before = torch.zeros(model.cfg.loops, dtype=torch.bool, device=device)
    reset_before[: min(3, model.cfg.loops)] = True
    reset_logits, _ = model(
        operands,
        visibility,
        reset_before=reset_before,
    )
    baseline_accuracy = float(
        baseline_logits[:, -1].argmax(dim=-1).eq(labels).float().mean().item()
    )
    reset_accuracy = float(
        reset_logits[:, -1].argmax(dim=-1).eq(labels).float().mean().item()
    )
    return {
        "condition": condition,
        "examples": operands.shape[0],
        "baseline_accuracy": baseline_accuracy,
        "reset_accuracy": reset_accuracy,
        "reset_effect": baseline_accuracy - reset_accuracy,
        "reset_before": reset_before.detach().cpu().tolist(),
    }


@torch.no_grad()
def evaluate_future_corruption(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    visibility: torch.Tensor,
    *,
    operand_index: int,
    delta: int = 1,
) -> dict[str, Any]:
    if not 0 <= operand_index < 3:
        raise ValueError("operand_index must be 0, 1, or 2")
    device = _model_device(model)
    operands = operands.to(device)
    visibility = visibility.to(device)
    corrupt = operands.clone()
    corrupt[:, operand_index] = (
        corrupt[:, operand_index] + delta
    ).remainder(model.cfg.p)
    model.eval()
    clean_logits, clean_states = model(operands, visibility)
    corrupt_logits, corrupt_states = model(corrupt, visibility)
    state_delta = (clean_states - corrupt_states).abs().amax(dim=-1)
    logit_delta = (clean_logits - corrupt_logits).abs().amax(dim=-1)
    reveal_loops: list[int] = []
    pre_state: list[torch.Tensor] = []
    pre_logit: list[torch.Tensor] = []
    post_state: list[torch.Tensor] = []
    post_logit: list[torch.Tensor] = []
    for sample_index in range(operands.shape[0]):
        revealed = visibility[sample_index, :, operand_index].nonzero(as_tuple=False)
        reveal_loop = (
            int(revealed[0, 0].item()) if revealed.numel() else model.cfg.loops
        )
        reveal_loops.append(reveal_loop)
        if reveal_loop > 0:
            pre_state.append(state_delta[sample_index, :reveal_loop])
            pre_logit.append(logit_delta[sample_index, :reveal_loop])
        if reveal_loop < model.cfg.loops:
            post_state.append(state_delta[sample_index, reveal_loop:])
            post_logit.append(logit_delta[sample_index, reveal_loop:])

    def maximum(parts: list[torch.Tensor]) -> float:
        if not parts:
            return 0.0
        return float(torch.cat(parts).max().item())

    return {
        "operand_index": operand_index,
        "delta": delta,
        "reveal_loops": reveal_loops,
        "max_pre_reveal_state_delta": maximum(pre_state),
        "max_pre_reveal_logit_delta": maximum(pre_logit),
        "max_post_reveal_state_delta": maximum(post_state),
        "max_post_reveal_logit_delta": maximum(post_logit),
    }


@torch.no_grad()
def evaluate_hybrid_patch(
    model: TriadicShortageModel,
    *,
    donor_operands: torch.Tensor,
    receiver_operands: torch.Tensor,
    condition: str,
    reveal_counts: tuple[int, ...] = (1, 2),
    seed: int = 71_001,
) -> dict[str, Any]:
    if donor_operands.shape != receiver_operands.shape:
        raise ValueError("donor and receiver operands must have the same shape")
    device = _model_device(model)
    donor_operands = donor_operands.to(device)
    receiver_operands = receiver_operands.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    visibility = make_visibility_mask(
        condition,
        donor_operands.shape[0],
        model.cfg.loops,
        device,
        generator=generator,
    )
    model.eval()
    _, donor_states = model(donor_operands, visibility)
    rows: list[dict[str, Any]] = []
    for reveal_count in reveal_counts:
        if not 1 <= reveal_count < model.cfg.loops:
            raise ValueError("reveal_count must leave at least one continuation loop")
        visited = visibility[:, :reveal_count].any(dim=1)
        target = hybrid_sum_target(
            donor_operands,
            receiver_operands,
            visited,
            p=model.cfg.p,
        )
        workspace = donor_states[:, reveal_count - 1]
        suffix_logits, _ = model.continue_from_workspace(
            receiver_operands,
            visibility,
            workspace=workspace,
            start_loop=reveal_count,
        )
        hybrid_accuracy = float(
            suffix_logits[:, -1]
            .argmax(dim=-1)
            .eq(target)
            .float()
            .mean()
            .item()
        )
        permutation = torch.roll(
            torch.arange(workspace.shape[0], device=device),
            shifts=1,
        )
        random_logits, _ = model.continue_from_workspace(
            receiver_operands,
            visibility,
            workspace=workspace[permutation],
            start_loop=reveal_count,
        )
        random_accuracy = float(
            random_logits[:, -1]
            .argmax(dim=-1)
            .eq(target)
            .float()
            .mean()
            .item()
        )
        rows.append(
            {
                "reveal_count": reveal_count,
                "visited_fraction": float(visited.float().mean().item()),
                "hybrid_accuracy": hybrid_accuracy,
                "random_workspace_accuracy": random_accuracy,
                "target_preview": target[: min(8, target.numel())]
                .detach()
                .cpu()
                .tolist(),
            }
        )
    return {
        "condition": condition,
        "examples": donor_operands.shape[0],
        "rows": rows,
    }


def fourier_bucket_fractions(values: np.ndarray, p: int) -> dict[str, float]:
    if values.ndim != 2 or values.shape[0] != p**3:
        raise ValueError("values must have shape [p**3, features]")
    array = values.reshape(p, p, p, -1)
    spectrum = np.fft.fftn(array, axes=(0, 1, 2))
    energy = (np.abs(spectrum) ** 2).sum(axis=-1)
    total = max(float(energy.sum()), 1e-12)
    bias = float(energy[0, 0, 0])
    single_a = sum(float(energy[k, 0, 0]) for k in range(1, p))
    single_b = sum(float(energy[0, k, 0]) for k in range(1, p))
    single_c = sum(float(energy[0, 0, k]) for k in range(1, p))
    pair_ab = sum(float(energy[k, k, 0]) for k in range(1, p))
    pair_ac = sum(float(energy[k, 0, k]) for k in range(1, p))
    pair_bc = sum(float(energy[0, k, k]) for k in range(1, p))
    final_sum = sum(float(energy[k, k, k]) for k in range(1, p))
    explained = (
        bias
        + single_a
        + single_b
        + single_c
        + pair_ab
        + pair_ac
        + pair_bc
        + final_sum
    )
    return {
        "bias_fraction": bias / total,
        "single_a_fraction": single_a / total,
        "single_b_fraction": single_b / total,
        "single_c_fraction": single_c / total,
        "single_total_fraction": (single_a + single_b + single_c) / total,
        "pair_sum_ab_fraction": pair_ab / total,
        "pair_sum_ac_fraction": pair_ac / total,
        "pair_sum_bc_fraction": pair_bc / total,
        "pair_sum_total_fraction": (pair_ab + pair_ac + pair_bc) / total,
        "final_sum_fraction": final_sum / total,
        "other_fraction": max(0.0, 1.0 - explained / total),
    }


@torch.no_grad()
def collect_exhaustive_states(
    model: TriadicShortageModel,
    *,
    condition: str,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _model_device(model)
    operands, labels = all_triples(model.cfg.p, device=device)
    all_states: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for start in range(0, operands.shape[0], batch_size):
        batch = operands[start : start + batch_size]
        visibility = make_visibility_mask(
            condition,
            batch.shape[0],
            model.cfg.loops,
            device,
            generator=generator,
        )
        logits, states = model(batch, visibility)
        all_logits.append(logits.detach().cpu())
        all_states.append(states.detach().cpu())
    return operands.detach().cpu(), labels.detach().cpu(), torch.cat(all_states, dim=0)


@torch.no_grad()
def evaluate_order_sweep(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if model.cfg.loops < 3:
        raise ValueError("order sweep requires at least three loops")
    device = _model_device(model)
    operands = operands.to(device)
    labels = labels.to(device)
    indices = indices.to(device)
    rows = []
    for order in itertools.permutations(range(3)):
        correct = 0
        total = 0
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start : start + batch_size]
            visibility = torch.zeros(
                (batch_indices.numel(), model.cfg.loops, 3),
                dtype=torch.bool,
                device=device,
            )
            for loop_index, operand_index in enumerate(order):
                visibility[:, loop_index, operand_index] = True
            logits, _ = model(operands[batch_indices], visibility)
            correct += int(
                logits[:, -1]
                .argmax(dim=-1)
                .eq(labels[batch_indices])
                .sum()
                .item()
            )
            total += batch_indices.numel()
        rows.append(
            {
                "order": list(order),
                "accuracy": correct / total,
                "examples": total,
            }
        )
    accuracies = [row["accuracy"] for row in rows]
    return {
        "rows": rows,
        "mean_accuracy": float(np.mean(accuracies)),
        "min_accuracy": min(accuracies),
        "max_accuracy": max(accuracies),
    }


def load_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[TriadicShortageModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = TriadicShortageConfig.from_dict(payload["config"])
    model = TriadicShortageModel(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def analyze_checkpoint(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    sample_size: int = 1024,
    batch_size: int = 1024,
    seed: int = 81_001,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    condition = str(payload["condition"])
    operands, labels = all_triples(model.cfg.p, device=device)
    heldout_idx = payload["split"]["heldout_idx"].to(device)
    subset_idx = heldout_idx[: min(sample_size, heldout_idx.numel())]
    subset_operands = operands[subset_idx]
    subset_labels = labels[subset_idx]
    behavior = evaluate(
        model,
        operands,
        labels,
        heldout_idx,
        condition=condition,
        batch_size=batch_size,
        seed=seed,
    )
    order_sweep = evaluate_order_sweep(
        model,
        operands,
        labels,
        heldout_idx,
        batch_size=batch_size,
    )
    canonical_condition = (
        "sequential" if condition == "sequential_shuffled" else condition
    )
    reset = evaluate_reset(
        model,
        subset_operands,
        subset_labels,
        condition=canonical_condition,
        seed=seed + 1,
    )
    visibility = make_visibility_mask(
        canonical_condition,
        subset_operands.shape[0],
        model.cfg.loops,
        device,
        generator=torch.Generator(device=device).manual_seed(seed + 2),
    )
    future = [
        evaluate_future_corruption(
            model,
            subset_operands,
            visibility,
            operand_index=operand_index,
        )
        for operand_index in range(3)
    ]
    hybrid = evaluate_hybrid_patch(
        model,
        donor_operands=subset_operands,
        receiver_operands=subset_operands.roll(1, dims=0),
        condition=canonical_condition,
        seed=seed + 3,
    )
    _, _, states = collect_exhaustive_states(
        model,
        condition=canonical_condition,
        batch_size=batch_size,
        seed=seed + 4,
    )
    fourier = []
    for loop_index in range(model.cfg.loops):
        row: dict[str, Any] = {"loop": loop_index + 1}
        row.update(
            fourier_bucket_fractions(
                states[:, loop_index].numpy().astype(np.float64),
                model.cfg.p,
            )
        )
        fourier.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "reset.json", reset)
    _write_json(out_dir / "future_corruption.json", future)
    _write_json(out_dir / "hybrid_patch.json", hybrid)
    _write_json(out_dir / "fourier_trajectory.json", fourier)
    _write_json(out_dir / "order_sweep.json", order_sweep)
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "condition": condition,
        "canonical_analysis_condition": canonical_condition,
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "behavior": behavior,
        "reset_effect": reset["reset_effect"],
        "hybrid_accuracy_mean": float(
            np.mean([row["hybrid_accuracy"] for row in hybrid["rows"]])
        ),
        "random_workspace_accuracy_mean": float(
            np.mean([row["random_workspace_accuracy"] for row in hybrid["rows"]])
        ),
        "max_pre_reveal_state_delta": max(
            row["max_pre_reveal_state_delta"] for row in future
        ),
        "max_pre_reveal_logit_delta": max(
            row["max_pre_reveal_logit_delta"] for row in future
        ),
        "order_sweep_mean_accuracy": order_sweep["mean_accuracy"],
        "order_sweep_min_accuracy": order_sweep["min_accuracy"],
        "order_sweep_max_accuracy": order_sweep["max_accuracy"],
        "fourier_trajectory": fourier,
    }
    _write_json(out_dir / "scorecard.json", scorecard)
    return scorecard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze triadic shortage checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=81_001)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scorecard = analyze_checkpoint(
        args.checkpoint,
        args.out_dir,
        device=pick_device(args.device),
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(scorecard, indent=2), flush=True)


if __name__ == "__main__":
    main()
