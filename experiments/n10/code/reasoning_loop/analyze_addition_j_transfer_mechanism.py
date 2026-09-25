"""Test which Addition J singular modes transfer beyond its training lengths.

This is a frozen-backbone, answer-only diagnostic.  It evaluates exact SVD
keep/delete and norm-matched random-write variants of Delta W = W - I, then
measures how the full J acts on the real pre-final-J hidden states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    PaperBatch,
    answer_cross_entropy,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
    set_seed,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lengths", type=int, nargs="+", default=(30, 40, 50, 60))
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=888001)
    parser.add_argument("--random-write-draws", type=int, default=3)
    return parser.parse_args(argv)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FrozenAffine(torch.nn.Module):
    def __init__(self, weight: np.ndarray, bias: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.weight + self.bias


def random_orthonormal(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    return basis[:, :rank]


def build_svd_variants(
    delta: np.ndarray,
    bias: np.ndarray,
    *,
    random_write_draws: int,
    seed: int,
) -> tuple[dict[str, FrozenAffine | None], dict[str, Any]]:
    dimension = delta.shape[0]
    identity = np.eye(dimension)
    left, singular, right_t = np.linalg.svd(delta)
    variants: dict[str, FrozenAffine | None] = {"raw": None}
    for rank in (1, 2, 4, 8, 16, 32, 48):
        kept = left[:, :rank] @ np.diag(singular[:rank]) @ right_t[:rank]
        variants[f"top{rank}"] = FrozenAffine(identity + kept, bias)
    for rank in (4, 8):
        kept = left[:, :rank] @ np.diag(singular[:rank]) @ right_t[:rank]
        variants[f"delete_top{rank}"] = FrozenAffine(identity + delta - kept, bias)
        for draw in range(random_write_draws):
            rng = np.random.default_rng(seed + 1000 * rank + draw)
            random_output = random_orthonormal(dimension, rank, rng)
            rotated = left[:, :rank] @ np.diag(singular[:rank]) @ random_output.T
            variants[f"random_write_top{rank}_d{draw}"] = FrozenAffine(
                identity + rotated, bias
            )
    variants["bias_only"] = FrozenAffine(identity, bias)
    variants["top4_no_bias"] = FrozenAffine(
        identity + left[:, :4] @ np.diag(singular[:4]) @ right_t[:4],
        np.zeros_like(bias),
    )
    energy = np.square(singular)
    metadata = {
        "singular_values": singular.tolist(),
        "ambient_energy_fraction": {
            str(rank): float(energy[:rank].sum() / energy.sum())
            for rank in (1, 2, 4, 8, 16, 32, 48)
        },
    }
    return variants, metadata


def arithmetic_mask(batch: PaperBatch) -> torch.Tensor:
    return batch.answer_mask & (batch.targets == 0) | batch.answer_mask & (batch.targets == 1)


def prediction_metrics(logits: torch.Tensor, batch: PaperBatch) -> dict[str, float]:
    predictions = logits.argmax(-1)
    mask = arithmetic_mask(batch)
    correct = predictions.eq(batch.targets)
    target_logits = logits.gather(-1, batch.targets[..., None]).squeeze(-1)
    wrong = logits.clone()
    wrong.scatter_(-1, batch.targets[..., None], float("-inf"))
    margin = target_logits - wrong.max(-1).values
    masked_correct = correct | ~mask
    sequence_margin = margin.masked_fill(~mask, float("inf")).min(-1).values
    first_index = mask.float().argmax(-1)
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    return {
        "exact_match": exact_match(logits, batch),
        "answer_cross_entropy": answer_cross_entropy(logits, batch),
        "arithmetic_exact_match": float(masked_correct.all(-1).float().mean()),
        "arithmetic_token_accuracy": float(correct[mask].float().mean()),
        "final_carry_accuracy": float(
            correct[batch_index, first_index].float().mean()
        ),
        "mean_arithmetic_margin": float(margin[mask].mean()),
        "positive_sequence_margin_fraction": float((sequence_margin > 0).float().mean()),
    }


@torch.no_grad()
def final_state(
    model,
    batch: PaperBatch,
    *,
    steps: int,
    controller: torch.nn.Module | None,
) -> torch.Tensor:
    state = None
    for state in model.iter_states(
        batch.inputs,
        steps=steps,
        controller=controller,
        controller_start_step=1 if controller is not None else None,
    ):
        pass
    if state is None:
        raise RuntimeError("empty recurrence")
    return state


def vector_rms(values: torch.Tensor) -> float:
    return float(values.float().square().sum(-1).mean().sqrt())


@torch.no_grad()
def full_trajectory_hidden_metrics(
    *,
    model,
    controller,
    batch: PaperBatch,
    delta: torch.Tensor,
    left: torch.Tensor,
    singular: torch.Tensor,
) -> dict[str, float]:
    embedded = model.input_embeddings(batch.inputs)
    state = torch.zeros_like(embedded)
    target_steps = int(batch.target_steps[0])
    if not bool(batch.target_steps.eq(target_steps).all()):
        raise ValueError("hidden diagnostic requires one fixed length")
    pre_final_j = None
    skip_final_next = None
    full_final_next = None
    for step in range(1, target_steps + 1):
        if step > 1:
            if step == target_steps:
                pre_final_j = state
                controlled = controller(state)
                skip_final_next = model.recurrent_step(state, embedded)
                full_final_next = model.recurrent_step(controlled, embedded)
                state = full_final_next
                continue
            state = controller(state)
        state = model.recurrent_step(state, embedded)
    if pre_final_j is None or skip_final_next is None or full_final_next is None:
        raise RuntimeError("failed to capture final J application")
    mask = arithmetic_mask(batch)
    h = pre_final_j[mask].float()
    controlled_h = controller(pre_final_j)[mask].float()
    correction = controlled_h - h
    linear = h @ delta
    result: dict[str, float] = {
        "pre_state_vector_rms": vector_rms(h),
        "j_correction_vector_rms": vector_rms(correction),
        "j_correction_relative_state": vector_rms(correction) / max(vector_rms(h), 1e-30),
        "linear_correction_vector_rms": vector_rms(linear),
        "bias_vector_rms": vector_rms(controller.bias[None, :].expand_as(h)),
        "post_executor_difference_vector_rms": vector_rms(
            (full_final_next - skip_final_next)[mask]
        ),
        "post_executor_difference_relative_skip_state": vector_rms(
            (full_final_next - skip_final_next)[mask]
        )
        / max(vector_rms(skip_final_next[mask]), 1e-30),
    }
    total_energy = float(linear.square().sum())
    for rank in (1, 2, 4, 8, 16, 32, 48):
        coefficients = (h @ left[:, :rank]) * singular[:rank]
        result[f"real_top{rank}_linear_energy_fraction"] = float(
            coefficients.square().sum() / max(total_energy, 1e-30)
        )
        result[f"real_top{rank}_input_coefficient_vector_rms"] = vector_rms(
            h @ left[:, :rank]
        )
    full_logits = model.decode(full_final_next).float()
    skip_logits = model.decode(skip_final_next).float()
    for prefix, logits in (("with_final_J", full_logits), ("skip_final_J", skip_logits)):
        for key, value in prediction_metrics(logits, batch).items():
            result[f"{prefix}_{key}"] = value
    return result


@torch.no_grad()
def evaluate(
    *,
    model,
    spec,
    controller,
    variants: dict[str, FrozenAffine | None],
    lengths: Sequence[int],
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch-size")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = controller.matrix().detach().float()
    identity = torch.eye(weight.shape[0], device=device)
    delta = weight - identity
    left, singular, _ = torch.linalg.svd(delta)
    task_totals: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    hidden_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for length in lengths:
        for batch_index in range(examples // batch_size):
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            for name, variant in variants.items():
                if variant is not None:
                    variant = variant.to(device)
                state = final_state(
                    model,
                    batch,
                    steps=length + spec.step_offset,
                    controller=variant,
                )
                metrics = prediction_metrics(model.decode(state).float(), batch)
                for key, value in metrics.items():
                    task_totals[(length, name)][key] += value
                task_totals[(length, name)]["batches"] += 1
            hidden = full_trajectory_hidden_metrics(
                model=model,
                controller=controller,
                batch=batch,
                delta=delta,
                left=left,
                singular=singular,
            )
            for key, value in hidden.items():
                hidden_totals[length][key] += value
            hidden_totals[length]["batches"] += 1
            print(
                json.dumps(
                    {
                        "event": "batch_complete",
                        "length": length,
                        "batch": batch_index + 1,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    task_rows = []
    for (length, name), values in sorted(task_totals.items()):
        batches = values.pop("batches")
        task_rows.append(
            {
                "length": length,
                "target_steps": length + spec.step_offset,
                "variant": name,
                "examples": examples,
                **{key: value / batches for key, value in values.items()},
            }
        )
    hidden_rows = []
    for length, values in sorted(hidden_totals.items()):
        batches = values.pop("batches")
        hidden_rows.append(
            {
                "length": length,
                "target_steps": length + spec.step_offset,
                "examples": examples,
                **{key: value / batches for key, value in values.items()},
            }
        )
    return task_rows, hidden_rows


def plot_results(task_rows: Sequence[dict[str, Any]], hidden_rows: Sequence[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.7), dpi=180)
    selected = ("raw", "full", "top1", "top4", "top8", "top16", "delete_top4")
    for name in selected:
        rows = sorted((row for row in task_rows if row["variant"] == name), key=lambda row: row["length"])
        axes[0].plot(
            [row["length"] for row in rows],
            [row["exact_match"] for row in rows],
            marker="o",
            label=name,
        )
    axes[0].set(title="Strict whole-answer EM", xlabel="logical length", ylabel="accuracy", ylim=(-0.03, 1.03))
    for name in ("raw", "full", "top4", "top8", "top16"):
        rows = sorted((row for row in task_rows if row["variant"] == name), key=lambda row: row["length"])
        axes[1].plot(
            [row["length"] for row in rows],
            [row["arithmetic_token_accuracy"] for row in rows],
            marker="o",
            label=name,
        )
    axes[1].set(title="Arithmetic-token accuracy", xlabel="logical length", ylabel="accuracy", ylim=(-0.03, 1.03))
    for rank in (1, 2, 4, 8, 16, 32, 48):
        axes[2].plot(
            [row["length"] for row in hidden_rows],
            [row[f"real_top{rank}_linear_energy_fraction"] for row in hidden_rows],
            marker="o",
            label=f"top-{rank}",
        )
    axes[2].set(title="Real-state linear-correction energy", xlabel="logical length", ylabel="fraction", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    atomic_json(
        manifest_path,
        {
            "status": "running",
            "pid": os.getpid(),
            "checkpoint": str(args.checkpoint),
            "controller": str(args.controller),
            "started_unix_time": time.time(),
        },
    )
    device = pick_device(args.device)
    set_seed(args.seed)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if spec.name != "addition":
        raise ValueError("this diagnostic is fixed to Addition")
    if int(controller_payload["anchor_step"]) != 1:
        raise ValueError("this diagnostic expects anchor_step=1")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight = controller.matrix().detach().float().cpu().double().numpy()
    delta = weight - np.eye(weight.shape[0])
    bias = controller.bias.detach().float().cpu().double().numpy()
    variants, svd_metadata = build_svd_variants(
        delta,
        bias,
        random_write_draws=args.random_write_draws,
        seed=args.seed,
    )
    variants["full"] = controller
    variants["no_AB"] = ControllerView(controller, mode="no_AB")
    variants["identity_D"] = ControllerView(controller, mode="identity_D")
    variants["no_bias"] = ControllerView(controller, mode="no_bias")
    task_rows, hidden_rows = evaluate(
        model=model,
        spec=spec,
        controller=controller,
        variants=variants,
        lengths=tuple(args.lengths),
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed + 100,
        device=device,
    )
    write_csv(args.out_dir / "variant_metrics.csv", task_rows)
    write_csv(args.out_dir / "real_hidden_metrics.csv", hidden_rows)
    plot_results(task_rows, hidden_rows, args.out_dir / "addition_j_transfer_mechanism.png")
    result = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": backbone_payload.get("step"),
        "controller": str(args.controller),
        "controller_sha256": hashlib.sha256(args.controller.read_bytes()).hexdigest(),
        "controller_seed": controller_payload.get("seed"),
        "controller_training_lengths": controller_payload.get("controller_sampled_logical_lengths"),
        "loss_placement": controller_payload.get("loss"),
        "lengths": list(args.lengths),
        "examples_per_length": args.examples,
        "svd": svd_metadata,
        "claim_boundary": (
            "SVD keep/delete and norm-matched random-write variants test a causal role for Addition J modes. "
            "This one-backbone local-MPS diagnostic does not establish a complete Addition circuit or a cross-seed law."
        ),
    }
    atomic_json(args.out_dir / "summary.json", result)
    atomic_json(
        manifest_path,
        {**result, "completed_unix_time": time.time(), "out_dir": str(args.out_dir)},
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
