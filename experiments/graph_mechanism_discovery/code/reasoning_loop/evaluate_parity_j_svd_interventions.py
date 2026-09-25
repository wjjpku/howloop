"""Causal trajectory tests for the parity controller's SVD components."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.paper_length_telomere import (
    answer_cross_entropy,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lengths", nargs="+", type=int, default=(20, 40, 50, 75, 100))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--post-target-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=430001)
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="reject checkpoints outside the corrected input-once Parity protocol",
    )
    return parser.parse_args()


class AffineMap(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() @ self.weight + self.bias


def truncated_delta(
    u: torch.Tensor,
    singular: torch.Tensor,
    vh: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank == 0:
        return torch.zeros(
            (u.shape[0], vh.shape[1]), device=u.device, dtype=u.dtype
        )
    return (u[:, :rank] * singular[:rank]) @ vh[:rank]


def random_rank_delta(
    *,
    singular: torch.Tensor,
    dimension: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    left = torch.linalg.qr(
        torch.randn(dimension, rank, generator=generator), mode="reduced"
    ).Q.to(device)
    right = torch.linalg.qr(
        torch.randn(dimension, rank, generator=generator), mode="reduced"
    ).Q.to(device)
    return (left * singular[:rank]) @ right.T


def random_basis(
    *, dimension: int, rank: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.linalg.qr(
        torch.randn(dimension, rank, generator=generator), mode="reduced"
    ).Q.to(device)


def build_variants(controller, device: torch.device) -> tuple[dict[str, torch.nn.Module | None], dict[str, Any]]:
    dimension = controller.dimension
    identity = torch.eye(dimension, device=device)
    diagonal = controller.diagonal.float()
    a = controller.A.float()
    b_factor = controller.B.float()
    bias = controller.bias.float()
    ab = a @ b_factor
    weight = torch.diag(diagonal) + ab
    delta = weight - identity
    u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    zero_bias = torch.zeros_like(bias)
    variants: dict[str, torch.nn.Module | None] = {
        "raw_no_J": None,
        "full_J": AffineMap(weight, bias),
        "identity_D": AffineMap(identity + ab, bias),
        "no_AB": AffineMap(torch.diag(diagonal), bias),
        "no_bias": AffineMap(weight, zero_bias),
        "AB_only": AffineMap(identity + ab, zero_bias),
        "bias_only": AffineMap(identity, bias),
        "D_only": AffineMap(torch.diag(diagonal), zero_bias),
    }
    energy = singular.square()
    total_energy = energy.sum().clamp_min(1e-20)
    rank_metadata = {}
    for rank in (1, 2, 4, 8, 16, 32, 48):
        top = truncated_delta(u, singular, vh, rank)
        variants[f"top{rank}"] = AffineMap(identity + top, bias)
        rank_metadata[str(rank)] = {
            "ambient_delta_energy": float((energy[:rank].sum() / total_energy).item()),
            "operator_norm": float(singular[0].item()),
        }
    top4 = truncated_delta(u, singular, vh, 4)
    variants["delete_top4"] = AffineMap(identity + delta - top4, bias)
    variants["top4_no_bias"] = AffineMap(identity + top4, zero_bias)
    for mode in range(4):
        single = (
            u[:, mode : mode + 1] * singular[mode : mode + 1]
        ) @ vh[mode : mode + 1]
        variants[f"single_mode{mode + 1}"] = AffineMap(identity + single, bias)
        variants[f"top4_without_mode{mode + 1}"] = AffineMap(
            identity + top4 - single, bias
        )
        variants[f"full_without_mode{mode + 1}"] = AffineMap(
            identity + delta - single, bias
        )
    for seed in (314159, 271828, 161803):
        random_delta = random_rank_delta(
            singular=singular,
            dimension=dimension,
            rank=4,
            seed=seed,
            device=device,
        )
        variants[f"random_top4_s{seed}"] = AffineMap(identity + random_delta, bias)
        random_input = random_basis(
            dimension=dimension,
            rank=4,
            seed=seed + 1000,
            device=device,
        )
        random_output = random_basis(
            dimension=dimension,
            rank=4,
            seed=seed + 2000,
            device=device,
        )
        variants[f"random_input_top4_s{seed}"] = AffineMap(
            identity + (random_input * singular[:4]) @ vh[:4],
            bias,
        )
        variants[f"random_output_top4_s{seed}"] = AffineMap(
            identity + (u[:, :4] * singular[:4]) @ random_output.T,
            bias,
        )
    return variants, {
        "delta_singular_values": singular.detach().cpu().tolist(),
        "ranks": rank_metadata,
        "random_control": (
            "rank 4 and the learned top-4 singular values are preserved; "
            "both-random controls do not match real hidden-state effect. "
            "Random-output controls keep learned U and singular values, so "
            "the weight-correction norm is exactly matched on every h."
        ),
    }


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


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(
        args.checkpoint, device=device, paper_mode=args.paper_mode
    )
    controller, controller_payload = load_controller(args.controller, device=device)
    if spec.name != "parity" or controller_payload["anchor_step"] != 1:
        raise ValueError("diagnostic expects the anchor-1 parity controller")
    variants, matrix_metadata = build_variants(controller, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    totals: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    counts: dict[tuple[str, int, int], int] = defaultdict(int)

    for length in args.lengths:
        maximum_step = length + args.post_target_steps
        for _ in range(args.batches):
            batch = generate_paper_batch(
                spec,
                batch_size=args.batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            token_embeddings = model.read_in(batch.inputs)
            for label, affine in variants.items():
                state = torch.zeros_like(token_embeddings)
                for step in range(1, maximum_step + 1):
                    embedded = model.input_embeddings(
                        batch.inputs, step_index=step
                    )
                    if affine is not None and step > 1:
                        state = affine(state)
                    state = model.recurrent_step(state, embedded)
                    if step < max(1, length - 8):
                        continue
                    logits = model.decode(state).float()
                    key = (label, length, step)
                    totals[key]["correct"] += exact_match(logits, batch) * args.batch_size
                    totals[key]["answer_nll"] += answer_cross_entropy(logits, batch) * args.batch_size
                    counts[key] += args.batch_size

    rows: list[dict[str, Any]] = []
    for (label, length, step), values in sorted(totals.items()):
        count = counts[(label, length, step)]
        rows.append(
            {
                "variant": label,
                "length": length,
                "step": step,
                "relative_to_target": step - length,
                "exact_match": values["correct"] / count,
                "answer_nll": values["answer_nll"] / count,
                "examples": count,
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "trajectory_metrics.csv", rows)
    summaries: list[dict[str, Any]] = []
    for label in variants:
        for length in args.lengths:
            selected = [
                row for row in rows if row["variant"] == label and row["length"] == length
            ]
            target = next(row for row in selected if row["relative_to_target"] == 0)
            plus_one = next(row for row in selected if row["relative_to_target"] == 1)
            post = [row for row in selected if 0 <= row["relative_to_target"] <= args.post_target_steps]
            summaries.append(
                {
                    "variant": label,
                    "length": length,
                    "target_exact_match": target["exact_match"],
                    "target_answer_nll": target["answer_nll"],
                    "target_plus_one_exact_match": plus_one["exact_match"],
                    "post_target_accuracy_auc_0_to_8": float(
                        np.mean([row["exact_match"] for row in post])
                    ),
                    "examples": target["examples"],
                }
            )
    write_csv(args.out_dir / "variant_summary.csv", summaries)
    payload = {
        "status": "complete",
        "execution": {
            "device": str(device),
            "kind": "local frozen-model trajectory intervention",
            "seed": args.seed,
            "examples_per_length": args.batch_size * args.batches,
            "paper_mode": bool(args.paper_mode),
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": backbone_payload["step"],
            "backbone_seed": backbone_payload["seed"],
            "shared_physical_layers": spec.block_layers,
            "trained_logical_lengths": [1, spec.train_max_length],
        },
        "controller": {
            "artifact": str(args.controller),
            "seed": controller_payload["seed"],
            "anchor_step": controller_payload["anchor_step"],
            "trained_logical_lengths": controller_payload[
                "controller_sampled_logical_lengths"
            ],
            "loss": controller_payload["loss"],
            "state_loss_weight": controller_payload["state_loss_weight"],
        },
        "pre_registered_questions": {
            "top4_causal_core": (
                "supported only if top4 retains most full-J target performance "
                "and delete_top4 loses it"
            ),
            "bias_role": "compare full_J/no_bias and top4/top4_no_bias",
            "directional_alignment": (
                "spectrum-matched random top4 is diagnostic only because actual "
                "hidden-effect magnitude is not matched"
            ),
        },
        "matrix": matrix_metadata,
        "summaries": summaries,
        "evidence_boundary": (
            "Full-trajectory causal component tests on one backbone seed and "
            "one controller seed. Random controls preserve spectrum but not "
            "actual hidden-state effect magnitude."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(rows, summaries, args.out_dir)
    return payload


def plot_results(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    variants = sorted({str(row["variant"]) for row in summaries})
    lengths = sorted({int(row["length"]) for row in summaries})
    target = np.zeros((len(variants), len(lengths)))
    plus_one = np.zeros_like(target)
    for variant_index, variant in enumerate(variants):
        for length_index, length in enumerate(lengths):
            row = next(
                value
                for value in summaries
                if value["variant"] == variant and value["length"] == length
            )
            target[variant_index, length_index] = row["target_exact_match"]
            plus_one[variant_index, length_index] = row["target_plus_one_exact_match"]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 8.2))
    image = axes[0].imshow(target, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("Exact match at registered target T")
    axes[0].set_xticks(np.arange(len(lengths)), labels=[f"L{x}" for x in lengths])
    axes[0].set_yticks(np.arange(len(variants)), labels=variants, fontsize=8)
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    image = axes[1].imshow(plus_one, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("Exact match at T+1")
    axes[1].set_xticks(np.arange(len(lengths)), labels=[f"L{x}" for x in lengths])
    axes[1].set_yticks(np.arange(len(variants)), labels=variants, fontsize=8)
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.suptitle("Parity J SVD and component interventions")
    figure.tight_layout()
    figure.savefig(out_dir / "variant_accuracy_heatmaps.png", dpi=190)
    plt.close(figure)

    focus = (
        "raw_no_J",
        "full_J",
        "top1",
        "top2",
        "top4",
        "top8",
        "delete_top4",
        "no_AB",
        "no_bias",
        "top4_no_bias",
        "random_top4_s314159",
    )
    length = max(lengths)
    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    for variant in focus:
        selected = [
            row
            for row in rows
            if row["variant"] == variant and row["length"] == length
        ]
        axis.plot(
            [row["relative_to_target"] for row in selected],
            [row["exact_match"] for row in selected],
            marker="o",
            markersize=2.8,
            label=variant,
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
    axis.set_title(f"Parity L{length}: phase curve around target")
    axis.set_xlabel("step minus registered target")
    axis.set_ylabel("exact match")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(out_dir / "l100_phase_curve.png", dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    payload = evaluate(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "device": payload["execution"]["device"],
                "examples_per_length": payload["execution"]["examples_per_length"],
                "variants": len({row["variant"] for row in payload["summaries"]}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
