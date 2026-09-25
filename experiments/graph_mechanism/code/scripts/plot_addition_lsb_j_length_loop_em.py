#!/usr/bin/env python3
"""Evaluate and plot Addition EM over the full (input length, loop count) grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scripts.compare_addition_readout_order_attention import (
        answer_positions_lsb_to_carry,
    )
except ModuleNotFoundError:
    from compare_addition_readout_order_attention import (
        answer_positions_lsb_to_carry,
    )

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)


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


@torch.inference_mode()
def evaluate_grid(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module,
    controller_start_step: int,
    maximum_length: int,
    maximum_step: int,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logical_length in range(1, maximum_length + 1):
        ordinary_positions = answer_positions_lsb_to_carry(
            spec, logical_length
        )[:logical_length]
        position_index = torch.tensor(ordinary_positions, device=device)
        exact_correct = {
            "raw": torch.zeros(maximum_step, dtype=torch.long),
            "full": torch.zeros(maximum_step, dtype=torch.long),
        }
        bit_correct = {
            "raw": torch.zeros(maximum_step, dtype=torch.long),
            "full": torch.zeros(maximum_step, dtype=torch.long),
        }
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + logical_length)
        remaining = examples
        while remaining:
            batch = generate_paper_batch(
                spec,
                batch_size=min(batch_size, remaining),
                min_length=logical_length,
                max_length=logical_length,
                fixed_length=logical_length,
                generator=generator,
            ).to(device)
            targets = batch.targets.index_select(1, position_index)
            variants = (
                ("raw", None, None),
                ("full", controller, controller_start_step),
            )
            for variant, active_controller, start_step in variants:
                for step_index, state in enumerate(
                    model.iter_states(
                        batch.inputs,
                        steps=maximum_step,
                        controller=active_controller,
                        controller_start_step=start_step,
                    )
                ):
                    predictions = model.decode(state).argmax(dim=-1).index_select(
                        1, position_index
                    )
                    correct = predictions.eq(targets)
                    exact_correct[variant][step_index] += (
                        correct.all(dim=1).sum().cpu()
                    )
                    bit_correct[variant][step_index] += correct.sum().cpu()
            remaining -= batch.inputs.shape[0]

        for variant in ("raw", "full"):
            for step in range(1, maximum_step + 1):
                rows.append(
                    {
                        "variant": variant,
                        "input_length": logical_length,
                        "loop_count": step,
                        "ordinary_digit_exact_match": (
                            float(exact_correct[variant][step - 1]) / examples
                        ),
                        "ordinary_digit_bit_accuracy": (
                            float(bit_correct[variant][step - 1])
                            / (examples * logical_length)
                        ),
                        "examples": examples,
                        "carry_in_metric": False,
                    }
                )
    return rows


def plot_grid(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    metric_label: str,
    metric_title: str,
    maximum_length: int,
    maximum_step: int,
    train_maximum_length: int,
    controller_updates: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), sharey=True)
    labels = {"raw": "raw backbone", "full": f"J ({controller_updates:,} updates)"}
    image = None
    for axis, variant in zip(axes, ("raw", "full")):
        matrix = np.zeros((maximum_step, maximum_length), dtype=float)
        for row in rows:
            if row["variant"] != variant:
                continue
            matrix[int(row["loop_count"]) - 1, int(row["input_length"]) - 1] = float(
                row[metric]
            )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            extent=(0.5, maximum_length + 0.5, 0.5, maximum_step + 0.5),
        )
        diagonal_maximum = min(maximum_length, maximum_step)
        axis.plot(
            [1, diagonal_maximum],
            [1, diagonal_maximum],
            color="white",
            linestyle="--",
            linewidth=1.7,
            label="registered T(n)=n",
        )
        axis.axvline(
            train_maximum_length + 0.5,
            color="#ff9f1c",
            linestyle="--",
            linewidth=1.7,
            label=f"train max n={train_maximum_length}",
        )
        axis.set(
            title=labels[variant],
            xlabel="input length n",
            xlim=(0.5, maximum_length + 0.5),
            ylim=(0.5, maximum_step + 0.5),
        )
        axis.set_xticks(range(1, maximum_length + 1))
        axis.set_yticks(range(1, maximum_step + 1, 2))
        axis.legend(loc="upper left")
    axes[0].set_ylabel("executed loop count t")
    if image is not None:
        colorbar_axis = figure.add_axes((0.925, 0.11, 0.015, 0.77))
        figure.colorbar(
            image,
            cax=colorbar_axis,
            label=metric_label,
        )
    figure.suptitle(
        f"Addition LSB→MSB, causal, NoPE: length × loop {metric_title} landscape\n"
        "each cell evaluates native length n after exactly t loops; carry omitted",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.10, top=0.88, wspace=0.13)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if spec.name != "addition" or not spec.addition_lsb_first:
        raise ValueError("requires an LSB-first Addition checkpoint")
    if spec.step_offset != 0:
        raise ValueError("requires the T(n)=n protocol")
    if model.config.attention_mode != "causal" or model.config.position_embedding != "none":
        raise ValueError("requires causal attention without position embeddings")
    full = ControllerView(controller, mode="full").to(device).eval()
    rows = evaluate_grid(
        model=model,
        spec=spec,
        controller=full,
        controller_start_step=int(controller_payload["anchor_step"]),
        maximum_length=args.maximum_length,
        maximum_step=args.maximum_step,
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "length_loop_em_grid.csv"
    em_figure_path = args.out_dir / "length_loop_em_n1to20_t1to40.png"
    accuracy_figure_path = args.out_dir / "length_loop_accuracy_n1to20_t1to40.png"
    write_csv(csv_path, rows)
    controller_updates = int(
        controller_payload["training_budget"]["total_optimizer_updates"]
    )
    plot_grid(
        rows,
        metric="ordinary_digit_exact_match",
        metric_label="ordinary-digit exact match (carry excluded)",
        metric_title="EM",
        maximum_length=args.maximum_length,
        maximum_step=args.maximum_step,
        train_maximum_length=int(controller_payload["controller_logical_max_length"]),
        controller_updates=controller_updates,
        output=em_figure_path,
    )
    plot_grid(
        rows,
        metric="ordinary_digit_bit_accuracy",
        metric_label="ordinary-digit mean bit accuracy (carry excluded)",
        metric_title="accuracy",
        maximum_length=args.maximum_length,
        maximum_step=args.maximum_step,
        train_maximum_length=int(controller_payload["controller_logical_max_length"]),
        controller_updates=controller_updates,
        output=accuracy_figure_path,
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller),
        "controller_updates": controller_updates,
        "maximum_length": args.maximum_length,
        "maximum_step": args.maximum_step,
        "examples_per_cell": args.examples,
        "metrics": [
            "ordinary_digit_exact_match",
            "ordinary_digit_bit_accuracy",
        ],
        "carry_in_metric": False,
        "target_rule": "T(n)=n",
        "figures": {
            "exact_match": str(em_figure_path),
            "bit_accuracy": str(accuracy_figure_path),
        },
        "csv": str(csv_path),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--maximum-length", type=int, default=20)
    parser.add_argument("--maximum-step", type=int, default=40)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=884001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
