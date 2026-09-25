#!/usr/bin/env python3
"""Plot the registered T(m)=m Addition readout frontier, excluding carry.

Unlike a fixed-n trajectory, row m here is evaluated on native length-m inputs
at their registered endpoint loop T(m)=m.  This is the heatmap aligned with the
adaptive-length backbone training protocol.
"""

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


def final_state(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    steps: int,
    controller: torch.nn.Module | None,
    controller_start_step: int | None,
) -> torch.Tensor:
    state = None
    for state in model.iter_states(
        inputs,
        steps=steps,
        controller=controller,
        controller_start_step=controller_start_step,
    ):
        pass
    if state is None:
        raise RuntimeError("registered endpoint must contain at least one loop")
    return state


@torch.inference_mode()
def evaluate_registered_frontier(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module,
    controller_start_step: int,
    maximum_length: int,
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
        correct_by_variant = {
            "raw": torch.zeros(logical_length, dtype=torch.long),
            "full": torch.zeros(logical_length, dtype=torch.long),
        }
        exact_by_variant = {"raw": 0, "full": 0}
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
            states = {
                "raw": final_state(
                    model,
                    batch.inputs,
                    steps=logical_length,
                    controller=None,
                    controller_start_step=None,
                ),
                "full": final_state(
                    model,
                    batch.inputs,
                    steps=logical_length,
                    controller=controller,
                    controller_start_step=controller_start_step,
                ),
            }
            for variant, state in states.items():
                predictions = model.decode(state).argmax(dim=-1).index_select(
                    1, position_index
                )
                correct = predictions.eq(targets)
                correct_by_variant[variant] += correct.sum(dim=0).cpu()
                exact_by_variant[variant] += int(correct.all(dim=1).sum())
            remaining -= batch.inputs.shape[0]

        for variant in ("raw", "full"):
            exact_match = exact_by_variant[variant] / examples
            bit_accuracies = correct_by_variant[variant].float() / examples
            for position in range(logical_length):
                rows.append(
                    {
                        "variant": variant,
                        "logical_length": logical_length,
                        "registered_step": logical_length,
                        "position_from_lsb": position,
                        "bit_accuracy": float(bit_accuracies[position]),
                        "ordinary_digit_exact_match": exact_match,
                        "ordinary_digit_mean_accuracy": float(
                            bit_accuracies.mean()
                        ),
                        "examples": examples,
                    }
                )
    return rows


def plot_frontier(
    rows: Sequence[dict[str, Any]],
    *,
    maximum_length: int,
    train_maximum_length: int,
    controller_updates: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.2))
    colors = {"raw": "#6c757d", "full": "#d62728"}
    labels = {"raw": "raw", "full": f"J ({controller_updates:,} updates)"}
    for variant in ("raw", "full"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        endpoints = {
            int(row["logical_length"]): row for row in variant_rows
        }
        lengths = np.asarray(sorted(endpoints))
        em = np.asarray(
            [float(endpoints[length]["ordinary_digit_exact_match"]) for length in lengths]
        )
        bit_accuracy = np.asarray(
            [
                float(endpoints[length]["ordinary_digit_mean_accuracy"])
                for length in lengths
            ]
        )
        axes[0, 0].plot(
            lengths, em, marker="o", linewidth=2, color=colors[variant],
            label=labels[variant]
        )
        axes[0, 1].plot(
            lengths, bit_accuracy, marker="o", linewidth=2,
            color=colors[variant], label=labels[variant]
        )

    for axis, title, ylabel in (
        (axes[0, 0], "Ordinary-digit exact match at T(m)=m", "m-digit EM"),
        (axes[0, 1], "Ordinary-digit mean accuracy at T(m)=m", "mean bit accuracy"),
    ):
        axis.axvspan(0.5, train_maximum_length + 0.5, color="#4c78a8", alpha=0.08)
        axis.axvline(train_maximum_length + 0.5, color="#4c78a8", linestyle="--")
        axis.set(xlim=(1, maximum_length), ylim=(-0.02, 1.02), xlabel="native logical length m", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    image = None
    for axis, variant in zip(axes[1], ("raw", "full")):
        matrix = np.full((maximum_length, maximum_length), np.nan)
        for row in rows:
            if row["variant"] != variant:
                continue
            length = int(row["logical_length"])
            position = int(row["position_from_lsb"])
            matrix[length - 1, position] = float(row["bit_accuracy"])
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(
            masked,
            origin="lower",
            aspect="equal",
            vmin=0,
            vmax=1,
            cmap="viridis",
            extent=(-0.5, maximum_length - 0.5, 0.5, maximum_length + 0.5),
        )
        axis.axhline(
            train_maximum_length + 0.5,
            color="white",
            linestyle="--",
            linewidth=1.4,
        )
        axis.set(
            title=f"{labels[variant]} registered endpoints",
            xlabel="supervised answer bit: LSB → MSB",
            ylabel="native length m and registered loop T(m)=m",
        )
        axis.set_xticks(range(maximum_length))
        axis.set_xticklabels(["LSB"] + [str(i) for i in range(1, maximum_length)], rotation=45)
        axis.set_yticks(range(1, maximum_length + 1))

    if image is not None:
        colorbar_axis = figure.add_axes((0.945, 0.075, 0.012, 0.34))
        figure.colorbar(image, cax=colorbar_axis, label="ordinary-bit accuracy")
    figure.suptitle(
        "Addition LSB→MSB, causal, NoPE: registered T(m)=m frontier\n"
        "row m uses native length-m inputs at loop m; carry omitted everywhere",
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.07, right=0.92, bottom=0.07, top=0.91,
        hspace=0.30, wspace=0.22
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if spec.name != "addition" or not spec.addition_lsb_first:
        raise ValueError("requires an LSB-first Addition checkpoint")
    if spec.step_offset != 0:
        raise ValueError("requires the T(m)=m protocol")
    if model.config.attention_mode != "causal" or model.config.position_embedding != "none":
        raise ValueError("requires causal attention without position embeddings")
    full = ControllerView(controller, mode="full").to(device).eval()
    rows = evaluate_registered_frontier(
        model=model,
        spec=spec,
        controller=full,
        controller_start_step=int(controller_payload["anchor_step"]),
        maximum_length=args.maximum_length,
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "registered_frontier.csv"
    figure_path = args.out_dir / "registered_tn_frontier_n1to20.png"
    write_csv(csv_path, rows)
    controller_updates = int(
        controller_payload["training_budget"]["total_optimizer_updates"]
    )
    plot_frontier(
        rows,
        maximum_length=args.maximum_length,
        train_maximum_length=int(controller_payload["controller_logical_max_length"]),
        controller_updates=controller_updates,
        output=figure_path,
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller),
        "controller_updates": controller_updates,
        "maximum_length": args.maximum_length,
        "examples_per_length": args.examples,
        "target_rule": "T(m)=m",
        "carry_in_figure": False,
        "figure": str(figure_path),
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
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=784001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
