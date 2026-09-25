#!/usr/bin/env python3
"""Plot native-length Addition accuracy over input length and recurrent depth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.paper_length_telomere import (
    ControllerView,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)
from scripts.compare_addition_readout_order_attention import (
    answer_positions_lsb_to_carry,
)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module | None,
    controller_start_step: int | None,
    maximum_length: int,
    maximum_step: int,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    variants: tuple[tuple[str, torch.nn.Module | None, int | None], ...]
    variants = (("raw", None, None),)
    if controller is not None:
        variants += (("full", controller, controller_start_step),)

    rows: list[dict[str, Any]] = []
    for logical_length in range(1, maximum_length + 1):
        positions = answer_positions_lsb_to_carry(spec, logical_length)
        position_index = torch.tensor(positions, device=device)
        correct_bits = {
            name: torch.zeros(maximum_step, dtype=torch.long)
            for name, _, _ in variants
        }
        correct_examples = {
            name: torch.zeros(maximum_step, dtype=torch.long)
            for name, _, _ in variants
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
            for name, active_controller, start_step in variants:
                trajectory = model.iter_states(
                    batch.inputs,
                    steps=maximum_step,
                    controller=active_controller,
                    controller_start_step=start_step,
                )
                for step_index, state in enumerate(trajectory):
                    predictions = model.decode(state).argmax(dim=-1).index_select(
                        1, position_index
                    )
                    correct = predictions.eq(targets)
                    correct_bits[name][step_index] += correct.sum().cpu()
                    correct_examples[name][step_index] += correct.all(dim=1).sum().cpu()
            remaining -= batch.inputs.shape[0]

        bit_count = examples * (logical_length + 1)
        for name, _, _ in variants:
            for step in range(1, maximum_step + 1):
                rows.append(
                    {
                        "variant": name,
                        "input_length": logical_length,
                        "loop_count": step,
                        "full_arithmetic_bit_accuracy": (
                            float(correct_bits[name][step - 1]) / bit_count
                        ),
                        "full_arithmetic_exact_match": (
                            float(correct_examples[name][step - 1]) / examples
                        ),
                        "ordinary_digit_count": logical_length,
                        "final_carry_in_metric": True,
                        "examples": examples,
                    }
                )
    return rows


def matrix_for(
    rows: Sequence[dict[str, Any]],
    *,
    variant: str,
    maximum_length: int,
    maximum_step: int,
) -> np.ndarray:
    matrix = np.full((maximum_step, maximum_length), np.nan, dtype=float)
    for row in rows:
        if row["variant"] == variant:
            matrix[int(row["loop_count"]) - 1, int(row["input_length"]) - 1] = float(
                row["full_arithmetic_bit_accuracy"]
            )
    return matrix


def decorate(
    axis: plt.Axes,
    *,
    maximum_length: int,
    maximum_step: int,
    backbone_train_maximum_length: int,
    controller_train_maximum_length: int | None,
    step_offset: int,
) -> None:
    end = min(maximum_length, maximum_step - step_offset)
    if end >= 1:
        axis.plot(
            [1, end],
            [1 + step_offset, end + step_offset],
            color="white",
            linestyle="--",
            linewidth=1.8,
            label=f"registered T(n)=n+{step_offset}",
        )
    axis.axvline(
        backbone_train_maximum_length + 0.5,
        color="#ff9f1c",
        linestyle="--",
        linewidth=1.8,
        label=f"backbone train max n={backbone_train_maximum_length}",
    )
    if (
        controller_train_maximum_length is not None
        and controller_train_maximum_length != backbone_train_maximum_length
    ):
        axis.axvline(
            controller_train_maximum_length + 0.5,
            color="#ef476f",
            linestyle="-.",
            linewidth=1.8,
            label=f"J train max n={controller_train_maximum_length}",
        )
    axis.set(
        xlabel="input logical length n",
        xlim=(0.5, maximum_length + 0.5),
        ylim=(0.5, maximum_step + 0.5),
    )
    axis.set_xticks(range(1, maximum_length + 1))
    axis.set_yticks(range(1, maximum_step + 1, 2))
    axis.legend(loc="upper left", fontsize=9)


def plot_raw(
    rows: Sequence[dict[str, Any]],
    *,
    title_prefix: str,
    maximum_length: int,
    maximum_step: int,
    backbone_train_maximum_length: int,
    step_offset: int,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 8.5), constrained_layout=True)
    image = axis.imshow(
        matrix_for(
            rows,
            variant="raw",
            maximum_length=maximum_length,
            maximum_step=maximum_step,
        ),
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
        extent=(0.5, maximum_length + 0.5, 0.5, maximum_step + 0.5),
    )
    decorate(
        axis,
        maximum_length=maximum_length,
        maximum_step=maximum_step,
        backbone_train_maximum_length=backbone_train_maximum_length,
        controller_train_maximum_length=None,
        step_offset=step_offset,
    )
    axis.set_ylabel("executed loop count t")
    axis.set_title(
        f"{title_prefix}: original/raw loop-depth accuracy\n"
        "color = mean bit accuracy over n sum bits + final carry; PAD excluded"
    )
    figure.colorbar(image, ax=axis, label="full-arithmetic bit accuracy")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_comparison(
    rows: Sequence[dict[str, Any]],
    *,
    title_prefix: str,
    maximum_length: int,
    maximum_step: int,
    backbone_train_maximum_length: int,
    controller_train_maximum_length: int,
    step_offset: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True)
    image = None
    for axis, variant, label in zip(
        axes,
        ("raw", "full"),
        ("original/raw", "Diag+LoRA J"),
        strict=True,
    ):
        image = axis.imshow(
            matrix_for(
                rows,
                variant=variant,
                maximum_length=maximum_length,
                maximum_step=maximum_step,
            ),
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            extent=(0.5, maximum_length + 0.5, 0.5, maximum_step + 0.5),
        )
        decorate(
            axis,
            maximum_length=maximum_length,
            maximum_step=maximum_step,
            backbone_train_maximum_length=backbone_train_maximum_length,
            controller_train_maximum_length=controller_train_maximum_length,
            step_offset=step_offset,
        )
        axis.set_title(label)
    axes[0].set_ylabel("executed loop count t")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="full-arithmetic bit accuracy (sum bits + final carry)",
            shrink=0.86,
        )
    figure.suptitle(
        f"{title_prefix}: length × loop accuracy; native layout, PAD excluded",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.06, right=0.90, bottom=0.09, top=0.90, wspace=0.12)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("requires an Addition checkpoint")
    controller = None
    controller_payload = None
    controller_start_step = None
    if args.controller is not None:
        loaded_controller, controller_payload = load_controller(
            args.controller, device=device
        )
        controller = ControllerView(loaded_controller, mode="full").to(device).eval()
        controller_start_step = int(controller_payload["anchor_step"])

    rows = evaluate(
        model=model,
        spec=spec,
        controller=controller,
        controller_start_step=controller_start_step,
        maximum_length=args.maximum_length,
        maximum_step=args.maximum_step,
        examples=args.examples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "length_loop_accuracy.csv"
    raw_path = args.out_dir / "original_raw_loop_depth_accuracy.png"
    comparison_path = args.out_dir / "raw_vs_j_loop_depth_accuracy.png"
    write_csv(csv_path, rows)
    architecture = (
        f"Addition {'LSB→MSB' if spec.addition_lsb_first else 'MSB→LSB'}, "
        f"{model.config.attention_mode}, {model.config.position_embedding}"
    )
    plot_raw(
        rows,
        title_prefix=architecture,
        maximum_length=args.maximum_length,
        maximum_step=args.maximum_step,
        backbone_train_maximum_length=int(spec.train_max_length),
        step_offset=int(spec.step_offset),
        output=raw_path,
    )
    if controller is not None:
        plot_comparison(
            rows,
            title_prefix=architecture,
            maximum_length=args.maximum_length,
            maximum_step=args.maximum_step,
            backbone_train_maximum_length=int(spec.train_max_length),
            controller_train_maximum_length=int(
                controller_payload["controller_logical_max_length"]
            ),
            step_offset=int(spec.step_offset),
            output=comparison_path,
        )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller) if args.controller is not None else None,
        "architecture": architecture,
        "maximum_length": args.maximum_length,
        "maximum_step": args.maximum_step,
        "examples_per_cell": args.examples,
        "metric": "mean bit accuracy over native n sum bits plus final carry",
        "padding_in_metric": False,
        "registered_target_rule": f"T(n)=n+{spec.step_offset}",
        "csv": str(csv_path),
        "figures": {
            "original_raw": str(raw_path),
            "raw_vs_j": str(comparison_path) if controller is not None else None,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--maximum-length", type=int, default=20)
    parser.add_argument("--maximum-step", type=int, default=25)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=884001)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
