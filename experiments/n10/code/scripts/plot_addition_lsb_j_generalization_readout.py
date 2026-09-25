#!/usr/bin/env python3
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
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
def readout_trajectory(
    *,
    model: torch.nn.Module,
    spec: Any,
    controller: torch.nn.Module | None,
    controller_start_step: int | None,
    logical_length: int,
    maximum_step: int,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    positions = answer_positions_lsb_to_carry(spec, logical_length)
    position_index = torch.tensor(positions, device=device)
    position_correct = torch.zeros(
        maximum_step, logical_length + 1, dtype=torch.long
    )
    supervised_exact_correct = torch.zeros(maximum_step, dtype=torch.long)
    full_exact_correct = torch.zeros(maximum_step, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
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
        for step_index, state in enumerate(
            model.iter_states(
                batch.inputs,
                steps=maximum_step,
                controller=controller,
                controller_start_step=controller_start_step,
            ),
            start=1,
        ):
            predictions = model.decode(state).argmax(dim=-1).index_select(
                1, position_index
            )
            correct = predictions.eq(targets)
            position_correct[step_index - 1] += correct.sum(dim=0).cpu()
            supervised_exact_correct[step_index - 1] += (
                correct[:, :logical_length].all(dim=1).sum().cpu()
            )
            full_exact_correct[step_index - 1] += correct.all(dim=1).sum().cpu()
        remaining -= batch.inputs.shape[0]
    rows: list[dict[str, Any]] = []
    for step in range(1, maximum_step + 1):
        for numerical_position in range(logical_length + 1):
            rows.append(
                {
                    "step": step,
                    "position_from_lsb": numerical_position,
                    "is_final_carry": numerical_position == logical_length,
                    "bit_accuracy": float(
                        position_correct[step - 1, numerical_position]
                    )
                    / examples,
                    "supervised_digit_exact_match": float(
                        supervised_exact_correct[step - 1]
                    )
                    / examples,
                    "full_arithmetic_exact_match": float(
                        full_exact_correct[step - 1]
                    )
                    / examples,
                }
            )
    return rows


def trajectory_matrix(
    rows: Sequence[dict[str, Any]], maximum_step: int, logical_length: int
) -> np.ndarray:
    indexed = {
        (int(row["step"]), int(row["position_from_lsb"])): float(
            row["bit_accuracy"]
        )
        for row in rows
    }
    return np.asarray(
        [
            [indexed[(step, position)] for position in range(logical_length)]
            for step in range(1, maximum_step + 1)
        ]
    )


def endpoint_series(
    rows: Sequence[dict[str, str]], variant: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["variant"] == variant),
        key=lambda row: int(row["length"]),
    )
    return (
        np.asarray([int(row["length"]) for row in selected]),
        np.asarray([float(row[metric]) for row in selected]),
    )


def plot_summary(
    *,
    endpoint_rows: Sequence[dict[str, str]],
    raw_rows: Sequence[dict[str, Any]],
    full_rows: Sequence[dict[str, Any]],
    logical_length: int,
    maximum_step: int,
    target_step: int,
    target_rule: str,
    controller_updates: int,
    j_train_min_length: int,
    j_train_max_length: int,
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    colors = {"raw": "#6c757d", "full": "#d62728"}
    labels = {"raw": "raw", "full": f"J ({controller_updates:,} updates)"}
    for variant in ("raw", "full"):
        lengths, digit_em = endpoint_series(
            endpoint_rows, variant, "supervised_digit_exact_match"
        )
        _, digit_bit_accuracy = endpoint_series(
            endpoint_rows, variant, "supervised_digit_bit_accuracy"
        )
        axes[0, 0].plot(
            lengths,
            digit_em,
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=colors[variant],
            label=labels[variant],
        )
        axes[0, 1].plot(
            lengths,
            digit_bit_accuracy,
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=colors[variant],
            label=labels[variant],
        )
    for axis, title, ylabel in (
        (
            axes[0, 0],
            f"Supervised-digit length generalization at {target_rule}",
            "m-digit EM (carry excluded)",
        ),
        (
            axes[0, 1],
            f"Supervised-digit bit accuracy at {target_rule}",
            "ordinary-digit bit accuracy",
        ),
    ):
        axis.axvspan(
            j_train_min_length - 0.5,
            j_train_max_length + 0.5,
            color="#4c78a8",
            alpha=0.08,
            label=f"J train: m={j_train_min_length}..{j_train_max_length}",
        )
        axis.axvline(
            j_train_max_length + 0.5,
            color="#4c78a8",
            linestyle="--",
            linewidth=1.2,
        )
        axis.set(title=title, xlabel="logical addition length n", ylabel=ylabel)
        axis.set_xlim(1, max(int(row["length"]) for row in endpoint_rows))
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    matrices = (
        (trajectory_matrix(raw_rows, maximum_step, logical_length), "raw readout"),
        (trajectory_matrix(full_rows, maximum_step, logical_length), "J readout"),
    )
    image = None
    for axis, (matrix, title) in zip(axes[1], matrices):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            extent=(-0.5, logical_length - 0.5, 0.5, maximum_step + 0.5),
        )
        axis.axhspan(
            target_step - 0.5,
            target_step + 0.5,
            facecolor="none",
            edgecolor="white",
            linestyle="--",
            linewidth=1.5,
        )
        axis.set(
            title=(
                f"fixed n={logical_length}: {title}; white box = T(n)="
                f"{target_step}"
            ),
            xlabel="supervised numerical answer bit: LSB → MSB",
            ylabel="loop readout",
        )
        axis.set_xticks(range(logical_length))
        axis.set_xticklabels(
            ["LSB"] + [str(i) for i in range(1, logical_length)]
        )
        axis.set_yticks(range(1, maximum_step + 1, 2))
    if image is not None:
        colorbar_axis = figure.add_axes((0.945, 0.075, 0.012, 0.34))
        figure.colorbar(image, cax=colorbar_axis, label="per-bit accuracy")
    figure.suptitle(
        "Addition LSB→MSB, causal, NoPE: frozen-backbone J behavior\n"
        f"each heatmap row is the same fixed n={logical_length} input length; "
        "carry omitted everywhere",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.07, top=0.92, hspace=0.30, wspace=0.22)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    controller, controller_payload = load_controller(args.controller, device=device)
    if spec.name != "addition" or not spec.addition_lsb_first:
        raise ValueError("this plot requires an LSB-first Addition checkpoint")
    if model.config.attention_mode != "causal" or model.config.position_embedding != "none":
        raise ValueError("this plot requires causal attention without position embeddings")
    full = ControllerView(controller, mode="full").to(device).eval()
    common = {
        "model": model,
        "spec": spec,
        "logical_length": args.logical_length,
        "maximum_step": args.maximum_step,
        "examples": args.examples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
    }
    raw_rows = readout_trajectory(
        controller=None, controller_start_step=None, **common
    )
    full_rows = readout_trajectory(
        controller=full,
        controller_start_step=int(controller_payload["anchor_step"]),
        **common,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tagged_rows = [
        {"variant": variant, **row}
        for variant, rows in (("raw", raw_rows), ("full", full_rows))
        for row in rows
    ]
    write_csv(args.out_dir / "readout_trajectory.csv", tagged_rows)
    endpoint_rows = read_csv(args.endpoint_csv)
    step_offset = int(spec.step_offset)
    target_step = args.logical_length + step_offset
    target_rule = "T(m)=m" if step_offset == 0 else f"T(m)=m+{step_offset}"
    controller_updates = int(
        controller_payload["training_budget"]["total_optimizer_updates"]
    )
    plot_summary(
        endpoint_rows=endpoint_rows,
        raw_rows=raw_rows,
        full_rows=full_rows,
        logical_length=args.logical_length,
        maximum_step=args.maximum_step,
        target_step=target_step,
        target_rule=target_rule,
        controller_updates=controller_updates,
        j_train_min_length=int(controller_payload["controller_logical_min_length"]),
        j_train_max_length=int(controller_payload["controller_logical_max_length"]),
        output=args.out_dir / "lsb_causal_nope_raw_j_generalization_readout.png",
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "controller": str(args.controller),
        "controller_updates": controller_updates,
        "controller_anchor_step": int(controller_payload["anchor_step"]),
        "controller_post_final_j": bool(
            controller_payload.get("controller_post_final_j", False)
        ),
        "logical_length": args.logical_length,
        "registered_target_loop": target_step,
        "target_loop_rule": target_rule,
        "maximum_step": args.maximum_step,
        "examples": args.examples,
        "loss_placement": {
            "backbone": backbone_payload["supervision"],
            "controller": controller_payload["loss"],
        },
        "figure": str(
            args.out_dir / "lsb_causal_nope_raw_j_generalization_readout.png"
        ),
        "claim_boundary": "behavioral readout localization; not a complete circuit claim",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--endpoint-csv", type=Path, required=True)
    parser.add_argument("--logical-length", type=int, default=10)
    parser.add_argument("--maximum-step", type=int, default=16)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=684001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
