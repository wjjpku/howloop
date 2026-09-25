#!/usr/bin/env python3
"""Causal position/latency maps for the fixed-width Addition checkpoint.

The clean/corrupt pairs differ only in whether a carry is generated at a
chosen low-order column.  At each recurrent boundary we patch clean residual
states into the corrupt run and measure recovery of a more-significant answer
bit.  Two maps are produced:

1. which token positions contain causally usable carry information;
2. how many further shared-block applications are needed after patching the
   operand carry-chain positions before the target answer bit changes.

This is a position-level causal test, not a minimal head/MLP circuit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_addition_carry_state_patching import (
    make_carry_pairs,
    position_groups,
    target_metrics,
    trajectory,
)
from reasoning_loop.paper_length_telomere import (
    atomic_json_dump,
    load_backbone,
    pick_device,
    write_csv,
)


def token_labels(width: int) -> list[str]:
    return (
        [f"a{i}" for i in range(width)]
        + ["+"]
        + [f"b{i}" for i in range(width)]
        + ["carry"]
        + [f"sum{i}" for i in range(width)]
        + [f"pad{i}" for i in range(4)]
    )


def score(
    logits: torch.Tensor,
    *,
    target_position: int,
    clean_target: torch.Tensor,
    corrupt_target: torch.Tensor,
) -> torch.Tensor:
    target_logits = logits[:, target_position]
    clean_logits = target_logits.gather(-1, clean_target[:, None]).squeeze(-1)
    corrupt_logits = target_logits.gather(-1, corrupt_target[:, None]).squeeze(-1)
    return clean_logits - corrupt_logits


@torch.inference_mode()
def finish(
    model: torch.nn.Module,
    state: torch.Tensor,
    inputs: torch.Tensor,
    *,
    first_step: int,
    final_step: int,
) -> torch.Tensor:
    live = state
    for step_index in range(first_step + 1, final_step + 1):
        embedded = model.input_embeddings(inputs, step_index=step_index)
        live = model.recurrent_step(live, embedded)
    return live


@torch.inference_mode()
def position_patch_rows(
    *,
    model: torch.nn.Module,
    clean_states: list[torch.Tensor],
    corrupt_states: list[torch.Tensor],
    corrupt_inputs: torch.Tensor,
    target_position: int,
    clean_target: torch.Tensor,
    corrupt_target: torch.Tensor,
    denominator: float,
    carry_span: int,
    target_steps: int,
    position_chunk: int,
) -> list[dict[str, Any]]:
    examples, sequence_length, dimension = corrupt_states[0].shape
    rows: list[dict[str, Any]] = []
    labels = token_labels(sequence_length // 3 - 2)
    if len(labels) != sequence_length:
        raise RuntimeError("token label count does not match sequence length")
    for completed_steps in range(1, target_steps + 1):
        corrupt_score = score(
            model.decode(corrupt_states[target_steps - 1]).float(),
            target_position=target_position,
            clean_target=clean_target,
            corrupt_target=corrupt_target,
        ).mean()
        for start in range(0, sequence_length, position_chunk):
            positions = list(range(start, min(start + position_chunk, sequence_length)))
            count = len(positions)
            patched = (
                corrupt_states[completed_steps - 1][None]
                .expand(count, -1, -1, -1)
                .clone()
            )
            donor = clean_states[completed_steps - 1]
            for local_index, position in enumerate(positions):
                patched[local_index, :, position] = donor[:, position]
            flat_state = patched.reshape(count * examples, sequence_length, dimension)
            flat_inputs = (
                corrupt_inputs[None]
                .expand(count, -1, -1, -1)
                .reshape(count * examples, sequence_length, -1)
            )
            final_state = finish(
                model,
                flat_state,
                flat_inputs,
                first_step=completed_steps,
                final_step=target_steps,
            )
            logits = model.decode(final_state).float().reshape(
                count, examples, sequence_length, -1
            )
            repeated_clean = clean_target[None].expand(count, -1).reshape(-1)
            repeated_corrupt = corrupt_target[None].expand(count, -1).reshape(-1)
            patched_score = score(
                logits.reshape(count * examples, sequence_length, -1),
                target_position=target_position,
                clean_target=repeated_clean,
                corrupt_target=repeated_corrupt,
            ).reshape(count, examples)
            predictions = logits[:, :, target_position].argmax(dim=-1)
            for local_index, position in enumerate(positions):
                rows.append(
                    {
                        "carry_span": carry_span,
                        "completed_steps": completed_steps,
                        "remaining_steps": target_steps - completed_steps,
                        "position": position,
                        "position_label": labels[position],
                        "normalized_final_logit_recovery": float(
                            (patched_score[local_index].mean() - corrupt_score)
                            / denominator
                        ),
                        "clean_target_accuracy": float(
                            predictions[local_index].eq(clean_target).float().mean()
                        ),
                    }
                )
    return rows


@torch.inference_mode()
def latency_rows(
    *,
    model: torch.nn.Module,
    clean_states: list[torch.Tensor],
    corrupt_states: list[torch.Tensor],
    corrupt_inputs: torch.Tensor,
    target_position: int,
    clean_target: torch.Tensor,
    corrupt_target: torch.Tensor,
    denominator: float,
    carry_span: int,
    target_steps: int,
    groups: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for completed_steps in range(1, target_steps + 1):
        for group_name in (
            "operand_chain_slots",
            "target_answer_slot",
            "neighbor_answer_slot_control",
        ):
            positions = groups[group_name]
            patched = corrupt_states[completed_steps - 1].clone()
            patched[:, positions] = clean_states[completed_steps - 1][:, positions]
            for readout_step in range(completed_steps, target_steps + 1):
                final_state = finish(
                    model,
                    patched,
                    corrupt_inputs,
                    first_step=completed_steps,
                    final_step=readout_step,
                )
                patched_score = score(
                    model.decode(final_state).float(),
                    target_position=target_position,
                    clean_target=clean_target,
                    corrupt_target=corrupt_target,
                ).mean()
                corrupt_score = score(
                    model.decode(corrupt_states[readout_step - 1]).float(),
                    target_position=target_position,
                    clean_target=clean_target,
                    corrupt_target=corrupt_target,
                ).mean()
                rows.append(
                    {
                        "carry_span": carry_span,
                        "position_group": group_name,
                        "position_count": len(positions),
                        "patch_boundary": completed_steps,
                        "readout_boundary": readout_step,
                        "additional_loops": readout_step - completed_steps,
                        "normalized_logit_effect": float(
                            (patched_score - corrupt_score) / denominator
                        ),
                    }
                )
    return rows


def plot_position_maps(
    rows: list[dict[str, Any]], *, width: int, target_steps: int, out_dir: Path
) -> None:
    spans = sorted({int(row["carry_span"]) for row in rows})
    labels = token_labels(width)
    figure, axes = plt.subplots(
        len(spans), 1, figsize=(15, 3.6 * len(spans)), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, span in zip(axes, spans, strict=True):
        values = np.full((target_steps, len(labels)), np.nan)
        for row in rows:
            if int(row["carry_span"]) == span:
                values[int(row["completed_steps"]) - 1, int(row["position"])] = float(
                    row["normalized_final_logit_recovery"]
                )
        image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_title(f"carry span {span}: clean→corrupt single-position patch")
        axis.set_ylabel("patch after loop")
        axis.set_yticks(range(target_steps), range(1, target_steps + 1))
        axis.axvline(2 * width + 0.5, color="black", linewidth=1.0)
        figure.colorbar(image, ax=axis, label="normalized final-logit recovery")
    axes[-1].set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    axes[-1].set_xlabel("residual token position (a0/b0/sum0 are MSB)")
    figure.suptitle("Where carry information is causally available across recurrent loops")
    figure.savefig(out_dir / "position_carry_migration.png", dpi=180)
    figure.savefig(out_dir / "position_carry_migration.pdf")
    plt.close(figure)


def plot_latency(rows: list[dict[str, Any]], *, target_steps: int, out_dir: Path) -> None:
    spans = sorted({int(row["carry_span"]) for row in rows})
    figure, axes = plt.subplots(
        2,
        len(spans),
        figsize=(5 * len(spans), 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    groups = ("operand_chain_slots", "target_answer_slot")
    for column, span in enumerate(spans):
        for row_index, group in enumerate(groups):
            axis = axes[row_index, column]
            values = np.full((target_steps, target_steps), np.nan)
            for row in rows:
                if int(row["carry_span"]) == span and row["position_group"] == group:
                    values[int(row["readout_boundary"]) - 1, int(row["patch_boundary"]) - 1] = float(
                        row["normalized_logit_effect"]
                    )
            image = axis.imshow(values, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
            axis.set_title(f"span {span}, {group.replace('_slots', '')}")
            axis.set_xlabel("patch boundary")
            axis.set_ylabel("readout boundary")
            axis.set_xticks(range(target_steps), range(1, target_steps + 1))
            axis.set_yticks(range(target_steps), range(1, target_steps + 1))
            figure.colorbar(image, ax=axis, label="normalized causal effect")
    figure.suptitle("Causal carry transfer requires multiple recurrent applications")
    figure.savefig(out_dir / "carry_transfer_latency.png", dpi=180)
    figure.savefig(out_dir / "carry_transfer_latency.pdf")
    plt.close(figure)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError("this analysis requires an Addition checkpoint")
    target_steps = args.width + int(spec.step_offset)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    all_position_rows: list[dict[str, Any]] = []
    all_latency_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for carry_span in args.carry_spans:
        source_position = args.target_operand_position + carry_span
        clean, corrupt = make_carry_pairs(
            examples=args.examples,
            width=args.width,
            target_operand_position=args.target_operand_position,
            carry_source_position=source_position,
            generator=generator,
        )
        clean = clean.to(device)
        corrupt = corrupt.to(device)
        clean_states = trajectory(model, clean, target_steps)
        corrupt_states = trajectory(model, corrupt, target_steps)
        target_position = 2 * args.width + 1 + args.target_operand_position + 1
        clean_target = clean.targets[:, target_position]
        corrupt_target = corrupt.targets[:, target_position]
        clean_metrics = target_metrics(
            model.decode(clean_states[-1]).float(),
            target_position=target_position,
            clean_target=clean_target,
            corrupt_target=corrupt_target,
        )
        corrupt_metrics = target_metrics(
            model.decode(corrupt_states[-1]).float(),
            target_position=target_position,
            clean_target=clean_target,
            corrupt_target=corrupt_target,
        )
        denominator = (
            clean_metrics["mean_clean_minus_corrupt_logit"]
            - corrupt_metrics["mean_clean_minus_corrupt_logit"]
        )
        if abs(denominator) < 1e-6:
            raise RuntimeError("clean/corrupt final-logit denominator is too small")
        baseline_rows.append(
            {
                "carry_span": carry_span,
                "target_position": target_position,
                "denominator": denominator,
                "clean_accuracy": clean_metrics["clean_target_accuracy"],
                "corrupt_accuracy": corrupt_metrics["corrupt_target_accuracy"],
            }
        )
        all_position_rows.extend(
            position_patch_rows(
                model=model,
                clean_states=clean_states,
                corrupt_states=corrupt_states,
                corrupt_inputs=corrupt.inputs,
                target_position=target_position,
                clean_target=clean_target,
                corrupt_target=corrupt_target,
                denominator=denominator,
                carry_span=carry_span,
                target_steps=target_steps,
                position_chunk=args.position_chunk,
            )
        )
        groups = position_groups(
            width=args.width,
            target_operand_position=args.target_operand_position,
            carry_source_position=source_position,
        )
        all_latency_rows.extend(
            latency_rows(
                model=model,
                clean_states=clean_states,
                corrupt_states=corrupt_states,
                corrupt_inputs=corrupt.inputs,
                target_position=target_position,
                clean_target=clean_target,
                corrupt_target=corrupt_target,
                denominator=denominator,
                carry_span=carry_span,
                target_steps=target_steps,
                groups=groups,
            )
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "position_patching.csv", all_position_rows)
    write_csv(args.out_dir / "transfer_latency.csv", all_latency_rows)
    plot_position_maps(
        all_position_rows,
        width=args.width,
        target_steps=target_steps,
        out_dir=args.out_dir,
    )
    plot_latency(all_latency_rows, target_steps=target_steps, out_dir=args.out_dir)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "loss_placement": "fixed n=10 final-only CE at loop 11",
        "shared_block_layers": int(model.config.block_layers),
        "target_steps": target_steps,
        "effective_transformer_depth": target_steps * int(model.config.block_layers),
        "examples_per_span": args.examples,
        "carry_spans": list(args.carry_spans),
        "patch_direction": "clean carry -> corrupt no-carry",
        "hook_site": "residual state after each complete shared-block recurrence",
        "baselines": baseline_rows,
        "claim_boundary": (
            "The maps test position-level causal availability and transfer latency. "
            "They do not identify a minimal head/MLP circuit or prove a unique algorithm."
        ),
    }
    atomic_json_dump(summary, args.out_dir / "summary.json")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--target-operand-position", type=int, default=4)
    parser.add_argument("--carry-spans", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--position-chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=284101)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
