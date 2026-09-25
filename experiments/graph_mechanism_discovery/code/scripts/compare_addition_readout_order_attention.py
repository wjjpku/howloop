#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    PaperBatch,
    PaperTaskSpec,
    _binary_addition,
    generate_paper_batch,
    load_backbone,
    pick_device,
)


@dataclass(frozen=True)
class SystemSpec:
    label: str
    checkpoint: Path


def parse_system(raw: str) -> SystemSpec:
    label, separator, checkpoint = raw.partition("=")
    if not separator or not label or not checkpoint:
        raise argparse.ArgumentTypeError("systems must use LABEL=CHECKPOINT")
    return SystemSpec(label=label, checkpoint=Path(checkpoint))


def answer_positions_lsb_to_carry(
    spec: PaperTaskSpec, logical_length: int
) -> list[int]:
    width = spec.addition_layout_width(logical_length)
    answer_start = 2 * width + 1
    if spec.addition_lsb_first:
        return list(range(answer_start, answer_start + logical_length + 1))
    answer_offset = width - logical_length
    stored_carry_to_lsb = list(
        range(
            answer_start + answer_offset,
            answer_start + width + 1,
        )
    )
    return list(reversed(stored_carry_to_lsb))


def make_long_carry_batch(
    spec: PaperTaskSpec, logical_length: int
) -> PaperBatch:
    width = spec.addition_layout_width(logical_length)
    first = torch.ones(logical_length, dtype=torch.long)
    second = torch.zeros(logical_length, dtype=torch.long)
    second[-1] = 1
    answer = _binary_addition(first, second)
    sequence_length = spec.sequence_length(logical_length)
    token_ids = torch.full((1, sequence_length), 3, dtype=torch.long)
    targets = torch.full((1, sequence_length), 3, dtype=torch.long)
    answer_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
    token_ids[0, :width] = 0
    token_ids[0, width + 1 : 2 * width + 1] = 0
    if spec.addition_lsb_first:
        token_ids[0, :logical_length] = first.flip(0)
        token_ids[0, width + 1 : width + 1 + logical_length] = second.flip(0)
    else:
        token_ids[0, width - logical_length : width] = first
        token_ids[
            0,
            2 * width + 1 - logical_length : 2 * width + 1,
        ] = second
    token_ids[0, width] = 2
    answer_start = 2 * width + 1
    token_ids[0, answer_start] = 5
    targets[0, :answer_start] = 4
    targets[0, answer_start : answer_start + width + 1] = 0
    if spec.addition_lsb_first:
        targets[0, answer_start : answer_start + logical_length + 1] = (
            answer.flip(0)
        )
    else:
        answer_offset = width - logical_length
        targets[
            0,
            answer_start + answer_offset : answer_start + width + 1,
        ] = answer
    answer_mask[0, answer_start:] = True
    return PaperBatch(
        inputs=F.one_hot(token_ids, num_classes=spec.vocab_size).float(),
        targets=targets,
        answer_mask=answer_mask,
        lengths=torch.tensor([logical_length]),
        target_steps=torch.tensor([logical_length + spec.step_offset]),
    )


def advance(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    state: torch.Tensor,
    step: int,
) -> torch.Tensor:
    injected = model.input_embeddings(inputs, step_index=step)
    return model.recurrent_step(state, injected)


@torch.inference_mode()
def evaluate_system(
    system: SystemSpec,
    *,
    logical_length: int,
    maximum_step: int,
    examples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model, spec, payload = load_backbone(system.checkpoint, device=device)
    if spec.name != "addition":
        raise ValueError(f"{system.label} is not an Addition checkpoint")
    positions = answer_positions_lsb_to_carry(spec, logical_length)
    position_correct = torch.zeros(
        maximum_step, logical_length + 1, dtype=torch.long
    )
    exact_correct = torch.zeros(maximum_step, dtype=torch.long)
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
        state = torch.zeros(
            batch.inputs.shape[0],
            batch.inputs.shape[1],
            model.config.d_model,
            device=device,
        )
        position_index = torch.tensor(positions, device=device)
        targets = batch.targets.index_select(1, position_index)
        for step in range(1, maximum_step + 1):
            state = advance(model, batch.inputs, state, step)
            predictions = model.decode(state).argmax(dim=-1).index_select(
                1, position_index
            )
            correct = predictions.eq(targets)
            position_correct[step - 1] += correct.sum(dim=0).cpu()
            exact_correct[step - 1] += correct.all(dim=1).sum().cpu()
        remaining -= batch.inputs.shape[0]

    random_rows: list[dict[str, Any]] = []
    for step in range(1, maximum_step + 1):
        for numerical_position in range(logical_length + 1):
            random_rows.append(
                {
                    "system": system.label,
                    "step": step,
                    "position_from_lsb": numerical_position,
                    "is_final_carry": numerical_position == logical_length,
                    "accuracy": (
                        float(position_correct[step - 1, numerical_position])
                        / examples
                    ),
                    "exact_match": float(exact_correct[step - 1]) / examples,
                }
            )

    carry_batch = make_long_carry_batch(spec, logical_length).to(device)
    carry_positions = torch.tensor(positions, device=device)
    carry_targets = carry_batch.targets.index_select(1, carry_positions)
    state = torch.zeros(
        1,
        carry_batch.inputs.shape[1],
        model.config.d_model,
        device=device,
    )
    carry_rows: list[dict[str, Any]] = []
    for step in range(1, maximum_step + 1):
        state = advance(model, carry_batch.inputs, state, step)
        logits = model.decode(state).index_select(1, carry_positions)
        predictions = logits.argmax(dim=-1)
        probabilities = logits.softmax(dim=-1).gather(
            -1, carry_targets.unsqueeze(-1)
        ).squeeze(-1)
        for numerical_position in range(logical_length + 1):
            carry_rows.append(
                {
                    "system": system.label,
                    "step": step,
                    "position_from_lsb": numerical_position,
                    "is_final_carry": numerical_position == logical_length,
                    "target": int(carry_targets[0, numerical_position]),
                    "prediction": int(predictions[0, numerical_position]),
                    "correct": int(
                        predictions[0, numerical_position]
                        == carry_targets[0, numerical_position]
                    ),
                    "target_probability": float(
                        probabilities[0, numerical_position]
                    ),
                }
            )

    metadata = {
        "system": system.label,
        "checkpoint": str(system.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "token_order": (
            "LSB_to_MSB" if spec.addition_lsb_first else "MSB_to_LSB"
        ),
        "attention_mode": model.config.attention_mode,
        "position_embedding": model.config.position_embedding,
        "position_injection": model.config.position_injection,
        "step_offset": int(spec.step_offset),
        "registered_target_loop": logical_length + int(spec.step_offset),
        "addition_answer_supervision": spec.addition_answer_supervision,
        "answer_positions_lsb_to_carry": positions,
    }
    return metadata, random_rows, carry_rows


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


def matrix(
    rows: Sequence[dict[str, Any]],
    *,
    label: str,
    maximum_step: int,
    logical_length: int,
    key: str,
) -> np.ndarray:
    indexed = {
        (int(row["step"]), int(row["position_from_lsb"])): float(row[key])
        for row in rows
        if row["system"] == label
    }
    return np.asarray(
        [
            [
                indexed[(step, position)]
                for position in range(logical_length + 1)
            ]
            for step in range(1, maximum_step + 1)
        ]
    )


def plot_panels(
    *,
    systems: Sequence[SystemSpec],
    rows: Sequence[dict[str, Any]],
    maximum_step: int,
    logical_length: int,
    target_step: int,
    key: str,
    title: str,
    output: Path,
    cmap: str,
) -> None:
    columns = 2
    rows_count = (len(systems) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(14, 5.5 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    last_image = None
    for axis, system in zip(axes.flat, systems, strict=False):
        values = matrix(
            rows,
            label=system.label,
            maximum_step=maximum_step,
            logical_length=logical_length,
            key=key,
        )
        last_image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap=cmap,
        )
        axis.axhline(
            target_step - 1,
            color="white",
            linestyle="--",
            linewidth=1.2,
        )
        axis.set_title(f"{system.label} (target loop {target_step})")
        axis.set_xlabel("numerical answer bit: LSB → final carry")
        axis.set_ylabel("loop readout")
        axis.set_xticks(
            range(logical_length + 1),
            ["LSB"]
            + [str(index) for index in range(1, logical_length)]
            + ["carry"],
        )
        axis.set_yticks(
            np.arange(0, maximum_step, 2),
            np.arange(1, maximum_step + 1, 2),
        )
    for axis in axes.flat[len(systems) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    if last_image is not None:
        figure.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.78)
    figure.savefig(output, dpi=210)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", type=parse_system, required=True)
    parser.add_argument("--logical-length", type=int, default=10)
    parser.add_argument("--maximum-step", type=int, default=16)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=284001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    metadata: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    carry_rows: list[dict[str, Any]] = []
    for system in args.system:
        system_metadata, system_random, system_carry = evaluate_system(
            system,
            logical_length=args.logical_length,
            maximum_step=args.maximum_step,
            examples=args.examples,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        metadata.append(system_metadata)
        random_rows.extend(system_random)
        carry_rows.extend(system_carry)
    target_steps = {
        int(system_metadata["registered_target_loop"])
        for system_metadata in metadata
    }
    if len(target_steps) != 1:
        raise ValueError("all compared systems must share one target loop")
    target_step = next(iter(target_steps))
    write_csv(args.out_dir / "random_position_accuracy.csv", random_rows)
    write_csv(args.out_dir / "all_ones_plus_one_trajectory.csv", carry_rows)
    plot_panels(
        systems=args.system,
        rows=random_rows,
        maximum_step=args.maximum_step,
        logical_length=args.logical_length,
        target_step=target_step,
        key="accuracy",
        title="Random additions: per-bit readout accuracy aligned by numerical bit",
        output=args.out_dir / "random_position_accuracy_aligned.png",
        cmap="viridis",
    )
    plot_panels(
        systems=args.system,
        rows=carry_rows,
        maximum_step=args.maximum_step,
        logical_length=args.logical_length,
        target_step=target_step,
        key="correct",
        title="Carry-heavy example (all ones + 1): correct bits by loop",
        output=args.out_dir / "all_ones_plus_one_correctness_aligned.png",
        cmap="RdYlGn",
    )
    summary = {
        "status": "complete",
        "logical_length": args.logical_length,
        "registered_target_loop": target_step,
        "evaluated_loops": [1, args.maximum_step],
        "random_examples_per_system": args.examples,
        "alignment": "columns are numerical LSB through final carry regardless of token order",
        "systems": metadata,
        "claim_boundary": (
            "Direct-readout localization only; causal mechanism requires matched "
            "component or state interventions."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
