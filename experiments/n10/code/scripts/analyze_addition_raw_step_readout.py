#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.paper_length_telomere import (
    PAPER_TASKS,
    PaperBatch,
    _binary_addition,
    load_backbone,
    load_controller,
    pick_device,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_batch(
    *,
    spec: Any,
    logical_length: int,
    batch_size: int,
    generator: torch.Generator,
) -> PaperBatch:
    from reasoning_loop.paper_length_telomere import generate_paper_batch

    return generate_paper_batch(
        spec,
        batch_size=batch_size,
        min_length=logical_length,
        max_length=logical_length,
        fixed_length=logical_length,
        generator=generator,
    )


def empty_step_totals(logical_length: int) -> dict[str, Any]:
    return {
        "examples": 0,
        "exact": 0,
        "correct_digits": 0,
        "digits": 0,
        "carry_correct": 0,
        "padding_exact": 0,
        "low_order_frontier": 0,
        "ce_sum": 0.0,
        "min_margin_sum": 0.0,
        "position_correct_lsb_to_carry": [0] * (logical_length + 1),
    }


def load_system(
    args: argparse.Namespace,
) -> tuple[Any, Any, torch.nn.Module | None, dict[str, Any] | None]:
    device = pick_device(args.device)
    model, spec, _ = load_backbone(args.checkpoint, device=device)
    controller = None
    controller_payload = None
    if args.controller is not None:
        controller, controller_payload = load_controller(args.controller, device=device)
        named_checkpoint = Path(str(controller_payload["checkpoint"]))
        if named_checkpoint != args.checkpoint:
            raise ValueError(
                "controller metadata names a different backbone: "
                f"{named_checkpoint} != {args.checkpoint}"
            )
    return model, spec, controller, controller_payload


def advance_state(
    *,
    model: Any,
    controller: torch.nn.Module | None,
    controller_payload: dict[str, Any] | None,
    state: torch.Tensor,
    embedded: torch.Tensor,
    step: int,
) -> torch.Tensor:
    if (
        controller is not None
        and controller_payload is not None
        and step > int(controller_payload["anchor_step"])
    ):
        state = controller(state)
    return model.recurrent_step(state, embedded)


def update_totals(
    totals: dict[str, Any],
    *,
    logits: torch.Tensor,
    batch: PaperBatch,
    logical_length: int,
) -> None:
    answer_start = 2 * logical_length + 1
    answer_end = answer_start + logical_length + 1
    actual_logits = logits[:, answer_start:answer_end].float()
    actual_targets = batch.targets[:, answer_start:answer_end]
    predictions = actual_logits.argmax(dim=-1)
    correct = predictions.eq(actual_targets)
    exact = correct.all(dim=1)
    reversed_correct = correct.flip(dims=(1,))
    frontier = torch.cumprod(reversed_correct.to(torch.int64), dim=1).sum(dim=1)
    target_logits = actual_logits.gather(-1, actual_targets.unsqueeze(-1)).squeeze(-1)
    distractor_logits = actual_logits.masked_fill(
        F.one_hot(actual_targets, num_classes=actual_logits.shape[-1]).bool(),
        -torch.inf,
    ).amax(dim=-1)
    minimum_margin = (target_logits - distractor_logits).amin(dim=1)
    full_predictions = logits.argmax(dim=-1)
    padding_exact = full_predictions[:, answer_end:].eq(
        batch.targets[:, answer_end:]
    ).all(dim=1)

    totals["examples"] += int(correct.shape[0])
    totals["exact"] += int(exact.sum())
    totals["correct_digits"] += int(correct.sum())
    totals["digits"] += int(correct.numel())
    totals["carry_correct"] += int(correct[:, 0].sum())
    totals["padding_exact"] += int(padding_exact.sum())
    totals["low_order_frontier"] += int(frontier.sum())
    totals["ce_sum"] += float(
        F.cross_entropy(
            actual_logits.flatten(0, 1),
            actual_targets.flatten(),
            reduction="sum",
        )
    )
    totals["min_margin_sum"] += float(minimum_margin.sum())
    position = reversed_correct.sum(dim=0).tolist()
    totals["position_correct_lsb_to_carry"] = [
        left + int(right)
        for left, right in zip(
            totals["position_correct_lsb_to_carry"], position, strict=True
        )
    ]


@torch.inference_mode()
def analyze_random_readouts(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = pick_device(args.device)
    model, spec, controller, controller_payload = load_system(args)
    if spec.name != "addition":
        raise ValueError("this analysis requires Addition")
    rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for logical_length in args.lengths:
        totals = {
            step: empty_step_totals(logical_length)
            for step in range(1, args.maximum_step + 1)
        }
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 1009 * logical_length)
        remaining = args.examples
        while remaining:
            batch = make_batch(
                spec=spec,
                logical_length=logical_length,
                batch_size=min(args.batch_size, remaining),
                generator=generator,
            ).to(device)
            embedded = model.input_embeddings(batch.inputs)
            state = torch.zeros_like(embedded)
            for step in range(1, args.maximum_step + 1):
                state = advance_state(
                    model=model,
                    controller=controller,
                    controller_payload=controller_payload,
                    state=state,
                    embedded=embedded,
                    step=step,
                )
                update_totals(
                    totals[step],
                    logits=model.decode(state),
                    batch=batch,
                    logical_length=logical_length,
                )
            remaining -= batch.inputs.shape[0]
        target_step = logical_length + int(spec.step_offset)
        for step, values in totals.items():
            examples = values["examples"]
            rows.append(
                {
                    "length": logical_length,
                    "step": step,
                    "target_step": target_step,
                    "step_offset": step - target_step,
                    "examples": examples,
                    "actual_answer_exact_match": values["exact"] / examples,
                    "actual_answer_token_accuracy": values["correct_digits"] / values["digits"],
                    "final_carry_accuracy": values["carry_correct"] / examples,
                    "padding_exact_match": values["padding_exact"] / examples,
                    "mean_correct_low_order_digits": values["low_order_frontier"] / examples,
                    "mean_correct_low_order_fraction": values["low_order_frontier"] / (examples * (logical_length + 1)),
                    "actual_answer_cross_entropy": values["ce_sum"] / values["digits"],
                    "mean_sequence_min_margin": values["min_margin_sum"] / examples,
                }
            )
            for position_from_lsb, correct_count in enumerate(
                values["position_correct_lsb_to_carry"]
            ):
                position_rows.append(
                    {
                        "length": logical_length,
                        "step": step,
                        "target_step": target_step,
                        "step_offset": step - target_step,
                        "position_from_lsb": position_from_lsb,
                        "is_final_carry": position_from_lsb == logical_length,
                        "accuracy": correct_count / examples,
                    }
                )
    return rows, position_rows


def make_long_carry_batch(spec: Any, logical_length: int) -> tuple[PaperBatch, str, str, str]:
    first = torch.ones(logical_length, dtype=torch.long)
    second = torch.zeros(logical_length, dtype=torch.long)
    second[-1] = 1
    answer = _binary_addition(first, second)
    sequence_length = spec.sequence_length(logical_length)
    token_ids = torch.full((1, sequence_length), 3, dtype=torch.long)
    targets = torch.full((1, sequence_length), 3, dtype=torch.long)
    answer_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
    token_ids[0, :logical_length] = first
    token_ids[0, logical_length] = 2
    token_ids[0, logical_length + 1 : 2 * logical_length + 1] = second
    answer_start = 2 * logical_length + 1
    token_ids[0, answer_start] = 5
    targets[0, :answer_start] = 4
    targets[0, answer_start : answer_start + logical_length + 1] = answer
    answer_mask[0, answer_start:] = True
    return (
        PaperBatch(
            inputs=F.one_hot(token_ids, num_classes=spec.vocab_size).float(),
            targets=targets,
            answer_mask=answer_mask,
            lengths=torch.tensor([logical_length]),
            target_steps=torch.tensor([logical_length + spec.step_offset]),
        ),
        "".join(map(str, first.tolist())),
        "".join(map(str, second.tolist())),
        "".join(map(str, answer.tolist())),
    )


@torch.inference_mode()
def analyze_long_carry_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = pick_device(args.device)
    model, spec, controller, controller_payload = load_system(args)
    rows: list[dict[str, Any]] = []
    for logical_length in args.example_lengths:
        batch, first, second, target = make_long_carry_batch(spec, logical_length)
        batch = batch.to(device)
        answer_start = 2 * logical_length + 1
        answer_end = answer_start + logical_length + 1
        embedded = model.input_embeddings(batch.inputs)
        state = torch.zeros_like(embedded)
        previous_prediction: str | None = None
        for step in range(1, args.maximum_step + 1):
            state = advance_state(
                model=model,
                controller=controller,
                controller_payload=controller_payload,
                state=state,
                embedded=embedded,
                step=step,
            )
            prediction_tokens = model.decode(state).argmax(dim=-1)[0, answer_start:answer_end]
            prediction = "".join(map(str, prediction_tokens.tolist()))
            rows.append(
                {
                    "length": logical_length,
                    "step": step,
                    "target_step": logical_length + int(spec.step_offset),
                    "first": first,
                    "second": second,
                    "target": target,
                    "prediction": prediction,
                    "exact": prediction == target,
                    "changed_from_previous": previous_prediction is not None and prediction != previous_prediction,
                    "position_correct": json.dumps(
                        [int(left == right) for left, right in zip(prediction, target, strict=True)]
                    ),
                }
            )
            previous_prediction = prediction
    return rows


@torch.inference_mode()
def analyze_random_example_trajectories(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = pick_device(args.device)
    model, spec, controller, controller_payload = load_system(args)
    logical_length = args.trace_length
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 7919 * logical_length)
    candidates = make_batch(
        spec=spec,
        logical_length=logical_length,
        batch_size=max(64, args.trace_examples * 8),
        generator=generator,
    ).to(device)
    answer_start = 2 * logical_length + 1
    answer_end = answer_start + logical_length + 1
    carry = candidates.targets[:, answer_start]
    selected: list[int] = []
    for carry_value in (0, 1):
        matching = torch.nonzero(carry.eq(carry_value), as_tuple=False).flatten()
        selected.extend(matching[: args.trace_examples].tolist())
    index = torch.tensor(selected, device=device, dtype=torch.long)
    batch = PaperBatch(
        inputs=candidates.inputs.index_select(0, index),
        targets=candidates.targets.index_select(0, index),
        answer_mask=candidates.answer_mask.index_select(0, index),
        lengths=candidates.lengths.index_select(0, index),
        target_steps=candidates.target_steps.index_select(0, index),
    )
    input_tokens = batch.inputs.argmax(dim=-1)
    target_tokens = batch.targets
    embedded = model.input_embeddings(batch.inputs)
    state = torch.zeros_like(embedded)
    rows: list[dict[str, Any]] = []
    for step in range(1, args.maximum_step + 1):
        state = advance_state(
            model=model,
            controller=controller,
            controller_payload=controller_payload,
            state=state,
            embedded=embedded,
            step=step,
        )
        predictions = model.decode(state).argmax(dim=-1)
        for sample_index in range(batch.inputs.shape[0]):
            first = input_tokens[sample_index, :logical_length]
            second = input_tokens[
                sample_index, logical_length + 1 : 2 * logical_length + 1
            ]
            target = target_tokens[sample_index, answer_start:answer_end]
            prediction = predictions[sample_index, answer_start:answer_end]
            rows.append(
                {
                    "sample": sample_index,
                    "final_carry": int(target[0]),
                    "step": step,
                    "target_step": logical_length + int(spec.step_offset),
                    "first": "".join(map(str, first.tolist())),
                    "second": "".join(map(str, second.tolist())),
                    "target": "".join(map(str, target.tolist())),
                    "prediction": "".join(map(str, prediction.tolist())),
                    "exact": bool(prediction.eq(target).all()),
                }
            )
    return rows


def matrix_from_rows(
    rows: list[dict[str, Any]],
    lengths: list[int],
    maximum_step: int,
    key: str,
) -> np.ndarray:
    indexed = {(int(row["length"]), int(row["step"])): float(row[key]) for row in rows}
    return np.asarray(
        [[indexed[(length, step)] for step in range(1, maximum_step + 1)] for length in lengths]
    )


def plot_heatmaps(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    specs = (
        ("actual_answer_exact_match", "Arithmetic exact match"),
        ("actual_answer_token_accuracy", "Arithmetic digit accuracy"),
        ("final_carry_accuracy", "Final carry accuracy"),
    )
    for axis, (key, title) in zip(axes, specs, strict=True):
        matrix = matrix_from_rows(rows, args.lengths, args.maximum_step, key)
        image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.scatter(
            [length for length in args.lengths],
            np.arange(len(args.lengths)),
            marker="x",
            color="white",
            s=28,
            linewidths=1.2,
            label="registered T(n)=n+1",
        )
        axis.set_title(title)
        axis.set_xlabel("loop step")
        axis.set_xticks(np.arange(0, args.maximum_step, 2), np.arange(1, args.maximum_step + 1, 2))
        axis.set_yticks(np.arange(len(args.lengths)), args.lengths)
        axis.set_ylabel("logical length n")
        axis.legend(frameon=False, loc="lower right")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(args.out_dir / "readout_heatmaps.png", dpi=180)
    figure.savefig(args.out_dir / "readout_heatmaps.pdf")
    plt.close(figure)


def plot_example_correctness(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis, logical_length in zip(axes.flat, args.example_lengths[-4:], strict=True):
        selected = [row for row in rows if int(row["length"]) == logical_length]
        matrix = np.asarray([json.loads(row["position_correct"]) for row in selected])
        axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
        axis.axhline(logical_length - 0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{logical_length}-bit all-ones + 1")
        axis.set_xlabel("answer position: carry/MSB to LSB")
        axis.set_ylabel("loop step")
        axis.set_yticks(np.arange(0, args.maximum_step, 2), np.arange(1, args.maximum_step + 1, 2))
    figure.savefig(args.out_dir / "long_carry_position_correctness.png", dpi=180)
    figure.savefig(args.out_dir / "long_carry_position_correctness.pdf")
    plt.close(figure)


def plot_position_accuracy(
    args: argparse.Namespace, position_rows: list[dict[str, Any]]
) -> None:
    selected_lengths = args.example_lengths[-4:]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis, logical_length in zip(axes.flat, selected_lengths, strict=True):
        indexed = {
            (int(row["step"]), int(row["position_from_lsb"])): float(row["accuracy"])
            for row in position_rows
            if int(row["length"]) == logical_length
        }
        # Reverse the stored LSB-to-carry order so the displayed x-axis follows
        # the actual target sequence: carry/MSB first, then toward the LSB.
        matrix = np.asarray(
            [
                [indexed[(step, position)] for position in range(logical_length, -1, -1)]
                for step in range(1, args.maximum_step + 1)
            ]
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        axis.axhline(logical_length - 0.5, color="white", linestyle="--", linewidth=1)
        axis.set_title(f"Random {logical_length}-bit additions")
        axis.set_xlabel("answer slot: carry/MSB to LSB")
        axis.set_ylabel("loop step")
        axis.set_yticks(
            np.arange(0, args.maximum_step, 2),
            np.arange(1, args.maximum_step + 1, 2),
        )
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(args.out_dir / "position_accuracy_front.png", dpi=180)
    figure.savefig(args.out_dir / "position_accuracy_front.pdf")
    plt.close(figure)


def summarize(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_length = []
    for length in args.lengths:
        group = [row for row in rows if int(row["length"]) == length]
        target = next(row for row in group if int(row["step"]) == length + 1)
        first_perfect = next(
            (int(row["step"]) for row in group if float(row["actual_answer_exact_match"]) == 1.0),
            None,
        )
        perfect_steps = [int(row["step"]) for row in group if float(row["actual_answer_exact_match"]) == 1.0]
        best = max(group, key=lambda row: (float(row["actual_answer_exact_match"]), float(row["actual_answer_token_accuracy"])))
        per_length.append(
            {
                "length": length,
                "target_step": length + 1,
                "target_arithmetic_exact_match": target["actual_answer_exact_match"],
                "target_arithmetic_digit_accuracy": target["actual_answer_token_accuracy"],
                "target_final_carry_accuracy": target["final_carry_accuracy"],
                "first_perfect_step": first_perfect,
                "perfect_steps": perfect_steps,
                "best_step": int(best["step"]),
                "best_exact_match": best["actual_answer_exact_match"],
            }
        )
    return {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "controller": str(args.controller) if args.controller is not None else None,
        "model": args.model_label,
        "loss_placement": "final answer-region CE only at each sample's T(n)=n+1",
        "shared_physical_layers": 3,
        "evaluated_steps": [1, args.maximum_step],
        "examples_per_length": args.examples,
        "per_length": per_length,
        "claim_boundary": "Readout localization only; no component intervention was performed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--model-label", default="official Addition adaptive-step seed0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--example-lengths", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--maximum-step", type=int, default=30)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=281001)
    parser.add_argument("--trace-length", type=int, default=20)
    parser.add_argument(
        "--trace-examples",
        type=int,
        default=3,
        help="number of examples retained for each final-carry value",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, position_rows = analyze_random_readouts(args)
    example_rows = analyze_long_carry_examples(args)
    trace_rows = analyze_random_example_trajectories(args)
    write_csv(args.out_dir / "readout_by_length_step.csv", rows)
    write_csv(args.out_dir / "position_accuracy.csv", position_rows)
    write_csv(args.out_dir / "long_carry_examples.csv", example_rows)
    write_csv(args.out_dir / "random_example_trajectories.csv", trace_rows)
    payload = summarize(args, rows)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plot_heatmaps(args, rows)
    plot_position_accuracy(args, position_rows)
    plot_example_correctness(args, example_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
