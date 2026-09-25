"""Causally separate initial input access from recurrent token replay in Parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from reasoning_loop.paper_length_telomere import (
    PaperLoopedTransformer,
    atomic_json_dump,
    generate_paper_batch,
    load_backbone,
    pick_device,
    write_csv,
)


CONDITIONS = (
    "registered_initial_only",
    "no_input",
    "legacy_every_loop",
    "single_clean_replay_after_endpoint",
    "single_shuffled_replay_after_endpoint",
)


def scheduled_input(
    model: PaperLoopedTransformer,
    inputs: torch.Tensor,
    shuffled_inputs: torch.Tensor,
    *,
    condition: str,
    step_index: int,
    endpoint_step: int,
) -> torch.Tensor:
    """Return the input signal for one explicitly named counterfactual schedule."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown input-path condition: {condition}")
    natural = model.input_embeddings(inputs, step_index=step_index)
    if condition == "registered_initial_only":
        return natural
    if condition == "no_input":
        return torch.zeros_like(natural)
    if condition == "legacy_every_loop":
        return model.input_embeddings(inputs, step_index=1)
    if step_index != endpoint_step + 1:
        return natural
    replay_inputs = (
        inputs
        if condition == "single_clean_replay_after_endpoint"
        else shuffled_inputs
    )
    return model.input_embeddings(replay_inputs, step_index=1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(20, 24, 32, 40, 64, 100)
    )
    parser.add_argument("--relative-start", type=int, default=-4)
    parser.add_argument("--relative-end", type=int, default=8)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026082701)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def answer_statistics(
    model: PaperLoopedTransformer,
    state: torch.Tensor,
    labels: torch.Tensor,
    position: int,
) -> tuple[float, float, float]:
    logits = model.decode(state).float()[:, position]
    rows = torch.arange(logits.shape[0], device=logits.device)
    correct = logits[rows, labels]
    competing = logits.clone()
    competing[rows, labels] = -torch.inf
    margin = correct - competing.max(dim=1).values
    probability = logits.softmax(dim=1)[rows, labels]
    predictions = logits.argmax(dim=1)
    return (
        float(predictions.eq(labels).sum().cpu()),
        float(margin.sum().cpu()),
        float(probability.sum().cpu()),
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.relative_start > args.relative_end:
        raise ValueError("relative-start must not exceed relative-end")
    device = pick_device(args.device)
    model, spec, payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "parity":
        raise ValueError("this ablation is defined only for Parity")
    if model.config.token_embedding_injection != "initial_only":
        raise ValueError("the primary comparison requires an input-once checkpoint")
    if model.config.position_embedding != "none":
        raise ValueError("the current causal labels assume the authoritative NoPE model")

    totals: dict[tuple[str, int, int], dict[str, float]] = {}
    for length_index, length in enumerate(args.lengths):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 1009 * length_index)
        remaining = args.examples
        while remaining:
            current_batch = min(args.batch_size, remaining)
            batch = generate_paper_batch(
                spec,
                batch_size=current_batch,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            labels = batch.targets[:, length]
            shuffled_inputs = batch.inputs.roll(shifts=1, dims=0)
            states = {
                condition: torch.zeros_like(model.read_in(batch.inputs))
                for condition in CONDITIONS
            }
            final_step = length + args.relative_end
            for step_index in range(1, final_step + 1):
                for condition in CONDITIONS:
                    injected = scheduled_input(
                        model,
                        batch.inputs,
                        shuffled_inputs,
                        condition=condition,
                        step_index=step_index,
                        endpoint_step=length,
                    )
                    states[condition] = model.recurrent_step(
                        states[condition], injected
                    )
                    relative_depth = step_index - length
                    if not args.relative_start <= relative_depth <= args.relative_end:
                        continue
                    correct, margin, probability = answer_statistics(
                        model, states[condition], labels, length
                    )
                    key = (condition, length, step_index)
                    values = totals.setdefault(
                        key,
                        {
                            "examples": 0.0,
                            "correct": 0.0,
                            "margin_sum": 0.0,
                            "probability_sum": 0.0,
                            "injection_square_sum": 0.0,
                            "injection_values": 0.0,
                        },
                    )
                    values["examples"] += current_batch
                    values["correct"] += correct
                    values["margin_sum"] += margin
                    values["probability_sum"] += probability
                    values["injection_square_sum"] += float(
                        injected.float().square().sum().cpu()
                    )
                    values["injection_values"] += injected.numel()
            remaining -= current_batch

    rows: list[dict[str, Any]] = []
    for (condition, length, step_index), values in sorted(totals.items()):
        rows.append(
            {
                "condition": condition,
                "length": length,
                "step": step_index,
                "relative_depth": step_index - length,
                "examples": int(values["examples"]),
                "accuracy": values["correct"] / values["examples"],
                "mean_oriented_margin": values["margin_sum"] / values["examples"],
                "mean_correct_probability": (
                    values["probability_sum"] / values["examples"]
                ),
                "input_injection_rms": (
                    values["injection_square_sum"] / values["injection_values"]
                )
                ** 0.5,
            }
        )
    summary = {
        "status": "complete",
        "analysis": "parity_input_path_causal_ablation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(payload["step"]),
        "backbone_seed": int(payload["seed"]),
        "frozen_backbone": True,
        "controller": None,
        "model": {
            "token_embedding_injection": model.config.token_embedding_injection,
            "position_embedding": model.config.position_embedding,
            "position_injection": model.config.position_injection,
        },
        "lengths": list(args.lengths),
        "relative_depth_window": [args.relative_start, args.relative_end],
        "examples_per_length": args.examples,
        "evaluation_seed": args.seed,
        "conditions": {
            "registered_initial_only": "checkpoint-declared schedule",
            "no_input": "remove the only token-entry event, including call 1",
            "legacy_every_loop": "counterfactually replay the clean token embedding every call",
            "single_clean_replay_after_endpoint": "one extra clean replay at call n+1",
            "single_shuffled_replay_after_endpoint": "one extra different-example replay at call n+1",
        },
        "evidence_boundary": (
            "This isolates access through the explicit input-addition edge. "
            "It does not remove token information already stored in the recurrent state."
        ),
    }
    return rows, summary


def plot_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    lengths = sorted({int(row["length"]) for row in rows})
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for axis, length in zip(axes.flat, lengths):
        selected = [row for row in rows if int(row["length"]) == length]
        for condition in CONDITIONS:
            curve = sorted(
                [row for row in selected if row["condition"] == condition],
                key=lambda row: int(row["relative_depth"]),
            )
            axis.plot(
                [int(row["relative_depth"]) for row in curve],
                [float(row["accuracy"]) for row in curve],
                marker="o",
                markersize=3,
                label=condition,
            )
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"n={length}")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("relative depth d=t-n")
    for axis in axes[:, 0]:
        axis.set_ylabel("accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Parity input-path causal ablation (frozen input-once backbone)")
    figure.tight_layout(rect=(0, 0.11, 1, 0.96))
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(args)
    write_csv(args.out_dir / "input_path_ablation.csv", rows)
    plot_rows(rows, args.out_dir / "input_path_ablation.png")
    summary["files"] = {
        "table": "input_path_ablation.csv",
        "plot": "input_path_ablation.png",
    }
    atomic_json_dump(summary, args.out_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
