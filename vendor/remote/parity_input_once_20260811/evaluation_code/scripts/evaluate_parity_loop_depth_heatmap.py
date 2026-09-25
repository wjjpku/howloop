"""Evaluate dense Parity length-by-loop readout maps for a frozen backbone.

The script intentionally reports behavior, not a circuit claim.  It follows the
same fixed examples through every recurrent step and records both strict answer
exact match and the parity-token accuracy at the separator position.  An
optional J controller is evaluated on the identical examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--max-loop", type=int, default=60)
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=284001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reliable-threshold", type=float, default=0.9)
    parser.add_argument("--reliable-window", type=int, default=3)
    return parser.parse_args()


def exact_successes(logits: torch.Tensor, batch: Any) -> int:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(batch.targets) | ~batch.answer_mask
    return int(correct.all(dim=1).sum().item())


def parity_token_statistics(
    logits: torch.Tensor,
    batch: Any,
    answer_position: int,
) -> tuple[int, float, float]:
    answer_logits = logits[:, answer_position].float()
    targets = batch.targets[:, answer_position]
    predictions = answer_logits.argmax(dim=-1)
    correct = int(predictions.eq(targets).sum().item())
    probabilities = answer_logits.softmax(dim=-1)
    target_probability = float(
        probabilities.gather(1, targets[:, None]).sum().item()
    )
    correct_logits = answer_logits.gather(1, targets[:, None]).squeeze(1)
    competing_logits = answer_logits.clone()
    competing_logits.scatter_(1, targets[:, None], -torch.inf)
    signed_margin = correct_logits - competing_logits.max(dim=1).values
    return correct, target_probability, float(signed_margin.sum().item())


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "parity":
        raise ValueError("this evaluator is defined only for Parity")
    controller = None
    controller_payload = None
    if args.controller is not None:
        controller, controller_payload = load_controller(args.controller, device=device)
        if controller_payload["checkpoint"] != str(args.checkpoint):
            expected_remote = Path(controller_payload["checkpoint"]).name
            actual_local = args.checkpoint.name
            if expected_remote != actual_local:
                # Local paper-analysis copies retain an authoritative remote path
                # in controller metadata but may rename the copied backbone.
                model_signature = controller_payload["model"]
                if (
                    int(model_signature["d_model"]) != model.config.d_model
                    or int(model_signature["n_heads"]) != model.config.n_heads
                ):
                    raise ValueError("controller belongs to a different backbone")
        controller = ControllerView(controller, mode="full").to(device).eval()

    variants: dict[str, torch.nn.Module | None] = {"raw": None}
    if controller is not None:
        variants["J"] = controller
    totals: dict[tuple[str, int, int], dict[str, float]] = {}

    for length in range(args.min_length, args.max_length + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 1009 * length)
        remaining = args.examples
        while remaining:
            current_batch_size = min(args.batch_size, remaining)
            batch = generate_paper_batch(
                spec,
                batch_size=current_batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            token_embeddings = model.read_in(batch.inputs)
            states = {
                variant: torch.zeros_like(token_embeddings) for variant in variants
            }
            for loop in range(1, args.max_loop + 1):
                embedded = model.input_embeddings(batch.inputs, step_index=loop)
                for variant, live_controller in variants.items():
                    state = states[variant]
                    if live_controller is not None and loop > int(
                        controller_payload["anchor_step"]
                    ):
                        state = live_controller(state)
                    state = model.recurrent_step(state, embedded)
                    states[variant] = state
                    logits = model.decode(state).float()
                    parity_correct, probability_sum, margin_sum = (
                        parity_token_statistics(logits, batch, length)
                    )
                    values = totals.setdefault(
                        (variant, length, loop),
                        {
                            "examples": 0.0,
                            "exact_successes": 0.0,
                            "parity_correct": 0.0,
                            "correct_probability_sum": 0.0,
                            "parity_margin_sum": 0.0,
                        },
                    )
                    values["examples"] += current_batch_size
                    values["exact_successes"] += exact_successes(logits, batch)
                    values["parity_correct"] += parity_correct
                    values["correct_probability_sum"] += probability_sum
                    values["parity_margin_sum"] += margin_sum
            remaining -= current_batch_size

    rows: list[dict[str, Any]] = []
    for (variant, length, loop), values in sorted(totals.items()):
        examples = values["examples"]
        rows.append(
            {
                "variant": variant,
                "length": length,
                "target_loop": length,
                "loop": loop,
                "relative_loop": loop - length,
                "examples": int(examples),
                "exact_match": values["exact_successes"] / examples,
                "parity_token_accuracy": values["parity_correct"] / examples,
                "correct_parity_probability": (
                    values["correct_probability_sum"] / examples
                ),
                "mean_parity_margin": values["parity_margin_sum"] / examples,
            }
        )

    summary = summarize(
        rows,
        threshold=args.reliable_threshold,
        window=args.reliable_window,
        train_max_length=spec.train_max_length,
    )
    summary.update(
        {
            "status": "complete",
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": int(backbone_payload["step"]),
            "backbone_seed": int(backbone_payload["seed"]),
            "backbone_loss_placement": backbone_payload["supervision"],
            "model": {
                "d_model": model.config.d_model,
                "n_heads": model.config.n_heads,
                "shared_physical_layers": model.config.block_layers,
            },
            "controller": str(args.controller) if args.controller else None,
            "controller_anchor_step": (
                int(controller_payload["anchor_step"])
                if controller_payload is not None
                else None
            ),
            "evaluation": {
                "lengths": [args.min_length, args.max_length],
                "loops": [1, args.max_loop],
                "examples_per_cell": args.examples,
                "seed": args.seed,
                "device": str(device),
            },
            "claim_boundary": (
                "A diagonal or sloped accuracy frontier is evidence for a "
                "length-dependent recurrent computation schedule. It does not "
                "by itself identify the internal parity circuit or prove that "
                "one input bit is causally consumed per loop."
            ),
        }
    )
    return rows, summary


def summarize(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    window: int,
    train_max_length: int,
) -> dict[str, Any]:
    variants = sorted({str(row["variant"]) for row in rows})
    output: dict[str, Any] = {
        "reliable_threshold": threshold,
        "reliable_window": window,
        "variants": {},
    }
    for variant in variants:
        selected_variant = [row for row in rows if row["variant"] == variant]
        lengths = sorted({int(row["length"]) for row in selected_variant})
        per_length: dict[str, Any] = {}
        onset_pairs: list[tuple[int, int]] = []
        for length in lengths:
            selected = sorted(
                (row for row in selected_variant if row["length"] == length),
                key=lambda row: int(row["loop"]),
            )
            accuracies = [float(row["parity_token_accuracy"]) for row in selected]
            onset = None
            for index in range(0, len(selected) - window + 1):
                if min(accuracies[index : index + window]) >= threshold:
                    onset = int(selected[index]["loop"])
                    onset_pairs.append((length, onset))
                    break
            best_accuracy = max(accuracies)
            best_loop = min(
                int(row["loop"])
                for row in selected
                if float(row["parity_token_accuracy"]) == best_accuracy
            )
            by_loop = {int(row["loop"]): row for row in selected}
            target = by_loop.get(length)
            pre = by_loop.get(length - 1)
            post = by_loop.get(length + 1)
            survival = 0
            for loop in range(length, int(selected[-1]["loop"]) + 1):
                row = by_loop.get(loop)
                if row is None or float(row["parity_token_accuracy"]) < threshold:
                    break
                survival += 1
            per_length[str(length)] = {
                "split": "trained_length" if length <= train_max_length else "OOD_length",
                "first_reliable_loop": onset,
                "first_reliable_offset": onset - length if onset is not None else None,
                "best_loop": best_loop,
                "best_offset": best_loop - length,
                "best_parity_token_accuracy": best_accuracy,
                "pre_target_accuracy": (
                    float(pre["parity_token_accuracy"]) if pre else None
                ),
                "target_accuracy": (
                    float(target["parity_token_accuracy"]) if target else None
                ),
                "post_target_accuracy": (
                    float(post["parity_token_accuracy"]) if post else None
                ),
                "contiguous_reliable_loops_from_target": survival,
            }

        fit: dict[str, float | int | None]
        if len(onset_pairs) >= 2:
            x = np.asarray([pair[0] for pair in onset_pairs], dtype=np.float64)
            y = np.asarray([pair[1] for pair in onset_pairs], dtype=np.float64)
            slope, intercept = np.polyfit(x, y, deg=1)
            predicted = slope * x + intercept
            residual = float(np.square(y - predicted).sum())
            total = float(np.square(y - y.mean()).sum())
            fit = {
                "n_lengths": len(onset_pairs),
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": 1.0 - residual / total if total > 0 else None,
                "mean_absolute_offset_from_target": float(
                    np.abs(y - x).mean()
                ),
            }
        else:
            fit = {
                "n_lengths": len(onset_pairs),
                "slope": None,
                "intercept": None,
                "r_squared": None,
                "mean_absolute_offset_from_target": None,
            }
        output["variants"][variant] = {
            "reliable_onset_fit": fit,
            "per_length": per_length,
        }
    return output


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(rows: list[dict[str, Any]], out_dir: Path) -> None:
    variants = sorted({str(row["variant"]) for row in rows}, key=lambda x: x != "raw")
    lengths = sorted({int(row["length"]) for row in rows})
    loops = sorted({int(row["loop"]) for row in rows})
    figure, axes = plt.subplots(
        len(variants),
        2,
        figsize=(16, 5.5 * len(variants)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row_index, variant in enumerate(variants):
        selected = [row for row in rows if row["variant"] == variant]
        for column_index, (metric, title) in enumerate(
            (
                ("parity_token_accuracy", "parity answer-token accuracy"),
                ("exact_match", "strict answer-region exact match"),
            )
        ):
            matrix = np.full((len(lengths), len(loops)), np.nan)
            for row in selected:
                length_index = lengths.index(int(row["length"]))
                loop_index = loops.index(int(row["loop"]))
                matrix[length_index, loop_index] = float(row[metric])
            axis = axes[row_index, column_index]
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(loops[0] - 0.5, loops[-1] + 0.5, lengths[0] - 0.5, lengths[-1] + 0.5),
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            diagonal_max = min(lengths[-1], loops[-1])
            axis.plot(
                [max(lengths[0], loops[0]), diagonal_max],
                [max(lengths[0], loops[0]), diagonal_max],
                color="white",
                linestyle="--",
                linewidth=1.1,
                label="registered T(n)=n",
            )
            axis.axhline(20.5, color="white", linewidth=0.8, alpha=0.8)
            axis.set_title(f"{variant}: {title}")
            axis.set_xlabel("executed recurrent loops")
            axis.set_ylabel("input length n")
            axis.legend(loc="lower right", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    figure.suptitle(
        "Parity loop-depth map (white horizontal line: train-length boundary)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(out_dir / "parity_loop_depth_heatmaps.png", dpi=210)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.examples < 1 or args.batch_size < 1:
        raise ValueError("examples and batch-size must be positive")
    if args.min_length < 1 or args.max_length < args.min_length:
        raise ValueError("invalid length range")
    if args.max_loop < 1:
        raise ValueError("max-loop must be positive")
    if not 0.0 <= args.reliable_threshold <= 1.0:
        raise ValueError("reliable-threshold must be in [0,1]")
    if args.reliable_window < 1:
        raise ValueError("reliable-window must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(args)
    write_rows(args.out_dir / "loop_depth_metrics.csv", rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_heatmaps(rows, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
