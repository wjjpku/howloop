"""Evaluate Parity only in a narrow band around the registered t=n readout.

The recurrent state still advances from loop 1, but logits are decoded only for
|t-n| <= half_width.  Raw and J states use identical examples and are advanced
in one concatenated backbone call per loop for efficiency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--length-step", type=int, default=1)
    parser.add_argument("--max-loop", type=int, default=500)
    parser.add_argument("--half-width", type=int, default=10)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=285001)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def statistics_from_answer_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    strict = predictions.eq(targets).all(dim=1)
    parity_logits = logits[:, 0].float()
    parity_targets = targets[:, 0]
    parity_predictions = parity_logits.argmax(dim=-1)
    probabilities = parity_logits.softmax(dim=-1)
    correct_logits = parity_logits.gather(1, parity_targets[:, None]).squeeze(1)
    competing = parity_logits.clone()
    competing.scatter_(1, parity_targets[:, None], -torch.inf)
    return {
        "examples": float(logits.shape[0]),
        "strict_successes": float(strict.sum().item()),
        "parity_successes": float(parity_predictions.eq(parity_targets).sum().item()),
        "correct_probability_sum": float(
            probabilities.gather(1, parity_targets[:, None]).sum().item()
        ),
        "margin_sum": float((correct_logits - competing.max(dim=1).values).sum().item()),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not (1 <= args.min_length <= args.max_length):
        raise ValueError("invalid length range")
    if args.length_step < 1 or args.max_loop < 1 or args.half_width < 0:
        raise ValueError("length-step/max-loop/half-width are invalid")
    if args.examples < 1 or args.batch_size < 1:
        raise ValueError("examples and batch-size must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(args.checkpoint, device=device)
    if spec.name != "parity":
        raise ValueError("this evaluator is defined only for Parity")
    controller, controller_payload = load_controller(args.controller, device=device)
    controller = ControllerView(controller, mode="full").to(device).eval()
    anchor_step = int(controller_payload["anchor_step"])
    if int(controller_payload["model"]["d_model"]) != model.config.d_model:
        raise ValueError("controller and backbone d_model differ")

    lengths = list(range(args.min_length, args.max_length + 1, args.length_step))
    manifest: dict[str, Any] = {
        "status": "running",
        "task": "Parity t approximately n diagonal band",
        "checkpoint": str(args.checkpoint),
        "controller": str(args.controller),
        "checkpoint_step": int(backbone_payload["step"]),
        "backbone_seed": int(backbone_payload["seed"]),
        "backbone_loss_placement": backbone_payload["supervision"],
        "controller_anchor_step": anchor_step,
        "lengths": lengths,
        "maximum_loop": args.max_loop,
        "half_width": args.half_width,
        "examples_per_cell": args.examples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "metric": "strict exact match on the complete answer region and parity-token accuracy",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    totals: dict[tuple[str, int, int], dict[str, float]] = {}
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for length_index, length in enumerate(lengths, start=1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + 1009 * length)
        first_scored_loop = max(1, length - args.half_width)
        last_scored_loop = min(args.max_loop, length + args.half_width)
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
            raw_state = torch.zeros_like(token_embeddings)
            controlled_state = torch.zeros_like(token_embeddings)
            answer_targets = batch.targets[:, length:]
            for loop in range(1, last_scored_loop + 1):
                embedded = model.input_embeddings(batch.inputs, step_index=loop)
                doubled_embeddings = torch.cat((embedded, embedded), dim=0)
                if loop > anchor_step:
                    controlled_state = controller(controlled_state)
                combined_state = torch.cat((raw_state, controlled_state), dim=0)
                combined_state = model.recurrent_step(combined_state, doubled_embeddings)
                raw_state, controlled_state = combined_state.split(current_batch_size, dim=0)
                if loop < first_scored_loop:
                    continue
                for variant, state in (("raw", raw_state), ("J", controlled_state)):
                    answer_logits = model.read_out(state[:, length:, :]).float()
                    batch_stats = statistics_from_answer_logits(answer_logits, answer_targets)
                    values = totals.setdefault(
                        (variant, length, loop),
                        {
                            "examples": 0.0,
                            "strict_successes": 0.0,
                            "parity_successes": 0.0,
                            "correct_probability_sum": 0.0,
                            "margin_sum": 0.0,
                        },
                    )
                    for key, value in batch_stats.items():
                        values[key] += value
            remaining -= current_batch_size

        elapsed = time.monotonic() - started
        progress = {
            "event": "length_complete",
            "length": length,
            "length_index": length_index,
            "length_count": len(lengths),
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_memory_mib": (
                round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
                if device.type == "cuda"
                else None
            ),
        }
        print(json.dumps(progress), flush=True)

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
                "exact_match": values["strict_successes"] / examples,
                "parity_token_accuracy": values["parity_successes"] / examples,
                "correct_parity_probability": values["correct_probability_sum"] / examples,
                "mean_parity_margin": values["margin_sum"] / examples,
            }
        )
    with (args.out_dir / "diagonal_band_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.monotonic() - started
    manifest.update(
        {
            "status": "complete",
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
            "row_count": len(rows),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
