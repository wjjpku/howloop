"""Canonical audit of natural D8L8 continuation from the exact recurrent state.

The audit deliberately separates two target semantics:

* endpoint hold: prediction equals the trained f^8(start) endpoint;
* successor continuation: prediction equals f^(8+r)(start), excluding graph
  cycles for which that target aliases the trained endpoint.

It also verifies that uninterrupted unrolling, cached-H8 continuation, and an
independently reconstructed aligned H8 produce numerically identical states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import cache_states_with_initial
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def continuation_counts(
    prediction: torch.Tensor,
    endpoint: torch.Tensor,
    successor_target: torch.Tensor,
) -> dict[str, int]:
    """Return alias-aware behavior counts for one continuation depth."""

    strict = successor_target.ne(endpoint)
    return {
        "examples": int(prediction.numel()),
        "strict_examples": int(strict.sum()),
        "hold_correct": int(prediction.eq(endpoint).sum()),
        "successor_correct": int(prediction.eq(successor_target).sum()),
        "strict_successor_correct": int(prediction[strict].eq(successor_target[strict]).sum()),
        "strict_hold_correct": int(prediction[strict].eq(endpoint[strict]).sum()),
        "strict_wrong": int(
            (prediction[strict].ne(successor_target[strict]) & prediction[strict].ne(endpoint[strict])).sum()
        ),
    }


def add_counts(receiver: dict[str, float], update: dict[str, int]) -> None:
    for key, value in update.items():
        receiver[key] += value


def rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize(accumulators: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for extra_loop, counts in sorted(accumulators.items()):
        rows.append(
            {
                "extra_loop": extra_loop,
                **{key: int(value) for key, value in counts.items()},
                "endpoint_hold_accuracy": rate(counts["hold_correct"], counts["examples"]),
                "aliased_successor_accuracy": rate(counts["successor_correct"], counts["examples"]),
                "strict_successor_accuracy": rate(
                    counts["strict_successor_correct"], counts["strict_examples"]
                ),
                "strict_hold_accuracy": rate(counts["strict_hold_correct"], counts["strict_examples"]),
                "strict_wrong_rate": rate(counts["strict_wrong"], counts["strict_examples"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    x = np.asarray([row["extra_loop"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.55), constrained_layout=True)
    axes[0].plot(x, [row["endpoint_hold_accuracy"] for row in rows], marker="o", label="Endpoint hold")
    axes[0].plot(x, [row["strict_successor_accuracy"] for row in rows], marker="s", label="Strict successor")
    axes[0].plot(x, [row["strict_wrong_rate"] for row in rows], marker="^", label="Other output")
    axes[0].set_xlabel("Extra shared-component calls after loop 8")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("A  Natural terminal-state behavior")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, [row["aliased_successor_accuracy"] for row in rows], marker="o", label="Observed moving-target accuracy")
    aliases = [1 - row["strict_examples"] / row["examples"] for row in rows]
    axes[1].plot(x, aliases, linestyle="--", marker="x", label="Endpoint/target collision rate")
    axes[1].set_xlabel("Extra shared-component calls after loop 8")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("B  Why the old periodic curve appeared")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026080901)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--examples-per-seed", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--extra-loops", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.examples_per_seed % args.batch_size:
        raise ValueError("examples-per-seed must be divisible by batch-size")
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    if cfg.block_schedule != "all_blocks":
        raise ValueError("this exact-equivalence audit currently requires all_blocks")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    accumulators = defaultdict(lambda: defaultdict(float))
    input_digest = hashlib.sha256()
    maximum_differences = {
        "cached_vs_full_h8": 0.0,
        "cached_continuation_vs_full": 0.0,
        "aligned_vs_natural_h8": 0.0,
        "aligned_continuation_vs_full": 0.0,
    }
    state_rmse = defaultdict(list)
    for seed_offset in args.evaluation_seeds:
        data_seed = args.seed + int(seed_offset)
        set_seed(data_seed)
        for _ in range(args.examples_per_seed // args.batch_size):
            tokens, targets, successors, start = fixed_depth_batch(
                cfg,
                args.batch_size,
                device,
                path_positions=cfg.max_depth + args.extra_loops,
            )
            input_digest.update(tensor_bytes(tokens))
            input_digest.update(tensor_bytes(successors))
            input_digest.update(tensor_bytes(start))

            full = cache_states_with_initial(
                model, tokens, loops=cfg.max_loops + args.extra_loops
            )
            cached = cache_states_with_initial(model, tokens, loops=cfg.max_loops)
            cached_h8 = cached[-1]
            maximum_differences["cached_vs_full_h8"] = max(
                maximum_differences["cached_vs_full_h8"],
                float((cached_h8 - full[cfg.max_loops]).abs().max()),
            )
            endpoint = targets[:, cfg.max_depth - 1]
            aligned = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=cfg.max_loops,
                phase_position=cfg.max_depth,
            )
            maximum_differences["aligned_vs_natural_h8"] = max(
                maximum_differences["aligned_vs_natural_h8"],
                float((aligned - cached_h8).abs().max()),
            )
            cached_state = cached_h8.clone()
            aligned_state = aligned.clone()
            for extra_loop in range(1, args.extra_loops + 1):
                loop_index = cfg.max_loops + extra_loop - 1
                cached_state = model.apply_loop(cached_state, loop_index=loop_index)
                aligned_state = model.apply_loop(aligned_state, loop_index=loop_index)
                full_state = full[cfg.max_loops + extra_loop]
                cached_difference = float((cached_state - full_state).abs().max())
                aligned_difference = float((aligned_state - full_state).abs().max())
                maximum_differences["cached_continuation_vs_full"] = max(
                    maximum_differences["cached_continuation_vs_full"], cached_difference
                )
                maximum_differences["aligned_continuation_vs_full"] = max(
                    maximum_differences["aligned_continuation_vs_full"], aligned_difference
                )
                state_rmse[extra_loop].append(
                    float((full_state.float() - full[cfg.max_loops].float()).square().mean().sqrt())
                )
                prediction = logits_from_raw_state(model, full_state).argmax(dim=-1)
                target = targets[:, cfg.max_depth + extra_loop - 1]
                add_counts(
                    accumulators[extra_loop], continuation_counts(prediction, endpoint, target)
                )

    rows = summarize(accumulators)
    for row in rows:
        row["hidden_rmse_from_h8_mean"] = float(np.mean(state_rmse[row["extra_loop"]]))
        row["hidden_rmse_from_h8_sem"] = float(
            np.std(state_rmse[row["extra_loop"]], ddof=1) / math.sqrt(len(state_rmse[row["extra_loop"]]))
        )
    write_csv(args.out_dir / "continuation_rows.csv", rows)
    plot(rows, args.out_dir / "natural_overloop_protocol_audit.pdf")

    equivalence_passed = all(value <= 1e-7 for value in maximum_differences.values())
    holding_passed = all(row["strict_hold_accuracy"] >= 0.99 for row in rows)
    successor_rejected = all(row["strict_successor_accuracy"] <= 0.01 for row in rows)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": checkpoint_payload["config"],
        "loss_placement": "final-only CE at loop 8",
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "evaluation_seed_base": args.seed,
        "evaluation_seed_offsets": list(args.evaluation_seeds),
        "examples": args.examples_per_seed * len(args.evaluation_seeds),
        "input_batch_sha256": input_digest.hexdigest(),
        "target_semantics": {
            "endpoint_hold": "prediction equals f^8(start)",
            "aliased_successor": "prediction equals f^(8+r)(start), including graph-cycle collisions",
            "strict_successor": "same successor target after excluding samples where f^(8+r)(start)=f^8(start)",
        },
        "maximum_absolute_hidden_differences": maximum_differences,
        "decisions": {
            "cached_full_and_aligned_protocols_equivalent": equivalence_passed,
            "terminal_endpoint_holding_supported": holding_passed,
            "strict_successor_continuation_supported": not successor_rejected,
            "old_periodic_curve_explained_by_target_aliasing": all(
                abs(
                    row["aliased_successor_accuracy"]
                    - (1 - row["strict_examples"] / row["examples"])
                )
                <= 0.01
                for row in rows
            ),
        },
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
