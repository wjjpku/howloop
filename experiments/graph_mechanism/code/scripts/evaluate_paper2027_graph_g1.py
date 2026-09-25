#!/usr/bin/env python3
"""Locked-test G1 phenotype evaluation for final-only N8 D8L8 Graph models.

The top-level test cluster is a graph permutation, not an individual start
node.  The locked test set contains 512 independently sampled permutations and
all eight possible starts per permutation.  This makes post-horizon estimates
and later hierarchical bootstrap intervals well-defined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from reasoning_loop.graph_path_depth_circuit import (
    checkpoint_loss_mode,
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device
from reasoning_loop.graph_path_temporal_intervention import (
    cache_raw_states,
    logits_from_raw_state,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_or_load_locked_test(
    *,
    cfg: GraphPathConfig,
    path: Path,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Create an immutable graph-permutation test set, or validate its lock."""
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "node_count": cfg.node_count,
            "max_depth": cfg.max_depth,
            "permutations": permutations,
            "seed": seed,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"locked Graph test metadata mismatch: expected={expected}, observed={observed}"
            )
        successors = payload.get("successors")
        if not isinstance(successors, torch.Tensor) or successors.shape != (
            permutations,
            cfg.node_count,
        ):
            raise ValueError("locked Graph test has invalid successor tensor")
        return payload

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    successors = torch.rand(
        permutations, cfg.node_count, generator=generator, dtype=torch.float32
    ).argsort(dim=-1).to(torch.long)
    payload = {
        "kind": "paper2027_graph_g1_locked_test",
        "node_count": cfg.node_count,
        "max_depth": cfg.max_depth,
        "permutations": permutations,
        "seed": seed,
        "successors": successors,
        "starts_per_permutation": list(range(cfg.node_count)),
    }
    _save_atomic(payload, path)
    return payload


def _counts(
    *, prediction: torch.Tensor, endpoint: torch.Tensor, successor: torch.Tensor
) -> dict[str, int]:
    strict = successor.ne(endpoint)
    return {
        "examples": int(prediction.numel()),
        "strict_examples": int(strict.sum()),
        "endpoint_hold_correct": int(prediction.eq(endpoint).sum()),
        "moving_successor_correct": int(prediction.eq(successor).sum()),
        "strict_successor_correct": int(prediction[strict].eq(successor[strict]).sum()),
        "strict_endpoint_hold_correct": int(prediction[strict].eq(endpoint[strict]).sum()),
        "strict_other": int(
            (prediction[strict].ne(successor[strict]) & prediction[strict].ne(endpoint[strict])).sum()
        ),
    }


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accumulators: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        receiver = accumulators[int(row["call"])]
        for key in (
            "examples",
            "strict_examples",
            "endpoint_hold_correct",
            "moving_successor_correct",
            "strict_successor_correct",
            "strict_endpoint_hold_correct",
            "strict_other",
        ):
            receiver[key] += float(row[key])
    result: list[dict[str, Any]] = []
    for call, counts in sorted(accumulators.items()):
        result.append(
            {
                "call": call,
                **{key: int(value) for key, value in counts.items()},
                "endpoint_hold_accuracy": _rate(
                    counts["endpoint_hold_correct"], counts["examples"]
                ),
                "moving_successor_accuracy": _rate(
                    counts["moving_successor_correct"], counts["examples"]
                ),
                "strict_successor_accuracy": _rate(
                    counts["strict_successor_correct"], counts["strict_examples"]
                ),
                "strict_endpoint_hold_accuracy": _rate(
                    counts["strict_endpoint_hold_correct"], counts["strict_examples"]
                ),
                "strict_other_rate": _rate(
                    counts["strict_other"], counts["strict_examples"]
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict[str, Any]], path: Path) -> None:
    calls = [int(row["call"]) for row in summary]
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for key, label in (
        ("endpoint_hold_accuracy", "hold f^8(start)"),
        ("strict_successor_accuracy", "strict f^t(start)"),
        ("strict_other_rate", "strict other"),
    ):
        axis.plot(calls, [float(row[key]) for row in summary], label=label)
    axis.axvline(8, color="black", linestyle="--", linewidth=1, label="trained call")
    axis.set(xlabel="recurrent call t", ylabel="rate", ylim=(-0.03, 1.03))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    expected = {
        "node_count": 8,
        "max_depth": 8,
        "d_model": 256,
        "n_heads": 4,
        "d_mlp": 1024,
        "n_layers": 2,
        "max_loops": 8,
        "block_schedule": "all_blocks",
        "inner_norm_style": "pre_layernorm",
    }
    observed = {key: getattr(cfg, key) for key in expected}
    mismatch = {key: (expected[key], observed[key]) for key in expected if expected[key] != observed[key]}
    if mismatch:
        raise ValueError(f"G1 requires the registered N8 D8L8 protocol: {mismatch}")
    if checkpoint_loss_mode(args.checkpoint) != "final_only":
        raise ValueError("G1 requires a final-only Graph backbone")
    if args.max_call < cfg.max_loops:
        raise ValueError("max_call must include the trained call")
    if args.batch_size % cfg.node_count:
        raise ValueError("batch_size must be divisible by node_count so graph clusters stay intact")

    locked = make_or_load_locked_test(
        cfg=cfg,
        path=args.locked_test,
        permutations=args.permutations,
        seed=args.test_seed,
    )
    successors_all = locked["successors"]
    rows: list[dict[str, Any]] = []
    for first in range(0, args.permutations, args.batch_size // cfg.node_count):
        last = min(args.permutations, first + args.batch_size // cfg.node_count)
        successors = successors_all[first:last].repeat_interleave(cfg.node_count, dim=0).to(device)
        starts = torch.arange(cfg.node_count).repeat(last - first).to(device)
        tokens, targets, _, _ = fixed_depth_batch(
            cfg,
            successors.shape[0],
            device,
            path_positions=args.max_call,
            successors=successors,
            start=starts,
        )
        states = cache_raw_states(model, tokens, max_loop=args.max_call)
        endpoint = targets[:, cfg.max_depth - 1]
        for call, state in enumerate(states, start=1):
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            successor = targets[:, call - 1]
            for local_graph in range(last - first):
                begin, end = local_graph * cfg.node_count, (local_graph + 1) * cfg.node_count
                rows.append(
                    {
                        "permutation": first + local_graph,
                        "call": call,
                        **_counts(
                            prediction=prediction[begin:end],
                            endpoint=endpoint[begin:end],
                            successor=successor[begin:end],
                        ),
                    }
                )
    summary_rows = summarize_clusters(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "permutation_clusters.csv", rows)
    write_csv(args.out_dir / "aggregate_calls.csv", summary_rows)
    plot(summary_rows, args.out_dir / "continuation_curve.png")
    result = {
        "status": "complete",
        "protocol_id": "paper2027.graph.g1.phenotype.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "config": asdict(cfg),
        "loss_placement": "final-only CE at call 8",
        "locked_test": str(args.locked_test),
        "locked_test_sha256": sha256(args.locked_test),
        "test_seed": args.test_seed,
        "permutations": args.permutations,
        "starts_per_permutation": cfg.node_count,
        "top_level_cluster": "graph permutation",
        "max_call": args.max_call,
        "aggregate_rows": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locked-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--test-seed", type=int, default=2026093001)
    parser.add_argument("--max-call", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
