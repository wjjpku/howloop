#!/usr/bin/env python3
"""Final-lock phenotype evaluation for a G4 graph-disjoint backbone."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_temporal_intervention import cache_raw_states, logits_from_raw_state
from reasoning_loop.paper2027_graph_g4_backbone import PROTOCOL_ID
from reasoning_loop.paper2027_graph_g4_protocol import load_unique_lock, sha256
from scripts.evaluate_paper2027_graph_g1 import _counts, plot, summarize_clusters, write_csv


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model, cfg, checkpoint = load_checkpoint(args.checkpoint, torch.device(args.device))
    if checkpoint.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("G4 evaluation requires a registered graph-disjoint checkpoint")
    lock = load_unique_lock(args.final_test_lock)
    if lock["role"] != "final_test" or lock["node_count"] != cfg.node_count:
        raise ValueError("evaluation must use the registered final N8 lock")
    expected_sha = checkpoint.get("excluded_locks", {}).get("final_test_sha256")
    observed_sha = sha256(args.final_test_lock)
    if expected_sha != observed_sha:
        raise ValueError("the evaluated final lock was not the lock excluded during backbone training")
    if args.max_call < cfg.max_loops or args.batch_size % cfg.node_count:
        raise ValueError("max call must include 8 and batch must preserve graph clusters")
    successors_all = lock["successors"]
    if not isinstance(successors_all, torch.Tensor):
        raise ValueError("lock lacks successors")
    rows: list[dict[str, Any]] = []
    graphs = int(lock["permutations"])
    for first in range(0, graphs, args.batch_size // cfg.node_count):
        last = min(graphs, first + args.batch_size // cfg.node_count)
        successors = successors_all[first:last].repeat_interleave(cfg.node_count, dim=0).to(args.device)
        starts = torch.arange(cfg.node_count, device=args.device).repeat(last - first)
        tokens, targets, _, _ = fixed_depth_batch(
            cfg, successors.shape[0], torch.device(args.device), path_positions=args.max_call,
            successors=successors, start=starts,
        )
        states = cache_raw_states(model, tokens, max_loop=args.max_call)
        endpoint = targets[:, cfg.max_depth - 1]
        for call, state in enumerate(states, start=1):
            prediction = logits_from_raw_state(model, state).argmax(dim=-1)
            successor = targets[:, call - 1]
            for local_graph in range(last - first):
                begin, end = local_graph * cfg.node_count, (local_graph + 1) * cfg.node_count
                rows.append({"permutation": first + local_graph, "call": call,
                             **_counts(prediction=prediction[begin:end], endpoint=endpoint[begin:end], successor=successor[begin:end])})
    summary_rows = summarize_clusters(rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.out_dir / "permutation_clusters.csv", rows)
    write_csv(args.out_dir / "aggregate_calls.csv", summary_rows)
    plot(summary_rows, args.out_dir / "continuation_curve.png")
    result = {
        "status": "complete", "protocol_id": "paper2027.graph.g4.final_lock_phenotype.v1",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"), "config": asdict(cfg),
        "loss_placement": "final-only CE at call 8", "final_test_lock": str(args.final_test_lock),
        "final_test_lock_sha256": observed_sha, "excluded_during_backbone_training": True,
        "permutations": graphs, "starts_per_permutation": cfg.node_count,
        "top_level_cluster": "unique graph permutation", "max_call": args.max_call,
        "aggregate_rows": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--final-test-lock", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-call", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), sort_keys=True))


if __name__ == "__main__":
    main()
