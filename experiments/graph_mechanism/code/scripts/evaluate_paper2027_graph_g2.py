#!/usr/bin/env python3
"""Locked matched-interface test for the fresh final-only N8 D8L8 Graph cohort.

At a late raw call t, the receiver's answer residual is replaced by the call-8
answer residual from the *same graph* whose query start has been advanced
t-8 successors.  Thus that donor encodes the same current graph node as the
receiver, but at the trained interface phase.  One frozen shared executor is
then applied and tested against f^(t+1)(start).  It is a causal role test, not
an age scalar or an inversion claim.
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

from reasoning_loop.graph_path_depth_circuit import checkpoint_loss_mode, fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import GraphPathConfig, pick_device
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state, replace_answer_state
from scripts.evaluate_paper2027_graph_g1 import make_or_load_locked_test, write_csv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_config(cfg: GraphPathConfig, checkpoint: Path) -> None:
    expected = {
        "node_count": 8, "max_depth": 8, "d_model": 256, "n_heads": 4,
        "d_mlp": 1024, "n_layers": 2, "max_loops": 8,
        "block_schedule": "all_blocks", "inner_norm_style": "pre_layernorm",
    }
    mismatch = {key: (value, getattr(cfg, key)) for key, value in expected.items() if getattr(cfg, key) != value}
    if mismatch:
        raise ValueError(f"G2 requires registered final-only N8 D8L8 Graph: {mismatch}")
    if checkpoint_loss_mode(checkpoint) != "final_only":
        raise ValueError("G2 requires a final-only Graph backbone")


def initial(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model.token_embed(tokens) + model.pos_embed.unsqueeze(0)


def advance_starts(successors: torch.Tensor, starts: torch.Tensor, steps: int) -> torch.Tensor:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    result = starts
    for _ in range(steps):
        result = successors.gather(1, result[:, None]).squeeze(1)
    return result


def norm_matched_random_answer(state: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    """Random last-position residual with the donor's per-example norm."""
    answer = state[:, -1, :]
    random = torch.randn(answer.shape, dtype=answer.dtype, device=answer.device, generator=generator)
    random = random / random.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    random = random * answer.norm(dim=-1, keepdim=True)
    result = state.clone()
    result[:, -1, :] = random
    return result


def permute_graph_clusters(state: torch.Tensor, *, node_count: int) -> torch.Tensor:
    if state.shape[0] % node_count:
        raise ValueError("batch must contain complete graph clusters")
    graph_count = state.shape[0] // node_count
    if graph_count < 2:
        raise ValueError("cross-graph control requires at least two graph clusters")
    return state.reshape(graph_count, node_count, *state.shape[1:]).roll(1, 0).reshape_as(state)


def permute_within_graph(state: torch.Tensor, *, node_count: int) -> torch.Tensor:
    if state.shape[0] % node_count:
        raise ValueError("batch must contain complete graph clusters")
    return state.reshape(-1, node_count, *state.shape[1:]).roll(1, 1).reshape_as(state)


def counts(prediction: torch.Tensor, current: torch.Tensor, next_target: torch.Tensor) -> dict[str, int]:
    strict = next_target.ne(current)
    return {
        "examples": int(prediction.numel()),
        "strict_examples": int(strict.sum()),
        "next_successor_correct": int(prediction.eq(next_target).sum()),
        "current_hold_correct": int(prediction.eq(current).sum()),
        "strict_next_successor_correct": int(prediction[strict].eq(next_target[strict]).sum()),
        "strict_current_hold_correct": int(prediction[strict].eq(current[strict]).sum()),
        "strict_other": int((prediction[strict].ne(next_target[strict]) & prediction[strict].ne(current[strict])).sum()),
    }


def rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        bucket = total[(int(row["source_call"]), str(row["mode"]))]
        for key in ("examples", "strict_examples", "next_successor_correct", "current_hold_correct", "strict_next_successor_correct", "strict_current_hold_correct", "strict_other"):
            bucket[key] += float(row[key])
    result = []
    for (source_call, mode), values in sorted(total.items()):
        result.append({
            "source_call": source_call, "mode": mode,
            **{key: int(value) for key, value in values.items()},
            "next_successor_accuracy": rate(values["next_successor_correct"], values["examples"]),
            "current_hold_accuracy": rate(values["current_hold_correct"], values["examples"]),
            "strict_next_successor_accuracy": rate(values["strict_next_successor_correct"], values["strict_examples"]),
            "strict_current_hold_accuracy": rate(values["strict_current_hold_correct"], values["strict_examples"]),
            "strict_other_rate": rate(values["strict_other"], values["strict_examples"]),
        })
    return result


def plot(summary: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    modes = sorted({str(row["mode"]) for row in summary})
    for mode in modes:
        subset = [row for row in summary if row["mode"] == mode]
        axis.plot([row["source_call"] for row in subset], [row["strict_next_successor_accuracy"] for row in subset], marker="o", label=mode)
    axis.set(xlabel="late receiver call t; score after one executor", ylabel="strict next-successor accuracy", ylim=(-.03, 1.03))
    axis.grid(alpha=.2); axis.legend(frameon=False, ncol=2, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220); figure.savefig(path.with_suffix(".pdf")); plt.close(figure)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    verify_config(cfg, args.checkpoint)
    source_calls = sorted(set(args.source_calls))
    if not source_calls or min(source_calls) < cfg.max_loops:
        raise ValueError("source_calls must be at least the trained call 8")
    if args.batch_size % cfg.node_count:
        raise ValueError("batch_size must contain complete graph clusters")
    locked = make_or_load_locked_test(cfg=cfg, path=args.locked_test, permutations=args.permutations, seed=args.test_seed)
    generator = torch.Generator(device=device); generator.manual_seed(args.random_seed)
    rows: list[dict[str, Any]] = []
    graph_batch = args.batch_size // cfg.node_count
    for first in range(0, args.permutations, graph_batch):
        last = min(args.permutations, first + graph_batch)
        successors = locked["successors"][first:last].repeat_interleave(cfg.node_count, dim=0).to(device)
        starts = torch.arange(cfg.node_count, device=device).repeat(last - first)
        receiver_tokens, receiver_targets, _, _ = fixed_depth_batch(cfg, successors.shape[0], device, path_positions=max(source_calls) + 1, successors=successors, start=starts)
        receiver = initial(model, receiver_tokens)
        for call in range(1, max(source_calls) + 1):
            receiver = model.apply_loop(receiver, loop_index=call - 1)
            if call not in source_calls:
                continue
            matched_start = advance_starts(successors, starts, call - cfg.max_loops)
            wrong_start = advance_starts(successors, starts, call - cfg.max_loops + 1)
            donor_tokens, _, _, _ = fixed_depth_batch(cfg, successors.shape[0], device, path_positions=cfg.max_loops, successors=successors, start=matched_start)
            wrong_tokens, _, _, _ = fixed_depth_batch(cfg, successors.shape[0], device, path_positions=cfg.max_loops, successors=successors, start=wrong_start)
            donor, wrong = initial(model, donor_tokens), initial(model, wrong_tokens)
            for donor_call in range(cfg.max_loops):
                donor = model.apply_loop(donor, loop_index=donor_call)
                wrong = model.apply_loop(wrong, loop_index=donor_call)
            candidates = {
                "raw_late_next": receiver,
                "matched_young": replace_answer_state(receiver, donor),
                "wrong_current": replace_answer_state(receiver, wrong),
                "shuffled_donor": replace_answer_state(receiver, permute_within_graph(donor, node_count=cfg.node_count)),
                "cross_graph": replace_answer_state(receiver, permute_graph_clusters(donor, node_count=cfg.node_count)),
                "norm_matched_random": replace_answer_state(receiver, norm_matched_random_answer(donor, generator=generator)),
                "immediate_readout": replace_answer_state(receiver, donor),
            }
            current, next_target = receiver_targets[:, call - 1], receiver_targets[:, call]
            for mode, interface in candidates.items():
                output = interface if mode == "immediate_readout" else model.apply_loop(interface, loop_index=call)
                prediction = logits_from_raw_state(model, output).argmax(dim=-1)
                for local_graph in range(last - first):
                    begin, end = local_graph * cfg.node_count, (local_graph + 1) * cfg.node_count
                    rows.append({"permutation": first + local_graph, "source_call": call, "mode": mode, **counts(prediction[begin:end], current[begin:end], next_target[begin:end])})
    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "permutation_clusters.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", summary)
    plot(summary, args.out_dir / "matched_interface_curve.png")
    result = {
        "status": "complete", "protocol_id": "paper2027.graph.g2.matched_interface.v1",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_step": checkpoint_payload.get("step"), "config": asdict(cfg),
        "loss_placement": "final-only CE at call 8", "locked_test": str(args.locked_test), "locked_test_sha256": sha256(args.locked_test),
        "test_seed": args.test_seed, "permutations": args.permutations, "starts_per_permutation": cfg.node_count, "top_level_cluster": "graph permutation",
        "source_calls": source_calls, "interface": "same graph; donor query advanced t-8 then raw call-8 answer residual replaces receiver answer residual", "aggregate_rows": summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--locked-test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-calls", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--test-seed", type=int, default=2026093001)
    parser.add_argument("--random-seed", type=int, default=2026094001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
