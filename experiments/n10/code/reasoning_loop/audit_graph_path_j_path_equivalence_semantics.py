"""Machine-readable audit of H1-start F/J state and graph-target semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import (
    _execute_word,
    _natural_h1,
    action_semantics,
    sample_equivalent_word_pair,
    single_rollback_word_from_h1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=829801)
    parser.add_argument("--random-pairs-per-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=str(payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    bank = bank.frozen()
    rng = np.random.default_rng(args.seed + 1)
    symbolic_checks = 0
    pair_distinct = 0
    per_k: dict[str, dict[str, int]] = {}
    paired_back_counts = tuple(range(2, 25))
    for back_count in paired_back_counts:
        mandatory_counts = {source_age: 0 for source_age in range(2, 9)}
        for index in range(args.random_pairs_per_k):
            mandatory = 2 + index % 7
            left, right = sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=mandatory,
            )
            left_semantics = action_semantics(left)
            right_semantics = action_semantics(right)
            for semantics in (left_semantics, right_semantics):
                assert semantics.end_age == 8
                assert semantics.forward_count == back_count + 7
                assert semantics.back_count == back_count
                assert min(semantics.age_path) >= 1 and max(semantics.age_path) <= 8
                assert mandatory in semantics.rollback_sources
                symbolic_checks += 1
            mandatory_counts[mandatory] += 2
            pair_distinct += int(left != right)
        per_k[str(back_count)] = {
            "pairs": args.random_pairs_per_k,
            "distinct_pairs": pair_distinct - sum(
                value["distinct_pairs"] for value in per_k.values()
            ),
            "forward_count": back_count + 7,
            "mandatory_source_min_count": min(mandatory_counts.values()),
            "mandatory_source_max_count": max(mandatory_counts.values()),
        }
    one_j = {
        str(source_age): list(action_semantics(single_rollback_word_from_h1(source_age)).rollback_sources)
        for source_age in range(2, 9)
    }

    tokens, _, successors, start = fixed_depth_batch(
        cfg, args.batch_size, device, path_positions=cfg.max_depth
    )
    h1 = _natural_h1(model, tokens)
    h1_current = advance_nodes(successors, start, steps=1)
    positions = tuple(range(cfg.seq_len))
    empirical = []
    for back_count in (2, 5, 8, 12, 24):
        left, right = sample_equivalent_word_pair(
            rng=rng, back_count=back_count, mandatory_source_age=8
        )
        left_state, _, left_target = _execute_word(
            model=model,
            bank=bank,
            initial_state=h1,
            initial_current=h1_current,
            successors=successors,
            actions=left,
            positions=positions,
        )
        right_state, _, right_target = _execute_word(
            model=model,
            bank=bank,
            initial_state=h1,
            initial_current=h1_current,
            successors=successors,
            actions=right,
            positions=positions,
        )
        oracle = advance_nodes(successors, start, steps=back_count + 8)
        empirical.append(
            {
                "back_count": back_count,
                "left_right_target_equal": bool(left_target.eq(right_target).all()),
                "target_matches_successor_power_k_plus_8": bool(left_target.eq(oracle).all()),
                "left_final_age": action_semantics(left).end_age,
                "right_final_age": action_semantics(right).end_age,
                "final_state_relative_difference": float(
                    (left_state.float() - right_state.float()).norm()
                    / right_state.float().norm().clamp_min(1e-12)
                ),
            }
        )
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "initial_state": "natural H1 from raw tokens",
        "state_rule": "F increments logical phase; J decrements logical phase; all words stay in H1..H8",
        "target_rule": "graph current advances only under F; k J calls imply k+8 graph hops from the raw start",
        "loss_rule": "trainer applies final graph CE at semantic H8 only",
        "symbolic_checks": symbolic_checks,
        "all_pairs_distinct": pair_distinct == len(paired_back_counts) * args.random_pairs_per_k,
        "per_k": per_k,
        "one_J_rollback_sources": one_j,
        "empirical": empirical,
        "all_empirical_targets_valid": all(
            row["left_right_target_equal"]
            and row["target_matches_successor_power_k_plus_8"]
            for row in empirical
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
