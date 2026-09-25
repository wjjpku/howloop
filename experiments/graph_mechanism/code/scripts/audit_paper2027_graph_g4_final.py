#!/usr/bin/env python3
"""Fail-closed integrity and numerical audit for the eligible G4 final tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aggregate-dir", default="aggregate_rerun_v1")
    parser.add_argument("--evaluation-dir", default="g4_evaluation_rerun_v1")
    parser.add_argument("--evaluation-manifest-prefix", default="evaluation_rerun_v1_seed")
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=list(range(100, 112)))
    args = parser.parse_args()

    aggregate_dir = args.root / args.aggregate_dir
    summary = read_json(aggregate_dir / "summary.json")
    assert summary["status"] == "complete"
    assert summary["protocol_id"] == "paper2027.graph.g4.disjoint_aggregate.v1"
    assert summary["evaluation_dir"] == args.evaluation_dir
    assert summary["evaluation_manifest_prefix"] == args.evaluation_manifest_prefix
    assert summary["included_backbone_seeds"] == args.expected_seeds
    assert summary["completed_backbone_count"] == len(args.expected_seeds)
    assert summary["registered_backbone_count"] == len(args.expected_seeds)
    assert summary["selection_rule"] == "none; every completed backbone and both controller replicas"
    assert summary["final_lock_excluded_from_backbone_and_controller_training"] is True

    final_hash = summary["final_test_lock_sha256"]
    selection_hash = summary["selection_lock_sha256"]
    assert sha256(args.root / "locks" / "final_test_permutations_512.pt") == final_hash
    assert sha256(args.root / "locks" / "selection_permutations_512.pt") == selection_hash

    for seed in args.expected_seeds:
        backbone = read_json(args.root / "manifests" / f"backbone_seed{seed}.json")
        controller = read_json(args.root / "manifests" / f"controller_seed{seed}.json")
        evaluation = read_json(args.root / "manifests" / f"{args.evaluation_manifest_prefix}{seed}.json")
        assert backbone["status"] == controller["status"] == evaluation["status"] == "complete"
        assert backbone["final_test_lock_sha256"] == controller["final_test_lock_sha256"] == evaluation["final_test_lock_sha256"] == final_hash
        assert backbone["selection_lock_sha256"] == controller["selection_lock_sha256"] == selection_hash
        assert evaluation["evaluation_dir"] == args.evaluation_dir
        assert evaluation["evaluation_manifest_prefix"] == args.evaluation_manifest_prefix
        for path_key, hash_key in (("raw_source", "raw_source_sha256"), ("controller_source", "controller_source_sha256"), ("runner_source", "runner_source_sha256")):
            source = Path(evaluation[path_key])
            assert source.is_file() and sha256(source) == evaluation[hash_key]

        base = args.root / args.evaluation_dir / f"seed{seed}"
        for relative in (
            "raw/summary.json", "rank48_seed1/full/summary.json", "rank48_seed2/full/summary.json",
            "rank48_seed1/executor_off/summary.json", "rank48_seed1/batch_shuffle/summary.json",
            "rank48_seed1/D_only/summary.json", "rank48_seed1/no_AB/summary.json",
            "rank48_seed1/identity_D/summary.json", "rank48_seed1/AB_only/summary.json",
            "rank48_seed1/no_bias/summary.json", "rank48_seed1/mean_D/summary.json",
            "rank48_seed1/shuffle_D/summary.json", "rank48_seed1/spectrum_matched_random_delta/summary.json",
        ):
            assert (base / relative).is_file(), (seed, relative)

    with (aggregate_dir / "seed_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["backbone_seed"]) for row in rows] == args.expected_seeds
    raw_auc = summary["raw_strict_successor_auc_9_128"]["point_estimate"]
    full_auc = summary["full_strict_successor_auc_9_128"]["point_estimate"]
    assert raw_auc < 0.005, raw_auc
    assert full_auc > 0.20, full_auc
    assert summary["full_strict_successor_auc_9_128"]["ci95_low"] > summary["raw_strict_successor_auc_9_128"]["ci95_high"]
    assert summary["controls_strict_successor_auc_9_128"]["executor_off"]["ci95_high"] < summary["full_strict_successor_auc_9_128"]["ci95_low"]
    assert summary["controls_strict_successor_auc_9_128"]["batch_shuffle"]["ci95_high"] < summary["full_strict_successor_auc_9_128"]["ci95_low"]

    print(json.dumps({
        "status": "PASS", "backbones": len(args.expected_seeds),
        "raw_successor_auc": raw_auc, "controller_successor_auc": full_auc,
        "controller_ci": [summary["full_strict_successor_auc_9_128"]["ci95_low"], summary["full_strict_successor_auc_9_128"]["ci95_high"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
