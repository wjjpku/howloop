#!/usr/bin/env python3
"""Aggregate the fully graph-disjoint G4 cohort with nested clustering.

The primary statistical unit is the independently trained backbone.  Within a
backbone, controller replicas and then graph permutations are resampled.  No
endpoint qualification is applied: every completed backbone/controller pair
is included in the denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_recorded_source(payload: dict[str, Any], *, path_key: str, hash_key: str, context: str) -> None:
    """Fail closed if an input source was changed after its manifest was made."""
    raw_path, expected = payload.get(path_key), payload.get(hash_key)
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"{context} lacks {path_key}/{hash_key}")
    path = Path(raw_path)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{context} source hash does not match {path_key}")


def require_raw_final_lock_summary(path: Path, *, final_lock_sha: str, checkpoint_sha: str, context: str) -> None:
    if not path.is_file():
        raise ValueError(f"{context} lacks raw summary")
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or payload.get("final_test_lock_sha256") != final_lock_sha:
        raise ValueError(f"{context} raw evaluation is not bound to the final lock")
    if payload.get("checkpoint_sha256") != checkpoint_sha or payload.get("excluded_during_backbone_training") is not True:
        raise ValueError(f"{context} raw evaluation does not certify its frozen, held-out backbone")


def require_controlled_final_lock_summary(path: Path, *, final_lock_sha: str, checkpoint_sha: str, context: str) -> None:
    if not path.is_file():
        raise ValueError(f"{context} lacks controlled summary")
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or payload.get("locked_test_sha256") != final_lock_sha:
        raise ValueError(f"{context} controlled evaluation is not bound to the final lock")
    if payload.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{context} controlled evaluation changed the frozen backbone")
    training = payload.get("controller_training")
    excluded = training.get("excluded_locks") if isinstance(training, dict) else None
    if not isinstance(excluded, dict) or final_lock_sha not in set(excluded.values()):
        raise ValueError(f"{context} controller payload does not certify final-lock exclusion")


def per_permutation_auc(
    rows: Sequence[dict[str, str]], *, metric: str, first_call: int = 9, last_call: int = 128
) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        call = int(row["call"])
        if first_call <= call <= last_call:
            denominator = float(row["strict_examples"] if metric.startswith("strict_") else row["examples"])
            if denominator:
                grouped[int(row["permutation"])].append(float(row[metric]) / denominator)
    if not grouped:
        raise ValueError(f"no finite values for {metric}")
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite graph metric {metric}")
    return values


def call8_accuracy(rows: Sequence[dict[str, str]]) -> float:
    selected = [row for row in rows if int(row["call"]) == 8]
    if not selected:
        raise ValueError("evaluation lacks call 8")
    return sum(float(row["moving_successor_correct"]) for row in selected) / sum(float(row["examples"]) for row in selected)


def nested_bootstrap(
    per_seed_replica_graph: Sequence[np.ndarray], *, draws: int, seed: int
) -> dict[str, float | int]:
    """Bootstrap backbone, controller replica, and graph clusters in that order."""
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    if not per_seed_replica_graph:
        raise ValueError("need at least one backbone")
    arrays = [np.asarray(value, dtype=float) for value in per_seed_replica_graph]
    if any(value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1 for value in arrays):
        raise ValueError("each seed must provide [replica, graph] values")
    generator = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    n_seed = len(arrays)
    for draw in range(draws):
        total = 0.0
        for chosen_seed in generator.integers(0, n_seed, size=n_seed):
            values = arrays[int(chosen_seed)]
            chosen_replica = int(generator.integers(0, values.shape[0]))
            graph = generator.integers(0, values.shape[1], size=values.shape[1])
            total += float(values[chosen_replica, graph].mean())
        samples[draw] = total / n_seed
    point = float(np.mean([value.mean() for value in arrays]))
    return {"point_estimate": point, "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)), "bootstrap_draws": draws,
            "backbone_count": n_seed, "controller_replicas_per_backbone": int(arrays[0].shape[0]),
            "graphs_per_replica": int(arrays[0].shape[1])}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty aggregate")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-dir",
        default="g4_evaluation",
        help="named evaluation subtree; retries use a fresh subtree rather than overwriting a failed run",
    )
    parser.add_argument(
        "--evaluation-manifest-prefix",
        default="evaluation_seed",
        help="manifest prefix paired with --evaluation-dir",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026081203)
    args = parser.parse_args()

    manifests: dict[int, dict[str, Any]] = {}
    for path in sorted((args.root / "manifests").glob("backbone_seed*.json")):
        payload = json.loads(path.read_text())
        manifests[int(payload["backbone_seed"])] = payload
    completed = sorted(seed for seed, payload in manifests.items() if payload.get("status") == "complete")
    if not completed:
        raise ValueError("no complete G4 backbone manifests")
    final_lock = args.root / "locks" / "final_test_permutations_512.pt"
    selection_lock = args.root / "locks" / "selection_permutations_512.pt"
    if not final_lock.is_file() or not selection_lock.is_file():
        raise FileNotFoundError("G4 aggregate requires both immutable lock files")
    final_lock_sha, selection_lock_sha = sha256(final_lock), sha256(selection_lock)
    seed_rows: list[dict[str, Any]] = []
    primary_successor: list[np.ndarray] = []
    primary_hold: list[np.ndarray] = []
    control_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed in completed:
        backbone_manifest = manifests[seed]
        if backbone_manifest.get("protocol_id") != "paper2027.graph.g4.disjoint_backbone.v1":
            raise ValueError(f"seed {seed} is not a registered G4 backbone")
        require_recorded_source(backbone_manifest, path_key="source", hash_key="source_sha256", context=f"seed {seed} backbone manifest")
        if backbone_manifest.get("final_test_lock_sha256") != final_lock_sha or backbone_manifest.get("selection_lock_sha256") != selection_lock_sha:
            raise ValueError(f"seed {seed} backbone manifest has wrong held-out lock hashes")
        controller_manifest_path = args.root / "manifests" / f"controller_seed{seed}.json"
        evaluation_manifest_path = args.root / "manifests" / f"{args.evaluation_manifest_prefix}{seed}.json"
        if not controller_manifest_path.is_file() or not evaluation_manifest_path.is_file():
            raise ValueError(f"seed {seed} lacks controller/evaluation manifests")
        controller_manifest = json.loads(controller_manifest_path.read_text())
        evaluation_manifest = json.loads(evaluation_manifest_path.read_text())
        if controller_manifest.get("status") != "complete" or evaluation_manifest.get("status") != "complete":
            raise ValueError(f"seed {seed} controller/evaluation manifest is incomplete")
        require_recorded_source(controller_manifest, path_key="analysis_source", hash_key="analysis_source_sha256", context=f"seed {seed} controller manifest")
        for path_key, hash_key in (
            ("raw_source", "raw_source_sha256"),
            ("controller_source", "controller_source_sha256"),
            ("runner_source", "runner_source_sha256"),
        ):
            require_recorded_source(evaluation_manifest, path_key=path_key, hash_key=hash_key, context=f"seed {seed} evaluation manifest")
        if evaluation_manifest.get("evaluation_dir") != args.evaluation_dir or evaluation_manifest.get("evaluation_manifest_prefix") != args.evaluation_manifest_prefix:
            raise ValueError(f"seed {seed} evaluation manifest is not bound to the requested retry tree")
        if controller_manifest.get("final_test_lock_sha256") != final_lock_sha or controller_manifest.get("selection_lock_sha256") != selection_lock_sha:
            raise ValueError(f"seed {seed} controller manifest has wrong held-out lock hashes")
        if evaluation_manifest.get("final_test_lock_sha256") != final_lock_sha:
            raise ValueError(f"seed {seed} evaluation manifest has wrong final lock hash")
        checkpoint_sha = controller_manifest.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str):
            raise ValueError(f"seed {seed} controller manifest lacks its frozen checkpoint hash")
        base = args.root / args.evaluation_dir / f"seed{seed}"
        raw_path = base / "raw" / "permutation_clusters.csv"
        replicas = [base / f"rank48_seed{replica}" / "full" / "permutation_clusters.csv" for replica in (1, 2)]
        if not raw_path.is_file() or not all(path.is_file() for path in replicas):
            raise ValueError(f"complete backbone seed {seed} lacks complete G4 evaluation")
        require_raw_final_lock_summary(
            base / "raw" / "summary.json", final_lock_sha=final_lock_sha,
            checkpoint_sha=checkpoint_sha, context=f"seed {seed}",
        )
        for replica in (1, 2):
            require_controlled_final_lock_summary(
                base / f"rank48_seed{replica}" / "full" / "summary.json",
                final_lock_sha=final_lock_sha, checkpoint_sha=checkpoint_sha,
                context=f"seed {seed} replica {replica}",
            )
        raw = read_csv(raw_path)
        raw_successor = per_permutation_auc(raw, metric="strict_successor_correct")
        raw_hold = per_permutation_auc(raw, metric="endpoint_hold_correct")
        replica_successor = np.stack([per_permutation_auc(read_csv(path), metric="strict_successor_correct") for path in replicas])
        replica_hold = np.stack([per_permutation_auc(read_csv(path), metric="endpoint_hold_correct") for path in replicas])
        if replica_successor.shape[1] != raw_successor.shape[0]:
            raise ValueError("raw/controller final-lock graph counts differ")
        primary_successor.append(replica_successor)
        primary_hold.append(replica_hold)
        seed_rows.append({"backbone_seed": seed, "raw_call8_accuracy": call8_accuracy(raw),
                          "raw_strict_successor_auc_9_128": float(raw_successor.mean()),
                          "raw_endpoint_hold_auc_9_128": float(raw_hold.mean()),
                          "full_strict_successor_auc_9_128": float(replica_successor.mean()),
                          "full_endpoint_hold_auc_9_128": float(replica_hold.mean()),
                          "full_minus_raw_successor_auc": float(replica_successor.mean() - raw_successor.mean()),
                          "full_minus_raw_hold_auc": float(replica_hold.mean() - raw_hold.mean()),
                          "graphs": int(raw_successor.shape[0]), "controller_replicas": 2})
        for mode in ("executor_off", "batch_shuffle", "D_only", "no_AB", "identity_D", "AB_only", "no_bias", "mean_D", "shuffle_D", "spectrum_matched_random_delta"):
            path = base / "rank48_seed1" / mode / "permutation_clusters.csv"
            if not path.is_file():
                raise ValueError(f"seed {seed} missing required control {mode}")
            require_controlled_final_lock_summary(
                base / "rank48_seed1" / mode / "summary.json",
                final_lock_sha=final_lock_sha, checkpoint_sha=checkpoint_sha,
                context=f"seed {seed} control {mode}",
            )
            control_arrays[mode].append(per_permutation_auc(read_csv(path), metric="strict_successor_correct")[None, :])
    outcome = {"status": "complete", "protocol_id": "paper2027.graph.g4.disjoint_aggregate.v1",
               "registered_backbone_count": len(manifests), "completed_backbone_count": len(completed),
               "included_backbone_seeds": completed, "selection_rule": "none; every completed backbone and both controller replicas",
               "evaluation_dir": args.evaluation_dir, "evaluation_manifest_prefix": args.evaluation_manifest_prefix,
               "final_lock_excluded_from_backbone_and_controller_training": True, "seed_rows": seed_rows,
               "selection_lock_sha256": selection_lock_sha, "final_test_lock_sha256": final_lock_sha,
               "raw_call8_accuracy": {"point_estimate": float(np.mean([row["raw_call8_accuracy"] for row in seed_rows]))},
               "full_strict_successor_auc_9_128": nested_bootstrap(primary_successor, draws=args.bootstrap_draws, seed=args.bootstrap_seed),
               "full_endpoint_hold_auc_9_128": nested_bootstrap(primary_hold, draws=args.bootstrap_draws, seed=args.bootstrap_seed + 1),
               "raw_strict_successor_auc_9_128": nested_bootstrap([np.asarray([per_permutation_auc(read_csv(args.root / args.evaluation_dir / f"seed{seed}" / "raw" / "permutation_clusters.csv"), metric="strict_successor_correct")]) for seed in completed], draws=args.bootstrap_draws, seed=args.bootstrap_seed + 2),
               "controls_strict_successor_auc_9_128": {mode: nested_bootstrap(values, draws=args.bootstrap_draws, seed=args.bootstrap_seed + 10 + index) for index, (mode, values) in enumerate(sorted(control_arrays.items()))}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "seed_summary.csv", seed_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(outcome, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
