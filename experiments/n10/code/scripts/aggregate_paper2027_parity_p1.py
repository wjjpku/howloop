#!/usr/bin/env python3
"""Aggregate the corrected input-once Parity P1 replication cohort.

Endpoint records are analysed by backbone seed, with a binomial resample of
the fixed 512 held-out examples nested within each sampled backbone.  The
three deep seeds are kept explicitly separate from the twelve-seed behavioral
population: their phase analysis is mechanism evidence, not twelve-seed proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEEDS = list(range(3, 15))
DEEP_SEEDS = [3, 4, 5]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_manifest_hashes(
    *, root: Path, seed: int, mode: str, checkpoint: Path, out_root: Path
) -> None:
    """Verify frozen source/checkpoint and result inputs before aggregation."""
    backbone = read_json(root / "manifests" / f"backbone_seed{seed}.json")
    if not isinstance(backbone.get("source"), str) or backbone.get("source_sha256") != sha256(Path(backbone["source"])):
        raise ValueError(f"P1 backbone seed {seed} source hash no longer matches")
    recorded_checkpoint = Path(str(backbone.get("output_dir", ""))) / "final.pt"
    if not checkpoint.is_file() or checkpoint != recorded_checkpoint:
        raise ValueError(f"P1 backbone seed {seed} final checkpoint is unavailable or differs from its manifest")
    evaluation = read_json(root / "manifests" / f"evaluation_seed{seed}_{mode}.json")
    if evaluation.get("checkpoint_sha256") != sha256(checkpoint) or evaluation.get("paper_mode") is not True:
        raise ValueError(f"P1 {mode} evaluation seed {seed} is not bound to its input-once checkpoint")
    analysis_root = evaluation.get("analysis_code_root")
    sources = evaluation.get("analysis_source_sha256")
    if not isinstance(analysis_root, str) or not isinstance(sources, dict) or not sources:
        raise ValueError(f"P1 {mode} evaluation seed {seed} lacks analysis source provenance")
    for relative, expected in sources.items():
        path = Path(analysis_root) / relative
        if not isinstance(expected, str) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"P1 {mode} evaluation seed {seed} analysis source changed: {relative}")
    required = ["endpoint_horizon/horizon.csv", "endpoint_horizon/summary.json"]
    if mode == "deep":
        required.extend(["four_phase/summary.json", "loop_depth_heatmap/summary.json", "prospective_boundary.json"])
    outputs = evaluation.get("output_sha256")
    if not isinstance(outputs, dict):
        raise ValueError(f"P1 {mode} evaluation seed {seed} lacks output hashes")
    for relative in required:
        path = out_root / relative
        if not path.is_file() or outputs.get(relative) != sha256(path):
            raise ValueError(f"P1 {mode} evaluation seed {seed} artifact hash mismatch: {relative}")


def status(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(read_json(path).get("status"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def hierarchical_binary_bootstrap(successes: np.ndarray, totals: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    if successes.ndim != 1 or totals.shape != successes.shape or len(successes) < 1:
        raise ValueError("successes/totals must be nonempty matching vectors")
    if np.any(totals <= 0) or np.any(successes < 0) or np.any(successes > totals):
        raise ValueError("invalid binary counts")
    generator = np.random.default_rng(seed)
    # Sample backbones, then the binary test examples represented by each
    # aggregate count. This is exactly the nonparametric bootstrap for a
    # Bernoulli row when individual outcomes are exchangeable within a seed.
    selected = generator.integers(len(successes), size=(draws, len(successes)))
    selected_successes = successes[selected]
    selected_totals = totals[selected]
    probabilities = selected_successes / selected_totals
    sampled_successes = generator.binomial(selected_totals.astype(np.int64), probabilities)
    samples = sampled_successes.sum(axis=1) / selected_totals.sum(axis=1)
    point = successes.sum() / totals.sum()
    return {"estimate": float(point), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def seed_auc(rows: list[dict[str, str]], *, min_length: int) -> float:
    selected = sorted((int(row["length"]), float(row["exact_match"])) for row in rows if row["variant"] == "raw" and int(row["length"]) >= min_length)
    if len(selected) < 2:
        raise ValueError("need at least two endpoint lengths for an AUC")
    x = np.asarray([pair[0] for pair in selected], dtype=float)
    y = np.asarray([pair[1] for pair in selected], dtype=float)
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(y, x) / (x[-1] - x[0]))


def seed_bootstrap(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("need a nonempty seed vector")
    generator = np.random.default_rng(seed)
    sample = values[generator.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return {"estimate": float(values.mean()), "ci_low": float(np.quantile(sample, .025)), "ci_high": float(np.quantile(sample, .975))}


def _phase_record(seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    heldout = summary.get("dynamics", {}).get("raw", {}).get("heldout_evaluation", {})
    return {
        "seed": seed,
        "period_calls": heldout.get("polar_rotation_period_calls"),
        "signed_angle_radians": heldout.get("polar_rotation_signed_angle_radians"),
        "transition_r2": heldout.get("transition_r2"),
        "shared_phase_plane_energy_fraction": heldout.get("shared_phase_plane_energy_fraction"),
        "scaled_four_step_relative_error": heldout.get(
            "scaled_four_step_relative_error"
        ),
        "mlp_skip_previous_phase_fraction": summary.get("mlp_skip_hidden_lag", {}).get("fraction_closer_to_clean_previous_phase"),
        "attention_skip_previous_phase_fraction": summary.get("attention_skip_hidden_lag", {}).get("fraction_closer_to_clean_previous_phase"),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    endpoint_records: list[dict[str, Any]] = []
    per_seed: dict[int, list[dict[str, str]]] = {}
    phase_records: list[dict[str, Any]] = []
    for seed in args.seeds:
        if status(args.root / "manifests" / f"backbone_seed{seed}.json") != "complete":
            raise ValueError(f"P1 backbone seed {seed} is not complete")
        mode = "deep" if seed in args.deep_seeds else "endpoint"
        if status(args.root / "manifests" / f"evaluation_seed{seed}_{mode}.json") != "complete":
            raise ValueError(f"P1 {mode} evaluation seed {seed} is not complete")
        checkpoint = args.root / "backbones" / f"parity_input_once_seed{seed}" / "final.pt"
        out_root = args.root / "evaluation" / f"seed{seed}"
        require_manifest_hashes(root=args.root, seed=seed, mode=mode, checkpoint=checkpoint, out_root=out_root)
        rows = read_csv(out_root / "endpoint_horizon" / "horizon.csv")
        raw = [row for row in rows if row["variant"] == "raw"]
        if not raw:
            raise ValueError(f"seed {seed} endpoint horizon has no raw rows")
        per_seed[seed] = raw
        for row in raw:
            endpoint_records.append({"seed": seed, **row})
        if seed in args.deep_seeds:
            phase_records.append(_phase_record(seed, read_json(out_root / "four_phase" / "summary.json")))
    lengths = sorted({int(row["length"]) for rows in per_seed.values() for row in rows})
    statistics: dict[str, Any] = {}
    for index, length in enumerate(lengths):
        selected = [next((row for row in per_seed[seed] if int(row["length"]) == length), None) for seed in args.seeds]
        if any(row is None for row in selected):
            raise ValueError(f"all P1 seeds must evaluate common length {length}")
        successes = np.asarray([float(row["exact_successes"]) for row in selected])
        totals = np.asarray([float(row["examples"]) for row in selected])
        statistics[f"raw_exact_match_length_{length}"] = hierarchical_binary_bootstrap(successes, totals, draws=args.bootstrap_draws, seed=args.seed + index)
    aucs = np.asarray([seed_auc(per_seed[seed], min_length=24) for seed in args.seeds])
    statistics["raw_exact_match_auc_lengths_24_to_500"] = seed_bootstrap(aucs, draws=args.bootstrap_draws, seed=args.seed + 1001)
    result = {
        "status": "complete",
        "protocol_id": "paper2027.parity.p1.input_once.v2",
        "backbone_seeds": args.seeds,
        "deep_mechanism_seeds": args.deep_seeds,
        "statistics": statistics,
        "uncertainty": "endpoint: backbone then held-out Bernoulli example bootstrap; AUC: backbone bootstrap",
        "bootstrap_draws": args.bootstrap_draws,
        "endpoint_metric": "exact sequence match at registered T(n)=n",
        "phase_evidence_boundary": "phase records are reported only for the prespecified deep seeds; they are not pooled as independent backbone replications",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "endpoint_by_seed.csv", endpoint_records)
    write_csv(args.out_dir / "phase_deep_seed_table.csv", phase_records)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--deep-seeds", type=int, nargs="+", default=DEEP_SEEDS)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026099001)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(aggregate(parse_args()), indent=2, sort_keys=True))
