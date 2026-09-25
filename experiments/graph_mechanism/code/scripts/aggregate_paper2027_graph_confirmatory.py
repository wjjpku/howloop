#!/usr/bin/env python3
"""Hierarchical statistics for the frozen paper2027 Graph G1/G2/G3 cohort.

The top-level unit is a backbone seed; controller replicas are nested inside a
qualified backbone; graph permutations are the lowest-level clustered test
unit.  This script refuses to silently analyse only successful controllers or
only available seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_SEEDS = list(range(100, 112))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def cluster_rate(
    rows: Iterable[dict[str, str]],
    *,
    calls: Iterable[int],
    numerator: str,
    denominator: str,
    call_key: str = "call",
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-permutation aggregate rate and IDs, with calls pooled per cluster."""
    wanted = set(calls)
    totals: dict[int, list[float]] = {}
    for row in rows:
        if mode is not None and row.get("mode") != mode:
            continue
        if int(row[call_key]) not in wanted:
            continue
        key = int(row["permutation"])
        record = totals.setdefault(key, [0.0, 0.0])
        record[0] += float(row[numerator]); record[1] += float(row[denominator])
    if not totals:
        raise ValueError("no rows matched requested cluster metric")
    identifiers = np.asarray(sorted(totals), dtype=np.int64)
    numerators = np.asarray([totals[int(key)][0] for key in identifiers])
    denominators = np.asarray([totals[int(key)][1] for key in identifiers])
    if np.any(denominators <= 0):
        raise ValueError("cluster metric has non-positive denominator")
    return numerators / denominators, identifiers


def paired_cluster_delta(
    rows: list[dict[str, str]],
    *,
    calls: Iterable[int],
    numerator: str,
    denominator: str,
    treated: str,
    control: str,
    call_key: str = "call",
) -> tuple[np.ndarray, np.ndarray]:
    treated_values, treated_ids = cluster_rate(rows, calls=calls, numerator=numerator, denominator=denominator, call_key=call_key, mode=treated)
    control_values, control_ids = cluster_rate(rows, calls=calls, numerator=numerator, denominator=denominator, call_key=call_key, mode=control)
    if not np.array_equal(treated_ids, control_ids):
        raise ValueError("paired modes do not cover identical graph permutations")
    return treated_values - control_values, treated_ids


def bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("bootstrap_mean expects a nonempty vector")
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {"estimate": float(values.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def nested_bootstrap(
    values: list[list[np.ndarray]],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Backbone -> controller -> graph-permutation hierarchical bootstrap."""
    if not values or any(not controllers for controllers in values):
        raise ValueError("every included backbone must have at least one controller vector")
    if any(vector.ndim != 1 or not len(vector) for controllers in values for vector in controllers):
        raise ValueError("controller vectors must be nonempty one-dimensional arrays")
    point = float(np.mean([np.mean([vector.mean() for vector in controllers]) for controllers in values]))
    generator = np.random.default_rng(seed)
    samples = np.empty(draws)
    backbone_count = len(values)
    for draw in range(draws):
        sampled_backbones = generator.integers(backbone_count, size=backbone_count)
        backbone_means: list[float] = []
        for backbone_index in sampled_backbones:
            controllers = values[int(backbone_index)]
            sampled_controllers = generator.integers(len(controllers), size=len(controllers))
            controller_means = []
            for controller_index in sampled_controllers:
                vector = controllers[int(controller_index)]
                cluster_indices = generator.integers(len(vector), size=len(vector))
                controller_means.append(float(vector[cluster_indices].mean()))
            backbone_means.append(float(np.mean(controller_means)))
        samples[draw] = float(np.mean(backbone_means))
    return {"estimate": point, "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def _expect_status(path: Path, *allowed: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    if payload.get("status") not in allowed:
        raise ValueError(f"unexpected status in {path}: {payload.get('status')!r}, expected one of {allowed}")
    return payload


def _format_interval(statistic: dict[str, float]) -> str:
    return f"{statistic['estimate']:.4f} [{statistic['ci_low']:.4f}, {statistic['ci_high']:.4f}]"


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root
    seeds = args.seeds
    backbone_rows: list[dict[str, Any]] = []
    g1_posthorizon: list[list[np.ndarray]] = []
    g2_effects: list[list[np.ndarray]] = []
    g3_effects: list[list[np.ndarray]] = []
    component_rows: list[dict[str, Any]] = []
    complete_g1 = 0
    qualified = 0
    for seed in seeds:
        g1_manifest = _expect_status(root / "manifests" / f"backbone_seed{seed}.json", "complete")
        g1_evaluation = _expect_status(root / "manifests" / f"evaluation_seed{seed}.json", "complete")
        del g1_manifest, g1_evaluation
        complete_g1 += 1
        g1_rows = read_csv(root / "evaluation" / f"seed{seed}" / "permutation_clusters.csv")
        endpoint, endpoint_ids = cluster_rate(g1_rows, calls=[8], numerator="endpoint_hold_correct", denominator="examples")
        posthorizon, post_ids = cluster_rate(g1_rows, calls=range(17, 65), numerator="strict_successor_correct", denominator="strict_examples")
        if len(endpoint_ids) != args.permutations or len(post_ids) != args.permutations:
            raise ValueError(f"seed {seed} does not contain the locked {args.permutations}-permutation test")
        is_qualified = bool(endpoint.mean() >= .95)
        backbone_rows.append({"seed": seed, "call8_endpoint_hold_accuracy": float(endpoint.mean()), "posthorizon_strict_successor_auc_17_64": float(posthorizon.mean()), "qualified": is_qualified})
        g1_posthorizon.append([posthorizon])
        g2_manifest_path = root / "manifests" / f"g2_interface_seed{seed}.json"
        g3_manifest_path = root / "manifests" / f"g3_controller_seed{seed}.json"
        if not is_qualified:
            _expect_status(g2_manifest_path, "endpoint_failed")
            _expect_status(g3_manifest_path, "endpoint_failed")
            continue
        qualified += 1
        _expect_status(g2_manifest_path, "complete")
        _expect_status(g3_manifest_path, "complete")
        _expect_status(root / "manifests" / f"g3_evaluation_seed{seed}.json", "complete")
        g2_rows = read_csv(root / "g2_interface" / f"seed{seed}" / "permutation_clusters.csv")
        g2_delta, g2_ids = paired_cluster_delta(g2_rows, calls=[16, 32, 64], numerator="strict_next_successor_correct", denominator="strict_examples", treated="matched_young", control="raw_late_next", call_key="source_call")
        if len(g2_ids) != args.permutations:
            raise ValueError(f"G2 seed {seed} is missing locked graph permutations")
        g2_effects.append([g2_delta])
        primary: list[np.ndarray] = []
        raw_rows = read_csv(root / "g3_evaluation" / f"seed{seed}" / "rank48_seed1" / "raw" / "permutation_clusters.csv")
        for replica in (1, 2):
            full_rows = read_csv(root / "g3_evaluation" / f"seed{seed}" / f"rank48_seed{replica}" / "full" / "permutation_clusters.csv")
            full, ids = cluster_rate(full_rows, calls=range(17, 65), numerator="strict_successor_correct", denominator="strict_examples")
            raw, raw_ids = cluster_rate(raw_rows, calls=range(17, 65), numerator="strict_successor_correct", denominator="strict_examples")
            if not np.array_equal(ids, raw_ids):
                raise ValueError(f"G3 seed {seed}, replica {replica}: raw/full permutation mismatch")
            primary.append(full - raw)
        g3_effects.append(primary)
        for mode in ("D_only", "no_AB", "identity_D", "AB_only", "no_bias", "mean_D", "shuffle_D", "batch_shuffle", "spectrum_matched_random_delta", "executor_off"):
            rows = read_csv(root / "g3_evaluation" / f"seed{seed}" / "rank48_seed1" / mode / "permutation_clusters.csv")
            value, ids = cluster_rate(rows, calls=range(17, 65), numerator="strict_successor_correct", denominator="strict_examples")
            if not np.array_equal(ids, raw_ids):
                raise ValueError(f"G3 component {mode} seed {seed}: permutation mismatch")
            component_rows.append({"seed": seed, "mode": mode, "strict_successor_auc_17_64": float(value.mean())})
    if complete_g1 != len(seeds):
        raise AssertionError("unreachable incomplete G1 accounting")
    if not qualified:
        raise ValueError("no endpoint-qualified backbones; no controller efficacy claim is estimable")
    result = {
        "protocol_ids": ["paper2027.graph.g1.phenotype.v1", "paper2027.graph.g2.matched_interface.v1", "paper2027.graph.g3.locked_evaluation.v1"],
        "expected_backbone_seeds": seeds,
        "complete_backbones": complete_g1,
        "endpoint_qualified_backbones": qualified,
        "endpoint_qualified_fraction": qualified / len(seeds),
        "statistics": {
            "g1_posthorizon_strict_successor_auc_17_64": nested_bootstrap(g1_posthorizon, draws=args.bootstrap_draws, seed=args.seed + 1),
            "g2_matched_young_minus_raw_next_strict_accuracy_calls_16_32_64": nested_bootstrap(g2_effects, draws=args.bootstrap_draws, seed=args.seed + 2),
            "g3_full_minus_raw_strict_successor_auc_17_64": nested_bootstrap(g3_effects, draws=args.bootstrap_draws, seed=args.seed + 3),
        },
        "uncertainty": "hierarchical nonparametric bootstrap: backbone, controller within backbone where applicable, graph permutation",
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "claim_boundary": "Only a finite, checkpoint-dependent locked-test continuation/interface result; no claim of indefinite closure or a universal inverse.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "backbone_table.csv", backbone_rows)
    write_csv(args.out_dir / "g3_component_table.csv", component_rows)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "summary.txt").write_text("\n".join([
        f"complete backbones: {complete_g1}/{len(seeds)}",
        f"endpoint-qualified backbones: {qualified}/{len(seeds)}",
        f"G1 post-horizon strict AUC: {_format_interval(result['statistics']['g1_posthorizon_strict_successor_auc_17_64'])}",
        f"G2 matched-young effect: {_format_interval(result['statistics']['g2_matched_young_minus_raw_next_strict_accuracy_calls_16_32_64'])}",
        f"G3 full-minus-raw effect: {_format_interval(result['statistics']['g3_full_minus_raw_strict_successor_auc_17_64'])}",
        "",
    ]), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026096001)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(aggregate(parse_args()), indent=2, sort_keys=True))
