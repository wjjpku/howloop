#!/usr/bin/env python3
"""Aggregate the completed three-seed P3 held-out phase interventions.

P3 is intentionally an evidentially narrow test.  The discovered plane comes
from disjoint lengths; its answer-token rotations are then measured on four
held-out lengths.  A full phase replacement may alter the answer residual
directly, so it is retained as a positive patching control, not causal evidence
for the plane.  The causal comparison is a pi rotation versus an equal-norm
rotation in a random orthogonal two-dimensional plane.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


SEEDS = (3, 4, 5)
CONDITIONS = (
    "receiver",
    "phase_rotate_pi_answer",
    "random2d_rotation_pi_answer",
    "phase_remove_answer",
    "phase_keep_answer",
    "phase_answer",
)
METRICS = (
    "phase_translation_recovery",
    "margin_translation_recovery",
    "mean_accuracy",
    "mean_target_accuracy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def causal_row(summary: dict[str, Any], condition: str) -> dict[str, Any]:
    rows = [
        row
        for row in summary["causal_summary"]
        if row["variant"] == "raw"
        and row["condition"] == condition
        and row["continuation_group"] == "continued"
        and row["donor_offset"] == "all"
    ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one continued/all raw row for {condition}, got {len(rows)}")
    return rows[0]


def validate(summary: dict[str, Any], manifest: dict[str, Any], seed: int) -> None:
    architecture = summary.get("architecture", {})
    required_architecture = {
        "token_embedding_injection": "initial_only",
        "position_embedding": "none",
        "position_injection": "initial_only",
        "final_layer_norm": "inside every recurrent call",
    }
    if summary.get("status") != "complete" or manifest.get("status") != "complete":
        raise ValueError(f"seed {seed} P3 run is incomplete")
    if summary.get("backbone_seed") != seed:
        raise ValueError(f"seed {seed} summary names a different backbone")
    if summary.get("paper_mode") is not True:
        raise ValueError(f"seed {seed} did not run paper mode")
    if any(architecture.get(key) != value for key, value in required_architecture.items()):
        raise ValueError(f"seed {seed} violates the input-once Parity architecture")
    if manifest.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise ValueError(f"seed {seed} manifest/checkpoint mismatch")
    protocol = summary.get("protocol", {})
    if protocol.get("causal_lengths") != [10, 22, 36, 72]:
        raise ValueError(f"seed {seed} has a non-registered causal length set")
    for condition in CONDITIONS:
        causal_row(summary, condition)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for seed in SEEDS:
        source_dir = args.input_root / f"seed{seed}"
        summary_path = source_dir / "summary.json"
        manifest_path = source_dir / "run_manifest.json"
        summary, manifest = read_json(summary_path), read_json(manifest_path)
        validate(summary, manifest, seed)
        source_hashes[str(summary_path)] = sha256(summary_path)
        source_hashes[str(manifest_path)] = sha256(manifest_path)
        for condition in CONDITIONS:
            original = causal_row(summary, condition)
            row = {"backbone_seed": seed, "condition": condition}
            row.update({metric: float(original[metric]) for metric in METRICS})
            rows.append(row)
        heldout = summary["dynamics"]["raw"]["heldout_evaluation"]
        skip_rows.append(
            {
                "backbone_seed": seed,
                "mlp_skip_fraction_closer_to_previous_phase": float(summary["mlp_skip_hidden_lag"]["fraction_closer_to_clean_previous_phase"]),
                "attention_skip_fraction_closer_to_previous_phase": float(summary["attention_skip_hidden_lag"]["fraction_closer_to_clean_previous_phase"]),
                "period_calls": float(heldout["polar_rotation_period_calls"]),
                "plane_energy": float(heldout["shared_phase_plane_energy_fraction"]),
                "transition_r2": float(heldout["transition_r2"]),
                "scaled_four_step_relative_error": float(heldout["scaled_four_step_relative_error"]),
            }
        )

    index = {(row["backbone_seed"], row["condition"]): row for row in rows}
    differences = {
        metric: [
            index[(seed, "phase_rotate_pi_answer")][metric]
            - index[(seed, "random2d_rotation_pi_answer")][metric]
            for seed in SEEDS
        ]
        for metric in ("phase_translation_recovery", "margin_translation_recovery")
    }
    summary = {
        "status": "complete_registered_three_seed_null",
        "protocol_id": "paper2027.parity.p3.phase_causality.v1",
        "backbone_seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "interpretation": (
            "A full answer-token phase replacement is a positive patching control, not a causal plane result. "
            "Across the three held-out backbone analyses, a pi rotation in the discovered plane has no consistent "
            "advantage over an equal-norm random-plane rotation."
        ),
        "pi_rotation_minus_random_control": {
            metric: {"per_seed": values, "mean": mean(values)}
            for metric, values in differences.items()
        },
        "means_by_condition": {
            condition: {metric: mean([row[metric] for row in rows if row["condition"] == condition]) for metric in METRICS}
            for condition in CONDITIONS
        },
        "skip_means": {
            key: mean([row[key] for row in skip_rows])
            for key in (
                "mlp_skip_fraction_closer_to_previous_phase",
                "attention_skip_fraction_closer_to_previous_phase",
                "period_calls",
                "plane_energy",
                "transition_r2",
                "scaled_four_step_relative_error",
            )
        },
        "source_sha256": source_hashes,
    }
    write_csv(args.out_dir / "causal_conditions.csv", rows)
    write_csv(args.out_dir / "skip_and_dynamics.csv", skip_rows)
    write_json(args.out_dir / "summary.json", summary)
    write_json(
        args.out_dir / "source_attestation.json",
        {"status": "PASS", "checked_source_sha256": source_hashes, "errors": []},
    )


if __name__ == "__main__":
    main()
