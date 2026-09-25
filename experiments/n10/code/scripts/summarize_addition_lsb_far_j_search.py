#!/usr/bin/env python3
"""Aggregate matched far-range Addition J snapshot selections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for label in args.labels:
        selection_path = args.search_root / "selection" / label / "selection.json"
        controller_summary_path = (
            args.search_root / "controllers" / label / "summary.json"
        )
        selection = json.loads(selection_path.read_text())
        summary = json.loads(controller_summary_path.read_text())
        if selection.get("status") != "complete" or summary.get("status") != "complete":
            raise ValueError(f"incomplete candidate: {label}")
        invariants = {
            "checkpoint": selection["checkpoint"],
            "checkpoint_step": selection["checkpoint_step"],
            "validation_lengths": selection["validation_lengths"],
            "validation_seed": selection["validation_seed"],
            "examples_per_length": selection["examples_per_length"],
            "id_gate": selection["id_gate"],
        }
        if reference is None:
            reference = invariants
        elif invariants != reference:
            raise ValueError(f"candidate protocol mismatch: {label}")
        metrics = selection["selection_metrics"]
        rows.append(
            {
                "label": label,
                "parameterization": summary["controller_parameterization"],
                "rank": summary.get("rank"),
                "learning_rate_multiplier": summary["learning_rate_multiplier"],
                "selected_snapshot_update": selection["selected_snapshot_update"],
                "id_gate_em": metrics["id_gate_em"],
                "id_gate_passed": metrics["id_gate_passed"],
                "mean_raw_digit_em": metrics["mean_raw_digit_em"],
                "mean_digit_em": metrics["mean_digit_em"],
                "mean_delta_digit_em": metrics["mean_delta_digit_em"],
                "mean_bit_accuracy": metrics["mean_bit_accuracy"],
                "peak_cuda_memory_reserved_gib": summary[
                    "peak_cuda_memory_reserved_gib"
                ],
                "parameter_count": summary["parameter_count"],
                "selected_controller": selection["selected_controller"],
            }
        )
    eligible = [row for row in rows if row["id_gate_passed"]]
    if not eligible:
        raise RuntimeError("no candidate passed the common ID gate")
    best = max(
        eligible,
        key=lambda row: (
            float(row["mean_digit_em"]),
            float(row["mean_bit_accuracy"]),
            -int(row["parameter_count"]),
        ),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "candidate_comparison.csv", rows)
    result = {
        "status": "complete",
        "protocol": reference,
        "candidates": rows,
        "selected": best,
        "unopened_test_lengths": list(range(21, 31)),
        "selection_rule": (
            "maximize matched held-out mean digit exact match among candidates "
            "passing the common ID-retention gate; break ties by bit accuracy, "
            "then fewer parameters"
        ),
        "claim_boundary": (
            "candidate selection used controller-exposed lengths 10..20 only; "
            "lengths 21..30 were not used in snapshot or candidate selection"
        ),
    }
    (args.out_dir / "candidate_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
