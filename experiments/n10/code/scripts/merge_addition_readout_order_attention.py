#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from compare_addition_readout_order_attention import SystemSpec, plot_panels, write_csv


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["system"]),
            int(row["step"]),
            int(row["position_from_lsb"]),
        )
        indexed[key] = row
    return list(indexed.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--system-order", nargs="+", required=True)
    parser.add_argument("--logical-length", type=int, default=10)
    parser.add_argument("--maximum-step", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    random_rows: list[dict[str, Any]] = []
    carry_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for input_dir in args.input_dir:
        random_rows.extend(read_rows(input_dir / "random_position_accuracy.csv"))
        carry_rows.extend(read_rows(input_dir / "all_ones_plus_one_trajectory.csv"))
        source_summaries.append(
            json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
        )
    random_rows = deduplicate(random_rows)
    carry_rows = deduplicate(carry_rows)
    available = {str(row["system"]) for row in random_rows}
    missing = [label for label in args.system_order if label not in available]
    if missing:
        raise ValueError(f"missing systems: {missing}")
    systems = [SystemSpec(label=label, checkpoint=Path(".")) for label in args.system_order]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "random_position_accuracy.csv", random_rows)
    write_csv(args.out_dir / "all_ones_plus_one_trajectory.csv", carry_rows)
    plot_panels(
        systems=systems,
        rows=random_rows,
        maximum_step=args.maximum_step,
        logical_length=args.logical_length,
        key="accuracy",
        title="Random additions: per-bit readout accuracy aligned by numerical bit",
        output=args.out_dir / "random_position_accuracy_aligned.png",
        cmap="viridis",
    )
    plot_panels(
        systems=systems,
        rows=carry_rows,
        maximum_step=args.maximum_step,
        logical_length=args.logical_length,
        key="correct",
        title="Carry-heavy example (all ones + 1): correct bits by loop",
        output=args.out_dir / "all_ones_plus_one_correctness_aligned.png",
        cmap="RdYlGn",
    )
    summary = {
        "status": "complete",
        "logical_length": args.logical_length,
        "maximum_step": args.maximum_step,
        "system_order": args.system_order,
        "input_dirs": [str(path) for path in args.input_dir],
        "source_summaries": source_summaries,
        "claim_boundary": "direct-readout localization; causal interpretation uses separate patching runs",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
