"""Pool paired late/young attention interventions across rollback checkpoints."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


METRICS = (
    "accuracy",
    "margin",
    "survival_steps",
    "forward_accuracy_auc",
    "destination_mass_h0",
    "destination_mass_h1",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_deltas(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    rows = list(rows)
    lookup = {
        (
            int(row["graph_seed"]),
            int(row["back_count"]),
            int(row["rollback_checkpoint"]),
            row["condition"],
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        direction = row["direction"]
        if direction == "young_to_late_rescue":
            baseline = "late_baseline"
            component = row["condition"].removesuffix(".young_to_late")
        elif direction == "late_to_young_disrupt":
            baseline = "young_reference"
            component = row["condition"].removesuffix(".late_to_young")
        else:
            continue
        reference = lookup[
            (
                int(row["graph_seed"]),
                int(row["back_count"]),
                int(row["rollback_checkpoint"]),
                baseline,
            )
        ]
        result: dict[str, Any] = {
            "graph_seed": int(row["graph_seed"]),
            "back_count": int(row["back_count"]),
            "rollback_checkpoint": int(row["rollback_checkpoint"]),
            "component": component,
            "direction": direction,
            "matched_examples": int(row["matched_examples"]),
        }
        for metric in METRICS:
            result[f"delta_{metric}"] = float(row[metric]) - float(reference[metric])
        output.append(result)
    return output


def pool(deltas: Sequence[dict[str, Any]], *, by_back_count: bool) -> list[dict[str, Any]]:
    seed_keys = ["graph_seed", "component", "direction"]
    final_keys = ["component", "direction"]
    if by_back_count:
        seed_keys.insert(1, "back_count")
        final_keys.insert(0, "back_count")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in deltas:
        grouped[tuple(row[key] for key in seed_keys)].append(row)

    per_seed: list[dict[str, Any]] = []
    for key, parts in grouped.items():
        result = dict(zip(seed_keys, key, strict=True))
        weights = np.asarray([row["matched_examples"] for row in parts], dtype=float)
        result["matched_examples"] = int(weights.sum())
        result["cells"] = len(parts)
        for metric in METRICS:
            values = np.asarray([row[f"delta_{metric}"] for row in parts], dtype=float)
            result[f"delta_{metric}"] = float(np.average(values, weights=weights))
        per_seed.append(result)

    grouped_final: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped_final[tuple(row[key] for key in final_keys)].append(row)

    output: list[dict[str, Any]] = []
    for key, parts in sorted(grouped_final.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(final_keys, key, strict=True))
        result["graph_seeds"] = len(parts)
        result["matched_examples"] = int(sum(row["matched_examples"] for row in parts))
        result["cells_per_seed"] = min(row["cells"] for row in parts)
        for metric in METRICS:
            values = np.asarray([row[f"delta_{metric}"] for row in parts], dtype=float)
            result[f"delta_{metric}_mean"] = float(values.mean())
            result[f"delta_{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        output.append(result)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    deltas = paired_deltas(read_rows(args.seed_summary))
    write_rows(args.out_dir / "paired_deltas.csv", deltas)
    write_rows(args.out_dir / "pooled_all_cells.csv", pool(deltas, by_back_count=False))
    write_rows(args.out_dir / "pooled_by_back_count.csv", pool(deltas, by_back_count=True))


if __name__ == "__main__":
    main()
