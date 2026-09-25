from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPLICATION_GATES = (
    {
        "claim": "compression_lowers_natural_qk",
        "contrast": "compression",
        "metric": "max_qk_destination_match_gap",
        "direction": "negative",
    },
    {
        "claim": "compression_increases_mlp_rescue",
        "contrast": "compression",
        "metric": "max_mlp_rescued_clamp_next_margin",
        "direction": "positive",
    },
    {
        "claim": "compression_does_not_increase_path_bundle_dependence",
        "contrast": "compression",
        "metric": "max_bundle_endpoint_specificity",
        "direction": "nonpositive",
    },
    {
        "claim": "stretch_cleans_natural_qk",
        "contrast": "stretch",
        "metric": "max_qk_destination_match_gap",
        "direction": "positive",
    },
    {
        "claim": "stretch_reduces_mlp_rescue",
        "contrast": "stretch",
        "metric": "max_mlp_rescued_clamp_next_margin",
        "direction": "negative",
    },
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["contrast"], row["metric"]): row for row in rows}


def _support(row: dict[str, str], direction: str) -> tuple[int, int]:
    pair_count = int(row["pair_count"])
    if direction == "positive":
        return int(row["positive_pair_count"]), pair_count
    if direction == "negative":
        return int(row["negative_pair_count"]), pair_count
    if direction == "nonpositive":
        return (
            int(row["negative_pair_count"]) + int(row["zero_pair_count"]),
            pair_count,
        )
    raise ValueError(f"unknown direction: {direction}")


def build_summary(
    d256_rows: list[dict[str, str]],
    d64_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    old = _index(d256_rows)
    new = _index(d64_rows)
    output: list[dict[str, Any]] = []
    for gate in REPLICATION_GATES:
        key = (gate["contrast"], gate["metric"])
        if key not in old or key not in new:
            raise KeyError(f"missing replication metric: {key}")
        old_support, old_pairs = _support(old[key], gate["direction"])
        new_support, new_pairs = _support(new[key], gate["direction"])
        required = max(1, new_pairs - 1)
        output.append(
            {
                **gate,
                "d256_median_delta": float(old[key]["median_delta"]),
                "d256_support_count": old_support,
                "d256_pair_count": old_pairs,
                "d64_median_delta": float(new[key]["median_delta"]),
                "d64_support_count": new_support,
                "d64_pair_count": new_pairs,
                "required_support_count": required,
                "replicated": int(new_support >= required),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired d_model=256 and d_model=64 graph effects."
    )
    parser.add_argument("--d256-summary", type=Path, required=True)
    parser.add_argument("--d64-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_summary(
        _read_rows(args.d256_summary),
        _read_rows(args.d64_summary),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.out_dir / "width_replication_summary.csv", rows)
    payload = {
        "gate_count": len(rows),
        "replicated_count": sum(int(row["replicated"]) for row in rows),
        "rows": rows,
    }
    (args.out_dir / "width_replication_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
