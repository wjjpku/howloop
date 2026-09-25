#!/usr/bin/env python3
"""Freeze the P2 controller repair band from raw validation heatmap data.

The rule is chosen before controller training: pick the first length at which
the registered readout drops below the threshold while a two-call neighbor is
at least ``minimum_neighbor_gain`` better.  The training band is then a fixed
multiple of that first diseased length; no controller outcome enters this
choice.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def choose_boundary(
    rows: list[dict[str, str]],
    *,
    minimum_length: int,
    maximum_length: int,
    registered_threshold: float,
    minimum_neighbor_gain: float,
    band_multiplier: float,
    band_cap: int,
) -> dict[str, Any]:
    if minimum_length < 1 or maximum_length < minimum_length:
        raise ValueError("invalid search interval")
    if not 0.0 < registered_threshold <= 1.0:
        raise ValueError("registered threshold must lie in (0, 1]")
    if minimum_neighbor_gain < 0.0:
        raise ValueError("minimum neighbor gain must be nonnegative")
    if band_multiplier <= 1.0:
        raise ValueError("band multiplier must exceed one")
    if band_cap < minimum_length:
        raise ValueError("band cap is below the search interval")

    by_length_loop: dict[tuple[int, int], float] = {}
    for row in rows:
        if row.get("variant", "raw") != "raw":
            continue
        length = int(row["length"])
        loop = int(row["loop"])
        if length < minimum_length or length > maximum_length:
            continue
        by_length_loop[length, loop] = float(row["parity_token_accuracy"])

    scanned: list[dict[str, Any]] = []
    for length in range(minimum_length, maximum_length + 1):
        registered = by_length_loop.get((length, length))
        lower_neighbor = by_length_loop.get((length, length - 2))
        upper_neighbor = by_length_loop.get((length, length + 2))
        neighbor = max(
            value for value in (lower_neighbor, upper_neighbor) if value is not None
        ) if lower_neighbor is not None or upper_neighbor is not None else None
        diseased = bool(
            registered is not None
            and neighbor is not None
            and registered < registered_threshold
            and neighbor >= registered + minimum_neighbor_gain
        )
        record = {
            "length": length,
            "registered_accuracy": registered,
            "minus_two_accuracy": lower_neighbor,
            "plus_two_accuracy": upper_neighbor,
            "best_neighbor_accuracy": neighbor,
            "diseased": diseased,
        }
        scanned.append(record)
        if diseased:
            # The search interval is only the prospective diagnostic window.
            # The registered controller band extends beyond that window when
            # necessary (for example a failure at n=64 discovered while
            # scanning through n=100 defines a repair band through n=128).
            band_maximum = min(band_cap, int(round(length * band_multiplier)))
            if band_maximum <= length:
                raise RuntimeError("derived repair band is empty")
            return {
                "status": "disease_detected",
                "protocol_id": "paper2027.parity.p2.prospective_boundary.v1",
                "first_diseased_length": length,
                "training_range": [length, band_maximum],
                "controller_ood_definition": f"n>{band_maximum}",
                "rule": {
                    "search_range": [minimum_length, maximum_length],
                    "registered_threshold": registered_threshold,
                    "neighbor_offset_calls": 2,
                    "minimum_neighbor_gain": minimum_neighbor_gain,
                    "band_multiplier": band_multiplier,
                    "band_cap": band_cap,
                },
                "trigger": record,
                "scanned": scanned,
            }
    return {
        "status": "no_disease_detected",
        "protocol_id": "paper2027.parity.p2.prospective_boundary.v1",
        "rule": {
            "search_range": [minimum_length, maximum_length],
            "registered_threshold": registered_threshold,
            "neighbor_offset_calls": 2,
            "minimum_neighbor_gain": minimum_neighbor_gain,
            "band_multiplier": band_multiplier,
            "band_cap": band_cap,
        },
        "scanned": scanned,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatmap-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-length", type=int, default=21)
    parser.add_argument("--maximum-length", type=int, default=200)
    parser.add_argument("--registered-threshold", type=float, default=0.90)
    parser.add_argument("--minimum-neighbor-gain", type=float, default=0.10)
    parser.add_argument("--band-multiplier", type=float, default=2.0)
    parser.add_argument("--band-cap", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.heatmap_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = choose_boundary(
        rows,
        minimum_length=args.minimum_length,
        maximum_length=args.maximum_length,
        registered_threshold=args.registered_threshold,
        minimum_neighbor_gain=args.minimum_neighbor_gain,
        band_multiplier=args.band_multiplier,
        band_cap=args.band_cap,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
