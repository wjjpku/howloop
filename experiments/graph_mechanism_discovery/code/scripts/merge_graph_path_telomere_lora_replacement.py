#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-artifact", type=Path, required=True)
    parser.add_argument("--replacement-artifact", type=Path, required=True)
    parser.add_argument("--main-training", type=Path, required=True)
    parser.add_argument("--replacement-training", type=Path, required=True)
    parser.add_argument("--out-artifact", type=Path, required=True)
    parser.add_argument("--out-training", type=Path, required=True)
    args = parser.parse_args()

    main = torch.load(args.main_artifact, map_location="cpu", weights_only=False)
    replacement = torch.load(
        args.replacement_artifact, map_location="cpu", weights_only=False
    )
    for key in ("kind", "checkpoint", "positions", "placement"):
        if main.get(key) != replacement.get(key):
            raise ValueError(f"artifacts disagree on {key}")
    labels = set(replacement["modules"])
    if not labels <= set(main["modules"]):
        raise ValueError("replacement contains labels absent from main artifact")
    merged = dict(main)
    merged["modules"] = dict(main["modules"])
    merged["modules"].update(replacement["modules"])
    merged["replacement_note"] = {
        "labels": sorted(labels),
        "reason": "rank-2 lr30 variants had non-finite CE; replaced by lr10 curriculum variants",
        "source": str(args.replacement_artifact),
    }
    args.out_artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_artifact.with_suffix(args.out_artifact.suffix + ".tmp")
    torch.save(merged, temporary)
    os.replace(temporary, args.out_artifact)

    main_rows = read_rows(args.main_training)
    replacement_rows = read_rows(args.replacement_training)
    kept_rows = [row for row in main_rows if row.get("variant") not in labels]
    write_rows(args.out_training, kept_rows + replacement_rows)
    print(f"replaced {len(labels)} variants: {sorted(labels)}")


if __name__ == "__main__":
    main()
