from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


def _group_name(job: str) -> str:
    if "_seed" not in job:
        raise ValueError(f"unrecognized job label: {job}")
    return job.rsplit("_seed", 1)[0]


def _seed(job: str) -> int:
    return int(job.rsplit("_seed", 1)[1])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    screen_root: Path,
    id_screen_root: Path,
    out_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary_path in sorted(screen_root.glob("*/summary.json")):
        job = summary_path.parent.name
        payload = json.loads(summary_path.read_text())
        raw_rows = [row for row in payload["rows"] if row["variant"] == "raw"]
        if len(raw_rows) != 1 or int(raw_rows[0]["length"]) != 10:
            raise ValueError(f"{summary_path} is not the n=10 carry screen")
        metric = raw_rows[0]
        id_payload = json.loads(
            (id_screen_root / job / "summary.json").read_text()
        )
        id_rows = [
            row for row in id_payload["rows"] if row["variant"] == "raw"
        ]
        if [int(row["length"]) for row in id_rows] != list(range(1, 11)):
            raise ValueError(f"{job} is missing the carry ID curve 1..10")
        row = {
            "group": _group_name(job),
            "job": job,
            "seed": _seed(job),
            "balanced_carry_accuracy_n10": float(metric["exact_match"]),
            "balanced_carry_ce_n10": float(metric["answer_cross_entropy"]),
            "mean_margin_n10": float(metric["mean_sequence_min_margin"]),
            "mean_balanced_carry_accuracy_id1to10": mean(
                float(item["exact_match"]) for item in id_rows
            ),
            "minimum_balanced_carry_accuracy_id1to10": min(
                float(item["exact_match"]) for item in id_rows
            ),
            "examples": int(metric["examples"]),
            "checkpoint": payload["checkpoint"],
        }
        rows.append(row)
        grouped[row["group"]].append(row)

    group_rows: list[dict[str, Any]] = []
    eligible_jobs: list[str] = []
    for group, members in sorted(grouped.items()):
        seeds = sorted(int(row["seed"]) for row in members)
        complete = seeds == [0, 1, 2]
        accuracies = [float(row["balanced_carry_accuracy_n10"]) for row in members]
        eligible = complete and min(accuracies) >= threshold
        if eligible:
            eligible_jobs.extend(str(row["job"]) for row in members)
        group_rows.append(
            {
                "group": group,
                "seeds": "/".join(map(str, seeds)),
                "complete_three_seed_group": complete,
                "mean_balanced_carry_accuracy_n10": mean(accuracies),
                "minimum_balanced_carry_accuracy_n10": min(accuracies),
                "maximum_balanced_carry_accuracy_n10": max(accuracies),
                "eligible_for_carry_j": eligible,
                "eligibility_threshold": threshold,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_seed.csv", rows)
    _write_csv(out_dir / "per_group.csv", group_rows)
    result = {
        "status": "complete",
        "screen_root": str(screen_root),
        "eligibility_rule": (
            "all three seeds have balanced final-carry accuracy at n=10 "
            f">= {threshold:.3f} under registered T(n)=n+1"
        ),
        "claim_boundary": (
            "Eligibility establishes a learned final-carry subcircuit only; "
            "it does not establish the full Addition algorithm."
        ),
        "eligible_groups": [
            row["group"] for row in group_rows if row["eligible_for_carry_j"]
        ],
        "eligible_jobs": sorted(eligible_jobs),
        "per_seed": rows,
        "per_group": group_rows,
    }
    (out_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    report_lines = [
        "# Addition balanced final-carry screen",
        "",
        result["eligibility_rule"] + ".",
        "",
        "| group | mean | min | max | eligible |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in group_rows:
        report_lines.append(
            f"| {row['group']} | {row['mean_balanced_carry_accuracy_n10']:.4f} "
            f"| {row['minimum_balanced_carry_accuracy_n10']:.4f} "
            f"| {row['maximum_balanced_carry_accuracy_n10']:.4f} "
            f"| {'yes' if row['eligible_for_carry_j'] else 'no'} |"
        )
    report_lines.extend(["", "Boundary: " + result["claim_boundary"], ""])
    (out_dir / "REPORT.md").write_text("\n".join(report_lines))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--id-screen-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.98)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        json.dumps(
            aggregate(
                args.screen_root,
                args.id_screen_root,
                args.out_dir,
                args.threshold,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
