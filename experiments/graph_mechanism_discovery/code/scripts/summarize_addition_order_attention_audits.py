#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_audit(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("audits must use LABEL=DIRECTORY")
    return label, Path(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", type=parse_audit, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows_by_system: dict[str, list[dict[str, str]]] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for label, directory in args.audit:
        rows = read_csv(directory / "audit_trajectory.csv")
        rows_by_system[label] = rows
        source_summaries[label] = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        for row in rows:
            if int(row["step"]) == int(row["target_step"]):
                endpoint_rows.append(
                    {
                        "system": label,
                        "variant": row["variant"],
                        "length": int(row["length"]),
                        "target_step": int(row["target_step"]),
                        "exact_match": float(row["exact_match"]),
                        "answer_nll": float(row["answer_nll"]),
                        "answer_predictive_entropy": float(
                            row["answer_predictive_entropy"]
                        ),
                    }
                )
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in endpoint_rows:
            if row["system"] != label:
                continue
            split = "ID_1_10" if int(row["length"]) <= 10 else "OOD_11_20"
            grouped[(str(row["variant"]), split)].append(float(row["exact_match"]))
        for (variant, split), values in sorted(grouped.items()):
            aggregate_rows.append(
                {
                    "system": label,
                    "variant": variant,
                    "split": split,
                    "mean_target_exact_match": float(np.mean(values)),
                    "minimum_target_exact_match": float(np.min(values)),
                    "maximum_target_exact_match": float(np.max(values)),
                    "length_count": len(values),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "target_endpoint_by_length.csv", endpoint_rows)
    write_csv(args.out_dir / "target_endpoint_split_summary.csv", aggregate_rows)

    systems = [label for label, _ in args.audit]
    figure, axes = plt.subplots(
        len(systems), 3, figsize=(17, 4.5 * len(systems)), constrained_layout=True, squeeze=False
    )
    for index, label in enumerate(systems):
        system_endpoints = [row for row in endpoint_rows if row["system"] == label]
        variants = ("raw", "full", "no_AB", "identity_D", "full_executor_off")
        colors = {
            "raw": "black",
            "full": "tab:blue",
            "no_AB": "tab:orange",
            "identity_D": "tab:green",
            "full_executor_off": "tab:red",
        }
        axis = axes[index, 0]
        for variant in ("raw", "full"):
            selected = sorted(
                (row for row in system_endpoints if row["variant"] == variant),
                key=lambda row: int(row["length"]),
            )
            axis.plot(
                [row["length"] for row in selected],
                [row["exact_match"] for row in selected],
                marker="o",
                label=variant,
                color=colors[variant],
            )
        axis.axvline(10.5, color="gray", linestyle="--", linewidth=1)
        axis.set(
            title=f"{label}: target-step EM",
            xlabel="logical length",
            ylabel="exact match",
            ylim=(-0.03, 1.03),
        )
        axis.legend()

        axis = axes[index, 1]
        by_key = {
            (str(row["variant"]), int(row["length"])): float(row["exact_match"])
            for row in system_endpoints
        }
        lengths = sorted({int(row["length"]) for row in system_endpoints})
        gain = [by_key[("full", length)] - by_key[("raw", length)] for length in lengths]
        axis.axhline(0.0, color="black", linewidth=1)
        axis.bar(lengths, gain, color=["tab:blue" if value >= 0 else "tab:red" for value in gain])
        axis.axvline(10.5, color="gray", linestyle="--", linewidth=1)
        axis.set(
            title="J gain at registered T(n)",
            xlabel="logical length",
            ylabel="full J EM - raw EM",
        )

        axis = axes[index, 2]
        n10 = [row for row in rows_by_system[label] if int(row["length"]) == 10]
        for variant in variants:
            selected = sorted(
                (row for row in n10 if row["variant"] == variant),
                key=lambda row: int(row["step"]),
            )
            axis.plot(
                [int(row["step"]) for row in selected],
                [float(row["exact_match"]) for row in selected],
                label=variant,
                color=colors[variant],
                linewidth=2 if variant in {"raw", "full"} else 1.2,
            )
        axis.axvline(11, color="gray", linestyle="--", linewidth=1)
        axis.set(
            title="n=10 loop trajectory",
            xlabel="readout loop",
            ylabel="exact match",
            ylim=(-0.03, 1.03),
        )
        axis.legend(fontsize=8)
    figure.suptitle("Addition order/attention controls: raw vs Diag+LoRA J")
    figure.savefig(args.out_dir / "audit_comparison.png", dpi=190)
    figure.savefig(args.out_dir / "audit_comparison.pdf")
    plt.close(figure)

    summary = {
        "status": "complete",
        "systems": systems,
        "row_vector_controller_convention": "J(h)=h@diag(D)+(h@A)@B+b",
        "split_summary": aggregate_rows,
        "source_summaries": source_summaries,
        "claim_boundary": (
            "Endpoint gains are behavioral. Component ablations and executor-off "
            "separate controller dependence from executor replacement, but do not "
            "identify a minimal circuit."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
