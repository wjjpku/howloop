#!/usr/bin/env python3
"""Render the appendix-only matched D8L6 one-versus-two stage result.

The unit of variation in this experiment is a controller fit on one frozen
backbone.  The plot therefore shows both fits directly and deliberately does
not attach a multi-backbone error bar or make a population claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fraction_row(path: Path, *, mode: str, readout: str) -> dict[str, float]:
    rows = [
        row
        for row in read_csv(path)
        if row["mode"] == mode and row["readout"] == readout
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {mode}/{readout} row in {path}")
    row = rows[0]
    return {
        key: float(row[f"{key}_fraction"])
        for key in ("endpoint", "one", "two", "other")
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hop in (1, 2):
        for replica in (1, 2):
            source = root / "evaluation" / f"hop{hop}_seed{replica}" / "aggregate.csv"
            record = {
                "hop": hop,
                "replica": replica,
                "source": str(source),
                "pre": fraction_row(source, mode="full", readout="pre_executor"),
                "post": fraction_row(source, mode="full", readout="post_executor"),
                "shuffle_post": fraction_row(
                    source, mode="batch_shuffle", readout="post_executor"
                ),
            }
            rows.append(record)
    return rows


def render(records: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.55), constrained_layout=True)
    categories = ("endpoint", "one", "two", "other")
    colors = {"endpoint": "#1f77b4", "one": "#2ca02c", "two": "#ff7f0e", "other": "#7f7f7f"}

    labels = [f"J{row['hop']}, fit {row['replica']}" for row in records]
    x = np.arange(len(records))
    base = np.zeros(len(records))
    for category in categories:
        values = np.asarray([row["post"][category] for row in records])
        axes[0].bar(x, values, bottom=base, label=category, color=colors[category])
        base += values
    axes[0].set(
        title="Post-executor top-1 distribution",
        xticks=x,
        xticklabels=labels,
        ylabel="fraction on collision-free 8-cycles",
        ylim=(0, 1.03),
    )
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    x2 = np.arange(2)
    for index, row in enumerate(records):
        color = "#2ca02c" if row["hop"] == 1 else "#ff7f0e"
        axes[1].scatter(
            [index // 2 - 0.13, index // 2 + 0.13],
            [row["pre"]["two"], row["post"]["two"]],
            color=color,
            s=45,
            zorder=3,
        )
    axes[1].set(
        title="Two-hop target before and after F",
        xticks=x2,
        xticklabels=("one-hop objective", "two-hop objective"),
        ylabel=r"fraction decoded as $f^2(c)$",
        ylim=(-0.03, 1.03),
    )
    axes[1].axhline(0, color="black", linewidth=.6)
    axes[1].grid(axis="y", alpha=.2)
    axes[1].text(
        .02,
        .03,
        "Each pair: pre-F (left), post-F (right)\ncolour: trained target",
        transform=axes[1].transAxes,
        fontsize=7.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=260)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("D8L6 S1 root is not complete")
    records = collect(args.root)
    output = args.out_dir / "d8l6_s1_matched_stage_selection.png"
    render(records, output)
    summary = {
        "status": "complete",
        "protocol_id": manifest["protocol_id"],
        "backbone_scope": "one frozen D8L6 backbone; controller fits are not backbone replications",
        "records": records,
        "figure": str(output),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
