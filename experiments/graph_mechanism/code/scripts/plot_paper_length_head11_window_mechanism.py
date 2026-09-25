#!/usr/bin/env python3
"""Render the seed-0 J20/J40 head-11 recurrence-window causal audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j20", type=Path, required=True)
    parser.add_argument("--j40", type=Path, required=True)
    parser.add_argument("--out-stem", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = {
        "J20": json.loads(args.j20.read_text(encoding="utf-8")),
        "J40": json.loads(args.j40.read_text(encoding="utf-8")),
    }
    for name, payload in payloads.items():
        if payload["status"] != "complete" or payload["examples"] != 128:
            raise ValueError(f"{name} is not the registered N=128 result")
        if payload["candidate_head"] != 11 or payload["control_head"] != 15:
            raise ValueError(f"{name} does not use the registered heads")

    rows = []
    for controller, payload in payloads.items():
        for row in payload["window_rows"]:
            rows.append(
                {
                    "controller": controller,
                    "window": f"{row['start_loop']}--{row['end_loop']}",
                    **row,
                }
            )
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        f"{row['start_loop']}–{row['end_loop']}"
        for row in payloads["J20"]["window_rows"]
    ]
    x = list(range(len(labels)))
    colors = {"J20": "#2F6B9A", "J40": "#D17A22"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)

    for controller, payload in payloads.items():
        window_rows = payload["window_rows"]
        axes[0].plot(
            x,
            [row["candidate_ablation_em_drop"] for row in window_rows],
            marker="o",
            linewidth=2.2,
            color=colors[controller],
            label=f"{controller}, head 11",
        )
        axes[0].plot(
            x,
            [row["control_ablation_em_drop"] for row in window_rows],
            marker="x",
            linewidth=1.5,
            linestyle="--",
            color=colors[controller],
            alpha=0.75,
            label=f"{controller}, head 15 control",
        )
        axes[1].plot(
            x,
            [row["candidate_patch_em_gain"] for row in window_rows],
            marker="o",
            linewidth=2.2,
            color=colors[controller],
            label=f"{controller}, head 11",
        )
        axes[1].plot(
            x,
            [row["control_patch_em_gain"] for row in window_rows],
            marker="x",
            linewidth=1.5,
            linestyle="--",
            color=colors[controller],
            alpha=0.75,
            label=f"{controller}, head 15 control",
        )

    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].set_title("Ablation: loss from full-J endpoint EM")
    axes[0].set_ylabel("EM drop at loop 100")
    axes[0].set_ylim(-0.5, 0.9)
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_title("Output transfer: gain over no-AB endpoint EM")
    axes[1].set_ylabel("EM gain at loop 100")
    axes[1].set_ylim(-0.05, 0.13)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Intervened recurrence window")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Seed-0 local causal audit (N=128): J20 and J40 repeatedly use head 11",
        fontsize=12,
    )
    fig.savefig(args.out_stem.with_suffix(".png"), dpi=220)
    fig.savefig(args.out_stem.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
