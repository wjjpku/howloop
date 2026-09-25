#!/usr/bin/env python3
"""Render compact appendix-only P4 mechanism diagnostics from audited CSVs.

The figures intentionally separate a causal component comparison (SVD variants)
from descriptive local measurements (phase advance and hidden-state correction).
No historical controller or image can enter these outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def phase_and_hidden(p4_root: Path, output: Path) -> None:
    phase = read_json(p4_root / "four_phase" / "summary.json")
    hidden = read_csv(p4_root / "hidden_effect" / "step_metrics.csv")
    progress = read_csv(p4_root / "four_phase" / "component_phase_progress.csv")
    dynamics = phase["dynamics"]["controlled"]["heldout_evaluation"]
    chosen = [row for row in progress if row["variant"] == "controlled"]
    stages = [row["stage"] for row in chosen]
    advances = [float(row["mean_signed_progress_along_full_phase_step"]) for row in chosen]
    answer = [row for row in hidden if row["scope"] == "answer_tokens"]
    by_rel: dict[int, list[float]] = {}
    for row in answer:
        by_rel.setdefault(int(row["relative_to_target"]), []).append(float(row["correction_to_source"]))
    rel = sorted(by_rel)
    values = [float(np.mean(by_rel[item])) for item in rel]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), constrained_layout=True)
    axes[0].bar(range(len(stages)), advances, color=["#0072B2" if x == "MLP" else "#999999" for x in stages])
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set(xticks=range(len(stages)), xticklabels=stages, ylabel="signed progress along full phase step")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].set_title("Within-call phase advance")
    axes[1].plot(rel, values, marker="o", color="#D55E00")
    axes[1].set(xlabel="step minus registered target", ylabel=r"$\|J(h)-h\|/\|h\|$", title="Answer-token correction magnitude")
    axes[1].grid(alpha=.22)
    fig.suptitle(
        f"P4: held-out phase period {dynamics['polar_rotation_period_calls']:.2f}; "
        f"four-step relative error {dynamics['scaled_four_step_relative_error']:.2f}",
        fontsize=10,
    )
    save(fig, output)


def svd_variants(p4_root: Path, output: Path) -> None:
    rows = read_csv(p4_root / "svd" / "variant_summary.csv")
    preferred = [
        "raw_no_J", "full_J", "top1", "top2", "top4", "top8",
        "delete_top4", "no_AB", "no_bias", "random_top4_s314159",
    ]
    labels = [label for label in preferred if any(row["variant"] == label for row in rows)]
    lengths = sorted({int(row["length"]) for row in rows})
    values = np.full((len(labels), len(lengths)), np.nan)
    for i, label in enumerate(labels):
        for j, length in enumerate(lengths):
            matching = [row for row in rows if row["variant"] == label and int(row["length"]) == length]
            if matching:
                values[i, j] = float(matching[0]["post_target_accuracy_auc_0_to_8"])
    fig, axis = plt.subplots(figsize=(7.3, 4.0), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set(
        xticks=np.arange(len(lengths)), xticklabels=[str(length) for length in lengths],
        yticks=np.arange(len(labels)), yticklabels=labels,
        xlabel="logical length", ylabel="pre-registered matrix intervention",
        title="Post-target accuracy AUC (P4; causal controller variants)",
    )
    fig.colorbar(image, ax=axis, label="accuracy")
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p4-root", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    required = [
        args.p4_root / "four_phase" / "summary.json",
        args.p4_root / "four_phase" / "component_phase_progress.csv",
        args.p4_root / "hidden_effect" / "step_metrics.csv",
        args.p4_root / "svd" / "variant_summary.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("P4 figure inputs missing: " + ", ".join(missing))
    phase_and_hidden(args.p4_root, args.figure_dir / "parity_p4_phase_and_hidden.png")
    svd_variants(args.p4_root, args.figure_dir / "parity_p4_svd_variants.png")


if __name__ == "__main__":
    main()
