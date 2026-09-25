from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/jiaju/Documents")
PAPER_ROOT = ROOT / "looped transformer的返老回童药"
LOOPLUS_ROOT = ROOT / "github" / "LooPlus"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    phase_path = (
        PAPER_ROOT
        / "results/parity_diagonal_band_n500_20260804/ridge_slope/parity_ridge_phase_by_length.csv"
    )
    slope_path = (
        PAPER_ROOT
        / "results/parity_diagonal_band_n500_20260804/ridge_slope/parity_ridge_slope_analysis.json"
    )
    transfer_path = LOOPLUS_ROOT / "results/parity_circuit_20260807/source_answer_transfer.csv"
    skip_path = LOOPLUS_ROOT / "results/parity_circuit_20260807/mlp_skip.csv"
    svd_path = (
        LOOPLUS_ROOT
        / "results/parity_j_hidden_effect_20260803/top4_modes_local_mps/variant_summary.csv"
    )
    output_path = PAPER_ROOT / "paper_mechanism_first_20260807/figures/parity_phase_circuit.pdf"

    phase_rows = [
        row for row in read_csv(phase_path) if row["metric"] == "exact_match"
    ]
    slope = json.loads(slope_path.read_text(encoding="utf-8"))["metrics"]["exact_match"]
    transfer_rows = [
        row
        for row in read_csv(transfer_path)
        if int(row["length"]) == 20 and int(row["bit_position"]) == 0
    ]
    skip_rows = [row for row in read_csv(skip_path) if int(row["skipped_loop"]) > 0]
    svd_rows = [row for row in read_csv(svd_path) if int(row["length"]) == 100]

    figure, axes = plt.subplots(2, 2, figsize=(7.05, 5.75))
    colors = {"raw": "#d97706", "J": "#0f766e"}

    axis = axes[0, 0]
    for variant in ("raw", "J"):
        selected = [row for row in phase_rows if row["variant"] == variant]
        x = np.asarray([float(row["length_n"]) for row in selected])
        y = np.asarray([float(row["unwrapped_offset_t_minus_n"]) for row in selected])
        equation = slope[variant]["fitted_equation_t_from_n"]
        drift = float(equation["t_per_n_slope"]) - 1.0
        intercept = float(equation["intercept"])
        axis.scatter(x, y, s=3.0, alpha=0.28, color=colors[variant])
        axis.plot(x, drift * x + intercept, color=colors[variant], linewidth=1.6, label=variant)
    axis.axhline(0, color="#6b7280", linewidth=0.7, linestyle="--")
    axis.set(xlabel="input length $n$", ylabel="ridge offset $t-n$")
    axis.set_title("A  $J$ locks the recurrent phase", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.text(0.03, 0.94, "raw: 1.548 loops / 100 tokens\n$J$: 0.085 loops / 100 tokens", transform=axis.transAxes, va="top", fontsize=7.2)

    axis = axes[0, 1]
    for site, label, color in (
        ("source_token", "patch source", "#d97706"),
        ("answer_token", "patch answer", "#2563eb"),
    ):
        selected = sorted(
            (row for row in transfer_rows if row["site"] == site),
            key=lambda row: int(row["loop"]),
        )
        axis.plot(
            [int(row["loop"]) for row in selected],
            [float(row["recovery"]) for row in selected],
            marker="o",
            markersize=2.6,
            linewidth=1.4,
            color=color,
            label=label,
        )
    axis.set(xlabel="patch loop", ylabel="normalized causal recovery", ylim=(-0.03, 1.03))
    axis.set_title("B  State moves from source to answer", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8, loc="center right")

    axis = axes[1, 0]
    for eval_loop, label, color in (
        (20, "read at $T$", "#d97706"),
        (21, "read at $T+1$", "#059669"),
    ):
        selected = sorted(
            (row for row in skip_rows if int(row["eval_loop"]) == eval_loop),
            key=lambda row: int(row["skipped_loop"]),
        )
        axis.plot(
            [int(row["skipped_loop"]) for row in selected],
            [float(row["accuracy"]) for row in selected],
            marker="o",
            markersize=2.5,
            linewidth=1.3,
            color=color,
            label=label,
        )
    axis.set(xlabel="skipped shared-MLP call", ylabel="parity-token accuracy", ylim=(-0.03, 1.03))
    axis.set_title("C  Skipping the MLP delays one loop", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8, loc="lower right")

    axis = axes[1, 1]
    indexed = {row["variant"]: row for row in svd_rows}
    random_write = [
        row for name, row in indexed.items() if name.startswith("random_output_top4")
    ]
    entries = [
        ("raw", float(indexed["raw_no_J"]["target_exact_match"]), float(indexed["raw_no_J"]["target_plus_one_exact_match"])),
        ("full $J$", float(indexed["full_J"]["target_exact_match"]), float(indexed["full_J"]["target_plus_one_exact_match"])),
        ("top-4", float(indexed["top4"]["target_exact_match"]), float(indexed["top4"]["target_plus_one_exact_match"])),
        ("delete\ntop-4", float(indexed["delete_top4"]["target_exact_match"]), float(indexed["delete_top4"]["target_plus_one_exact_match"])),
        ("random\nwrite", float(np.mean([float(row["target_exact_match"]) for row in random_write])), float(np.mean([float(row["target_plus_one_exact_match"]) for row in random_write]))),
    ]
    x = np.arange(len(entries))
    width = 0.36
    axis.bar(x - width / 2, [entry[1] for entry in entries], width, color="#2563eb", label="$T$")
    axis.bar(x + width / 2, [entry[2] for entry in entries], width, color="#9ca3af", label="$T+1$")
    axis.set_xticks(x, [entry[0] for entry in entries], fontsize=7)
    axis.set(ylabel="exact match at length 100", ylim=(0, 1.05))
    axis.set_title("D  Four oriented modes reproduce $J$", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")

    for axis in axes.ravel():
        axis.grid(axis="y", color="#d1d5db", linewidth=0.45, alpha=0.75)
        axis.tick_params(labelsize=7.5)
        axis.title.set_fontsize(8.6)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    figure.tight_layout(pad=0.9, h_pad=1.0, w_pad=1.0)
    figure.savefig(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
