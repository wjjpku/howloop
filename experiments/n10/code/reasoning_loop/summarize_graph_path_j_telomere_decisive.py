"""Summarize the decisive centered-residual, transplant, and stagewise tests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centered-dir", type=Path, required=True)
    parser.add_argument("--transplant-dir", type=Path, required=True)
    parser.add_argument("--stagewise-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def weighted_transplant_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    keys = ("back_count", "rollback_checkpoint", "condition")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    metrics = (
        "accuracy",
        "margin",
        "survival_steps",
        "forward_accuracy_auc",
        "first_destination_mass",
        "delta_rms",
    )
    output: list[dict[str, Any]] = []
    for group, parts in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        weights = np.asarray([float(part["matched_examples"]) for part in parts])
        result: dict[str, Any] = dict(zip(keys, group, strict=True))
        result["matched_examples"] = int(weights.sum())
        result["matched_event_rows"] = len(parts)
        for metric in metrics:
            values = np.asarray([float(part[metric]) for part in parts])
            result[f"{metric}_weighted_mean"] = float(np.average(values, weights=weights))
            seed_values = []
            for seed in sorted({part["graph_seed"] for part in parts}):
                selected = [part for part in parts if part["graph_seed"] == seed]
                local_weights = np.asarray(
                    [float(part["matched_examples"]) for part in selected]
                )
                local_values = np.asarray([float(part[metric]) for part in selected])
                seed_values.append(float(np.average(local_values, weights=local_weights)))
            result[f"{metric}_seed_sem"] = (
                float(np.std(seed_values, ddof=1) / np.sqrt(len(seed_values)))
                if len(seed_values) > 1
                else 0.0
            )
        output.append(result)
    return output


def paired_transplant_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    identity = (
        "graph_seed",
        "batch",
        "back_count",
        "word",
        "rollback_checkpoint",
        "source_age",
        "target_age",
    )
    lookup = {
        (tuple(row[key] for key in identity), row["condition"]): row for row in rows
    }
    pairs = {
        "late_bottom_excess_clean0p5": "late_baseline",
        "late_bottom_excess_clean1p0": "late_baseline",
        "late_random0_excess_clean1p0_statematch": "late_baseline",
        "late_random1_excess_clean1p0_statematch": "late_baseline",
        "late_top_excess_clean1p0_statematch": "late_baseline",
        "young_bottom_excess_inject0p5": "young_reference",
        "young_bottom_excess_inject1p0": "young_reference",
        "young_random0_excess_inject1p0_statematch": "young_reference",
        "young_random1_excess_inject1p0_statematch": "young_reference",
    }
    metrics = (
        "accuracy",
        "margin",
        "survival_steps",
        "forward_accuracy_auc",
        "first_destination_mass",
    )
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for (row_id, condition), row in lookup.items():
        if condition not in pairs:
            continue
        baseline = lookup.get((row_id, pairs[condition]))
        if baseline is None:
            continue
        grouped[(row["back_count"], row["rollback_checkpoint"], condition)].append(
            (row, baseline)
        )
    output: list[dict[str, Any]] = []
    for (back_count, checkpoint, condition), parts in sorted(grouped.items()):
        weights = np.asarray([float(row["matched_examples"]) for row, _ in parts])
        result: dict[str, Any] = {
            "back_count": int(back_count),
            "rollback_checkpoint": int(checkpoint),
            "condition": condition,
            "baseline_condition": pairs[condition],
            "matched_examples": int(weights.sum()),
            "paired_event_rows": len(parts),
        }
        for metric in metrics:
            deltas = np.asarray(
                [float(row[metric]) - float(base[metric]) for row, base in parts]
            )
            result[f"{metric}_delta_weighted_mean"] = float(
                np.average(deltas, weights=weights)
            )
            seed_values = []
            for seed in sorted({row["graph_seed"] for row, _ in parts}):
                selected = [pair for pair in parts if pair[0]["graph_seed"] == seed]
                local_weights = np.asarray(
                    [float(row["matched_examples"]) for row, _ in selected]
                )
                local_deltas = np.asarray(
                    [float(row[metric]) - float(base[metric]) for row, base in selected]
                )
                seed_values.append(
                    float(np.average(local_deltas, weights=local_weights))
                )
            result[f"{metric}_delta_seed_sem"] = (
                float(np.std(seed_values, ddof=1) / np.sqrt(len(seed_values)))
                if len(seed_values) > 1
                else 0.0
            )
        output.append(result)
    return output


def stagewise_comparison(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, float], list[dict[str, str]]] = defaultdict(list)
    baseline = {
        int(row["source_age"]): row for row in rows if row["condition"] == "baseline"
    }
    for row in rows:
        if row["mode"] in {"bottom", "random"}:
            groups[
                (int(row["source_age"]), int(row["rank"]), float(row["floor"]))
            ].append(row)
    output: list[dict[str, Any]] = []
    for (source_age, rank, floor), parts in sorted(groups.items()):
        bottom = next(part for part in parts if part["mode"] == "bottom")
        random = [part for part in parts if part["mode"] == "random"]
        base = baseline[source_age]
        result: dict[str, Any] = {
            "source_age": source_age,
            "J_index": source_age - 1,
            "target_age": source_age - 1,
            "rank": rank,
            "floor": floor,
            "random_draws": len(random),
        }
        for metric in (
            "post_J_age_signed_error_mean",
            "post_J_current_accuracy_mean",
            "next_F_accuracy_mean",
            "final_H8_accuracy_mean",
            "final_margin_mean",
        ):
            baseline_value = float(base[metric])
            bottom_value = float(bottom[metric])
            random_value = float(np.mean([float(part[metric]) for part in random]))
            result[f"baseline_{metric}"] = baseline_value
            result[f"bottom_{metric}"] = bottom_value
            result[f"random_{metric}"] = random_value
            result[f"bottom_delta_{metric}"] = bottom_value - baseline_value
            result[f"random_delta_{metric}"] = random_value - baseline_value
        output.append(result)
    return output


def plot_summary(
    *,
    centered: list[dict[str, str]],
    stagewise: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), dpi=180)
    for condition in (
        "baseline",
        "bottom_mean_shrink0p25",
        "bottom_linear_shrink0p25",
        "random0_linear_shrink0p25_statematch",
    ):
        selected = sorted(
            [row for row in centered if row["condition"] == condition],
            key=lambda row: int(row["back_count"]),
        )
        axes[0].plot(
            [int(row["back_count"]) for row in selected],
            [float(row["accuracy_mean"]) for row in selected],
            marker="o",
            label=condition,
        )
    axes[0].set(title="Centered residual cleaning", xlabel="J calls", ylabel="final accuracy", ylim=(0, 1.03))

    selected_stage = [
        row for row in stagewise if row["rank"] == 8 and row["floor"] == 0.5
    ]
    x = [int(row["J_index"]) for row in selected_stage]
    axes[1].plot(
        x,
        [float(row["bottom_delta_post_J_age_signed_error_mean"]) for row in selected_stage],
        "o-",
        label="bottom-8",
    )
    axes[1].plot(
        x,
        [float(row["random_delta_post_J_age_signed_error_mean"]) for row in selected_stage],
        "s--",
        label="random state-match",
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(title="Direct one-J age contamination", xlabel="J index", ylabel="age error delta")

    selected_pairs = [
        row
        for row in paired
        if row["condition"] in {
            "late_bottom_excess_clean1p0",
            "young_bottom_excess_inject1p0",
        }
    ]
    labels = [
        f"k{row['back_count']}/r{row['rollback_checkpoint']}\n{row['condition'].split('_')[0]}"
        for row in selected_pairs
    ]
    axes[2].bar(
        np.arange(len(selected_pairs)),
        [float(row["survival_steps_delta_weighted_mean"]) for row in selected_pairs],
    )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    axes[2].set(title="Late clean / young inject", ylabel="remaining-step delta")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis is not axes[2]:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    centered = read_csv(args.centered_dir / "behavior_summary.csv")
    transplant_raw = read_csv(args.transplant_dir / "branch_rows.csv")
    stagewise_raw = read_csv(args.stagewise_dir / "stagewise_summary.csv")
    transplant = weighted_transplant_summary(transplant_raw)
    paired = paired_transplant_deltas(transplant_raw)
    stagewise = stagewise_comparison(stagewise_raw)
    write_csv(args.out_dir / "centered_residual_summary.csv", centered)
    write_csv(args.out_dir / "transplant_weighted_summary.csv", transplant)
    write_csv(args.out_dir / "transplant_paired_deltas.csv", paired)
    write_csv(args.out_dir / "stagewise_comparison.csv", stagewise)
    plot_summary(
        centered=centered,
        stagewise=stagewise,
        paired=paired,
        path=args.out_dir / "telomere_decisive_summary.png",
    )
    summary = {
        "status": "complete",
        "centered_dir": str(args.centered_dir),
        "transplant_dir": str(args.transplant_dir),
        "stagewise_dir": str(args.stagewise_dir),
        "transplant_weighting": (
            "matched-example weighted means; SEM computed across graph-seed weighted means"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
